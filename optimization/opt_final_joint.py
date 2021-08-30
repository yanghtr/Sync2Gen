#!/usr/bin/env python
# coding=utf-8

import os
import ipdb
import copy
import time
import pickle
import argparse
import itertools
import numpy as np
import matplotlib.pyplot as plt
from functools import wraps
from loguru import logger

import torch
import torch.nn.functional as F
from torch import nn

import sys
sys.path.append('../')
sys.path.append('./LBFGS/functions/')
from LBFGS import FullBatchLBFGS
import fit_pxv
import fit_pxe
import utils
import overlap

np.set_printoptions(suppress=True, linewidth=200, threshold=sys.maxsize)
torch.set_printoptions(precision=2, linewidth=240, sci_mode=False)

def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        logger.debug('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper


def sigmoid(x, c, a=10):
    ''' use sigmoid to approximate step function
    '''
    if type(x) == np.ndarray:
        nh = np
    elif type(x) == torch.Tensor:
        nh = torch
    return 1 / (1 + nh.exp(-(x - c) * a))


def sigmoid_rescale(x, c=0.5, a=10, min_thres=0, max_thres=1):
    ''' map x to [min_thres, max_thres] with sigmoid shape
    '''
    if type(x) == np.ndarray:
        nh = np
    elif type(x) == torch.Tensor:
        nh = torch
    x_scale = (x - nh.min(x)) / (nh.max(x) - nh.min(x))
    x_scale = sigmoid(x_scale, c, a) * (max_thres - min_thres) + min_thres
    return x_scale


def GM(x, a):
    ''' Geman-Mcclure robust function
    '''
    xsquare = x * x
    return xsquare / (xsquare + a)


def normal(x, mu, sigma):
    x_normalize = (x - mu) / sigma
    y = (1 / (sigma * np.sqrt(2*np.pi))) * torch.exp(-0.5 * x_normalize * x_normalize)
    return y


def GMM_array(x, pi, mu, sigma):
    ''' ATTENTION: sigma == np.sqrt(covariance) !!
    @Args:
        x: (n,)
        pi/mu/sigma: (n, K)
    '''
    K = pi.shape[1]
    x = x.reshape(-1, 1).repeat(1, K) # (n, K)
    x_normalize = (x - mu) / sigma
    y = (1 / (sigma * np.sqrt(2*np.pi))) * torch.exp(-0.5 * x_normalize * x_normalize)
    px = torch.sum(y * pi, dim=1)
    return px


def pxv_cont_array(x, params):
    ''' Use GMM to fit discrete #objects distribution
    np.abs(pi): ensure > 0, sigma: square of sigma
    @Args:
        x: (C,) or (B, C)
        params: (C, 6), order: pi1, pi2, mu1, mu2, sigma1, sigma2
    '''
    return torch.abs(params[:, 0]) * torch.exp(- (x-params[:, 2]) * (x-params[:, 2]) / params[:, 4]) +\
           torch.abs(params[:, 1]) * torch.exp(- (x-params[:, 3]) * (x-params[:, 3]) / params[:, 5])


def pxe_cont_array(x, y, params):
    ''' Use GMM to fit discrete pair-wise #objects distribution
    @Args:
        x: (C,) or (B, C)
        y: (C,) or (B, C)
        params: (C, 13), order: A1, A2, A3, A4, mu1x, mu1y, mu2x, mu2y, mu3x, mu3y, mu4x, mu4y, a
    @Returns:
        g = A1 * torch.exp( - (a*((x-mu1x)*(x-mu1x)) + a*((y-mu1y)*(y-mu1y))) ) +\
            A2 * torch.exp( - (a*((x-mu2x)*(x-mu2x)) + a*((y-mu2y)*(y-mu2y))) ) +\
            A3 * torch.exp( - (a*((x-mu3x)*(x-mu3x)) + a*((y-mu3y)*(y-mu3y))) ) +\
            A4 * torch.exp( - (a*((x-mu4x)*(x-mu4x)) + a*((y-mu4y)*(y-mu4y))) )
    '''
    x1 = x - params[:, 4]
    x2 = x - params[:, 6]
    x3 = x - params[:, 8]
    x4 = x - params[:, 10]
    y1 = y - params[:, 5]
    y2 = y - params[:, 7]
    y3 = y - params[:, 9]
    y4 = y - params[:, 11]
    g = params[:, 0] * torch.exp( - ( params[:, 12]*x1*x1 + params[:, 12]*y1*y1 ) ) +\
        params[:, 1] * torch.exp( - ( params[:, 12]*x2*x2 + params[:, 12]*y2*y2 ) ) +\
        params[:, 2] * torch.exp( - ( params[:, 12]*x3*x3 + params[:, 12]*y3*y3 ) ) +\
        params[:, 3] * torch.exp( - ( params[:, 12]*x4*x4 + params[:, 12]*y4*y4 ) )
    return g


def global_rotation_align(rotation):
    '''
    @Args:
        rotation: (n, 2)
    @Returns:
        R_global: (2, 2). Rotate the whole scene to align with +x axis
    '''
    ros = rotation[:, :2] * np.sign(rotation[:, 0:1]) # make x positive
    ro_list = [] # all orientations are close to x axis
    for ro in ros:
        if np.abs(ro[1]) > np.abs(ro[0]):
            if ro[1] > 0:
                ro_list.append([ro[1], -ro[0]])
            else:
                ro_list.append([-ro[1], ro[0]])
        else:
            ro_list.append([ro[0], ro[1]])
    roa = np.array(ro_list)
    roa = roa[roa[:, 1].argsort()]
    ro_global = np.median(roa, axis=0) # robust
    theta = np.arcsin(ro_global[1])
    # 2 cases: orientation above/below x axis
    if theta > 0:
        theta = np.pi * 2 - theta
    R_global = np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    return R_global


class Solver_Xv(nn.Module):
    def __init__(self, hyper_params_dict, xv0, tn_rel_all0, pxv_params, pxe_params, coexist_pairs, num_class, num_each_class, use_cuda=False):
        '''
        @Args:
            xv0: (N,)
            pxv_params: (NUM_CLASS, 6)
            pxe_params: (NUM_CLASS, NUM_CLASS, 13)
        '''
        super(Solver_Xv, self).__init__()
        self.num_class = num_class
        self.num_each_class = num_each_class
        self.max_parts = num_class * num_each_class

        self.pxv_params = pxv_params
        self.pxe_params = pxe_params
        self.coexist_pairs = coexist_pairs

        if use_cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.xv0 = xv0
        self.tn_rel_all0 = tn_rel_all0 
        xv0_clip = torch.clamp(xv0, min=1e-6, max=1-1e-6)
        self.xv_free = nn.Parameter(torch.log(1/xv0_clip.clone() - 1)) # initialize xv_free such that self.xv is initialized to xv0
        self.xv = None

        self.sigma_cv = hyper_params_dict['func.joint_opt_func_xv.sigma_cv']
        self.alpha_cv = hyper_params_dict['func.joint_opt_func_xv.alpha_cv']
        self.sigma_ce = hyper_params_dict['func.joint_opt_func_xv.sigma_ce']
        self.alpha_ce = hyper_params_dict['func.joint_opt_func_xv.alpha_ce']

    # @timing
    def forward(self, tn_abs_all):
        '''
        @Args:
            tn_abs_all: (N, 3)
            tn_rel_all0: (N, N, 3)
        '''

        self.xv = 1 / (1 + torch.exp(self.xv_free)) # make sure xv is in (0, 1)

        assert(self.max_parts == tn_abs_all.shape[0] == self.tn_rel_all0.shape[0])

        sigma_cv_all = self.sigma_cv[:, None].repeat(1, self.num_each_class).reshape(-1) # (N,)
        diff_xv = self.xv - self.xv0
        term1 = torch.sum(diff_xv * diff_xv / sigma_cv_all)

        xvc = torch.sum(sigmoid(self.xv, c=0.5, a=10).reshape(self.num_class, self.num_each_class), dim=1)
        term2 = torch.sum( -torch.log(pxv_cont_array(xvc, self.pxv_params) + 1e-6) )

        if self.coexist_pairs.shape[0] != 0:
            term5 = torch.sum( -torch.log(pxe_cont_array(xvc[self.coexist_pairs[:, 0]], xvc[self.coexist_pairs[:, 1]], self.pxe_params[self.coexist_pairs[:, 0], self.coexist_pairs[:, 1]]) + 1e-6) )
            term5 = 1 * term5 / self.coexist_pairs.shape[0]
        else:
            term5 = 0

        txy_abs = tn_abs_all[:, :2]
        txy_rel0 = self.tn_rel_all0[:, :, :2]
        diff_g = txy_abs[None, :, :].repeat(self.max_parts, 1, 1) - txy_abs[:, None, :].repeat(1, self.max_parts, 1) - txy_rel0
        diff_g = diff_g * diff_g * (self.xv.reshape(-1, 1) * self.xv)[:, :, None]
        sigma_g = self.sigma_ce[:, None, :, None].repeat(1, self.num_each_class, 1, self.num_each_class).reshape(self.max_parts, self.max_parts)[:, :, None]
        alpha_g = self.alpha_ce[:, None, :, None].repeat(1, self.num_each_class, 1, self.num_each_class).reshape(self.max_parts, self.max_parts)[:, :, None]
        diff_g = diff_g / sigma_g
        term3 = torch.sum(GM(diff_g, alpha_g))

        if torch.isnan(term1) or torch.isnan(term2) or torch.isnan(term3):
            ipdb.set_trace()

        term1 = 1 * term1 / self.max_parts # 1
        term2 = 2 * term2 / self.max_parts # 2
        term3 = 20 * term3 / self.max_parts / self.max_parts # 0.5
        term4 = torch.mean(self.xv * (1 - self.xv)) * 10 # 10

        self.loss = term1 + term2 + term3 + term4 + term5

        return self.loss


class Solver_Gv(nn.Module):
    def __init__(self, hyper_params_dict, ro_abs_all0, tn_abs_all0, sz_abs_all0, rotation_class0, tn_rel_all0, same_size0, rel_size0, Pcv_abs, Pcv_rel, Povl, Prot_abs, Prot_rel, num_class, num_each_class, valid_thres=None, use_cuda=False):
        '''
        @Args:
            ro_abs_all0: (N, 2)
            tn_abs_all0: (N, 3)
            sz_abs_all0: (N, 3)

            tn_rel_all0: (N, N, 3)

            Povl: (C, C), class overlapping probability (binary) array
            Prot_abs: (C, 3)
            Prot_rel: (C, C, 3)

            valid_thres: (N,), repeat for C classes
        '''
        super(Solver_Gv, self).__init__()
        self.num_class = num_class
        self.num_each_class = num_each_class
        self.max_parts = num_class * num_each_class

        if use_cuda:
            self.dtype = torch.cuda.FloatTensor
        else:
            self.dtype = torch.FloatTensor

        self.Pcv_abs = Pcv_abs
        self.Pcv_rel = Pcv_rel
        self.Povl = Povl
        self.Prot_rel = Prot_rel[:, None, :, None, :].repeat(1, num_each_class, 1, num_each_class, 1).reshape(self.max_parts, self.max_parts, 3)

        if valid_thres is None:
            self.valid_thres = 0.5
        else:
            self.valid_thres = valid_thres

        self.ro_abs_free  = nn.Parameter(ro_abs_all0.clone())          # (N, 2), may not be normalized
        self.ro_abs = self.ro_abs_free / torch.norm(self.ro_abs_free, dim=1, keepdim=True) # (N, 2), must be normalized

        self.txy_abs      = nn.Parameter(tn_abs_all0[:, :2].clone())   # (N, 2)
        self.tz_abs       = nn.Parameter(tn_abs_all0[:, 2].clone())    # (N,)
        self.sz_abs       = nn.Parameter(sz_abs_all0.clone())          # (N, 3)

        self.ro_abs_all0 = ro_abs_all0
        self.tn_abs_all0 = tn_abs_all0
        self.sz_abs_all0 = sz_abs_all0

        self.rotation_class0 = rotation_class0
        self.tn_rel_all0 = tn_rel_all0
        self.same_size0 = same_size0
        self.rel_size0 = rel_size0

        self.sigma_cv = hyper_params_dict['func.joint_opt_func_gv.sigma_cv']
        self.alpha_cv = hyper_params_dict['func.joint_opt_func_gv.alpha_cv']
        self.sigma_ce = hyper_params_dict['func.joint_opt_func_gv.sigma_ce']
        self.alpha_ce = hyper_params_dict['func.joint_opt_func_gv.alpha_ce']


    def forward(self, xv, opt_mode, with_collision=True):
        if opt_mode == 'sync_size':
            self.loss = self.sync_size(xv)
        elif opt_mode == 'sync_translation':
            self.loss = self.sync_translation(xv, with_collision=with_collision)
        elif opt_mode == 'sync_rotation':
            self.loss = self.sync_rotation(xv)
        elif opt_mode == 'collision':
            self.loss = self.penalize_collision(xv)
        else:
            raise AssertionError('Not Implemented')
        return self.loss


    def _get_valid(self, xv):
        mask_abs = (xv > self.valid_thres)[:, None].float() # (N, 1)
        valid_abs_index = torch.where(mask_abs)[0]
        n = valid_abs_index.shape[0]
        if n <= 1:
            logger.debug('use lower threshold')
            for xv_thres in np.arange(0.4, 0.1, -0.1):
                mask_abs = torch.tensor(xv > self.valid_thres * xv_thres / 0.5).type(self.dtype)[:, None] # (N, 1)
                valid_abs_index = torch.where(mask_abs)[0]
                n = valid_abs_index.shape[0]
                if n > 1:
                    break
            logger.debug(n)
            assert(n > 0)
        return n, mask_abs, valid_abs_index


    def sync_rotation(self, xv, mask_rotation_thres=0.15):
        ''' 
        @Args:
            xv: (N, )
        '''
        n, mask_abs, valid_abs_index = self._get_valid(xv)
        if n <= 1:
            logger.debug('Too few objects. Do not sync')
            return torch.sum(0 * self.ro_abs_free)

        loss_v_recon = torch.zeros(1).type(self.dtype)
        loss_e_recon = torch.zeros(1).type(self.dtype)

        self.ro_abs = self.ro_abs_free / torch.norm(self.ro_abs_free, dim=1, keepdim=True) # (N, 2)
        ro_abs_valid = self.ro_abs[valid_abs_index]
        ro_abs_valid0 = self.ro_abs_all0[valid_abs_index]
        
        innerProduct = torch.abs(torch.sum(ro_abs_valid[:, None, :].repeat(1, n, 1) * ro_abs_valid[None, :, :].repeat(n, 1, 1), dim=2))
        # rotation_class0==1: parallel, labels=1. rotation_class0==2: orthogonal, labels=0
        innerProduct_init = (2 - self.rotation_class0)[valid_abs_index, :][:, valid_abs_index] # (n, n), 1: parallel, 0: orthogonal

        mask_rotation = (self.Prot_rel[valid_abs_index, :][:, valid_abs_index][:, :, 0] <= mask_rotation_thres) # (n, n). 1: sync R. 0: not sync R. Prior, whether sync or not
        mask_rotation_init = ( self.rotation_class0[valid_abs_index, :][:, valid_abs_index] != 0 ) # network prediction, whether sync or not
        mask_rotation = mask_rotation * mask_rotation_init # only consider mask
        loss_e_recon = torch.sum(torch.abs( (innerProduct - innerProduct_init) *  mask_rotation)) / torch.sum(mask_rotation)

        # Robust norm could be used
        diff_ro = ro_abs_valid - ro_abs_valid0
        loss_v_recon = torch.sum(diff_ro * diff_ro) / n 

        loss_rotation = loss_v_recon + 100 * loss_e_recon # 100

        return loss_rotation


    def sync_size(self, xv):
        ''' Currently we just sync same_size
        @Args:
            xv: (N, )
        '''
        n, mask_abs, valid_abs_index = self._get_valid(xv)
        if n <= 1:
            logger.debug('Too few objects. Do not sync')
            return torch.sum(0 * self.sz_abs)

        loss_v_recon = torch.zeros(1).type(self.dtype)
        loss_e_recon = torch.zeros(1).type(self.dtype)

        sz_abs_valid  = self.sz_abs[valid_abs_index]      # (n, 3)
        sz_abs_valid0 = self.sz_abs_all0[valid_abs_index] # (n, 3)
        same_size_valid_mask = self.same_size0[valid_abs_index, :][:, valid_abs_index] # (n, n)
        same_size_valid_mask = (same_size_valid_mask - torch.diag(torch.diag(same_size_valid_mask)))[:, :, None] # (n, n, 1)

        diff_sz_abs = sz_abs_valid - sz_abs_valid0
        loss_v_recon = torch.sum(diff_sz_abs * diff_sz_abs) / n

        diff_sz_rel = sz_abs_valid[:, None, :].repeat(1, n, 1) - sz_abs_valid[None, :, :].repeat(n, 1, 1)
        diff_sz_rel = diff_sz_rel * same_size_valid_mask
        loss_e_recon = torch.sum(diff_sz_rel * diff_sz_rel) / (torch.sum(same_size_valid_mask) + 1e-6)

        loss_size = loss_v_recon + 10 * loss_e_recon

        return loss_size


    # @timing
    def sync_translation(self, xv, with_collision=True):
        '''
        @Args:
            xv: (N, )
        '''
        n, mask_abs, valid_abs_index = self._get_valid(xv)
        if n <= 1:
            logger.debug('Too few objects. Do not sync')
            return torch.sum(0 * self.txy_abs)

        loss_v_recon = torch.zeros(1).type(self.dtype)
        loss_v_prior = torch.zeros(1).type(self.dtype)
        loss_e_recon = torch.zeros(1).type(self.dtype)
        loss_e_prior = torch.zeros(1).type(self.dtype)

        txy_abs0 = self.tn_abs_all0[:, :2]
        txy_rel0 = self.tn_rel_all0[:, :, :2]

        sigma_gv_all = self.sigma_cv[:, None].repeat(1, self.num_each_class).reshape(-1, 1) # (N, 1)
        alpha_gv_all = self.alpha_cv[:, None].repeat(1, self.num_each_class).reshape(-1, 1) # (N, 1)
        sigma_ge_all = self.sigma_ce[:, None, :, None].repeat(1, self.num_each_class, 1, self.num_each_class).reshape(self.max_parts, self.max_parts)[:, :, None]
        alpha_ge_all = self.alpha_ce[:, None, :, None].repeat(1, self.num_each_class, 1, self.num_each_class).reshape(self.max_parts, self.max_parts)[:, :, None]

        ''' ------------------------- Absolute terms ------------------------- '''
        self.weight_abs = torch.zeros_like(txy_abs0).type(self.dtype)
        self.pdf_abs_mat = torch.zeros_like(self.txy_abs).type(self.dtype) # (N, 2)

        params_abs = self.Pcv_abs[valid_abs_index // self.num_each_class] # (n, 2, 3, 6)

        self.weight_abs[valid_abs_index, 0] = GMM_array(txy_abs0[valid_abs_index, 0], params_abs[:, 0, 0], params_abs[:, 0, 1], torch.sqrt(params_abs[:, 0, 2]))
        self.weight_abs[valid_abs_index, 1] = GMM_array(txy_abs0[valid_abs_index, 1], params_abs[:, 1, 0], params_abs[:, 1, 1], torch.sqrt(params_abs[:, 1, 2]))

        self.pdf_abs_mat[valid_abs_index, 0] = GMM_array(self.txy_abs[valid_abs_index, 0], params_abs[:, 0, 0], params_abs[:, 0, 1], torch.sqrt(params_abs[:, 0, 2]))
        self.pdf_abs_mat[valid_abs_index, 1] = GMM_array(self.txy_abs[valid_abs_index, 1], params_abs[:, 1, 0], params_abs[:, 1, 1], torch.sqrt(params_abs[:, 1, 2]))

        loss_v_prior = torch.sum(-torch.log(self.pdf_abs_mat[valid_abs_index] + 1e-6)) # WARNING: must use valid_abs_index

        diff_gv_all = (self.txy_abs - txy_abs0) 
        diff_gv_all = diff_gv_all * diff_gv_all * mask_abs / sigma_gv_all
        loss_v_recon = torch.sum(GM(diff_gv_all, alpha_gv_all) * self.weight_abs)

        ''' ------------------------- Relative terms ------------------------- '''
        txy_rel = self.txy_abs[None, :, :].repeat(self.max_parts, 1, 1) - self.txy_abs[:, None, :].repeat(1, self.max_parts, 1) # (N, N, 2)

        # self.weight_rel is an upper triangular matrix
        self.weight_rel = torch.zeros_like(txy_rel0).type(self.dtype)
        self.pdf_rel_mat = torch.zeros_like(txy_rel).type(self.dtype) # (N, N, 2)

        index2 = torch.LongTensor(list(itertools.combinations(valid_abs_index, 2)))
        index2_i = index2[:, 0]
        index2_j = index2[:, 1]
        params_rel = self.Pcv_rel[index2[:, 0] // self.num_each_class, index2[:, 1] // self.num_each_class] # (nn, 2, 3, 8)

        self.weight_rel[index2_i, index2_j, 0] = GMM_array(txy_rel0[index2_i, index2_j, 0], params_rel[:, 0, 0], params_rel[:, 0, 1], torch.sqrt(params_rel[:, 0, 2]))
        self.weight_rel[index2_i, index2_j, 1] = GMM_array(txy_rel0[index2_i, index2_j, 1], params_rel[:, 1, 0], params_rel[:, 1, 1], torch.sqrt(params_rel[:, 1, 2]))

        self.pdf_rel_mat[index2_i, index2_j, 0] = GMM_array(txy_rel[index2_i, index2_j, 0], params_rel[:, 0, 0], params_rel[:, 0, 1], torch.sqrt(params_rel[:, 0, 2]))
        self.pdf_rel_mat[index2_i, index2_j, 1] = GMM_array(txy_rel[index2_i, index2_j, 1], params_rel[:, 1, 0], params_rel[:, 1, 1], torch.sqrt(params_rel[:, 1, 2]))

        loss_e_prior = torch.sum( -torch.log(self.pdf_rel_mat[index2_i, index2_j] + 1e-6) )

        diff_ge_all = (txy_rel - txy_rel0)
        self.diff_ge_all = diff_ge_all * diff_ge_all * (mask_abs * mask_abs.T)[:, :, None] / sigma_ge_all
        
        # loss_e_recon = torch.sum(GM(diff_ge_all, alpha_ge_all) * self.weight_rel * weight_rel_mask) # Only compute loss for the upper triangular part. Note that txy_rel0 is not necessarily symmetric, e.g. both are -0.1
        loss_e_recon = torch.sum(self.diff_ge_all * self.weight_rel * self.weight_rel * 4) # Only compute loss for the upper triangular part. Note that txy_rel0 is not necessarily symmetric, e.g. both are -0.1
        '''    two choices:
                sigma_ge_all=0.01: (diff_ge_all * weight)
                sigma_ge_all=1:    (GM(diff_ge_all, 0.01) * weight)
        '''

        ''' ------------------------- collision terms ------------------------- '''
        if with_collision:
            loss_collision = self.penalize_collision(xv)
            loss_collision = loss_collision * 50 # 10

        loss_v_recon = loss_v_recon / n * 2
        loss_v_prior = loss_v_prior / n * 0.01 # * 0.01
        # self.weight_rel is an upper triangular matrix
        loss_e_recon = loss_e_recon / (n * n - n) / 2 * 100 # 50
        loss_e_prior = loss_e_prior / (n * n - n) / 2 * 0.1 # * 0.1

        if with_collision:
            loss_translation = loss_v_recon + loss_v_prior + loss_e_recon + loss_e_prior + loss_collision
        else:
            loss_translation = loss_v_recon + loss_v_prior + loss_e_recon + loss_e_prior


        for v in [loss_v_recon, loss_v_prior, loss_e_recon, loss_e_prior]:
            if torch.isnan(v) or torch.isinf(v):
                ipdb.set_trace()

        return loss_translation


    # @timing
    def penalize_collision(self, xv, overlap_thres=0.7):
        '''
        @Args:
            xv: (N, )
            Povl: indicator 0/1. Probability of overlapping, whether penalizing collision
        self.ro_abs  = nn.Parameter(torch.tensor(ro_abs_all0))          # (N, 2)
        self.txy_abs = nn.Parameter(torch.tensor(tn_abs_all0[:, :2]))   # (N, 2)
        self.tz_abs  = nn.Parameter(torch.tensor(tn_abs_all0[:, 2]))    # (N,)
        self.sz_abs  = nn.Parameter(torch.tensor(sz_abs_all0))          # (N, 3)
        '''
        n, mask_abs, valid_abs_index = self._get_valid(xv)
        if n <= 1:
            logger.debug('Too few objects. Do not sync')
            return torch.sum(0 * self.txy_abs)

        loss_collision = torch.zeros(1).type(self.dtype)

        ''' ATTENTION! MUST DETACH HERE '''
        boxes = utils.params2box(self.ro_abs_all0.detach().clone(), self.txy_abs.detach().clone(), self.sz_abs_all0.detach().clone()) # (N, 4, 2)
        self.overlappingInfoMat = torch.zeros((n, n, 4)).type(self.dtype)
        index2 = torch.LongTensor(list(itertools.combinations(valid_abs_index, 2)))
        index2_i = index2[:, 0]
        index2_j = index2[:, 1]
        Povl_flat = self.Povl[index2_i // self.num_each_class, index2_j // self.num_each_class] # (nn)
        Povl_flat = (Povl_flat < overlap_thres).float()

        boxesi = boxes[index2_i]
        boxesj = boxes[index2_j]

        isOverlapping, select_dim, distance = overlap.overlappingInfo(boxesi, boxesj)
        isOverlapping = isOverlapping * Povl_flat
        delta = 0.1

        loss_collision = torch.sum( F.relu(distance + delta - torch.abs(self.txy_abs[index2_j, select_dim] - self.txy_abs[index2_i, select_dim])) * isOverlapping )
        num_ovl = torch.sum(isOverlapping)
        loss_collision = loss_collision / (num_ovl + 1e-8)

        return loss_collision



if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='bedroom or living')

    parser.add_argument('--dump_results', action='store_true', help='whether dump')
    parser.add_argument('--dump_dir', type=str, help='dump dir')
    parser.add_argument('--assets_dir', type=str, default='./assets/', help='assets dir')
    parser.add_argument('--data_dir', type=str, help='pred abs & rel dir')
    parser.add_argument('--log_dir', type=str, default='./log', help='log dir')
    parser.add_argument('--use_joint_params', action='store_true', help='use params from joint learning')
    parser.add_argument('--hyper_params_path', type=str, help='params from joint_hyper_opt')

    parser.add_argument('--start', type=int, default=0, help='start index of opt')
    parser.add_argument('--length', type=int, default=5, help='interval of opt')
    parser.add_argument('--debug', action='store_true', help='debug mode')
    parser.add_argument('--debug_id', type=int, help='debug mode')

    args = parser.parse_args()

    data_dir = args.data_dir
    assets_dir = f'{args.assets_dir}/{args.type}'
    assert(args.type in ['bedroom', 'living'])

    if not os.path.exists(args.log_dir):
        os.makedirs(args.log_dir)
    logger.add(f"{args.log_dir}/log.log")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if use_cuda else "cpu")

    THRESHOLD_RATIO = 1.0 # multiply wv0 to increase INITIAL threshold
    num_bins = 21
    halfRange = 6
    interval = 0.3
    NUM_CLASS = 20
    NUM_EACH_CLASS = 4

    Pxv = np.load(f'{assets_dir}/Pxv.npy')
    Pxe = np.load(f'{assets_dir}/Pxe.npy')
    pxv_params = fit_pxv.fit_pxv_cont(Pxv)
    pxe_params, coexist_pairs = fit_pxe.fit_pxe_cont(Pxe)
    Pcv_abs = np.load(f'{assets_dir}/Pcv_abs.npy')
    Pcv_rel = np.load(f'{assets_dir}/Pcv_rel.npy')
    Povl = np.load(f'{assets_dir}/Povl.npy')
    Prot_abs = None
    Prot_rel = np.load(f'{assets_dir}/Prot_rel.npy')

    pxv_params = torch.FloatTensor(pxv_params).to(device)
    pxe_params = torch.FloatTensor(pxe_params).to(device)
    coexist_pairs = torch.LongTensor(coexist_pairs).to(device)
    Pcv_abs = torch.FloatTensor(Pcv_abs).to(device)
    Pcv_rel = torch.FloatTensor(Pcv_rel).to(device)
    Povl = torch.FloatTensor(Povl).to(device) # Probability (binary) of overlapping
    Prot_rel = torch.FloatTensor(Prot_rel).to(device) # Probability of relative rotation type: 0: no, 1: parallel, 2: orthogonal

    ''' different threshold for different class '''
    num = np.load(f'{assets_dir}/num_unique.npy')
    cw_mat = sigmoid_rescale(num, c=0.5, a=10, min_thres=0.3, max_thres=0.5)
    valid_thres = torch.FloatTensor(cw_mat).to(device)[:, None].repeat(1, NUM_EACH_CLASS).reshape(-1)

    ''' hyper parameters '''
    hyper_params_dict = {}
    if args.use_joint_params:
        hyper_params_free_dict = torch.load(args.hyper_params_path, map_location=torch.device(device))['model_state_dict']
        for k, v in hyper_params_free_dict.items():
            hyper_params_dict[k[:-5]] = torch.clamp(v, min=0.001, max=2)
    else:
        wp = 1 - Pxv[:, 0]
        wp = torch.tensor(wp).float()

        if args.type == 'living':
            hyper_params_dict['func.joint_opt_func_xv.sigma_cv'] = wp
        else: # bedroom
            hyper_params_dict['func.joint_opt_func_xv.sigma_cv'] = sigmoid(wp, c=0.5, a=10)
        hyper_params_dict['func.joint_opt_func_xv.alpha_cv'] = 0.01 * torch.ones(NUM_CLASS).float() # 0.1       
        hyper_params_dict['func.joint_opt_func_xv.sigma_ce'] = torch.ones((NUM_CLASS, NUM_CLASS)).float()
        hyper_params_dict['func.joint_opt_func_xv.alpha_ce'] = 0.1 * torch.ones((NUM_CLASS, NUM_CLASS)).float()

        hyper_params_dict['func.joint_opt_func_gv.sigma_cv'] = torch.ones(NUM_CLASS).float()  # 1                               
        hyper_params_dict['func.joint_opt_func_gv.alpha_cv'] = 0.01 * torch.ones(NUM_CLASS).float() # 0.1                       
        hyper_params_dict['func.joint_opt_func_gv.sigma_ce'] = torch.ones((NUM_CLASS, NUM_CLASS)).float() # if use weight: 0.01 
        hyper_params_dict['func.joint_opt_func_gv.alpha_ce'] = 0.1 * torch.ones((NUM_CLASS, NUM_CLASS)).float() # 0.1           

    for key, val in hyper_params_dict.items():
        hyper_params_dict[key] = val.to(device)

    t0 = time.perf_counter()
    time_list = [time.perf_counter()]
    for scene_idx in range(args.start, args.start + args.length):
        logger.debug('=' * 50, scene_idx, '=' * 50)
        if args.debug:
            if scene_idx != args.debug_id:
                continue

        pred_abs_all = np.load(os.path.join(data_dir, str(scene_idx).zfill(4) + '_abs_pred.npy')) # (80, 16)
        pred_rel_all = np.load(os.path.join(data_dir, str(scene_idx).zfill(4) + '_rel_pred.npy')) # (80, 80, depend_on_which_network)
        N = pred_abs_all.shape[0]

        ''' 
            ---------------------- Initialization ----------------------
        '''
        # Initial of the ABSOLUTE parameters we want to optimize
        ro_abs_all0 = utils.parse_abs_angle_new(pred_abs_all) # (N, 2)
        tn_abs_all0 = pred_abs_all[:, 9:12] # (N, 3)
        sz_abs_all0 = pred_abs_all[:, 12:15] # (N, 3)
        wv0         = pred_abs_all[:, -1] * THRESHOLD_RATIO

        # RELATIVE parameters
        # relative translation
        tn_rel_all0 = np.zeros((N, N, 3))
        Ix = (pred_rel_all[:, :, 0] > 0.0).astype(np.float32)
        Iy = (pred_rel_all[:, :, 1] > 0.0).astype(np.float32)
        cx = np.argmax(pred_rel_all[:, :, 2:2+num_bins], axis=2)
        cy = np.argmax(pred_rel_all[:, :, 2+num_bins:2+2*num_bins], axis=2)
        rx = pred_rel_all[:, :, -9]
        ry = pred_rel_all[:, :, -8]
        z  = pred_rel_all[:, :, -7]
        tn_rel_all0[:, :, 0] = utils.disc2translation(Ix, cx, rx, halfRange, interval)
        tn_rel_all0[:, :, 1] = utils.disc2translation(Iy, cy, ry, halfRange, interval)
        tn_rel_all0[:, :, 2] = z
        # relative rotation
        rotation_class0 = np.argmax(pred_rel_all[:, :, -6:-3], axis=2)
        # relative size
        same_size0 = (pred_rel_all[:, :, -3] > 0.0).astype(np.float32)
        rel_size0 = (pred_rel_all[:, :, -2]).astype(np.float32)

        ro_abs_all0 = torch.FloatTensor(ro_abs_all0).to(device) # (N, 2)
        tn_abs_all0 = torch.FloatTensor(tn_abs_all0).to(device) # (N, 3)
        sz_abs_all0 = torch.FloatTensor(sz_abs_all0).to(device) # (N, 3)
        wv0         = torch.FloatTensor(wv0        ).to(device) # (N,)
        for i, cw in enumerate(cw_mat):
            wv0[i*NUM_EACH_CLASS : (i+1)*NUM_EACH_CLASS] = sigmoid(wv0[i*NUM_EACH_CLASS : (i+1)*NUM_EACH_CLASS], c=cw, a=10)

        rotation_class0 = torch.FloatTensor(rotation_class0).to(device)
        tn_rel_all0 = torch.FloatTensor(tn_rel_all0).to(device)
        same_size0 = torch.FloatTensor(same_size0).to(device)
        rel_size0 = torch.FloatTensor(rel_size0).to(device)

        '''
           ---------------------- Build Model ----------------------
        '''
        solver_xv = Solver_Xv(hyper_params_dict, wv0, tn_rel_all0, pxv_params, pxe_params, coexist_pairs, num_class=NUM_CLASS, num_each_class=NUM_EACH_CLASS, use_cuda=use_cuda)
        solver_gv = Solver_Gv(hyper_params_dict, ro_abs_all0, tn_abs_all0, sz_abs_all0, rotation_class0, tn_rel_all0, same_size0, rel_size0, Pcv_abs, Pcv_rel, Povl, Prot_abs, Prot_rel, num_class=NUM_CLASS, num_each_class=NUM_EACH_CLASS, valid_thres=valid_thres, use_cuda=use_cuda)
        solver_xv.to(device)
        solver_gv.to(device)

        learning_rate = 0.1 
        optimizer_xv = FullBatchLBFGS(solver_xv.parameters(), lr=learning_rate)
        optimizer_tn = FullBatchLBFGS([solver_gv.txy_abs, solver_gv.tz_abs], lr=learning_rate)
        optimizer_ro = FullBatchLBFGS([solver_gv.ro_abs_free], lr=learning_rate)
        optimizer_sz = FullBatchLBFGS([solver_gv.sz_abs], lr=learning_rate)
        optimizer_collision = FullBatchLBFGS([solver_gv.txy_abs], lr=learning_rate)


        '''
           ---------------------- Optimization ----------------------
        '''
        xv_opt_prev  = copy.deepcopy(wv0)                # (N,)
        ro_opt_prev  = copy.deepcopy(ro_abs_all0[:, :2]) # (N, 2)
        txy_opt_prev = copy.deepcopy(tn_abs_all0[:, :2]) # (N, 2)
        sz_opt_prev  = copy.deepcopy(sz_abs_all0)        # (N, 3)

        for it in range(30):
            ''' -------------- opt xv --------------'''
            def closure_xv():
                optimizer_xv.zero_grad()
                loss_xv = solver_xv(solver_gv.txy_abs.detach())
                return loss_xv
            loss_xv = closure_xv()
            loss_xv.backward()

            for it_xv in range(1):
                options_xv = {'closure': closure_xv, 'current_loss': loss_xv}
                loss_xv, _, lr, _, F_eval, G_eval, _, _ = optimizer_xv.step(options_xv)


            ''' -------------- opt rotation --------------
            '''
            def closure_ro():
                optimizer_ro.zero_grad()
                loss_ro = solver_gv(solver_xv.xv.detach(), opt_mode='sync_rotation')
                return loss_ro
            loss_ro = closure_ro()
            loss_ro.backward()

            for it_ro in range(1):
                options_ro = {'closure': closure_ro, 'current_loss': loss_ro}
                loss_ro, _, lr, _, F_eval, G_eval, _, _ = optimizer_ro.step(options_ro)


            ''' -------------- opt size --------------
            '''
            def closure_sz():
                optimizer_sz.zero_grad()
                loss_sz = solver_gv(solver_xv.xv.detach(), opt_mode='sync_size')
                return loss_sz
            loss_sz = closure_sz()
            loss_sz.backward()

            for it_sz in range(1):
                options_sz = {'closure': closure_sz, 'current_loss': loss_sz}
                loss_sz, _, lr, _, F_eval, G_eval, _, _ = optimizer_sz.step(options_sz)


            ''' -------------- opt translation --------------'''
            # Only sync translation
            def closure_tn1():
                optimizer_tn.zero_grad()
                loss_tn = solver_gv(solver_xv.xv.detach(), opt_mode='sync_translation', with_collision=False)
                return loss_tn
            loss_tn = closure_tn1()
            loss_tn.backward()

            for it_tn in range(1):
                options_tn1 = {'closure': closure_tn1, 'current_loss': loss_tn}
                loss_tn, _, lr, _, F_eval, G_eval, _, _ = optimizer_tn.step(options_tn1)

            # Only penalize collision
            def closure_tn2():
                optimizer_tn.zero_grad()
                loss_tn = solver_gv(solver_xv.xv.detach(), opt_mode='collision')
                return loss_tn
            loss_tn = closure_tn2()
            loss_tn.backward()

            for it_tn in range(1):
                options_tn2 = {'closure': closure_tn2, 'current_loss': loss_tn}
                loss_tn, _, lr, _, F_eval, G_eval, _, _ = optimizer_tn.step(options_tn2)

            # sync translation + penalize collision
            def closure_tn3():
                optimizer_tn.zero_grad()
                loss_tn = solver_gv(solver_xv.xv.detach(), opt_mode='sync_translation', with_collision=True)
                return loss_tn
            loss_tn = closure_tn3()
            loss_tn.backward()

            for it_tn in range(1):
                options_tn3 = {'closure': closure_tn3, 'current_loss': loss_tn}
                loss_tn, _, lr, _, F_eval, G_eval, _, _ = optimizer_tn.step(options_tn3)


            xv_opt  = solver_xv.xv.detach()
            ro_opt  = solver_gv.ro_abs.detach()
            txy_opt = solver_gv.txy_abs.detach()
            sz_opt  = solver_gv.sz_abs.detach()
            idx_opt = torch.where(xv_opt > 0.5)[0]

            # no rotation
            logger.debug("loss_xv= %.6f, loss_tn= %.6f, txy diff= %.6f, ro diff= %.6f, sz diff= %.6f" % 
                            (loss_xv.detach().cpu().numpy(), 
                             loss_tn.detach().cpu().numpy(), 
                             (torch.norm((txy_opt_prev - txy_opt)[idx_opt]) / idx_opt.shape[0]).detach().cpu().numpy(), 
                             (torch.norm((ro_opt_prev  - ro_opt )[idx_opt]) / idx_opt.shape[0]).detach().cpu().numpy(), 
                             (torch.norm((sz_opt_prev  - sz_opt )[idx_opt]) / idx_opt.shape[0]).detach().cpu().numpy()))

            tol = 1e-4
            if it > 0 and torch.norm((txy_opt_prev - txy_opt)[idx_opt]) / idx_opt.shape[0] < tol and\
                          torch.norm((ro_opt_prev  - ro_opt)[idx_opt]) / idx_opt.shape[0] < tol and\
                          torch.norm((sz_opt_prev  - sz_opt)[idx_opt]) / idx_opt.shape[0] < tol:
                break
            xv_opt_prev  = copy.deepcopy(xv_opt)
            ro_opt_prev  = copy.deepcopy(ro_opt)
            txy_opt_prev = copy.deepcopy(txy_opt)
            sz_opt_prev  = copy.deepcopy(sz_opt)

        logger.debug(f'{scene_idx} Init: ')
        logger.debug(torch.where(wv0 > valid_thres)[0].detach().cpu().numpy() // NUM_EACH_CLASS)
        logger.debug(f'{scene_idx} Opt: ')
        logger.debug(torch.where(xv_opt > valid_thres)[0].detach().cpu().numpy() // NUM_EACH_CLASS)

        # Note: compare with valid thres not 0.5
        # xv_opt_npy = (xv_opt.detach().cpu().numpy() > 0.5)
        xv_opt_npy = (xv_opt > valid_thres).detach().cpu().numpy()
        txy_opt_npy = (txy_opt.detach().cpu().numpy()) 
        tz_opt_npy = (solver_gv.tz_abs.detach().cpu().numpy())  # TODO: currently we do not optimize tz, so it remains initial values
        ro_opt_npy = (ro_opt.detach().cpu().numpy()) 
        sz_opt_npy = (sz_opt.detach().cpu().numpy()) 
        valid_abs_index = np.where(xv_opt_npy)[0]
        n = valid_abs_index.shape[0]
        if args.type == 'bedroom':
            # global rotation alignment:
            R_global = global_rotation_align(ro_opt_npy[valid_abs_index])
            ro_opt_npy = ro_opt_npy.dot(R_global.T)

        pred_abs_sync_all = np.zeros((N, 10))
        pred_abs_sync_all[valid_abs_index] = np.concatenate((ro_opt_npy[valid_abs_index], np.zeros((n, 1)),\
                                                             txy_opt_npy[valid_abs_index], tz_opt_npy.reshape(-1, 1)[valid_abs_index],\
                                                             sz_opt_npy[valid_abs_index], xv_opt_npy.reshape(-1, 1)[valid_abs_index]), axis=1)
        if args.dump_results:
            dump_sync_dir = args.dump_dir
            if not os.path.exists(dump_sync_dir):
                os.makedirs(dump_sync_dir)
            np.save(os.path.join(dump_sync_dir, str(scene_idx).zfill(4) + '_sync.npy'), pred_abs_sync_all)

        if args.debug:
            ipdb.set_trace()

        time_list += [time.perf_counter()]

    tlist = [time_list[i+1] - time_list[i] for i in range(len(time_list) - 1)]
    t1 = time.perf_counter()
    logger.info(t1 - t0)
    logger.info(tlist)

