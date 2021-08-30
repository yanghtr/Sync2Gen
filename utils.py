#!/usr/bin/env python
# coding=utf-8

import os
import sys
import pickle
import numpy as np

import torch

def angle2class(angle, num_class):
    ''' Convert continuous angle to discrete class
        [optinal] also small regression number from  
        class center angle to current angle.
       
        angle is from 0-2pi (or -pi~pi), class center at 0, 1*(2pi/N), 2*(2pi/N) ...  (N-1)*(2pi/N)
        return is class of int32 of 0,1,...,N-1 and a number such that
            class*(2pi/N) + number = angle
    '''
    angle = angle%(2*np.pi)
    assert(angle>=0 and angle<=2*np.pi)
    angle_per_class = 2*np.pi/float(num_class)
    shifted_angle = (angle+angle_per_class/2)%(2*np.pi)
    class_id = int(shifted_angle/angle_per_class)
    residual_angle = shifted_angle - (class_id*angle_per_class+angle_per_class/2)
    return class_id, residual_angle


def class2angle(pred_cls, residual, num_class):
    ''' Inverse function to angle2class 
    @Args:
        pred_cls: (same_shape)
        residual: (same_shape)
    @Returns:
        angle: (same_shape). 0~2pi
    '''
    angle_per_class = 2*np.pi/float(num_class)
    angle_center = pred_cls * angle_per_class
    angle = angle_center + residual
    # if angle>np.pi: # turn 0~2pi to -pi to pi
    #     angle = angle - 2*np.pi
    return angle


def parse_abs_angle(X_abs):
    '''
    @Args:
        X_abs: (N, dim=16), only use X_abs[:, :9]. ATTENTION: each line should be valid
    @Returns:
        n1:    (N, 2)
    '''
    N = X_abs.shape[0]
    X_abs = X_abs[:, :9]
    n1 = np.zeros((N, 2))
    pred_class = np.argmax(X_abs[:, :8], axis=1)
    for i, x_abs in enumerate(X_abs):
        angle_recon = class2angle(pred_class[i], x_abs[8], num_class=8)
        n1[i, 0] = np.cos(angle_recon)
        n1[i, 1] = np.sin(angle_recon)
    return n1


def parse_abs_angle_new(X_abs):
    ''' All array. faster
    @Args:
        X_abs: (N, dim=16), only use X_abs[:, :9]. ATTENTION: each line should be valid
    @Returns:
        n1:    (N, 2)
    '''
    N = X_abs.shape[0]
    X_abs = X_abs[:, :9]
    n1 = np.zeros((N, 2))
    pred_class = np.argmax(X_abs[:, :8], axis=1)
    angle_recon = class2angle(pred_class, X_abs[:, 8], num_class=8)
    n1[:, 0] = np.cos(angle_recon)
    n1[:, 1] = np.sin(angle_recon)
    return n1



def parse_abs_batch(X_abs):
    '''
    @Args:
        X_abs: (B, N, dim=16), only use X_abs[:, :9].
        mask:  (B, N), whether abs is valid
    @Returns:
        X_abs_parse: (B, N, 9). n1(2) + t(3) + s(3) + indicator(1) = 9
    '''
    B, N = X_abs.shape[0], X_abs.shape[1]
    n1 = torch.zeros((B, N, 2)).to(X_abs)
    pred_cls = torch.argmax(X_abs[:, :, :8], dim=2)
    angle = class2angle(pred_cls, X_abs[:, :, 8], num_class=8)
    n1[:, :, 0] = torch.cos(angle)
    n1[:, :, 1] = torch.sin(angle)

    X_abs_parse = torch.cat((n1, X_abs[:, :, 9:]), dim=2)
    # assert(X_abs_parse.shape == torch.Size([B, N, 9]))

    return X_abs_parse


def translation2class(tij, halfRange, interval):
    ''' discretize tij to bin, d = interval, s = num_bins / 2
    class_id:
        [0, d): s, [d, 2d): s+1, ..., [(s-1)d, \inf): 2s-1
        [-d, 0): 0, [-2d, -d): 1, ..., (-\inf, (s-1)d): s-1
    residual:
        always positive, distance to the interval point that is closest to 0, 
        e.g. -1.1 \in [-2, -1), residual = 0.1; 1.2 \in [1, 2), residual = 0.2
    @Args:
        tij: numpy array (same shape)
    @Returns:
        class_id/residual: (same shape)
    '''
    num_bins = 2 * halfRange / interval + 2
    assert(num_bins % 2 == 0)
    Iij = (tij >= 0).astype(np.float32)
    class_id = np.minimum(np.abs(tij) // interval, num_bins / 2 - 1) + Iij * num_bins / 2
    residual = np.abs(tij) - (class_id - Iij * num_bins / 2) * interval
    return class_id.astype(np.int), residual

def class2translation(class_id, residual, halfRange, interval):
    num_bins = 2 * halfRange / interval + 2
    Iij = (class_id >= num_bins / 2).astype(np.float32) # 1 or 0
    sij = (Iij - 0.5) * 2 # 1 or -1
    tij = sij * ((class_id - Iij * num_bins / 2) * interval + residual)
    return tij


def translation2disc(tij, halfRange, interval):
    ''' discretize tij to bin
    @Args:
        tij: numpy array (same shape)
    @Returns:
        Iij: (same shape), >0: 1; <=0: 0
        class_id/residual: (same shape)
    '''
    Iij = (tij > 0) # indicator, 1(positive) or 0(zero or negative)
    class_id = np.minimum(np.abs(tij), halfRange + interval - 1e-8) // interval
    residual = np.abs(tij) - class_id * interval
    return Iij.astype(np.float32), class_id.astype(np.int), residual

def disc2translation(Iij, class_id, residual, halfRange, interval):
    sij = (Iij - 0.5) * 2 # 1 or -1
    tij = sij * (class_id * interval + residual)
    return tij


def data2box(dataline):
    '''
    @Args:
        dataline: (10,) or (16,)
    @Returns:
        box: (4, 2). 4 corners
    '''
    abs_dim = dataline.shape[0]
    assert(abs_dim == 10 or abs_dim == 16)
    if abs_dim == 16:
        location = dataline[9:11]
        size = dataline[12:14]
        angle_recon = utils.class2angle(np.argmax(dataline[:8]), dataline[8], num_class=8)
        n1 = np.array([np.cos(angle_recon), np.sin(angle_recon)])
        n2 = np.array([-n1[1], n1[0]])

    elif abs_dim == 10:
        location = dataline[3:5]
        size = dataline[6:8]
        n1 = dataline[:2]
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.array([-n1[1], n1[0]])

    p1 = location + size[1]*n1/2.0 + size[0]*n2/2.0
    p2 = location + size[1]*n1/2.0 - size[0]*n2/2.0
    p3 = location - size[1]*n1/2.0 - size[0]*n2/2.0
    p4 = location - size[1]*n1/2.0 + size[0]*n2/2.0

    box = np.stack((p1, p2, p3, p4))

    return box


def params2box(ro, txy, sz):
    '''
    @Args: Only the first 2 columns are used
        ro:  (N, 2)
        txy: (N, 2)
        sz:  (N, 2)
    @Returns:
        box: (N, 4, 2)
    '''
    location = txy[:, :2]
    size = sz[:, :2]
    n1 = ro[:, :2] / torch.norm(ro[:, :2], dim=1, keepdim=True)
    n2 = torch.stack((-n1[:, 1], n1[:, 0]), dim=1)

    p1 = location + size[:, 1:2]*n1/2.0 + size[:, 0:1]*n2/2.0 # (N, 2)
    p2 = location + size[:, 1:2]*n1/2.0 - size[:, 0:1]*n2/2.0
    p3 = location - size[:, 1:2]*n1/2.0 - size[:, 0:1]*n2/2.0
    p4 = location - size[:, 1:2]*n1/2.0 + size[:, 0:1]*n2/2.0

    box = torch.stack((p1, p2, p3, p4), dim=1) # (N, 4, 2)

    return box


def params2box_batch(ro, txy, sz):
    '''
    @Args: Only the first 2 columns are used
        ro:  (B, N, 2)
        txy: (B, N, 2)
        sz:  (B, N, 2)
    @Returns:
        box: (B, N, 4, 2)
    '''
    location = txy[:, :, :2]
    size = sz[:, :, :2]
    n1 = ro[:, :, :2] / torch.norm(ro[:, :, :2], dim=-1, keepdim=True)
    n2 = torch.stack((-n1[:, :, 1], n1[:, :, 0]), dim=-1)

    p1 = location + size[:, :, 1:2]*n1/2.0 + size[:, :, 0:1]*n2/2.0 # (B, N, 2)
    p2 = location + size[:, :, 1:2]*n1/2.0 - size[:, :, 0:1]*n2/2.0
    p3 = location - size[:, :, 1:2]*n1/2.0 - size[:, :, 0:1]*n2/2.0
    p4 = location - size[:, :, 1:2]*n1/2.0 + size[:, :, 0:1]*n2/2.0

    box = torch.stack((p1, p2, p3, p4), dim=2) # (B, N, 4, 2)

    return box


