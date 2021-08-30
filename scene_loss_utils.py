#!/usr/bin/env python
# coding=utf-8
import os
import sys
import math
import importlib
from scipy.optimize import linear_sum_assignment
import torch
import torch.nn as nn
import numpy as np
import time
from functools import wraps


def timing(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.perf_counter()
        r = func(*args, **kwargs)
        end = time.perf_counter()
        print('{}.{} : {}'.format(func.__module__, func.__name__, end - start))
        return r
    return wrapper


def smooth_l1_loss(input, target, beta=1.0, size_average=True):
    """
    very similar to the smooth_l1_loss from pytorch, but with
    the extra beta parameter. Huber loss
    """
    n = torch.abs(input - target)
    cond = n < beta
    loss = torch.where(cond, 0.5 * n ** 2 / beta, n - 0.5 * beta)
    if size_average:
        return loss.mean()
    return loss.sum()



''' --------------------Absolute Pose Representation Loss---------------'''

def linear_assignment_class(distance_mat, row_counts=None, col_masks=None):
    '''
    If rows_counts & col_counts are both None, then distance matrix has
    the same size as distance_mat. Else if rows_counts != None, meaning 
    for each batch i, row size of the distance matrix is rows_counts[i].
    @Args:
        distance_mat: (B, num_class, num_each_class, num_each_class), row->col: from gt to pred
        row_counts:   (B, num_class), In our case is GT num of parts of i-th class in each batch
        col_masks:    (B, num_class, num_each_class), tensor of 1s and 0s.
    @Returns: 
        (batch_idx, row_ind) is the index of GT(MAX_NUM_PARTS, feat_dim)
        (batch_idx, col_ind) is the index of pred(MAX_NUM_PARTS, feat_dim)

        batch_ind:    list, len = \sum row_counts_i
        row_ind:      list, len = \sum row_counts_i
        col_ind:      list, len = \sum row_counts_i
    '''
    batch_size = distance_mat.shape[0]
    num_class = distance_mat.shape[1]
    num_each_class = distance_mat.shape[2]

    batch_ind = []
    row_ind = []
    col_ind = []
    for i in range(batch_size):
        for j in range(num_class):
            dmat = distance_mat[i, j, :, :]
            if row_counts is not None:
                dmat = dmat[:row_counts[i, j], :]
            if col_masks is not None:
                col_idx = torch.nonzero(col_masks[i, j])[:, 0]
                dmat = dmat[:, col_idx]

            rind, cind = linear_sum_assignment(dmat.detach().to('cpu').numpy())
            rind = list(rind + num_each_class * j)
            if col_masks is None:
                cind = list(cind + num_each_class * j)
            else:
                cind = list(col_idx[cind] + num_each_class * j)

            batch_ind += [i]*len(rind)
            row_ind += rind
            col_ind += cind    

    return batch_ind, row_ind, col_ind


def compute_abs_all_match_loss_classify_angle(X_abs_gt, X_abs_pred, num_class=30, num_each_class=4, is_unmatched_loss=False):
    ''' Loss for representations of parts
    @Args:
        X_abs_gt: (B, MAX_NUM_PARTS, dim=9)
        X_abs_pred: (B, MAX_NUM_PARTS, dim=16)
    @Returns:
    '''
    batch_size, max_num_parts, abs_dim = X_abs_gt.shape[0], X_abs_gt.shape[1], X_abs_gt.shape[2]
    num_class = num_class
    num_each_class = num_each_class
    assert(max_num_parts == num_class * num_each_class)

    X_abs_pred_prob = X_abs_pred[:, :, 9:]

    # Compute distance matrix
    dist_mat = torch.zeros((batch_size, num_class, num_each_class, num_each_class))

    for b in range(batch_size):
        for c in range(num_class):
            offset = c * num_each_class
            # x_gt_tile = X_abs_gt[b, offset : offset + num_each_class, :].unsqueeze(1).repeat(1, num_each_class, 1)  # (r, n, f)
            # x_pred_tile = X_abs_pred_prob[b, offset : offset + num_each_class, :].unsqueeze(0).repeat(num_each_class, 1, 1)  # (r, n, f)
            dist_mat[b, c]  = torch.norm(X_abs_gt[b, offset : offset + num_each_class, 9:].unsqueeze(1) - 
                                         X_abs_pred_prob[b, offset : offset + num_each_class, :].unsqueeze(0), dim=2)  # (n, n), only first r rows valid

    batch_idx, matched_gt_idx, matched_pred_idx = linear_assignment_class(dist_mat)

    if is_unmatched_loss:
        error = X_abs_gt[batch_idx, matched_gt_idx] - X_abs_pred_prob[batch_idx, matched_pred_idx]
        abs_rep_loss = torch.sum(error * error) / batch_size / max_num_parts
    else:
        X_abs_gt_cat_match = X_abs_gt[batch_idx, matched_gt_idx] # (batch_size * max_num_parts, feature_dim)
        X_abs_pred_cat_match = X_abs_pred[batch_idx, matched_pred_idx]

        X_abs_gt_cat = X_abs_gt_cat_match[:, 9:] # (batch_size * max_num_parts, feature_dim)
        X_abs_pred_cat = X_abs_pred_cat_match[:, 9:]
        mask = X_abs_gt_cat[:, -1:]
        X_abs_pred_cat_new = torch.cat([X_abs_pred_cat[:, :-1] * mask, X_abs_pred_cat[:, -1:]], dim=1)

        abs_rep_loss = smooth_l1_loss(X_abs_pred_cat_new, X_abs_gt_cat, beta=0.5)
        abs_rep_loss = abs_rep_loss * (abs_dim - 9)

        angle_pred = X_abs_pred_cat_match[:, :9]
        angle_gt = X_abs_gt_cat_match[:, :9]

        criterion_angle = nn.CrossEntropyLoss()
        angle_index = torch.nonzero(angle_gt[:, :8])
        angle_label = angle_index[:, 1]
        angle_class_loss = criterion_angle(angle_pred[angle_index[:, 0], :8], angle_label)

        angle_residual_loss = smooth_l1_loss(angle_pred[:, -1], angle_gt[:, -1])

    return angle_class_loss, angle_residual_loss, abs_rep_loss, batch_idx, matched_gt_idx, matched_pred_idx



def compute_rel_all_match_loss_discrete(X_rel_gt, X_rel_pred, batch_idx, 
                                        matched_gt_idx, matched_pred_idx, 
                                        num_class=30, num_each_class=4,
                                        halfRange=6, interval=0.3, room_type=None):
    ''' Loss for representations of parts
    @Args:
        X_rel_gt: (B, MAX_NUM_PARTS, MAX_NUM_PARTS, dim)
        X_rel_pred: (B, MAX_NUM_PARTS, MAX_NUM_PARTS, dim)
    @Returns:
    '''
    batch_size, max_num_parts = X_rel_gt.shape[0], X_rel_gt.shape[1]
    num_class = num_class
    num_each_class = num_each_class
    assert(max_num_parts == num_class * num_each_class)
    num_bins = int(halfRange / interval + 1)

    X_rel_pred_prob = X_rel_pred
    
    num_rel = 0
    num_rel_off = 0
    error_tn_I_x = torch.zeros(1).to(X_rel_gt.device)
    error_tn_I_y = torch.zeros(1).to(X_rel_gt.device)
    error_tn_class_x = torch.zeros(1).to(X_rel_gt.device)
    error_tn_class_y = torch.zeros(1).to(X_rel_gt.device)
    error_tn_res_x = torch.zeros(1).to(X_rel_gt.device)
    error_tn_res_y = torch.zeros(1).to(X_rel_gt.device)
    error_z = torch.zeros(1).to(X_rel_gt.device)
    error_rotation_class = torch.zeros(1).to(X_rel_gt.device)
    error_same_size = torch.zeros(1).to(X_rel_gt.device)
    error_rel_size = torch.zeros(1).to(X_rel_gt.device)
    criterion_tn_class = nn.CrossEntropyLoss(reduction='none')
    criterion_tn_I = nn.BCEWithLogitsLoss(reduction='none')
    criterion_rotation_class = nn.CrossEntropyLoss(reduction='none')
    criterion_same_size = nn.BCEWithLogitsLoss(reduction='none')

    for b in range(batch_size):
        ind = np.where(np.array(batch_idx)==b)[0]
        gt_ind = np.array(matched_gt_idx)[ind].tolist()
        pred_ind = np.array(matched_pred_idx)[ind].tolist()
        if len(pred_ind) == 0:
            continue

        gt = X_rel_gt[b][gt_ind, :][:, gt_ind]
        pred = X_rel_pred_prob[b][pred_ind, :][:, pred_ind]
        mask = gt[:, :, -1:]

        pred_masked = pred * mask
        num_rel += torch.sum(mask)

        error_z += smooth_l1_loss(pred_masked[:, :, -7], gt[:, :, -5], beta=0.5, size_average=False)

        error_tn_res_x += smooth_l1_loss(pred_masked[:, :, -9], gt[:, :, -7], beta=0.5, size_average=False)
        error_tn_res_y += smooth_l1_loss(pred_masked[:, :, -8], gt[:, :, -6], beta=0.5, size_average=False)
        error_rel_size += smooth_l1_loss(pred_masked[:, :, -2], gt[:, :, -2], beta=0.5, size_average=False)

        first_nonzero_index = torch.nonzero(mask)[0, 0]
        index = torch.where(mask[first_nonzero_index, :, 0])[0]
        n = index.shape[0]
        error_mat_tn_I_x = criterion_tn_I(pred_masked[index, :][:, index][None, :, :, 0], gt[index, :][:, index][None, :, :, 0])
        error_mat_tn_I_y = criterion_tn_I(pred_masked[index, :][:, index][None, :, :, 1], gt[index, :][:, index][None, :, :, 1])
        error_mat_tn_class_x = criterion_tn_class(pred_masked[index, :][:, index][None, :, :, 2          : 2+num_bins  ].permute(0, 3, 1, 2), gt[index, :][:, index][None, :, :, 2].long())
        error_mat_tn_class_y = criterion_tn_class(pred_masked[index, :][:, index][None, :, :, 2+num_bins : 2+2*num_bins].permute(0, 3, 1, 2), gt[index, :][:, index][None, :, :, 3].long())

        error_mat_rotation_class = criterion_rotation_class(pred_masked[index, :][:, index][None, :, :, -6:-3].permute(0, 3, 1, 2), gt[index, :][:, index][None, :, :, -4].long())
        error_mat_same_size = criterion_same_size(pred_masked[index, :][:, index][None, :, :, -3], gt[index, :][:, index][None, :, :, -3])

        valid_mask = (torch.ones((n, n)) - torch.eye(n))[None, ...].to(X_rel_gt.device)

        error_tn_I_x += torch.sum(error_mat_tn_I_x * valid_mask)
        error_tn_I_y += torch.sum(error_mat_tn_I_y * valid_mask)

        if room_type == 'bedroom':
            error_tn_class_x += torch.sum(error_mat_tn_class_x * valid_mask)
            error_tn_class_y += torch.sum(error_mat_tn_class_y * valid_mask)
        else: # livingroom
            error_tn_class_x += torch.sum(error_mat_tn_class_x)
            error_tn_class_y += torch.sum(error_mat_tn_class_y)

        error_rotation_class += torch.sum(error_mat_rotation_class * valid_mask)
        error_same_size += torch.sum(error_mat_same_size * valid_mask)
        num_rel_off += torch.sum(mask) - n


    loss_z = error_z / (num_rel + 1e-12)
    loss_tn_res = (error_tn_res_x + error_tn_res_y) / (num_rel + 1e-12)
    if room_type == 'bedroom':
        loss_tn_class = (error_tn_class_x + error_tn_class_y) / (num_rel_off + 1e-12)
    else:
        loss_tn_class = (error_tn_class_x + error_tn_class_y) / (num_rel + 1e-12)
    loss_tn_I = (error_tn_I_x + error_tn_I_y) / (num_rel_off + 1e-12)

    loss_rotation_class = error_rotation_class / (num_rel_off + 1e-12)
    loss_same_size = error_same_size / (num_rel_off + 1e-12)
    loss_rel_size = error_rel_size / (num_rel + 1e-12)

    return loss_tn_I, loss_tn_class, loss_tn_res, loss_z, loss_rotation_class, loss_same_size, loss_rel_size



def get_loss(room_type, X_abs_gt, X_abs_pred, X_rel_gt, X_rel_pred, num_class=30, num_each_class=4):

    BATCH_SIZE = X_abs_gt.shape[0]
    loss_dict = {}

    angle_class_loss, angle_residual_loss, abs_rep_loss, batch_idx, matched_gt_idx, matched_pred_idx = compute_abs_all_match_loss_classify_angle(X_abs_gt, X_abs_pred, num_class, num_each_class)

    loss_tn_I, loss_tn_class, loss_tn_res, loss_z, loss_rotation_class, loss_same_size, loss_rel_size = compute_rel_all_match_loss_discrete(X_rel_gt, X_rel_pred, batch_idx, matched_gt_idx, matched_pred_idx, num_class, num_each_class, room_type=room_type)
    # Adjust parameters
    angle_class_loss = angle_class_loss * 0.1
    angle_residual_loss = angle_residual_loss
    abs_rep_loss = abs_rep_loss
    loss_tn_I = loss_tn_I
    loss_tn_class = loss_tn_class * 0.1
    loss_tn_res = loss_tn_res
    loss_z = loss_z
    loss_rotation_class = loss_rotation_class * 0.1
    loss_same_size = loss_same_size
    loss_rel_size = loss_rel_size

    # Total loss
    loss = angle_class_loss + angle_residual_loss + abs_rep_loss + loss_tn_I + loss_tn_class + loss_tn_res + loss_z + loss_rotation_class + loss_same_size + loss_rel_size

    # Record loss
    loss_dict['angle_class_loss'] = angle_class_loss
    loss_dict['angle_residual_loss'] = angle_residual_loss
    loss_dict['abs_rep_loss'] = abs_rep_loss
    loss_dict['loss_tn_I'] = loss_tn_I
    loss_dict['loss_tn_class'] = loss_tn_class
    loss_dict['loss_tn_res'] = loss_tn_res
    loss_dict['loss_z'] = loss_z
    loss_dict['loss_rotation_class'] = loss_rotation_class
    loss_dict['loss_same_size'] = loss_same_size
    loss_dict['loss_rel_size'] = loss_rel_size
    loss_dict['loss'] = loss

    return loss, loss_dict, batch_idx, matched_gt_idx, matched_pred_idx


