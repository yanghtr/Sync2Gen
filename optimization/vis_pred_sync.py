#!/usr/bin/env python
# coding=utf-8

import os
import sys
import pickle
import argparse
import numpy as np
import matplotlib.pyplot as plt

from opt_final_joint import sigmoid_rescale
sys.path.append('../')
import vis_utils

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='bedroom or living')
    parser.add_argument('--start', type=int, default=0, help='start index of opt')
    parser.add_argument('--length', type=int, default=5, help='start index of opt')
    parser.add_argument('--vis_dir', type=str, help='vis dir')
    parser.add_argument('--assets_dir', type=str, default='./assets/', help='vis dir')
    parser.add_argument('--pred_dir', type=str, help='pred dir')
    parser.add_argument('--sync_dir', type=str, help='sync dir')
    args = parser.parse_args()

    ''' different threshold for different class '''
    assets_dir = f'{args.assets_dir}/{args.type}'
    num = np.load(f'{assets_dir}/num_unique.npy')
    cw_mat = sigmoid_rescale(num, c=0.5, a=10, min_thres=0.3, max_thres=0.5)


    pred_dir = args.pred_dir 
    sync_dir = args.sync_dir
    vis_dir  = args.vis_dir

    if not os.path.exists(vis_dir):
        os.makedirs(vis_dir)

    '''
        Visualize the 2D/3D scene before sync and after sync
    '''
    for i in range(args.start, args.start + args.length):
        print(i)
        X_pred = np.load(os.path.join(pred_dir, str(i).zfill(4) + '_abs_pred.npy'))
        X_sync = np.load(os.path.join(sync_dir, str(i).zfill(4) + '_sync.npy'))

        vis_utils.draw_scene_2D(X_pred, name=os.path.join(vis_dir, str(i).zfill(4) + '_2d_pred.png'), room_type=args.type, num_class=20, num_each_class=4, abs_dim=X_pred.shape[1], thres=cw_mat)
        vis_utils.draw_scene_3D(X_pred, name=os.path.join(vis_dir, str(i).zfill(4) + '_3d_pred.png'), num_class=20, num_each_class=4, abs_dim=X_pred.shape[1], thres=cw_mat)
        vis_utils.draw_scene_2D(X_sync, name=os.path.join(vis_dir, str(i).zfill(4) + '_2d_sync.png'), room_type=args.type, num_class=20, num_each_class=4, abs_dim=X_sync.shape[1], thres=0.5)
        vis_utils.draw_scene_3D(X_sync, name=os.path.join(vis_dir, str(i).zfill(4) + '_3d_sync.png'), num_class=20, num_each_class=4, abs_dim=X_sync.shape[1], thres=0.5)

