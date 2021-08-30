#!/usr/bin/env python
# coding=utf-8
import os
import sys
import json
import torch
import numpy as np
import pickle
import itertools
import argparse

import matplotlib.pyplot as plt

MAX_NUM_PARTS = 80
NUM_EACH_CLASS = 4
NUM_CLASS = 20
assert(MAX_NUM_PARTS == NUM_CLASS * NUM_EACH_CLASS)

parser = argparse.ArgumentParser()
parser.add_argument( '--type', type=str, help = 'bedroom or living')
args = parser.parse_args()

if __name__ == '__main__':

    if args.type == 'bedroom':
        raw_data_path = './outputs/Bedroom_train_val.npy'
        dump_dir = './data/bedroom/'
    elif args.type == 'living':
        raw_data_path = './outputs/Livingroom_train_val.npy'
        dump_dir = './data/living/'
    else:
        raise AssertionError('unknown type')

    matdata = np.load(raw_data_path)
    # Project such that angle is in 2D 
    matdata[:, :, 2] = 0
    matdata[:, :, :2] = matdata[:, :, :2] / (np.linalg.norm(matdata[:, :, :2], axis=-1, keepdims=True) + 1e-8)

    for scene_idx in range(matdata.shape[0]):
        print(scene_idx)

        scene = matdata[scene_idx]

        X_abs = np.zeros((MAX_NUM_PARTS, 10))
        X_abs_R = np.zeros((MAX_NUM_PARTS, 3, 3))

        ''' Calculate X_abs
            In the NUM_EACH_CLASS of the original data, data is not necessarily 
            arranged in the first n-th parts. The following make sure in the 
            NUM_EACH_CLASS block, the data is in the first n-th parts.
            TODO: remove it. The newly generated dataset always satisfies.
        '''
        for c in range(NUM_CLASS):
            offset = c * NUM_EACH_CLASS
            mask = np.where(scene[offset : offset + NUM_EACH_CLASS, -1] == 1)[0]
            for i, idx in enumerate(mask):
                idx_new = offset + i
                idx_ori = offset + idx

                xdir = scene[idx_ori, 0:3]
                ydir = np.zeros((3))
                ydir[:2] = [-xdir[1], xdir[0]]
                zdir = np.cross(xdir, ydir)

                rotmat = np.vstack([xdir, ydir, zdir]).T
                
                center = scene[idx_ori, 3:6]
                size = scene[idx_ori, 6:9]

                X_abs[idx_new, :9] = np.concatenate((xdir, center, size))
                X_abs[idx_new, -1] = 1
                X_abs_R[idx_new] = rotmat

        assert(np.all(matdata[scene_idx] == X_abs))
         
        ''' Calculate X_rel
        '''
        X_rel_dict = dict()
        for i in range(MAX_NUM_PARTS):
            X_rel_dict[i] = dict()
            for j in range(MAX_NUM_PARTS):

                if X_abs[i, -1] == 1 and X_abs[j, -1] == 1:
                    Ri = X_abs_R[i]
                    ti = X_abs[i][3:6]
                    si = X_abs[i][6:9]
                    Rj = X_abs_R[j]
                    tj = X_abs[j][3:6]
                    sj = X_abs[j][6:9]
                    
                    Rij = Rj.dot(Ri.T)
                    Rij_vec = Rij.T.reshape(-1)

                    tij = tj - ti

                    X_rel_dict[i][j] = np.hstack((Rij_vec[:2], tij, si, sj, 1)) # 2 + 3 + 6 + 1 = 12

        np.save(os.path.join(dump_dir, str(scene_idx) + '_abs.npy'), X_abs)
        with open(os.path.join(dump_dir, str(scene_idx) + '_rel.pkl'), 'wb') as f:
            pickle.dump(X_rel_dict, f)

    print("Done!")


