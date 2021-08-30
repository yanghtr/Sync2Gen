#!/usr/bin/env python
# coding=utf-8

import os
import argparse
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
from sklearn.mixture import GaussianMixture

select_sem_list = list(range(20))
NUM_CLASS = 20
NUM_EACH_CLASS = 4
NUM_DATA_EXAMPLES = 4000

def gen_trans_rel(dataset_dir, dump_dir):

    trans_dict = {}
    for i in select_sem_list:
        for j in select_sem_list:
            trans_dict[(i, j)] = []

    for scene_idx in range(NUM_DATA_EXAMPLES):
        scene = np.load(os.path.join(dataset_dir, str(scene_idx) + '_abs.npy'))
        print(scene_idx)
        index = np.where(scene[:, -1])[0]
        for i in index:
            for j in index:
                if i == j: # remove a lot of zeros
                    continue
                if ((i // NUM_EACH_CLASS)  in select_sem_list) and ((j // NUM_EACH_CLASS) in select_sem_list):
                    ti = scene[i, 3:6]
                    tj = scene[j, 3:6]
                    # tj = ti + tij
                    trans_dict[(i//NUM_EACH_CLASS, j//NUM_EACH_CLASS)].append(tj - ti)

    if not os.path.exists(f'{dump_dir}/trans_rel'):
        os.makedirs(f'{dump_dir}/trans_rel')

    for i in select_sem_list:
        for j in select_sem_list:
            if len(trans_dict[(i, j)]) <= 10: # In GMM, we need to make sure that n_samples >= n_components
                X = np.zeros((10, 3))
            else:
                X = np.stack(trans_dict[(i, j)])
            np.save(f'{dump_dir}/trans_rel/{i}_{j}_trans.npy', X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--type', type=str, help='bedroom or living')
    args = parser.parse_args()

    dataset_dir = f'{args.data_path}/{args.type}'
    dump_dir = f'./assets/{args.type}/'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    num_class = 20
    num_each_class = 4
    n_components = 8

    gen_trans_rel(dataset_dir, dump_dir)

