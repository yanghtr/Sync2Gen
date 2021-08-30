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

def gen_trans_abs(dataset_dir, dump_dir):

    trans_dict = {}
    for i in select_sem_list:
        trans_dict[i] = []

    for scene_idx in range(NUM_DATA_EXAMPLES):
        scene = np.load(os.path.join(dataset_dir, str(scene_idx) + '_abs.npy'))
        print(scene_idx)
        index = np.where(scene[:, -1])[0]
        for i in index:
            if (i//NUM_EACH_CLASS) in select_sem_list:
                trans_dict[i//NUM_EACH_CLASS].append(scene[i, 3:6])

    if not os.path.exists(f'{dump_dir}/trans_abs'):
        os.makedirs(f'{dump_dir}/trans_abs')

    for i in select_sem_list:
        X = np.stack(trans_dict[i])
        np.save(f'{dump_dir}/trans_abs/{i}_trans.npy', X)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--type', type=str, help='bedroom or living')
    args = parser.parse_args()

    dataset_dir = f'{args.data_path}/{args.type}'
    dump_dir = f'./assets/{args.type}/'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    gen_trans_abs(dataset_dir, dump_dir)
    
