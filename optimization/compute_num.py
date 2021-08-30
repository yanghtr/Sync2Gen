#!/usr/bin/env python
# coding=utf-8
import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import curve_fit
from scipy.interpolate import interp1d

import sys; np.set_printoptions(suppress=True, linewidth=240, threshold=sys.maxsize)
NUM_EACH_CLASS = 4
NUM_CLASS = 20
NUM_DATA_EXAMPLES = 4000


def compute_num(dataset_dir):
    distribution = np.zeros((NUM_CLASS)) #  distribution: num_objects per scene

    for scene_idx in range(NUM_DATA_EXAMPLES):
        print(scene_idx)
        scene = np.load(os.path.join(dataset_dir, str(scene_idx) + '_abs.npy'))
        num_sum = np.sum(scene[:, -1].reshape(NUM_CLASS, NUM_EACH_CLASS), axis=1)
        for c in range(NUM_CLASS):
            # distribution[c] += num_sum[c] # all counts
            distribution[c] += num_sum[c] > 0 # unique counts

    return distribution


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--type', type=str, help='bedroom or living')
    args = parser.parse_args()

    dataset_dir = f'{args.data_path}/{args.type}'
    dump_dir = f'./assets/{args.type}/'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    distribution = compute_num(dataset_dir)
    np.save(f'{dump_dir}/num_unique.npy', distribution)

