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


def compute_Pxv(dataset_dir, dump_dir):
    distribution = np.zeros((NUM_CLASS, NUM_EACH_CLASS + 1)) #  distribution: num_objects per scene

    for scene_idx in range(NUM_DATA_EXAMPLES):
        scene = np.load(os.path.join(dataset_dir, str(scene_idx) + '_abs.npy'))
        num_sum = np.sum(scene[:, -1].reshape(NUM_CLASS, NUM_EACH_CLASS), axis=1)
        for c in range(NUM_CLASS):
            distribution[c, int(num_sum[c])] += 1

    distribution = distribution / np.sum(distribution, axis=1, keepdims=True)
    np.save(f'{dump_dir}/Pxv.npy', distribution)

    return distribution


def pxv_cont(x, pi1, pi2, mu1, mu2, sigma1_2, sigma2_2):
    ''' Use GMM to fit discrete #objects distribution
    np.abs(pi): ensure > 0
    sigmai_2: square of sigma
    '''
    return np.abs(pi1) * np.exp(-(x-mu1) * (x-mu1) / sigma1_2) + np.abs(pi2) * np.exp(-(x-mu2) * (x-mu2) / sigma2_2)


def fit_pxv_cont(Pxv):
    ''' Use GMM to fit discrete Pxv
    @Args:
        Pxv: (NUM_CLASS, NUM_EACH_CLASS + 1)
    '''
    N, M = Pxv.shape[0], Pxv.shape[1]
    assert(N == NUM_CLASS)
    assert(M == NUM_EACH_CLASS + 1)
    params = np.zeros((N, 6)) # Each class the hyperparameters are pi1, pi2, mu1, mu2, sigma1_2, sigma2_2.
    for c in range(N):
        psort_idx = np.argsort(Pxv[c])
        pmax1_idx = psort_idx[-1] # max p
        pmax2_idx = psort_idx[-2] # second max p
        pmax3_idx = psort_idx[-3] # third max p
        print('-' * 10, '\n', c, pmax1_idx, pmax2_idx, pmax3_idx)
        x = np.arange(0, M, 1)
        alpha = 0.1

        def func1(x, pi1, pi2, sigma1_2, sigma2_2):
            return np.abs(pi1) * np.exp(-(x-pmax1_idx) * (x-pmax1_idx) / sigma1_2) + np.abs(pi2) * np.exp(-(x-pmax2_idx) * (x-pmax2_idx) / sigma2_2)

        def func2(x, pi1, pi2, sigma1_2, sigma2_2):
            mu1 = pmax1_idx + (pmax2_idx - pmax1_idx) * alpha
            mu2 = pmax2_idx + (pmax1_idx - pmax2_idx) * alpha
            return np.abs(pi1) * np.exp(-(x-mu1) * (x-mu1) / sigma1_2) + np.abs(pi2) * np.exp(-(x-mu2) * (x-mu2) / sigma2_2)

        def func3(x, pi1, sigma1_2):
            mu1 = pmax1_idx
            return np.abs(pi1) * np.exp(-(x-mu1) * (x-mu1) / sigma1_2)

        def func4(x, pi1, mu1, sigma1_2):
            return np.abs(pi1) * np.exp(-(x-mu1) * (x-mu1) / sigma1_2)

        if np.where(Pxv[c] < 0.01)[0].shape[0] >= 3:
            print('Note: 3/5 are almost zeros')
            popt, pcov = curve_fit(func3, x, Pxv[c])
            params[c] = np.array([popt[0], 0, pmax1_idx, 0, popt[1], 1])
            print('func3')
        else:
            if np.abs(pmax1_idx - pmax2_idx) == 1 and Pxv[c, pmax2_idx] > 0.1:
                popt, pcov = curve_fit(func2, x, Pxv[c])
                params[c] = np.array([popt[0], popt[1], pmax1_idx + (pmax2_idx - pmax1_idx) * alpha, pmax2_idx + (pmax1_idx - pmax2_idx) * alpha, popt[2], popt[3]])
                print('func2')
            else:
                popt, pcov = curve_fit(func1, x, Pxv[c])
                params[c] = np.array([popt[0], popt[1], pmax1_idx, pmax2_idx, popt[2], popt[3]])
                print('func1')

            if np.any(pcov == np.inf):
                print('visualize and check the pattern. Maybe need to use func4')

    return params
 

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--type', type=str, help='bedroom or living')
    args = parser.parse_args()

    dataset_dir = f'{args.data_path}/{args.type}'
    dump_dir = f'./assets/{args.type}/'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    compute_Pxv(dataset_dir, dump_dir)

