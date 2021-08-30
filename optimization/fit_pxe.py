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


def compute_Pxe(dataset_dir, dump_dir):
    distribution = np.zeros((NUM_CLASS, NUM_CLASS, NUM_EACH_CLASS + 1, NUM_EACH_CLASS + 1)) #  distribution: num_objects per scene

    for scene_idx in range(NUM_DATA_EXAMPLES):
        scene = np.load(os.path.join(dataset_dir, str(scene_idx) + '_abs.npy'))
        num_sum = np.sum(scene[:, -1].reshape(NUM_CLASS, NUM_EACH_CLASS), axis=1)
        for ci in range(NUM_CLASS):
            for cj in range(NUM_CLASS):
                if ci < cj:
                    distribution[ci, cj, int(num_sum[ci]), int(num_sum[cj])] += 1

    for ci in range(NUM_CLASS):
        for cj in range(NUM_CLASS):
            if ci < cj:
                distribution[ci, cj] = distribution[ci, cj] / np.sum(distribution[ci, cj])

    np.save(f'{dump_dir}/Pxe.npy', distribution)

    return distribution


def fit_pxe_cont(Pxe):
    ''' Use GMM to fit discrete Pxe
    @Args:
        Pxe: (NUM_CLASS, NUM_CLASS, NUM_EACH_CLASS + 1, NUM_EACH_CLASS + 1)
    '''
    N, M = Pxe.shape[0], Pxe.shape[2]
    assert(N == NUM_CLASS)
    assert(M == NUM_EACH_CLASS + 1)
    params = np.zeros((N, N, 13)) # Each pair of class the hyperparameters are A1, A2, A3, A4, mu1x, mu1y, mu2x, mu2y, mu3x, mu3y, mu4x, mu4y, a

    x = np.arange(NUM_EACH_CLASS + 1)
    y = np.arange(NUM_EACH_CLASS + 1)
    x, y = np.meshgrid(x, y)
    x1d = x.reshape(1, -1)
    y1d = y.reshape(1, -1)
    xy1d = np.stack((x1d, y1d))

    coexist_pairs_list = []
    for ci in range(NUM_CLASS):
        for cj in range(NUM_CLASS):
            if ci < cj:
                pe = Pxe[ci, cj]
                if (pe[2, 2] > 0.1) or ((pe[0, 0] > 0.2) and (pe[1, 1] > 0.2)):
                    coexist_pairs_list.append([ci, cj])
                    print(ci, cj)

                    pxy1d = Pxe[ci, cj].T.reshape(-1)

                    psort_idx = np.unravel_index(np.argsort(Pxe[ci, cj].ravel()), Pxe[ci, cj].shape)
                    mu1x = psort_idx[0][-1] # max p
                    mu1y = psort_idx[1][-1] # max p
                    A1 = Pxe[ci, cj][mu1x, mu1y]

                    mu2x = psort_idx[0][-2]
                    mu2y = psort_idx[1][-2]
                    A2 = Pxe[ci, cj][mu2x, mu2y]

                    mu3x = psort_idx[0][-3]
                    mu3y = psort_idx[1][-3]
                    A3 = Pxe[ci, cj][mu3x, mu3y]

                    mu4x = psort_idx[0][-4]
                    mu4y = psort_idx[1][-4]
                    A4 = Pxe[ci, cj][mu4x, mu4y]

                    print(mu1x, mu1y, mu2x, mu2y, mu3x, mu3y, mu4x, mu4y)

                    def GMM2D(xy, a):
                        x = xy[0]
                        y = xy[1]
                        g = A1 * np.exp( - (a*((x-mu1x)**2) + a*((y-mu1y)**2)) ) + A2 * np.exp( - (a*((x-mu2x)**2) + a*((y-mu2y)**2)) ) + A3 * np.exp( - (a*((x-mu3x)**2) + a*((y-mu3y)**2)) ) + A4 * np.exp( - (a*((x-mu4x)**2) + a*((y-mu4y)**2)) )
                        return g.ravel()

                    popt, pcov = curve_fit(GMM2D, xy1d, pxy1d)
                    params[ci, cj] = np.array([A1, A2, A3, A4, mu1x, mu1y, mu2x, mu2y, mu3x, mu3y, mu4x, mu4y, popt[0]])

    coexist_pairs = np.array(coexist_pairs_list)

    return params, coexist_pairs


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--type', type=str, help='bedroom or living')
    args = parser.parse_args()

    dataset_dir = f'{args.data_path}/{args.type}'
    dump_dir = f'./assets/{args.type}/'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    compute_Pxe(dataset_dir, dump_dir)

    Pxe = np.load(f'{dump_dir}/Pxe.npy')
    assert(NUM_CLASS == Pxe.shape[0] == Pxe.shape[1])
    params, coexist_pairs = fit_pxe_cont(Pxe)

