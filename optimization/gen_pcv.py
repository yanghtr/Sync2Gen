#!/usr/bin/env python
# coding=utf-8
import os
import argparse
import numpy as np
from sklearn.mixture import GaussianMixture

def compute_Pcv_abs(dump_dir, num_class, num_each_class, n_components=6):
    ''' Fit GMM to absolute translation
    '''
    data_dir = f'{dump_dir}/trans_abs'

    return_dict = {}
    for select_sem in range(num_class):
        print(select_sem)
        fname = os.path.join(data_dir, str(select_sem) + '_trans.npy')
        X = np.load(fname)
        gmm_x = GaussianMixture(n_components=n_components, random_state=0).fit(X[:, 0:1])
        gmm_y = GaussianMixture(n_components=n_components, random_state=0).fit(X[:, 1:2])
        return_dict[select_sem] = [gmm_x, gmm_y]

    return return_dict


def compute_Pcv_rel(dump_dir, num_class, num_each_class, n_components=8):
    ''' Fit GMM to relative translation for i < j 
    '''
    data_dir = f'{dump_dir}/trans_rel' # (i, i) = 0 same object is removed for same class

    return_dict = {}
    for i in range(num_class):
        for j in range(num_class):
            if i > j:
                continue
            select_sem = (i, j)
            print(select_sem)

            fname = os.path.join(data_dir, str(select_sem[0]) + '_' + str(select_sem[1]) + '_trans.npy')
            X = np.load(fname)
            gmm_x = GaussianMixture(n_components=n_components, random_state=0).fit(X[:, 0:1])
            gmm_y = GaussianMixture(n_components=n_components, random_state=0).fit(X[:, 1:2])

            return_dict[(select_sem[0], select_sem[1])] = [gmm_x, gmm_y]

    return return_dict


def dump_Pcv_array(dump_dir, num_class, num_each_class, n_components_abs, n_components_rel):
    Pcv_abs_dict = compute_Pcv_abs(dump_dir, num_class, num_each_class, n_components=n_components_abs)
    Pcv_rel_dict = compute_Pcv_rel(dump_dir, num_class, num_each_class, n_components=n_components_rel)
    Pcv_abs_array = np.zeros((num_class, 2, 3, n_components_abs))
    Pcv_rel_array = np.zeros((num_class, num_class, 2, 3, n_components_rel))

    for k, v in Pcv_abs_dict.items():
        pcvx = np.stack((v[0].weights_.reshape(-1), v[0].means_.reshape(-1), np.sqrt(v[0].covariances_).reshape(-1)), axis=0)
        pcvy = np.stack((v[1].weights_.reshape(-1), v[1].means_.reshape(-1), np.sqrt(v[1].covariances_).reshape(-1)), axis=0)
        Pcv_abs_array[k] = np.stack((pcvx, pcvy), axis=0)

    for k, v in Pcv_rel_dict.items():
        pcvx = np.stack((v[0].weights_.reshape(-1), v[0].means_.reshape(-1), np.sqrt(v[0].covariances_).reshape(-1)), axis=0)
        pcvy = np.stack((v[1].weights_.reshape(-1), v[1].means_.reshape(-1), np.sqrt(v[1].covariances_).reshape(-1)), axis=0)
        Pcv_rel_array[k[0], k[1]] = np.stack((pcvx, pcvy), axis=0)

    np.save(f'{dump_dir}/Pcv_abs.npy', Pcv_abs_array)
    np.save(f'{dump_dir}/Pcv_rel.npy', Pcv_rel_array)
    print('Finish dump !')

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--type', type=str, help='bedroom or living')
    args = parser.parse_args()
    dump_dir = f'./assets/{args.type}/'

    NUM_CLASS = 20
    NUM_EACH_CLASS = 4
    n_components_abs = 6
    n_components_rel = 8
    dump_Pcv_array(dump_dir, NUM_CLASS, NUM_EACH_CLASS, n_components_abs, n_components_rel)
