#!/usr/bin/env python
# coding=utf-8
import os
import sys
import pickle
import argparse
import numpy as np
sys.path.append('../dataloader')
from loader_discrete import DatasetDiscrete

NUM_CLASS = 20
NUM_EACH_CLASS = 4
NUM_PARTS = NUM_EACH_CLASS * NUM_CLASS


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--type', type=str, help='bedroom or living')
    args = parser.parse_args()

    dataset_dir = f'{args.data_path}/{args.type}'
    dump_dir = f'./assets/{args.type}/'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    datalist = f'{args.data_path}/train_{args.type}.txt'

    halfRange = 6
    interval = 0.3
    num_bins = int(halfRange / interval + 1)

    dataset = DatasetDiscrete(dataset_dir, datalist, halfRange=halfRange, interval=interval)

    rotation_dict = {}
    for i in range(NUM_CLASS):
        for j in range(NUM_CLASS):
            rotation_dict[(i, j)] = []

    for idx in range(len(dataset)):
        print(idx)
        example = dataset[idx]
        # Check the dataset
        X_abs = example['X_abs']
        X_rel = example['X_rel']
        for i in range(NUM_PARTS):
            for j in range(NUM_PARTS):
                if i != j and X_rel[i,j,-1] == 1:
                    rotation_dict[(i//NUM_EACH_CLASS, j//NUM_EACH_CLASS)].append(X_rel[i, j, -4])

    Prot = np.zeros((NUM_CLASS, NUM_CLASS, 3))
    for i in range(NUM_CLASS):
        for j in range(NUM_CLASS):
            n = len(rotation_dict[(i, j)])
            if n == 0:
                continue
            n0 = np.where(np.array(rotation_dict[(i, j)]) == 0)[0].shape[0]
            n1 = np.where(np.array(rotation_dict[(i, j)]) == 1)[0].shape[0]
            n2 = np.where(np.array(rotation_dict[(i, j)]) == 2)[0].shape[0]
            Prot[i, j, 0] = n0 / n
            Prot[i, j, 1] = n1 / n
            Prot[i, j, 2] = n2 / n

    '''
        0: no-relation
        1: parallel
        2: orthogonal
    '''
    np.save(f'{dump_dir}/Prot_rel.npy', Prot)

