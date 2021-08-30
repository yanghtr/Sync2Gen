#!/usr/bin/env python
# coding=utf-8
'''
    Torch version
'''

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import pickle
import torch
import sys
sys.path.append('../')
import utils


def isOverlapping1D(box1, box2):
    '''
    @Args: Order Matters!
        box1: [xmin1, xmax1], xmin1: (n,), xmax1: (n,)
        box2: [xmin2, xmax2], xmin2: (n,), xmax2: (n,)
    @Returns:
        is_overlapping: (n,)
    '''
    xmin1 = box1[0]
    xmax1 = box1[1]
    xmin2 = box2[0]
    xmax2 = box2[1]
    return (xmax1 >= xmin2) * (xmax2 >= xmin1)


def overlapLength1D(box1, box2):
    '''
    @Args: Order Matters!
        box1: [xmin1, xmax1], xmin1: (n,), xmax1: (n,)
        box2: [xmin2, xmax2], xmin2: (n,), xmax2: (n,)
    @Returns:
        overlapLength1D: (n,)
    '''
    xmin1 = box1[0]
    xmax1 = box1[1]
    xmin2 = box2[0]
    xmax2 = box2[1]
    xmat = torch.stack([xmin1, xmax1, xmin2, xmax2], dim=1) # (n, 4)
    xmat_sorted, _ = torch.sort(xmat)
    return xmat_sorted[:, 2] - xmat_sorted[:, 1]


def overlappingInfo(box1, box2):
    ''' Judge whether two boxes overlap. If overlap, then compute which dim 
    has the smaller overlapping_ratio.
    overlapping_ratio = overlap_length_dim / min{si_dim, sj_dim}, dim=x, y
    @Args:
        box1: (n, 4, 2)
        box2: (n, 4, 2)
    @Returns:
        distance: we need to min max{0, |tj_dim - ti_dim| - distance}
    '''
    xmin1 = torch.min(box1[:, :, 0], dim=1)[0] # (n,)
    xmax1 = torch.max(box1[:, :, 0], dim=1)[0] # (n,)
    xmin2 = torch.min(box2[:, :, 0], dim=1)[0] # (n,)
    xmax2 = torch.max(box2[:, :, 0], dim=1)[0] # (n,)

    ymin1 = torch.min(box1[:, :, 1], dim=1)[0] # (n,)
    ymax1 = torch.max(box1[:, :, 1], dim=1)[0] # (n,)
    ymin2 = torch.min(box2[:, :, 1], dim=1)[0] # (n,)
    ymax2 = torch.max(box2[:, :, 1], dim=1)[0] # (n,)

    overlap_x = isOverlapping1D([xmin1, xmax1], [xmin2, xmax2]) # (n,)
    overlap_y = isOverlapping1D([ymin1, ymax1], [ymin2, ymax2]) # (n,)
    isOverlapping = (overlap_x * overlap_y).float() # (n,)

    length_x = overlapLength1D([xmin1, xmax1], [xmin2, xmax2]) # (n,)
    length_y = overlapLength1D([ymin1, ymax1], [ymin2, ymax2]) # (n,)

    ratio_x = length_x / torch.min( torch.stack((xmax1 - xmin1, xmax2 - xmin2)), dim=0 )[0]
    ratio_y = length_y / torch.min( torch.stack((ymax1 - ymin1, ymax2 - ymin2)), dim=0 )[0]

    select_dim = (ratio_x > ratio_y).long() # (n,), x: 0, y: 1
    distance = ((xmax1 - xmin1) + (xmax2 - xmin2)) / 2 * (1 - select_dim) + ((ymax1 - ymin1) + (ymax2 - ymin2)) / 2 * select_dim

    return isOverlapping, select_dim, distance



def isOverlapping1D_ori(box1, box2):
    '''
    @Args: Order Matters!
        box1: [xmin1, xmax1]
        box2: [xmin2, xmax2]
    '''
    xmin1 = box1[0]
    xmax1 = box1[1]
    xmin2 = box2[0]
    xmax2 = box2[1]
    return xmax1 >= xmin2 and xmax2 >= xmin1

def overlapLength1D_ori(box1, box2):
    '''
    @Args: Order Matters!
        box1: [xmin1, xmax1]
        box2: [xmin2, xmax2]
    '''
    xmin1 = box1[0]
    xmax1 = box1[1]
    xmin2 = box2[0]
    xmax2 = box2[1]
    xlist = [xmin1, xmax1, xmin2, xmax2]
    xlist.sort()
    return xlist[2] - xlist[1]


def overlappingInfo_ori(box1, box2, is_debug=False):
    ''' Judge whether two boxes overlap. If overlap, then compute which dim 
    has the smaller overlapping_ratio.
    overlapping_ratio = overlap_length_dim / min{si_dim, sj_dim}, dim=x, y
    @Args:
        box1: (4, 2)
        box2: (4, 2)
    @Returns:
        distance: we need to min max{0, |tj_dim - ti_dim| - distance}
    '''
    xmin1 = torch.min(box1[:, 0])
    xmax1 = torch.max(box1[:, 0])
    xmin2 = torch.min(box2[:, 0])
    xmax2 = torch.max(box2[:, 0])

    ymin1 = torch.min(box1[:, 1])
    ymax1 = torch.max(box1[:, 1])
    ymin2 = torch.min(box2[:, 1])
    ymax2 = torch.max(box2[:, 1])

    overlap_x = isOverlapping1D_ori([xmin1, xmax1], [xmin2, xmax2])
    overlap_y = isOverlapping1D_ori([ymin1, ymax1], [ymin2, ymax2])
    isOverlapping = (overlap_x and overlap_y)

    select_dim = None
    distance = None
    if isOverlapping:
        length_x = overlapLength1D_ori([xmin1, xmax1], [xmin2, xmax2])
        length_y = overlapLength1D_ori([ymin1, ymax1], [ymin2, ymax2])
         
        ratio_x  = length_x / torch.min(torch.tensor((xmax1 - xmin1, xmax2 - xmin2)))
        ratio_y  = length_y / torch.min(torch.tensor((ymax1 - ymin1, ymax2 - ymin2)))
        # select the dimension with smaller ratio: 0: x dim. 1: y dim.
        # if ratio_x <= ratio_y: 0. if ratio_x > ratio_y: 1
        if ratio_x <= ratio_y + 1e-6:
            select_dim = 0
            distance = ((xmax1 - xmin1) + (xmax2 - xmin2)) / 2
        else:
            select_dim = 1
            distance = ((ymax1 - ymin1) + (ymax2 - ymin2)) / 2

    return isOverlapping, select_dim, distance




if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', type=str, help='data path')
    parser.add_argument('--type', type=str, help='bedroom or living')
    args = parser.parse_args()

    dataset_dir = f'{args.data_path}/{args.type}'
    dump_dir = f'./assets/{args.type}/'
    if not os.path.exists(dump_dir):
        os.makedirs(dump_dir)

    NUM_CLASS = 20
    NUM_EACH_CLASS = 4
    NUM_DATA_EXAMPLES = 4000

    overlap_dict = {}
    for i in range(NUM_CLASS):
        for j in range(NUM_CLASS):
            overlap_dict[(i, j)] = []

    for scene_idx in range(NUM_DATA_EXAMPLES):
        print(scene_idx)
        X_abs = np.load(os.path.join(dataset_dir, str(scene_idx) + '_abs.npy'))
        valid_abs_index = np.where(X_abs[:, -1])[0]
        for idx in valid_abs_index:
            for jdx in valid_abs_index:
                if idx == jdx:
                    continue
                Xi = X_abs[idx]
                Xj = X_abs[jdx]
                boxi = torch.tensor(utils.data2box(Xi))
                boxj = torch.tensor(utils.data2box(Xj))
                isOverlapping, _, _ = overlappingInfo_ori(boxi, boxj)
                overlap_dict[(idx // NUM_EACH_CLASS, jdx // NUM_EACH_CLASS)].append(isOverlapping.numpy())

    with open(f'{dump_dir}/overlap.pkl', 'wb') as f:
        pickle.dump(overlap_dict, f)

    from vis_utils import get_sem_list
    _, id2cat = get_sem_list(args.type)
    overlap_dict = pickle.load(open(f'{dump_dir}/overlap.pkl', 'rb'))
    Povl = np.zeros((NUM_CLASS, NUM_CLASS)) # Probability that two classes overlap, default: 0
    if args.type == 'bedroom':
        ignore_class_list = [3, 4]
    elif args.type == 'living':
        ignore_class_list = [1, 11]
    else:
        raise NotImplementedError()
    for i in range(NUM_CLASS):
        for j in range(NUM_CLASS):
            if (i in ignore_class_list) or (j in ignore_class_list):
                Povl[i, j] = 1
                continue
            
            if len(overlap_dict[(i, j)]) == 0:
                continue # 0

            n = np.sum(overlap_dict[(i, j)])
            N = len(overlap_dict[(i, j)])
            if True:
                Povl[i, j] = n / N

    np.save(f'{dump_dir}/Povl.npy', Povl)

    print('-' * 30, 'These classes can be overlapped: ')
    for i in range(NUM_CLASS):
        for j in range(NUM_CLASS):
            if i <= j and (i not in ignore_class_list) and (j not in ignore_class_list) and Povl[i, j] > 0.7:
                print(i, j, id2cat[i], id2cat[j], Povl[i, j])

