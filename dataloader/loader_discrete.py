import sys
import os

import torch
import pickle
import numpy as np
from torch.utils import data

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = os.path.dirname(BASE_DIR)
sys.path.append(ROOT_DIR)

import utils

def close(a, b, r=0.10001):
    return (np.abs(a - b) <= r)

class DatasetDiscrete(data.Dataset):

    def __init__(self, root, obj_list_path, halfRange=6, interval=0.3):
        
        self.root = root
        self.halfRange = halfRange
        self.interval = interval
        self.obj_id_list = []
        with open(obj_list_path, 'r') as f:
            for item in f.readlines():
                self.obj_id_list.append(item.rstrip())


    def __getitem__(self, index):

        X_abs = np.load(os.path.join(self.root, self.obj_id_list[index] + '_abs.npy')) #(N, 16)
        N = X_abs.shape[0]
        X_rel_np = np.zeros((N, N, 12)).astype(np.float32) # The relative size is 12
        with open(os.path.join(self.root, self.obj_id_list[index] + '_rel.pkl'), 'rb') as f:
            X_rel = pickle.load(f)

        for i in range(N):
            if len(X_rel[i]) > 0:
                for j, v in X_rel[i].items():
                    X_rel_np[i, j, :] = X_rel[i][j]

        X_abs_sep = np.zeros((N, 16)) # 8 + 1 + 3 + 3 + 1 = 9
        for i, x_abs in enumerate(X_abs):
            if x_abs[-1] > 0.5:
                angle = np.arctan2(x_abs[1], x_abs[0])
                class_id, residual_angle = utils.angle2class(angle, num_class=8)
                X_abs_sep[i, int(class_id)] = 1
                X_abs_sep[i, 8:] = np.concatenate(([residual_angle], x_abs[3:]))

                # angle_recon = utils.class2angle(class_id, residual_angle, num_class=8)
                # assert(np.abs(x_abs[0] - np.cos(angle_recon)) < 1e-6)
                # assert(np.abs(x_abs[1] - np.sin(angle_recon)) < 1e-6)

        ''' Translation '''
        tn_I_x, tn_class_x, tn_res_x = utils.translation2disc(X_rel_np[:, :, 2:3], halfRange=self.halfRange, interval=self.interval) 
        tn_I_y, tn_class_y, tn_res_y = utils.translation2disc(X_rel_np[:, :, 3:4], halfRange=self.halfRange, interval=self.interval) 
        # tx = utils.disc2translation(tn_I_x, tn_class_x, tn_res_x, halfRange=6, interval=0.3)
        # ty = utils.disc2translation(tn_I_y, tn_class_y, tn_res_y, halfRange=6, interval=0.3)

        ''' Rotation '''
        # 0: no relation. 1: parallel. 2: orthogonal
        thres_R1 = np.cos(np.deg2rad(10))
        thres_R2 = np.cos(np.deg2rad(80))
        abs_cos = np.abs(X_rel_np[:, :, 0:1])
        rotation_class = (abs_cos > thres_R1).astype(np.float32) + (abs_cos < thres_R2).astype(np.float32) * 2

        ''' Size '''
        thres_s1 = 0.1
        same_size_cond = close(X_rel_np[:, :, 7:8], X_rel_np[:, :, 10:11]) * ( (close(X_rel_np[:, :, 5:6], X_rel_np[:, :, 8:9]) * close(X_rel_np[:, :, 6:7], X_rel_np[:, :, 9:10])) + (close(X_rel_np[:, :, 5:6], X_rel_np[:, :, 9:10]) * close(X_rel_np[:, :, 6:7], X_rel_np[:, :, 8:9])) )
        same_size = (same_size_cond != 0).astype(np.float32)

        rel_size = np.linalg.norm(X_rel_np[:, :, 8:11], axis=2, keepdims=True) / (np.linalg.norm(X_rel_np[:, :, 5:8], axis=2, keepdims=True) + 1e-10)

        # (6 + 1 + 1 + 2 + 1): txy(6), tz(1), R(1), size(2), mask(1)
        X_rel_sep = np.concatenate((tn_I_x, tn_I_y, tn_class_x, tn_class_y, tn_res_x, tn_res_y, X_rel_np[:, :, 4:5], rotation_class, same_size, rel_size, X_rel_np[:, :, -1:]), axis=2)
        
        ret_dict = {}
        ret_dict['index'] = int(self.obj_id_list[index])
        ret_dict['X_abs'] = X_abs_sep.astype(np.float32)
        ret_dict['X_rel'] = X_rel_sep.astype(np.float32)
        
        return ret_dict


    def __len__(self):
        return len(self.obj_id_list)


if __name__ == '__main__':

    room_type = 'bedroom'
    # room_type = 'living'
    root = '/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/src/dataset/data'
    data_path = os.path.join(root, room_type)
    # datalist = f'{root}/train_{room_type}.txt'
    datalist = f'{root}/val_{room_type}.txt'

    halfRange = 6
    interval = 0.3
    num_bins = int(halfRange / interval + 1)

    dataset = DatasetDiscrete(data_path, datalist, halfRange=halfRange, interval=interval)

    for idx in range(len(dataset)):
        print(idx)
        example = dataset[idx]
        # Check the dataset
        X_abs = example['X_abs']
        X_rel = example['X_rel']
        MAX_NUM_PARTS = 80
        assert(np.all(X_rel != np.nan))
        assert(np.all(X_rel != np.inf))
        for i in range(MAX_NUM_PARTS):
            for j in range(MAX_NUM_PARTS):
                if X_rel[i,j,-1] == 1:
                    Xi = X_abs[i]
                    Xj = X_abs[j]

                    # '''
                    angle_recon = utils.class2angle(np.where(Xi[:8] == 1)[0], Xi[8], num_class=8)
                    xi = np.zeros((3))
                    yi = np.zeros((3))
                    xi[:2] = [np.cos(angle_recon), np.sin(angle_recon)]
                    yi[:2] = [-xi[1], xi[0]]
                    zi = np.cross(xi, yi)
                    Ri = np.vstack([xi,yi,zi]).T
                    
                    angle_recon = utils.class2angle(np.where(Xj[:8] == 1)[0], Xj[8], num_class=8)
                    xj = np.zeros((3))
                    yj = np.zeros((3))
                    xj[:2] = [np.cos(angle_recon), np.sin(angle_recon)]
                    yj[:2] = [-xj[1], xj[0]]
                    zj = np.cross(xj, yj)
                    Rj = np.vstack([xj,yj,zj]).T
                    
                    # xij = np.zeros(3)
                    # yij = np.zeros(3)
                    # xij[:2] = X_rel[i][j][:2]
                    # yij[:2] = [-xij[1], xij[0]]
                    # zij = np.cross(xij, yij)
                    # Rij = np.vstack([xij, yij, zij]).T
                    # '''

                    ti = Xi[9:12][:, None]
                    tj = Xj[9:12][:, None]
                    tijx = utils.disc2translation(X_rel[i, j, 0], X_rel[i, j, 2], X_rel[i, j, 4], halfRange, interval)
                    tijy = utils.disc2translation(X_rel[i, j, 1], X_rel[i, j, 3], X_rel[i, j, 5], halfRange, interval)
                    tijz = X_rel[i, j, 6]
                    tij = np.array([tijx, tijy, tijz])[:, None]

                    si = Xi[12:15]
                    sj = Xj[12:15]

                    try:
                        ''' Rotation '''
                        abs_cos = np.abs(np.sum(xi * xj))
                        if np.abs(abs_cos - np.cos(np.deg2rad(10))) < 1e-5 or np.abs(abs_cos - np.cos(np.deg2rad(80))) < 1e-5:
                            pass
                        else:
                            if abs_cos > np.cos(np.deg2rad(10)):
                                assert(X_rel[i, j, -4] == 1) # parallel
                            elif abs_cos < np.cos(np.deg2rad(80)):
                                assert(X_rel[i, j, -4] == 2) # orthogonal
                            else:
                                assert(X_rel[i, j, -4] == 0) # no relation
                    except:
                        print('rotation')

                    try:
                        ''' Size '''
                        ss = (close(si[2], sj[2]) and close(si[2], sj[2])) and ( (close(si[0], sj[0]) and close(si[1], sj[1])) or (close(si[0], sj[1]) and close(si[1], sj[0]))  )
                        assert(ss == X_rel[i, j, -3])

                        sij_ratio = np.linalg.norm(sj) / np.linalg.norm(si)
                        assert(np.abs(sij_ratio - X_rel[i, j, -2]) < 1e-4)
                    except:
                        print('size')

                    try:
                        ''' Translation '''
                        assert(np.abs(np.linalg.norm(tj - (ti + tij))) < 1e-4)
                    except:
                        print('translation')

    print(' \n done \n')

