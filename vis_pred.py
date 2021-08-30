#!/usr/bin/env python
# coding=utf-8
import os
import numpy as np
import argparse
import vis_utils

parser = argparse.ArgumentParser()
parser.add_argument('--type', type=str, help='bedroom or living')
parser.add_argument('--data_dir', type=str, help='data dir')
parser.add_argument('--dump_dir', type=str, help='dump dir')
parser.add_argument('--max_num', type=int, default=100, help='num of scenes to visualize')
args = parser.parse_args()

if not os.path.exists(args.dump_dir):
    os.makedirs(args.dump_dir)

flist = [v for v in os.listdir(args.data_dir) if v.endswith('_abs_pred.npy')]
flist.sort()
if len(flist) > args.max_num:
    flist = flist[args.max_num]

for i, fname in enumerate(flist):
    print(i, fname)
    pred_abs = np.load(os.path.join(args.data_dir, fname))
    vis_utils.draw_scene_2D(pred_abs, f'./{args.dump_dir}/{i:04d}_2D.png', room_type=args.type, num_class=20, num_each_class=4, abs_dim=16)
    vis_utils.draw_scene_3D(pred_abs, f'./{args.dump_dir}/{i:04d}_3D.png', num_class=20, num_each_class=4, abs_dim=16)



