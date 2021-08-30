import json
import pickle
import numpy as np
import math
import os,argparse
import math
from shutil import copyfile
import sys; np.set_printoptions(precision=6, suppress=True, linewidth=100, threshold=sys.maxsize)

import open3d as o3d

NUM_EACH_CLASS = 4


def read_obj_vertices(obj_path):
    ''' This is slow. Check obj file format. 
    @Returns:
        v: N_vertices x 3
    '''
    v_list = []
    with open(obj_path, 'r') as f:
        for line in f.readlines():
            if line[:2] == 'v ':
                v_list.append([float(a) for a in line.split()[1:]])
    return np.array(v_list)


def rotation_matrix(axis, theta):
    axis = np.asarray(axis)
    axis = axis / math.sqrt(np.dot(axis, axis))
    a = math.cos(theta / 2.0)
    b, c, d = -axis * math.sin(theta / 2.0)
    aa, bb, cc, dd = a * a, b * b, c * c, d * d
    bc, ad, ac, ab, bd, cd = b * c, a * d, a * c, a * b, b * d, c * d
    return np.array([[aa + bb - cc - dd, 2 * (bc + ad), 2 * (bd - ac)],
                     [2 * (bc - ad), aa + cc - bb - dd, 2 * (cd + ab)],
                     [2 * (bd + ac), 2 * (cd - ab), aa + dd - bb - cc]])


def modelInfo2dict(model_info_path):
    model_info_dict = {}
    with open(model_info_path, 'r') as f:
        info = json.load(f)
    for v in info:
        model_info_dict[v['model_id']] = v
    return model_info_dict


def gen_box_from_params(p):
    '''
    p: r(3), t(3), s(3)
    '''
    dir_1 = np.zeros((3))
    dir_1[:2] = p[:2]
    dir_1 = dir_1 / np.linalg.norm(dir_1)
    dir_2 = np.zeros((3))
    dir_2[:2] = [-dir_1[1], dir_1[0]]
    dir_3 = np.cross(dir_1, dir_2)

    center = p[3:6]
    size = p[6:9]

    cornerpoints = np.zeros([8, 3])
    d1 = 0.5*size[1]*dir_1
    d2 = 0.5*size[0]*dir_2
    d3 = 0.5*size[2]*dir_3
    #d3 = 0
    cornerpoints[0][:] = center - d1 - d2 - d3
    cornerpoints[1][:] = center - d1 + d2 - d3
    cornerpoints[2][:] = center + d1 - d2 - d3
    cornerpoints[3][:] = center + d1 + d2 - d3
    cornerpoints[4][:] = center - d1 - d2 + d3
    cornerpoints[5][:] = center - d1 + d2 + d3
    cornerpoints[6][:] = center + d1 - d2 + d3
    cornerpoints[7][:] = center + d1 + d2 + d3
    return cornerpoints


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument( '--future_path', default = '/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/3D-FUTURE-model', help = 'path to 3D FUTURE')
    parser.add_argument( '--json_path', default = '/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/3D-FRONT', help = 'path to 3D FRONT')
    parser.add_argument( '--model_info_path', default = '/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/model_info.json', help = 'path to model info')
    parser.add_argument( '--save_path', default = './outputs', help = 'path to save result dir')
    parser.add_argument( '--type', type=str, help = 'bedroom or living')
    args = parser.parse_args()

    with open(f'./assets/cat2id_{args.type}.pkl', 'rb') as f:
        cat2id_dict = pickle.load(f)

    model_info_dict = modelInfo2dict(args.model_info_path)

    files = os.listdir(args.json_path)
    if not os.path.exists(args.save_path):
        os.mkdir(args.save_path)

    if args.type == 'bedroom':
        room_types = ['Bedroom', 'MasterBedroom', 'SecondBedroom']
    if args.type == 'living':
        room_types = ['LivingDiningRoom', 'LivingRoom']
    layout_room_dict = {k: [] for k in room_types}

    for n_m, m in enumerate(files):
        with open(args.json_path+'/'+m, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model_jid = []
        model_uid = []
        model_bbox= []

        mesh_uid = []
        mesh_xyz = []
        mesh_faces = []
        print(n_m, m[:-5])
        # model_uid & model_jid store all furniture info of all rooms
        for ff in data['furniture']:
            if 'valid' in ff and ff['valid']:
                model_uid.append(ff['uid']) # used to access 3D-FUTURE-model
                model_jid.append(ff['jid'])
                model_bbox.append(ff['bbox'])
        for mm in data['mesh']: # mesh refers to wall/floor/etc
            mesh_uid.append(mm['uid'])
            mesh_xyz.append(np.reshape(mm['xyz'], [-1, 3]))
            mesh_faces.append(np.reshape(mm['faces'], [-1, 3]))
        scene = data['scene']
        room = scene['room']
        for r in room:
            if r['type'] not in room_types:
                continue

            layout = np.zeros((len(cat2id_dict) * NUM_EACH_CLASS, 10))
            layout_dict = {i: [] for i in range(len(cat2id_dict))}

            room_id = r['instanceid']
            meshes=[]
            children = r['children']
            number = 1
            for c in children:
                
                ref = c['ref']
                if ref not in model_uid: # mesh (wall/floor) not furniture
                    continue
                idx = model_uid.index(ref)

                if not os.path.exists(args.future_path+'/' + model_jid[idx]):
                    print(model_info_dict[model_jid[idx]]['category'])
                    continue

                # v, vt, _, faces, ftc, _ = igl.read_obj(args.future_path+'/' + model_jid[idx] + '/raw_model.obj')
                v = read_obj_vertices(args.future_path+'/' + model_jid[idx] + '/raw_model.obj')

                center = (np.max(v, axis=0) + np.min(v, axis=0)) / 2

                hsize = (np.max(v, axis=0) - np.min(v, axis=0)) / 2 # half size
                bbox = center + np.array([[-1, -1, -1], [-1, 1, -1], [1, -1, -1], [1, 1, -1], [-1, -1, 1], [-1, 1, 1], [1, -1, 1], [1, 1, 1]]) * hsize

                pos = c['pos']
                rot = c['rot'][1:]
                scale = c['scale']

                # GT box after transfomation
                bbox = bbox * scale
                dref = [0,0,1]
                axis = np.cross(dref, rot)
                # Note: in the raw 3dfront definition, angle is half?
                theta = np.arccos(np.dot(dref, rot))*2
                if np.sum(axis) != 0 and not math.isnan(theta):
                    R = rotation_matrix(axis, theta)
                    bbox = np.transpose(bbox)
                    bbox = np.matmul(R, bbox)
                    bbox = np.transpose(bbox)
                bbox = bbox + pos

                cn = np.mean(bbox, axis=0)
                dend = np.mean(bbox[4:, :], axis=0) # center of face z = +1
                di = dend - cn
                di = di / np.linalg.norm(di)
                sc = hsize * scale * 2

                # Our bbox definition
                bbox_me = gen_box_from_params(np.concatenate([[di[2], di[0], di[1]], [cn[2], cn[0], cn[1]], [sc[0], sc[2], sc[1]]]))

                cat_id = cat2id_dict.get(model_info_dict[model_jid[idx]]['category'], -1)
                if cat_id != -1:
                    layout_dict[cat_id].append(np.concatenate([[di[2], di[0], di[1]], [cn[2], cn[0], cn[1]], [sc[0], sc[2], sc[1]], [1]]))


            for k, v in layout_dict.items():
                if len(v) != 0:
                    lv = np.minimum(len(v), NUM_EACH_CLASS)
                    layout[k * NUM_EACH_CLASS : k * NUM_EACH_CLASS + lv] = np.stack(v[:lv], axis=0)

            layout_room_dict[r['type']].append(layout)

    for k, v in layout_room_dict.items():
        np.save(f'{args.save_path}/{k}.npy', np.stack(v, axis=0))



