import json
import pickle
import numpy as np
import math
import os,argparse
import math
from shutil import copyfile
import sys; np.set_printoptions(precision=6, suppress=True, linewidth=100, threshold=sys.maxsize)
from loguru import logger

NUM_EACH_CLASS = 4

def modelInfo2dict(model_info_path):
    model_info_dict = {}
    with open(model_info_path, 'r') as f:
        info = json.load(f)
    for v in info:
        model_info_dict[v['model_id']] = v
    return model_info_dict


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument( '--future_path', default = '/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/3D-FUTURE-model', help = 'path to 3D FUTURE')
    parser.add_argument( '--json_path', default = '/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/3D-FRONT', help = 'path to 3D FRONT')
    parser.add_argument( '--model_info_path', default = '/mnt/yanghaitao/Dataset/Scene_Dataset/3D-FRONT/model_info.json', help = 'path to model info')
    parser.add_argument( '--save_path', default = './', help = 'path to save result dir')
    parser.add_argument( '--type', type=str, help = 'bedroom or living')
    args = parser.parse_args()

    logger.add("file_{time}.log")

    with open(f'./assets/cat2id_all.pkl', 'rb') as f:
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

    cat_num_dict = {k: 0 for k in cat2id_dict}

    for n_m, m in enumerate(files):
        with open(args.json_path+'/'+m, 'r', encoding='utf-8') as f:
            data = json.load(f)
        model_jid = []
        model_uid = []
        model_bbox= []

        mesh_uid = []
        mesh_xyz = []
        mesh_faces = []
        logger.info(m[:-5])
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
                    logger.info('not exist: ', model_info_dict[model_jid[idx]]['category'], model_jid[idx])
                    continue

                cat = model_info_dict[model_jid[idx]]['category']
                cat_num_dict[cat] += 1

    cat_num_dict_sort_value = {k: v for k, v in sorted(cat_num_dict.items(), key=lambda item: item[1], reverse=True)}
    print(cat_num_dict_sort_value)

    i = 0
    select_cat = {}
    for k, v in cat_num_dict_sort_value.items():
        select_cat[k] = i
        i += 1
        if i > 19:
            break
    pickle.dump(select_cat, open(f'./assets/cat2id_{args.type}.pkl', 'wb'))


