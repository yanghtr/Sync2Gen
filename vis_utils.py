import os
import numpy as np
import matplotlib
# matplotlib.use("Agg")
from matplotlib import pyplot as plt
import matplotlib.cm as cmx
from matplotlib.patches import Polygon
import matplotlib.colors as colors
from mpl_toolkits.mplot3d import Axes3D
import torch
import utils

NUM_CLASS = 20
color_list = np.arange(NUM_CLASS)

def get_sem_list(room_type):
    if room_type == 'bedroom':
        cat2id = {'Nightstand': 0,
                  'Wardrobe': 1,
                  'King-size Bed': 2,
                  'Pendant Lamp': 3,
                  'Ceiling Lamp': 4,
                  'TV Stand': 5,
                  'Dressing Table': 6,
                  'Corner Table': 7,
                  'Drawer Chest': 8,
                  'Desk': 9,
                  'Lounge Chair': 10,
                  'Single bed': 11,
                  'Dining Chair': 12,
                  'Stool': 13,
                  'Bookcase': 14,
                  'Shelf': 15,
                  'Sideboard': 16,
                  'armchair': 17,
                  'Dressing Chair': 18,
                  'Kids Bed': 19}

    elif room_type == 'living':
        cat2id = {'Dining Chair': 0,
                  'Pendant Lamp': 1,
                  'Coffee Table': 2,
                  'TV Stand': 3,
                  'Dining Table': 4,
                  'Corner Table': 5,
                  'Multi-seat Sofa': 6,
                  'armchair': 7,
                  'Sideboard': 8,
                  'Lounge Chair': 9,
                  'Stool': 10,
                  'Ceiling Lamp': 11,
                  'Bookcase': 12,
                  'Drawer Chest': 13,
                  'Loveseat Sofa': 14,
                  'L-shaped Sofa': 15,
                  'Wine Cabinet': 16,
                  'Nightstand': 17,
                  'Barstool': 18,
                  'Round End Table': 19}
    else:
        raise AssertionError('unknown room type')

    id2cat = {val: key for key, val in cat2id.items()}
    sem_list = [id2cat[key] for key in list(range(NUM_CLASS))]
    return sem_list, id2cat


def set_axes_equal(ax):
    '''Make axes of 3D plot have equal scale so that spheres appear as spheres,
    cubes as cubes, etc..  This is one possible solution to Matplotlib's
    ax.set_aspect('equal') and ax.axis('equal') not working for 3D.

    Input
      ax: a matplotlib axis, e.g., as output from plt.gca().
    '''

    x_limits = ax.get_xlim3d()
    y_limits = ax.get_ylim3d()
    z_limits = ax.get_zlim3d()

    x_range = abs(x_limits[1] - x_limits[0])
    x_middle = np.mean(x_limits)
    y_range = abs(y_limits[1] - y_limits[0])
    y_middle = np.mean(y_limits)
    z_range = abs(z_limits[1] - z_limits[0])
    z_middle = np.mean(z_limits)

    # The plot bounding box is a sphere in the sense of the infinity
    # norm, hence I call half the max range the plot radius.
    plot_radius = 0.5*max([x_range, y_range, z_range])

    ax.set_xlim3d([x_middle - plot_radius, x_middle + plot_radius])
    ax.set_ylim3d([y_middle - plot_radius, y_middle + plot_radius])
    ax.set_zlim3d([z_middle - plot_radius, z_middle + plot_radius])


def draw_scene_3Dbox(ax, p, color, rot=None, abs_dim=16):

    if abs_dim == 16:
        angle_recon = utils.class2angle(np.argmax(p[:8]), p[8], num_class=8)
        dir_1 = np.array([np.cos(angle_recon), np.sin(angle_recon), 0])
        dir_2 = np.zeros((3))
        dir_2[:2] = [-dir_1[1], dir_1[0]]
        dir_3 = np.cross(dir_1, dir_2)

        center = p[9:12]
        size = p[12:15]

    elif abs_dim == 10:
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
    #import ipdb; ipdb.set_trace()
    ax.plot([cornerpoints[0][0], cornerpoints[1][0]], [cornerpoints[0][1], cornerpoints[1][1]],
            [cornerpoints[0][2], cornerpoints[1][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[2][0]], [cornerpoints[0][1], cornerpoints[2][1]],
            [cornerpoints[0][2], cornerpoints[2][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[3][0]], [cornerpoints[1][1], cornerpoints[3][1]],
            [cornerpoints[1][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[3][0]], [cornerpoints[2][1], cornerpoints[3][1]],
            [cornerpoints[2][2], cornerpoints[3][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[5][0]], [cornerpoints[4][1], cornerpoints[5][1]],
            [cornerpoints[4][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[4][0], cornerpoints[6][0]], [cornerpoints[4][1], cornerpoints[6][1]],
            [cornerpoints[4][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[5][0], cornerpoints[7][0]], [cornerpoints[5][1], cornerpoints[7][1]],
            [cornerpoints[5][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[6][0], cornerpoints[7][0]], [cornerpoints[6][1], cornerpoints[7][1]],
            [cornerpoints[6][2], cornerpoints[7][2]], c=color)
    ax.plot([cornerpoints[0][0], cornerpoints[4][0]], [cornerpoints[0][1], cornerpoints[4][1]],
            [cornerpoints[0][2], cornerpoints[4][2]], c=color)
    ax.plot([cornerpoints[1][0], cornerpoints[5][0]], [cornerpoints[1][1], cornerpoints[5][1]],
            [cornerpoints[1][2], cornerpoints[5][2]], c=color)
    ax.plot([cornerpoints[2][0], cornerpoints[6][0]], [cornerpoints[2][1], cornerpoints[6][1]],
            [cornerpoints[2][2], cornerpoints[6][2]], c=color)
    ax.plot([cornerpoints[3][0], cornerpoints[7][0]], [cornerpoints[3][1], cornerpoints[7][1]],
            [cornerpoints[3][2], cornerpoints[7][2]], c=color)

    return

def draw_scene_3D(data, name, num_class=30, num_each_class=4, is_torch=False, abs_dim=16, thres=None):
    cmap = cmx.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=num_class-1), cmap=plt.get_cmap('jet'))
    if is_torch:
        data = data.cpu().numpy()

    fig = plt.figure(0, figsize=(14,14))
    ax = fig.add_subplot(1, 1, 1, projection='3d')

    ax.set_xlim(-3, 3)
    ax.set_ylim(-3, 3)
    ax.set_zlim(0, 6)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel('z')
    for i in range(num_class):
        for j in range(num_each_class):
            dataline = data[i*num_each_class+j,:]
            if type(thres) == float:
                tt = thres
            else:
                tt = thres[i]
            if dataline[-1] > tt:
                draw_scene_3Dbox(ax=ax, p=dataline, color=cmap.to_rgba(color_list[i]), abs_dim=abs_dim)
    set_axes_equal(ax)                
    plt.tight_layout()
    plt.savefig(name)
    plt.close()


def draw_scene_2Dbox(ax, dataline, color, sem, abs_dim=16):
    if abs_dim == 16:
        location = dataline[9:11]
        size = dataline[12:14]
        angle_recon = utils.class2angle(np.argmax(dataline[:8]), dataline[8], num_class=8)
        n1 = np.array([np.cos(angle_recon), np.sin(angle_recon)])
        n2 = np.array([-n1[1], n1[0]])

    elif abs_dim == 10:
        location = dataline[3:5]
        size = dataline[6:8]
        n1 = dataline[:2]
        n1 = n1 / np.linalg.norm(n1)
        n2 = np.array([-n1[1], n1[0]])

    p1 = location + size[1]*n1/2.0 + size[0]*n2/2.0
    p2 = location + size[1]*n1/2.0 - size[0]*n2/2.0
    p3 = location - size[1]*n1/2.0 - size[0]*n2/2.0
    p4 = location - size[1]*n1/2.0 + size[0]*n2/2.0

    x = [p1[0], p2[0], p3[0], p4[0], p1[0]]
    y = [p1[1], p2[1], p3[1], p4[1], p1[1]]
    ax.plot(x, y, color=color)
    ax.text(np.mean(x[:4]), np.mean(y[:4]), sem, fontsize=20)
    o = location
    o1 = location + 0.2 * n1
    ax.plot([o[0], o1[0]], [o[1], o1[1]])


def draw_scene_2D(data, name, room_type, num_class=30, num_each_class=4, is_torch=False, abs_dim=16, thres=None, is_dump=True):
    sem_list, _ = get_sem_list(room_type)

    cmap = cmx.ScalarMappable(norm=matplotlib.colors.Normalize(vmin=0, vmax=num_class-1), cmap=plt.get_cmap('jet'))
    if is_torch:
        data = data.cpu().numpy()

    fig = plt.figure(figsize=(14,14))
    ax = fig.add_subplot(111)

    for i in range(num_class):
        for j in range(num_each_class):
            dataline = data[i*num_each_class+j,:]
            if type(thres) == float:
                tt = thres
            else:
                tt = thres[i]
            if dataline[-1] > tt:
                draw_scene_2Dbox(ax=ax, dataline=dataline, color=cmap.to_rgba(color_list[i]), sem=sem_list[i], abs_dim=abs_dim)
    ax.set_aspect('equal', 'datalim')

    plt.tight_layout()
    if is_dump:
        plt.savefig(name)
        plt.close()
    else:
        plt.show()


