import os
import math
import glob
import time

import h5py
import importlib
import numpy as np
import pandas as pd
from tqdm import tqdm
import pyreality as pr

import multiprocessing
from multiprocessing import Pool

import pycuda.driver as drv
from pycuda.compiler import SourceModule
from pycuda.tools import make_default_context

ANCHOR_SIZE = 1792
# ANCHOR_SIZE = 128000
anchors_pos = pr.fibonacci_sphere(ANCHOR_SIZE)

os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
# os.environ['CUDA_VISIBLE_DEVICES'] = str(min(os.getpid() % 4, 2))
importlib.import_module('pycuda.autoinit')

module_src = open('./datasets/pointar/preprocess/cuda/sphere.cu', 'r').read()
module_src = module_src.replace(
    '#define ANCHOR_SIZE 1280',
    f'#define ANCHOR_SIZE {ANCHOR_SIZE}')
module = SourceModule(module_src)
nn_search = module.get_function("nn_search")


def sphere_points(point_cloud):
    points = np.array(point_cloud[:, 0:3], dtype=np.float32, copy=True)
    colors = np.array(point_cloud[:, 3:6], dtype=np.float32, copy=True)
    ray_dirs = np.array(point_cloud[:, 6:], dtype=np.float32, copy=True)

    base = np.array([len(points), np.finfo(np.float32).max], dtype=np.float32)
    anchor_distance = np.zeros((ANCHOR_SIZE, 2), dtype=np.float32) + base

    # t = time.time_ns()

    # GPU nearest neighbor search
    nn_search(
        drv.InOut(anchor_distance),
        drv.In(points),
        drv.In(anchors_pos),
        grid=(len(points), 1, 1),
        block=(1024, 1, 1))

    colors_with_base = np.concatenate((colors, [[0, 0, 0]]), axis=0)
    rays_with_base = np.concatenate((ray_dirs, [[0, 0, 0]]), axis=0)
    p_idx = anchor_distance[:, 0].astype(np.int32)

    anchor_clr = colors_with_base[p_idx]
    anchor_ray = rays_with_base[p_idx]
    anchor_dst = anchor_distance[:, 1, np.newaxis].astype(np.float32)
    anchor_dst[anchor_dst == np.finfo(np.float32).max] = 0


    res = np.concatenate(
        (anchors_pos, anchor_clr, anchor_ray, anchor_dst), axis=-1)

    return res

    # This code for sanity check
    ######################################################

    # Print sorted order of selected indexes
    # tf = pd.DataFrame.from_dict({
    #     'a_idx': a_index,
    #     'p_idx': p_index
    # })
    # tf =tf.sort_values('a_idx')

    # -----------------------------------------------------

    # Numpy staright forward implemention

    # idx = np.arange(len(points), dtype=np.int)

    # # Reduction, color anchor points
    # a_color = np.zeros((ANCHOR_SIZE, 3), dtype=np.float32)

    # p_index_distance = np.array(p_index_distance, dtype=np.float32)
    # p_index = p_index_distance[:, 0].astype(np.int32)
    # p_distance = p_index_distance[:, 1].astype(np.float32)
    # p_distance[p_distance < (0.1 * 0.1)] = np.finfo(np.float32).max

    # c = 0
    # for i in range(ANCHOR_SIZE):
    #     mask = p_index == i
    #     seq = p_distance[mask]

    #     if len(seq) > 0:
    #         c += 1
    #         min_distance_idx = np.argmin(seq)
    #         min_idx = idx[mask][min_distance_idx]
    #         a_color[i, :] = colors[min_idx]


def uniform_points(point_cloud) -> np.array:
    idx = np.sort(np.random.choice(
        point_cloud.shape[0], ANCHOR_SIZE, replace=False))

    p = point_cloud[idx]
    p_pos, p_clr, p_ray = p[:, :3], p[:, 3:6], p[:, 6:]

    # Assume placed object has radius of 0.1
    p_dst = np.linalg.norm(p_pos, axis=-1, keepdims=True)
    p_pos = p_pos / p_dst

    p = np.concatenate((p_pos, p_clr, p_ray, p_dst), axis=-1)
    p = p.astype(np.float32)

    return p


def normalize_points(point_cloud) -> np.array:
    idx = np.sort(np.random.choice(
        point_cloud.shape[0], ANCHOR_SIZE, replace=False))

    p = point_cloud[idx]
    p_pos, p_clr, p_ray = p[:, :3], p[:, 3:6], p[:, 6:]
    p_dist = np.linalg.norm(p_pos, axis=-1, keepdims=True)

    p = np.concatenate((p_pos, p_clr, p_ray, p_dist), axis=-1)
    p = p.astype(np.float32)

    return p


def runner(args):
    dataset, i = args

    pc = np.load(
        f'./datasets/pointar/pointar-dataset' +
        f'/{dataset}/{i}/point_cloud.npy')
    pc -= np.array([0, 0.1, 0, 0, 0, 0, 0, 0, 0])

    u = uniform_points(pc)
    s = sphere_points(pc)
    n = normalize_points(pc)

    return u, s, n


def get_package(dataset):
    g = glob.glob(f'./datasets/pointar/pointar-dataset/{dataset}/*')

    uniform_points_npz = np.zeros((len(g), ANCHOR_SIZE, 10), dtype=np.float32)
    sphere_points_npz = np.zeros((len(g), ANCHOR_SIZE, 10), dtype=np.float32)
    normalize_points_npz = np.zeros(
        (len(g), ANCHOR_SIZE, 10), dtype=np.float32)

    args = [
        (dataset, i)
        for i in range((len(g)))
    ]

    multiprocessing.set_start_method('spawn')

    with Pool(10) as _p:
        result = list(tqdm(_p.imap(runner, args), total=len(args)))

    for i in range(len(result)):
        u, s, n = result[i]

        uniform_points_npz[i] = u
        sphere_points_npz[i] = s
        normalize_points_npz[i] = n

    return uniform_points_npz, sphere_points_npz, normalize_points_npz


def set_hdf5_dataset(group, dataset_name, data):
    if dataset_name in group.keys():
        group[dataset_name][::] = data
    else:
        group.create_dataset(dataset_name, data=data, compression="gzip")


def pack_npz(dataset, index='all'):

    if index != 'all':
        f = h5py.File(
            f'./datasets/pointar' +
            f'/package/pointar-dataset-debug.hdf5', 'a')

        pc = np.load(
            f'./datasets/pointar/pointar-dataset/{dataset}' +
            f'/{index}/point_cloud.npy')
        pc -= np.array([0, 0.1, 0, 0, 0, 0, 0, 0, 0])
        # t = time.time_ns()

        u = uniform_points(pc)
        s = sphere_points(pc)
        n = normalize_points(pc)

        g = f.require_group(f'{dataset}_{ANCHOR_SIZE}')

        set_hdf5_dataset(g, 'uniform', u)
        set_hdf5_dataset(g, 'sphere', s)
        set_hdf5_dataset(g, 'normalize', n)

    else:
        f = h5py.File(
            f'./datasets/pointar' +
            f'/package/pointar-dataset-points.hdf5', 'a')

        g = f.require_group(f'{dataset}_{ANCHOR_SIZE}')

        u, s, n = get_package(dataset)

        set_hdf5_dataset(g, 'uniform', u)
        set_hdf5_dataset(g, 'sphere', s)
        set_hdf5_dataset(g, 'normalize', n)
