import os
import math
import torch
import importlib
import numpy as np
import pyreality as pr
import pycuda.driver as drv
from pycuda.compiler import SourceModule


class UniSphereTorchSampler(torch.nn.Module):
    def __init__(self, n_samples, n_points):
        super(UniSphereTorchSampler, self).__init__()

        # precomputed anchors
        anchors = pr.fibonacci_sphere(n_samples)
        anchors = torch.transpose(anchors, 1, 0)
        self.register_buffer('anchors', anchors)

        # float value as the max bound
        self.dist_bound = torch.finfo(torch.float32).max

        # Pre computed dist grid
        dist_grid = torch.zeros(
            (n_samples, n_points),
            dtype=torch.float32)
        dist_grid += self.dist_bound
        self.register_buffer('dist_grid', dist_grid)

        # Index buffers
        self.register_buffer('point_idx', torch.arange(
            n_points, dtype=torch.long))
        self.register_buffer('anchor_idx', torch.arange(
            n_samples, dtype=torch.long))

    def forward(self, points):
        dev = self.anchors.device

        t_p = torch.from_numpy(points).to(dev)
        t_d = torch.linalg.norm(t_p, dim=-1)

        t_n = t_d.clone()
        t_n[t_n == 0] = 1
        t_p = t_p / torch.unsqueeze(t_n, -1)

        angel_cos_grid = t_p @ self.anchors
        angel_min_idx = torch.argmax(angel_cos_grid, dim=-1)
        self.dist_grid[angel_min_idx, self.point_idx] = t_d[self.point_idx]

        dist_min_idx = torch.argmin(self.dist_grid, dim=-1)
        dist_min = self.dist_grid[self.anchor_idx, dist_min_idx]
        dist_min_idx[dist_min == self.dist_bound] = -1

        self.dist_grid = self.dist_grid * 0 + self.dist_bound

        return dist_min_idx


class UniSphereCUDASampler:
    def __init__(self, n_anchors):
        os.environ['CUDA_DEVICE_ORDER'] = 'PCI_BUS_ID'
        importlib.import_module('pycuda.autoinit')

        module_src = open(
            './datasets/xihe/preprocess/cuda/sphere.cu', 'r').read()
        module_src = module_src.replace(
            '#define ANCHOR_SIZE 1280',
            f'#define ANCHOR_SIZE {n_anchors}')
        module = SourceModule(module_src)

        self.n_anchors = n_anchors
        self.nn_search = module.get_function("nn_search")
        self.anchors = pr.fibonacci_sphere(n_anchors)

    def forward(self, point_cloud):
        points = np.array(
            point_cloud[:, 0:3], dtype=np.float32, copy=True)
        colors = np.array(
            point_cloud[:, 3:6], dtype=np.float32, copy=True)

        base = np.array([
            len(points), np.finfo(np.float32).max
        ], dtype=np.float32)
        anchor_distance = np.zeros(
            (self.n_anchors, 2), dtype=np.float32) + base

        self.nn_search(
            drv.InOut(anchor_distance),
            drv.In(points),
            drv.In(self.anchors),
            grid=(len(points), 1, 1),
            block=(1024, 1, 1))

        colors_with_base = np.concatenate((colors, [[0, 0, 0]]), axis=0)

        p_idx = anchor_distance[:, 0].astype(np.int32)
        anchor_clr = colors_with_base[p_idx]

        return np.concatenate((self.anchors, anchor_clr), axis=-1)
