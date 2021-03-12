import h5py
import numpy as np

from torch import from_numpy
from torch.utils.data import Dataset


class BaseDataset(Dataset):
    arr_source: np.ndarray
    arr_target: np.ndarray
    arr_indices: np.ndarray
    distribution: str
    data_root = 'datasets/package'

    def __init__(self, distribution, use_hdr=False):
        super(BaseDataset, self).__init__()

        hdr_mark = 'hdr' if use_hdr else 'ldr'

        self.distribution = distribution
        self.f_points = h5py.File(
            f'{self.data_root}/xihe-dataset-points.hdf5', 'r')

        self.arr_target = np.moveaxis(self.arr_target[hdr_mark], 1, -1)
        self.arr_indices = np.arange(len(self.arr_target), dtype=np.int)

    def __getitem__(self, idx):
        # TODO: refactor this
        point_cloud = self.__load_source__(idx)
        target_shc = self.__load_target__(idx)

        point_cloud = np.moveaxis(point_cloud, 0, -1)

        xyz = point_cloud[:3, :]
        rgb = point_cloud[3:6, :]
        ray = point_cloud[6:9, :]
        dst = point_cloud[9:, :]

        # hotpatch
        dst[dst == np.finfo(np.float32).max] = 0

        # Original SH coefficients data is channel last
        # change to channel first as PyTorch use it
        target_shc = target_shc.reshape((-1))
        target = from_numpy(target_shc)

        if self.distribution == 'sphere':
            xyz_sphere = from_numpy(xyz)
            dst = from_numpy(dst)
            xyz_world = xyz_sphere * dst
            rgb = from_numpy(rgb)
            ray = from_numpy(ray)

            return (xyz_sphere, xyz_world, rgb, ray), target

        elif self.distribution == 'uniform':
            xyz = from_numpy(xyz * dst)
            rgb = from_numpy(rgb)

            return (xyz, xyz, rgb, rgb), target

    def __len__(self):
        return len(self.arr_indices)

    def __load_source__(self, idx):
        return self.arr_source[self.arr_indices[idx]]

    def __load_target__(self, idx):
        return self.arr_target[self.arr_indices[idx]]


class PointARTrainD10Dataset(BaseDataset):
    def __init__(self, distribution='sphere', n_anchors=1280, *args, **kwargs):
        self.arr_target = np.load(f'{self.data_root}/traind10-shc.npz')
        super().__init__(distribution, *args, **kwargs)
        self.arr_source = self.f_points[f'traind10_{n_anchors}'][distribution]


class PointARTrainDataset(BaseDataset):
    def __init__(self, distribution='sphere', n_anchors=1280, *args, **kwargs):
        self.arr_target = np.load(f'{self.data_root}/train-shc.npz')
        super().__init__(distribution, *args, **kwargs)
        self.arr_source = self.f_points[f'train_{n_anchors}'][distribution]


class PointARTestDataset(BaseDataset):
    def __init__(self, distribution='sphere', n_anchors=1280, *args, **kwargs):
        self.arr_target = np.load(f'{self.data_root}/test-shc.npz')
        super().__init__(distribution, *args, **kwargs)
        self.arr_source = self.f_points[f'test_{n_anchors}'][distribution]
