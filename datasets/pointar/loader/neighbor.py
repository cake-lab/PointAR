import math

import numpy as np
import pyreality as pr
# from PIL import Image, ImageCms

from torch import from_numpy  # pylint: disable=no-name-in-module
from torch.utils.data import Dataset
from sklearn.preprocessing import MinMaxScaler


class BaseDataset(Dataset):
    source_npz: np.ndarray
    target_npz: np.ndarray
    data_root = 'datasets/package'

    def __init__(self, n_neighbors, use_hdr=False):
        super(BaseDataset, self).__init__()

        hdr_mark = 'hdr' if use_hdr else 'ldr'

        self.source = np.moveaxis(self.source_npz['sphere'], 1, -1)
        self.target = np.moveaxis(self.target_npz[hdr_mark], 1, -1)

    def __getitem__(self, idx):
        point_cloud = self.__load_source__(idx)
        target_shc = self.__load_target__(idx)

        rgb = point_cloud[3:, :]

        # Original SH coefficients data is channel last
        # change to channel first as PyTorch use it
        target_shc = target_shc.reshape((-1))

        rgb = from_numpy(rgb)
        target = from_numpy(target_shc)

        return rgb, target

    def __len__(self):
        return len(self.source)

    def __load_source__(self, idx):
        return self.source[idx]

    def __load_target__(self, idx):
        return self.target[idx]


class XiheNeighborTrainD10Dataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.source_npz = np.load(f'{self.data_root}/traind10-points.npz')
        self.target_npz = np.load(f'{self.data_root}/traind10-shc.npz')

        super().__init__(*args, **kwargs)


class XiheNeighborTrainDataset(BaseDataset):
    def __init__(self,  *args, **kwargs):
        self.source_npz = np.load(f'{self.data_root}/train-points.npz')
        self.target_npz = np.load(f'{self.data_root}/train-shc.npz')

        super().__init__(*args, **kwargs)


class XiheNeighborTestDataset(BaseDataset):
    def __init__(self, *args, **kwargs):
        self.source_npz = np.load(f'{self.data_root}/test-points.npz')
        self.target_npz = np.load(f'{self.data_root}/test-shc.npz')

        super().__init__(*args, **kwargs)
