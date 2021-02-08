"""
Utility functions for build_point_cloud
=====

This is a PointAR preprocessing component for building point cloud from neural
illumination dataset and Matterport3D dataset.
"""

import io
import imageio
import numpy as np


def rgb2gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def read_image_from_zip(f, path):
    img_bytes = io.BytesIO(f.read(path))
    return imageio.imread(img_bytes)


def map_hdr(channel):
    channel = channel.astype(np.float32)

    mask = channel < 3000

    channel[mask] = channel[mask] * 8e-8
    channel[~mask] = 0.00024 * \
        1.0002 ** (channel[~mask] - 3000)

    return channel
