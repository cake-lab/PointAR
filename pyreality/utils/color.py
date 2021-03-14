import numpy as np


def rgb_to_gray(rgb):
    return np.dot(rgb[..., :3], [0.2989, 0.5870, 0.1140])


def srgb_to_linear(srgb):
    mask = srgb >= 0.04045
    srgb[mask] = ((srgb[mask] + 0.055) / 1.055)**2.4
    srgb[~mask] = srgb[~mask] / 12.92
