import numpy as np
import pyreality as pr

from sklearn.neighbors import NearestNeighbors


class LightingEstimationDelegate:
    def estimate(self):
        raise NotImplementedError('Must implement estimate method')


class PointsBasedLightProbe:
    data: np.ndarray
    coefficients: np.ndarray
    estimation_delegate: LightingEstimationDelegate

    def __init__(self, points_size=1280):
        self.coefficients = np.zeros((9, 3), dtype=np.float32)

    def add(self, new_points):
        raise NotImplementedError('Must implement add method')

    def update(self, index, points):
        raise NotImplementedError('Must implement update method')

    def remove(self, points_index):
        raise NotImplementedError('Must implement remove method')

    def project_sh_function(self):
        sh = pr.spherical_harmonics_from_sphere_points(self.data)
        self.coefficients = sh.coefficients

    def estimate_sh_coefficients(self):
        shc = self.estimation_delegate.estimate(self.data)
        self.coefficients = shc

        return shc


class DirectionalPointsLightProbe(PointsBasedLightProbe):
    def __init__(self, points_size=1280):
        super().__init__(points_size=points_size)

        self.sphere_anchors = pr.fibonacci_sphere(
            samples=points_size)
        self.sphere_colors = np.zeros_like(
            self.sphere_anchors, dtype=np.float32)
        self.sphere_weights = np.ones(points_size, dtype=np.float32)

        self.neigh = NearestNeighbors(n_neighbors=1, algorithm='kd_tree')
        self.neigh.fit(self.sphere_anchors)

    @property
    def data(self):
        return np.concatenate((self.sphere_anchors, self.sphere_colors), axis=-1)

    def analysis_omega_changes(self, old_state, new_state):
        pass

    def add(self, new_points):
        # original_data = np.empty_like(self.data)
        # original_data[:] = self.data

        p, c = pr.point_cloud_util_split(new_points)

        norms = np.linalg.norm(p, axis=-1)
        positions = p[norms != 0]
        colors = c[norms != 0]
        norms = norms[norms != 0]

        positions /= norms[:, np.newaxis]

        idx = self.neigh.kneighbors(positions, return_distance=False)

        np.add.at(self.sphere_colors, np.squeeze(idx, axis=-1), colors)
        np.add.at(self.sphere_weights, idx, 1)

        self.sphere_colors = self.sphere_colors / self.sphere_weights[:, np.newaxis]
        self.sphere_weights[:] = 1

        # self.analysis_omega_changes(original_data, self.data)

    def update(self):
        raise NotImplementedError('TODO update')

    def remove(self):
        raise NotImplementedError('TODO remove')


# Convenience
def directional_light_probe(points, points_size=1280):
    probe = DirectionalPointsLightProbe(points_size=points_size)
    probe.add(points)

    return probe

__all__ = [
    'LightingEstimationDelegate', 'PointsBasedLightProbe',
    'DirectionalPointsLightProbe', 'directional_light_probe'
]
