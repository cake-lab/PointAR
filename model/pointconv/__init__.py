"""Training control and evaluation
"""

import torch
import torch.nn.functional as F

from trainer.base import BaseTrainer
from model.pointconv.network import PointConvModel


class PointAR(PointConvModel, BaseTrainer):

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        shc_mse_raw = F.mse_loss(source, target)
        shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }

    def calculate_valid_metrics(self, x, y):
        x_xyz, x_rgb = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        shc_mse_raw = F.mse_loss(source, target)
        shc_mse = F.mse_loss(source_norm, target_norm)

        return shc_mse_raw, {
            'shc_mse': shc_mse.item()
        }
