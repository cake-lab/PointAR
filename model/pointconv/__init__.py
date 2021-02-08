import torch
import torch.nn.functional as F

from trainer.base import BaseTrainer
from model.pointconv.network import PointConvModel


class PointARPlus(PointConvModel, BaseTrainer):

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters())

    def calculate_train_metrics(self, x, y):
        _, x_xyz_world, x_rgb, _ = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz_world, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        shc_mse_raw = F.mse_loss(source, target)
        shc_rmse_raw = torch.sqrt(shc_mse_raw)

        shc_mse = F.mse_loss(source_norm, target_norm)
        shc_rmse = torch.sqrt(shc_mse)

        # shc_recon_mae_raw = shc_recon_mae_loss(
        #     source.cpu(), target.cpu())
        # shc_recon_mae = shc_recon_mae_loss(
        #     source_norm.cpu(), target_norm.cpu())

        return shc_rmse_raw, {
            'shc_rmse': shc_rmse.item(),
            # 'shc_recon_mae': shc_recon_mae.item()
        }

    def calculate_valid_metrics(self, x, y):
        _, x_xyz_world, x_rgb, _ = x

        n_scale = torch.Tensor(self.hparams['scale']).cuda()
        n_min = torch.Tensor(self.hparams['min']).cuda()

        source = self.forward(x_xyz_world, x_rgb)
        target = y

        source_norm = (source - n_min) / n_scale
        target_norm = (target - n_min) / n_scale

        shc_mse = F.mse_loss(source_norm, target_norm)
        shc_rmse = torch.sqrt(shc_mse)
        # shc_recon_mae = shc_recon_mae_loss(
        #     source_norm.cpu(), target_norm.cpu())

        return shc_rmse, {
            'shc_rmse': shc_rmse.item(),
            # 'shc_recon_mae': shc_recon_mae.item()
        }
