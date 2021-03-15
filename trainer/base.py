"""PointAR trainer, based on PyTorch Lightning
"""

import pytorch_lightning as pl


class BaseTrainer(pl.LightningModule):

    def __init__(self, *args, **kwargs):
        super(BaseTrainer, self).__init__(*args, **kwargs)

    # Methods for overwrite
    def calculate_train_metrics(self, x, y):
        return 0, {}

    def calculate_valid_metrics(self, x, y):
        return 0, {}

    # Training delegate
    def training_step(self, batch, batch_nb):
        x, y = batch

        loss, metrics = self.calculate_train_metrics(x, y)

        return {
            'loss': loss,
            'metrics': metrics
        }

    def training_epoch_end(self, step_outputs):
        if 'metrics' not in step_outputs[0]:
            return

        metrics_key = step_outputs[0]['metrics'].keys()
        metrics_mean = {f'train_{k}': 0 for k in metrics_key}

        for output in step_outputs:
            for k in metrics_key:
                metrics_mean[f'train_{k}'] += output['metrics'][k]

        for k in metrics_key:
            metrics_mean[f'train_{k}'] /= len(step_outputs)

        self.log_dict(metrics_mean)

    # Validation delegate
    def validation_step(self, batch, batch_idx, dataloader_idx):
        x, y = batch
        loss, metrics = self.calculate_valid_metrics(x, y)

        if dataloader_idx:
            tb_prefix = ['valid', 'test'][dataloader_idx]
        else:
            tb_prefix = 'valid'

        return {
            'val_loss': loss,
            'metrics': metrics,
            'dataset': tb_prefix
        }

    def validation_epoch_end(self, step_loaders_outputs):
        for loader_outputs in step_loaders_outputs:
            output_0 = loader_outputs[0]

            if 'metrics' not in output_0:
                continue

            name = output_0['dataset']
            metrics_key = output_0['metrics'].keys()
            metrics_mean = {f'{name}_{k}': 0 for k in metrics_key}

            for output in loader_outputs:
                for k in metrics_key:
                    metrics_mean[f'{name}_{k}'] += output['metrics'][k]

            for k in metrics_key:
                metrics_mean[f'{name}_{k}'] /= len(loader_outputs)

            self.log_dict(metrics_mean)
