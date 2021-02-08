import os
import fire
import torch
import pytorch_lightning as pl
from pytorch_lightning.callbacks.early_stopping import EarlyStopping

from model.pointconv import PointARPlus
from trainer.utils import train_valid_test_split

from datasets.pointar.loader import PointARTestDataset
from datasets.pointar.loader import PointARTrainDataset
from datasets.pointar.loader import PointARTrainD10Dataset


class ModelSavingCallback(pl.Callback):
    def __init__(self, sample_input):
        self.sample_input = sample_input

    def on_epoch_end(self, trainer, pl_module):
        dump_path = f'./dist/model_dumps'
        os.system(f'mkdir -p {dump_path}')

        # torch.onnx.export(
        #     pl_module, self.sample_input,
        #     f'{dump_path}/{trainer.current_epoch}.onnx',
        #     do_constant_folding=True,
        #     opset_version=11)

        trainer.save_checkpoint(f'{dump_path}/{trainer.current_epoch}.ckpt')


def train(debug=False,
          use_hdr=False,
          normalize=False,
          distribution='sphere',
          n_anchors=1280,
          num_workers=16,
          downsample_rate=1.0,
          batch_size=32):

    # Specify dataset
    TestDataset = PointARTestDataset
    TrainDataset = TestDataset if debug else PointARTrainD10Dataset

    # Get loaders ready
    loader_param = {'distribution': distribution,
                    'use_hdr': use_hdr, 'n_anchors': n_anchors}
    loaders, scaler = train_valid_test_split(
        TrainDataset, loader_param,
        TestDataset, loader_param,
        normalize=normalize, num_workers=num_workers, batch_size=batch_size)

    train_loader, valid_loader, test_loader = loaders

    # Get model ready
    model = PointARPlus(hparams={
        'num_shc': 27,
        'n_anchors': n_anchors,
        'downsample_rate': downsample_rate,
        'min': torch.from_numpy(scaler.min_) if normalize else torch.zeros((27)),
        'scale': torch.from_numpy(scaler.scale_) if normalize else torch.ones((27))
    })

    # Train
    sample_input = (
        torch.zeros((1, 3, n_anchors)).float().cuda(),
        torch.zeros((1, 3, n_anchors)).float().cuda())

    trainer = pl.Trainer(
        gpus=1,
        # accelerator='ddp',
        check_val_every_n_epoch=1,
        callbacks=[
            ModelSavingCallback(
                sample_input=sample_input
            ),
            EarlyStopping(monitor='valid_shc_rmse')
        ])

    # Start training
    trainer.fit(
        model,
        train_dataloader=train_loader,
        val_dataloaders=[valid_loader, test_loader])


if __name__ == '__main__':
    fire.Fire(train)
