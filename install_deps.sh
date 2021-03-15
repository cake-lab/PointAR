#!/usr/bin/env sh

mkdir -p etc/torch
mkdir -p etc/torch_cluster

wget --directory-prefix etc/torch/ https://download.pytorch.org/whl/cu110/torchvision-0.8.2%2Bcu110-cp38-cp38-linux_x86_64.whl
wget --directory-prefix etc/torch/ https://download.pytorch.org/whl/cu110/torch-1.7.1%2Bcu110-cp38-cp38-linux_x86_64.whl
wget --directory-prefix etc/torch_cluster/ https://pytorch-geometric.com/whl/torch-1.7.0+cu110/torch_cluster-1.5.8-cp38-cp38-linux_x86_64.whl

# pipenv install
pip install etc/torch/torch-1.7.1+cu110-cp38-cp38-linux_x86_64.whl
pip install etc/torch/torchvision-0.8.2+cu110-cp38-cp38-linux_x86_64.whl
pip install etc/torch_cluster/torch_cluster-1.5.8-cp38-cp38-linux_x86_64.whl
