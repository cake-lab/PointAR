# PointAR: Efficient Lighting Estimation for Mobile Augmented Reality ([Project Page](https://yiqinzhao.me/project/point-ar/))

[Yiqin Zhao](https://yiqinzhao.me), [Tian Guo](https://tianguo.info)

This is the official code release for [PointAR](https://arxiv.org/pdf/2004.00006.pdf) which was published in ECCV 2020. 

_If you are interested in PointAR 🤗 , please also checkout the GitHub Repo for our follow-up work [Xihe](https://github.com/cake-lab/Xihe)!😁_

## Overview 
We propose an efficient lighting estimation pipeline that is suitable to run on modern mobile devices, with comparable resource complexities to state-of-the-art on-device deep learning models. Our pipeline, referred to as **PointAR**, takes a single RGB-D image captured from the mobile camera and a 2D location in that image, and estimates a 2nd order spherical harmonics coefficients which can be directly utilized by rendering engines for indoor lighting in the context of augmented reality. Our key insight is to formulate the lighting estimation as a learning problem directly from point clouds, which is in part inspired by the Monte Carlo integration leveraged by real-time spherical harmonics lighting. While existing approaches estimate lighting information with complex deep learning pipelines, our method focuses on reducing the computational complexity. Through both quantitative and qualitative experiments, we demonstrate that PointAR achieves lower lighting estimation errors compared to state-of-the-art methods. Further, our method requires an order of magnitude lower resource, comparable to that of mobile-specific DNNs.

## Paper 

[PointAR: Efficient Lighting Estimation for Mobile Augmented Reality](https://arxiv.org/pdf/2004.00006.pdf).

If you use the PointAR data or code, please cite: 

```bibtex
@InProceedings{pointar_eccv2020,
    author="Zhao, Yiqin
    and Guo, Tian",
    title="PointAR: Efficient Lighting Estimation for Mobile Augmented Reality",
    booktitle="European Conference on Computer Vision (ECCV)",
    year="2020",
}
```


## How to use the repo

First, clone the repo.

```bash
git clone git@github.com:cake-lab/PointAR.git
cd PointAR
```

Then, install all the dependencies with `pipenv`:

```bash
pipenv install
pipenv shell

# The following script will automatically install PyTorch with CUDA 11
# If you are running a different CUDA version, please modify corresponding lines
./install_deps.sh
```

## Preprocess Steps

One of the key steps in reproducing our work is to generate the transformed point cloud datasets. We provide the scripts which can be found in `datasets/pointar` for users to generate their respective datasets. At the high level, 
- users should first obtain access to the two open-source datasets (i.e., [Matterport3D]( https://github.com/niessner/Matterport) and [Neural Illumination](https://illumination.cs.princeton.edu) datasets);
- download these two datasets to a desirable directory. For the Matterport3D dataset, unzip the downloaded zip files and place them in a directory with structure similar to `v1/scans/<SCENE_ID>/...`. For the Neural Illumination dataset, just store the downloaded zip files, i.e. `illummaps_<SCENE_ID>.zip`, directly in a directory.
- modify the corresponding the path variable in `config.py` file to reflect the local directory name;
- then use the `gen_data.py` script to start generation.

Note it can take a few hours to generate the entire dataset (~1.4TB) depending on the GPU devices. 


## Model Training

To train the model:

```
python train.py

# For getting help info of the training script
python train.py --help
```
Our point cloud training component leverages the [PointConv](https://github.com/DylanWusee/pointconv_pytorch). 


## Acknowledgement
This work is supported in part by National Science Foundation grants #1755659 and #1815619. 


