# PointAR: Efficient Lighting Estimation for Mobile Augmented Reality ([Project Page](https://yiqinzhao.me/project/point-ar/))

[Yiqin Zhao](https://yiqinzhao.me), [Tian Guo](https://tianguo.info)


This is the official code release of our ECCV 2020 paper [PointAR: Efficient Lighting Estimation for Mobile Augmented Reality](https://arxiv.org/pdf/2004.00006.pdf).

We propose an efficient lighting estimation pipeline that is suitable to run on modern mobile devices, with comparable resource complexities to state-of-the-art on-device deep learning models. Our pipeline, referred to as PointAR, takes a single RGB-D image captured from the mobile camera and a 2D location in that image, and estimates a 2nd order spherical harmonics coefficients which can be directly utilized by rendering engines for indoor lighting in the context of augmented reality. Our key insight is to formulate the lighting estimation as a learning problem directly from point clouds, which is in part inspired by the Monte Carlo integration leveraged by real-time spherical harmonics lighting. While existing approaches estimate lighting information with complex deep learning pipelines, our method focuses on reducing the computational complexity. Through both quantitative and qualitative experiments, we demonstrate that PointAR achieves lower lighting estimation errors compared to state-of-the-art methods. Further, our method requires an order of magnitude lower resource, comparable to that of mobile-specific DNNs.

## Using the Code

First, clone the repo and update all the submodules.

```bash
git clone git@github.com:cake-lab/PointAR.git
cd PointAR
git submodule update --init --recursive
```

Then, install all the dependencies with `pipenv`:

```bash
./install_deps.sh
```


Please notice that we do not provide generated dataset, but users can use the generation scripts in `datasets/pointar` to generate their own dataset. To do so, one must first request access to Matterport3D and Neural Illumination datasets, as links shown in the end of the document. Additionally, we provide an intermediate abstraction of the Matterport3D dataset in the `matterport_configs.zip` file, please extract the files and modify related path string in the code before conducting generation.

To train the model:

```
python train.py
```

## Links

- Matterport3D dataset: https://github.com/niessner/Matterport
- Neural Illumination: https://illumination.cs.princeton.edu
- PointConv: https://github.com/DylanWusee/pointconv_pytorch