import io
import os
import json
import math
import imageio
import importlib
import numpy as np
import pyreality as pr
from functools import lru_cache

from tqdm import tqdm
import multiprocessing
from multiprocessing import Pool
from datasets.pointar.preprocess.utils import map_hdr

# import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule

from datasets.matterport3d import Matterport3DList, matterport3d_root
from datasets.neural_illumination import NeuralIlluminationList
from datasets.neural_illumination import NeuralIlluminationZips

importlib.import_module('pycuda.autoinit')

ni_zips = NeuralIlluminationZips()
mt_list = Matterport3DList()
DS_RATE = 4
DEBUG = False
OUTPUT_PATH = '/mnt/SSD4T/yiqinzhao/Xihe/xihe-dataset'

module = SourceModule(
    open('./datasets/xihe/preprocess/cuda/preprocess.cu', 'r').read()
)

make_point_cloud = module.get_function("makePointCloud")
make_sh_coefficients = module.get_function("makeSHCoefficients")
cameraAdjustment = module.get_function("cameraAdjustment")


def get_batched_basis_at(normal_batch):
    batch_size = normal_batch.shape[0]
    x, y, z = normal_batch[:, 0], normal_batch[:, 1], normal_batch[:, 2]
    sh_basis = np.zeros((batch_size, 9))

    # band 0
    sh_basis[:, 0] = 0.282095

    # band 1
    sh_basis[:, 1] = 0.488603 * y
    sh_basis[:, 2] = 0.488603 * z
    sh_basis[:, 3] = 0.488603 * x

    # band 2
    sh_basis[:, 4] = 1.092548 * x * y
    sh_basis[:, 5] = 1.092548 * y * z
    sh_basis[:, 6] = 0.315392 * (3 * z * z - 1)
    sh_basis[:, 7] = 1.092548 * x * z
    sh_basis[:, 8] = 0.546274 * (x * x - y * y)

    return sh_basis


def srgb_to_linear(srgb):
    mask = srgb >= 0.04045
    srgb[mask] = ((srgb[mask] + 0.055) / 1.055)**2.4
    srgb[~mask] = srgb[~mask] / 12.92


def load_rgbd_as_tensor(scene_id, color_image_name, downsample_rate):
    color_image = f'{matterport3d_root}'\
        + f'/{scene_id}' \
        + f'/undistorted_color_images' \
        + f'/{color_image_name}.jpg'

    depth_image_name = color_image_name.replace('_i', '_d')
    depth_image = f'{matterport3d_root}'\
        + f'/{scene_id}' \
        + f'/undistorted_depth_images' \
        + f'/{depth_image_name}.png'

    # According to Matterport document, we need to flip up down the image
    org_color_img = imageio.imread(color_image)
    org_depth_img = imageio.imread(depth_image)

    img_color = np.flipud(org_color_img) / 255
    img_depth = np.flipud(org_depth_img) / 4000

    img_color = img_color[::downsample_rate, ::downsample_rate, :]
    img_depth = img_depth[::downsample_rate, ::downsample_rate]

    return img_color, img_depth


def fetch_data_file(dataset, index, item):
    gt = item['illumination_map']
    f_zip = ni_zips[item['scene_id']]
    ill_map = item['illumination_map']
    surface_category = item['surface_category']

    # Source image
    img_color, img_depth = load_rgbd_as_tensor(
        item['scene_id'], item['observation_image'], DS_RATE)
    img_color = img_color.astype(np.float32)
    img_depth = img_depth.astype(np.float32)

    # Fetch LDR Image data
    ill_map_ldr = imageio.imread(io.BytesIO(
        f_zip.read(f'illummaps/{surface_category}/{ill_map}.png'))) / 255
    ill_map_ldr = ill_map_ldr[::4, ::4, :]
    ill_map_ldr = np.fliplr(ill_map_ldr)
    ill_map_ldr = ill_map_ldr.astype(np.float32)

    # Fetch HDR image data
    ill_map_hdr_r = map_hdr(
        imageio.imread(io.BytesIO(
            f_zip.read(f'illummaps/{surface_category}/{ill_map}_r.png')))
    ) * 0.8
    ill_map_hdr_g = map_hdr(
        imageio.imread(io.BytesIO(
            f_zip.read(f'illummaps/{surface_category}/{ill_map}_g.png')))
    ) * 1.0
    ill_map_hdr_b = map_hdr(
        imageio.imread(io.BytesIO(
            f_zip.read(f'illummaps/{surface_category}/{ill_map}_b.png')))
    ) * 1.6

    ill_map_hdr = np.stack(
        (ill_map_hdr_r, ill_map_hdr_g, ill_map_hdr_b),
        axis=-1)
    ill_map_hdr = ill_map_hdr[::4, ::4, :]
    ill_map_hdr = ill_map_hdr.astype(np.float32)

    npz_package = {
        'ill_map_ldr': ill_map_ldr,
        'ill_map_hdr': ill_map_hdr,
        'img_color': img_color,
        'img_depth': img_depth
    }

    os.system(f'mkdir -p {OUTPUT_PATH}/{dataset}/{index}')
    np.savez_compressed(
        f'{OUTPUT_PATH}/{dataset}/{index}/source',
        **npz_package)


def process_input(source_package, dataset, index, item):
    depth_name = item['observation_image'].replace('_i', '_d')
    configs = mt_list.get_config(item['scene_id'], depth_name)

    fx = int(configs['intrinsics']['fx'] // DS_RATE)
    fy = int(configs['intrinsics']['fy'] // DS_RATE)
    cx = int(configs['intrinsics']['cx'] // DS_RATE)
    cy = int(configs['intrinsics']['cy'] // DS_RATE)

    mat_ctw = np.array(configs['camera_to_world_matrix'], dtype=np.float32)
    mat_ctw = mat_ctw.reshape((4, 4))

    img_color = source_package['img_color']
    img_depth = source_package['img_depth']

    # Generate Point Cloud
    u = np.arange(img_color.shape[1])
    v = np.arange(img_color.shape[0])
    uv = np.stack(np.meshgrid(u, v), axis=-1)

    xy = (uv - np.array([cx, cy]))
    xy = xy * np.stack([img_depth, img_depth], axis=-1) / np.array([fx, fy])
    z = -img_depth
    xyz = np.concatenate((xy, z[:, :, np.newaxis]), axis=-1)

    # Camera to world
    w = np.ones(xyz.shape[:2])[:, :, np.newaxis]
    xyzw = np.concatenate((xyz, w), axis=-1)
    xyzw_world = np.inner(mat_ctw, xyzw)
    xyzw_world = np.moveaxis(xyzw_world, 0, -1)
    xyz = xyzw_world[:, :, :3]

    ix = int(float(item['ix']) // DS_RATE)
    iy = int(float(item['iy']) // DS_RATE)

    xyz_i = xyz - xyz[iy, ix]
    xyz_r = pr.euler_rotation_xyz(xyz_i, angels=(math.radians(-90), 0, 0))
    xyz_r = xyz_r.astype(np.float32)

    # Save results
    os.system(f'mkdir -p {OUTPUT_PATH}/{dataset}/{index}')

    pc = np.concatenate((xyz_r, img_color), axis=-1)
    pc = pc.reshape((-1, 6)).astype(np.float32)
    np.save(f'{OUTPUT_PATH}/{dataset}/{index}/point_cloud', pc)


def process_input_gpu(source_package, dataset, index, item):
    depth_name = item['observation_image'].replace('_i', '_d')
    configs = mt_list.get_config(item['scene_id'], depth_name)

    fx = int(configs['intrinsics']['fx'] // DS_RATE)
    fy = int(configs['intrinsics']['fy'] // DS_RATE)
    cx = int(configs['intrinsics']['cx'] // DS_RATE)
    cy = int(configs['intrinsics']['cy'] // DS_RATE)

    mat_ctw = np.array(configs['camera_to_world_matrix'], dtype=np.float32)
    mat_ctw = mat_ctw.reshape((4, 4))
    mat_rotation = pr.get_euler_rotation_matrix((math.radians(-90), 0, 0))

    img_color = source_package['img_color']
    img_depth = source_package['img_depth']

    width = img_color.shape[1]
    height = img_color.shape[0]

    intrinsics = np.array(
        [fx, fy, cx, cy, width, height],
        dtype=np.float32)

    # Generate Point Cloud
    xyz_camera_space = np.empty_like(img_color, dtype=np.float32)

    make_point_cloud(
        drv.Out(xyz_camera_space),
        drv.In(img_depth),
        drv.In(intrinsics),
        grid=(
            (width + 32 - 1) // 32,
            (height + 32 - 1) // 32,
            1),
        block=(32, 32, 1)
    )

    xyz_camera_space = np.array(xyz_camera_space, dtype=np.float32)
    xyz_world_space = xyz_camera_space.reshape((-1, 3))

    cameraAdjustment(
        drv.InOut(xyz_world_space),
        drv.In(mat_ctw),
        drv.In(mat_rotation),
        drv.In(intrinsics),
        grid=(
            (width + 32 - 1) // 32,
            (height + 32 - 1) // 32,
            1),
        block=(32, 32, 1)
    )

    if DEBUG:
        img_color = img_color.reshape((-1, 3))
        pc = np.concatenate((xyz_world_space, img_color), axis=-1)
        pc = pc.reshape((-1, 6)).astype(np.float32)
        np.save(f'{OUTPUT_PATH}/{dataset}/{index}/point_cloud_no_shift', pc)

    ix = int(float(item['ix']) // DS_RATE)
    iy = int(float(item['iy']) // DS_RATE)

    # Prepare to save results
    os.system(f'mkdir -p {OUTPUT_PATH}/{dataset}/{index}')

    ray_dirs = xyz_camera_space.reshape((-1, 3))
    ray_dirs_norm = np.linalg.norm(ray_dirs, axis=-1)
    ray_dirs_norm[ray_dirs_norm == 0] = 1
    ray_dirs = ray_dirs / ray_dirs_norm[:, np.newaxis]

    idx = iy * width + ix
    xyz_shifted = xyz_world_space - xyz_world_space[idx]

    img_color = img_color.reshape((-1, 3))

    if DEBUG:
        pc = np.concatenate((xyz_shifted, 0 - ray_dirs), axis=-1)
        pc = pc.reshape((-1, 6)).astype(np.float32)
        np.save(f'{OUTPUT_PATH}/{dataset}/{index}/point_cloud_rays_inverse', pc)

    pc = np.concatenate((xyz_shifted, img_color, ray_dirs), axis=-1)
    pc = pc.reshape((-1, 9)).astype(np.float32)

    # This point cloud has ray direction
    np.save(f'{OUTPUT_PATH}/{dataset}/{index}/point_cloud', pc)


def process_output(source_package, dataset, index, item):
    ill_map_ldr = source_package['ill_map_ldr']
    ill_map_hdr = source_package['ill_map_hdr']

    # Make UV grid to generate a cubemap
    cubemap_resolution = 128
    sample_u = np.arange(cubemap_resolution) / (cubemap_resolution - 1)
    sample_v = np.arange(cubemap_resolution) / (cubemap_resolution - 1)
    sample_uv = np.stack(np.meshgrid(sample_u, sample_v),
                         axis=-1).astype(np.float32)
    sample_uv = sample_uv.reshape((-1, 2))

    cubemap_xyz = np.empty(
        (6, cubemap_resolution * cubemap_resolution, 3),
        dtype=np.float32)

    for i in range(6):
        cubemap_xyz[i] = pr.cube_uv_to_xyz(i, sample_uv)

    cubemap_xyz = cubemap_xyz.reshape((-1, 3))
    cubemap_sph = pr.cartesian_to_spherical(cubemap_xyz)
    cubemap_xyz = pr.euler_rotation_xyz(
        cubemap_xyz,
        (math.radians(-90), math.radians(0), 0))
    cubemap_xyz = pr.euler_rotation_xyz(
        cubemap_xyz,
        (math.radians(0), math.radians(90), 0))

    # Make cubemap canvas for LDR and HDR
    cubemap_color_ldr = np.empty((cubemap_xyz.shape[0], 3), dtype=np.float32)
    cubemap_color_hdr = np.empty((cubemap_xyz.shape[0], 3), dtype=np.float32)

    cubemap_sph_tmp = cubemap_sph[:, :2]  # select theta and phi
    cubemap_sph_tmp = cubemap_sph_tmp + np.array([0, math.pi])
    cubemap_sph_tmp = cubemap_sph_tmp / np.array([math.pi, math.pi * 2])
    s = ill_map_ldr.shape
    cubemap_sph_tmp = cubemap_sph_tmp * np.array([s[0] - 1, s[1] - 1])
    cubemap_sph_tmp = cubemap_sph_tmp.astype(np.int)

    ill_map_ldr_2d = ill_map_ldr.reshape((-1, 3))
    ill_map_hdr_2d = ill_map_hdr.reshape((-1, 3))

    idx = np.arange(cubemap_sph_tmp.shape[0], dtype=np.int)
    cubemap_idx = cubemap_sph_tmp[:, 0] * s[1] + cubemap_sph_tmp[:, 1]

    cubemap_color_ldr[idx, :] = ill_map_ldr_2d[cubemap_idx, :]
    cubemap_color_hdr[idx, :] = ill_map_hdr_2d[cubemap_idx, :]

    # LDR Image need to convert to linear color space
    srgb_to_linear(cubemap_color_ldr)

    # Debug point for dumping cubemap to point cloud
    if DEBUG:
        cubemap_pc = np.concatenate((cubemap_xyz, cubemap_color_ldr), axis=-1)
        np.save(f'{OUTPUT_PATH}/{dataset}/{index}/cubemap', cubemap_pc)

    cubemap_weight = cubemap_xyz.reshape((6, -1, 3))

    cubemap_tmp = np.sum(cubemap_xyz * cubemap_xyz, axis=-1)[:, np.newaxis]
    cubemap_weight = 4 / (np.sqrt(cubemap_tmp) * cubemap_tmp)
    cubemap_norm = cubemap_xyz / \
        np.linalg.norm(cubemap_xyz, axis=-1)[:, np.newaxis]
    cubemap_basis = get_batched_basis_at(cubemap_norm)

    cubemap_res_ldr = cubemap_color_ldr * cubemap_weight
    cubemap_res_hdr = cubemap_color_hdr * cubemap_weight
    shc_ldr = np.einsum('ij,ik->ijk', cubemap_basis,
                        cubemap_res_ldr).sum(axis=0)
    shc_hdr = np.einsum('ij,ik->ijk', cubemap_basis,
                        cubemap_res_hdr).sum(axis=0)

    # normalize
    norm = (4 * math.pi) / np.sum(cubemap_weight)
    shc_ldr = (shc_ldr * norm).reshape(-1)
    shc_hdr = (shc_hdr * norm).reshape(-1)

    f = open(f'{OUTPUT_PATH}/{dataset}/{index}/shc_ldr.json', 'w')
    f.write(json.dumps(shc_ldr.tolist()))
    f.close()

    f = open(f'{OUTPUT_PATH}/{dataset}/{index}/shc_hdr.json', 'w')
    f.write(json.dumps(shc_hdr.tolist()))
    f.close()


@lru_cache(maxsize=3)
def get_cube_idx(width, height):
    # Make UV grid to generate a cubemap
    cubemap_resolution = 128
    sample_u = np.arange(cubemap_resolution) / (cubemap_resolution - 1)
    sample_v = np.arange(cubemap_resolution) / (cubemap_resolution - 1)
    sample_uv = np.stack(np.meshgrid(sample_u, sample_v),
                         axis=-1).astype(np.float32)
    sample_uv = sample_uv.reshape((-1, 2))

    cubemap_xyz = np.empty(
        (6, cubemap_resolution * cubemap_resolution, 3),
        dtype=np.float32)

    for i in range(6):
        cubemap_xyz[i] = pr.cube_uv_to_xyz(i, sample_uv)

    cubemap_xyz_flt = cubemap_xyz.reshape((-1, 3))
    cubemap_sph = pr.cartesian_to_spherical(cubemap_xyz_flt)

    # Rotate XYZ for correcting orientation
    cubemap_xyz_flt = pr.euler_rotation_xyz(
        cubemap_xyz_flt,
        (math.radians(-90), math.radians(0), 0))
    cubemap_xyz_flt = pr.euler_rotation_xyz(
        cubemap_xyz_flt,
        (math.radians(0), math.radians(90), 0))

    cubemap_sph_tmp = cubemap_sph[:, :2]  # select theta and phi
    cubemap_sph_tmp = cubemap_sph_tmp + np.array([0, math.pi])
    cubemap_sph_tmp = cubemap_sph_tmp / np.array([math.pi, math.pi * 2])
    cubemap_sph_tmp = cubemap_sph_tmp * np.array([height - 1, width - 1])
    cubemap_sph_tmp = cubemap_sph_tmp.astype(np.int)

    idx = np.arange(cubemap_sph_tmp.shape[0], dtype=np.int)
    cubemap_idx = cubemap_sph_tmp[:, 0] * width + cubemap_sph_tmp[:, 1]

    cubemap_weight = cubemap_xyz_flt.reshape((6, -1, 3))
    cubemap_tmp = np.sum(cubemap_xyz_flt * cubemap_xyz_flt,
                         axis=-1)[:, np.newaxis]
    cubemap_weight = 4 / (np.sqrt(cubemap_tmp) * cubemap_tmp)
    cubemap_weight = cubemap_weight.astype(np.float32)

    cubemap_norm = cubemap_xyz_flt / \
        np.linalg.norm(cubemap_xyz_flt, axis=-1)[:, np.newaxis]
    cubemap_basis = get_batched_basis_at(cubemap_norm)
    cubemap_basis = cubemap_basis.astype(np.float32)

    shc_norm = (4 * math.pi) / np.sum(cubemap_weight)
    shc_norm = shc_norm.astype(np.float64)

    return cubemap_xyz_flt, idx, cubemap_idx, cubemap_weight, cubemap_basis, shc_norm


def process_output_gpu(source_package, dataset, index, item):
    ill_map_ldr = source_package['ill_map_ldr']
    ill_map_hdr = source_package['ill_map_hdr']

    # Make cubemap canvas for LDR and HDR
    cubemap_xyz_flt, idx, cubemap_idx, cubemap_weight, cubemap_basis, shc_norm = get_cube_idx(
        ill_map_ldr.shape[1], ill_map_ldr.shape[0])
    cubemap_len = cubemap_xyz_flt.shape[0]

    ill_map_ldr_2d = ill_map_ldr.reshape((-1, 3))
    ill_map_hdr_2d = ill_map_hdr.reshape((-1, 3))

    cubemap_color_ldr = np.empty((cubemap_len, 3), dtype=np.float32)
    cubemap_color_hdr = np.empty((cubemap_len, 3), dtype=np.float32)

    cubemap_color_ldr[idx, :] = ill_map_ldr_2d[cubemap_idx, :]
    cubemap_color_hdr[idx, :] = ill_map_hdr_2d[cubemap_idx, :]

    # LDR Image need to convert to linear color space
    srgb_to_linear(cubemap_color_ldr)

    # Debug point for dumping cubemap to point cloud
    if DEBUG:
        cubemap_pc = np.concatenate(
            (cubemap_xyz_flt, cubemap_color_ldr), axis=-1)
        np.save(f'{OUTPUT_PATH}/{dataset}/{index}/cubemap_gpu', cubemap_pc)

    # Calculate the SH coefficients
    cubemap_clr_ldr = cubemap_color_ldr * cubemap_weight
    cubemap_clr_hdr = cubemap_color_hdr * cubemap_weight

    cubemap_clr_ldr = cubemap_clr_ldr.astype(np.float32)
    cubemap_clr_hdr = cubemap_clr_hdr.astype(np.float32)

    len_pixels = cubemap_len // 6
    shc_hdr = np.zeros((9, 3), dtype=np.float64)
    shc_ldr = np.zeros((9, 3), dtype=np.float64)

    make_sh_coefficients(
        drv.InOut(shc_ldr),
        drv.InOut(shc_hdr),
        drv.In(cubemap_basis),
        drv.In(cubemap_clr_ldr),
        drv.In(cubemap_clr_hdr),
        grid=(6, (len_pixels + 1024 - 1) // 1024, 1),
        block=(1, 1024, 1))

    # normalize
    shc_ldr = (shc_ldr * shc_norm).reshape(-1).astype(np.float32)
    shc_hdr = (shc_hdr * shc_norm).reshape(-1).astype(np.float32)

    f = open(f'{OUTPUT_PATH}/{dataset}/{index}/shc_ldr.json', 'w')
    f.write(json.dumps(shc_ldr.tolist()))
    f.close()

    f = open(f'{OUTPUT_PATH}/{dataset}/{index}/shc_hdr.json', 'w')
    f.write(json.dumps(shc_hdr.tolist()))
    f.close()


def process(args):
    dataset, idx, item = args

    # fetch_data_file(dataset, idx, item)
    # return

    source_package = np.load(f'{OUTPUT_PATH}/{dataset}/{idx}/source.npz')
    # process_input(source_package, dataset, idx, item)
    process_input_gpu(source_package, dataset, idx, item)
    # process_output(source_package, dataset, idx, item)
    process_output_gpu(source_package, dataset, idx, item)


def generate(dataset, index='all', debug=False):
    global DEBUG
    DEBUG = debug

    ni_list = NeuralIlluminationList(dataset)

    if index == 'all':
        args = [(dataset, i, v) for i, v in enumerate(ni_list)]

        # Allow cuda execution
        multiprocessing.set_start_method('spawn')

        with Pool(80) as _p:
            list(tqdm(_p.imap(process, args), total=len(args)))

        # for arg in tqdm(args):
        #     process(arg)
    else:
        # fetch_data_file(dataset, index, ni_list[index])

        source_package = np.load(f'{OUTPUT_PATH}/{dataset}/{index}/source.npz')
        process_input_gpu(source_package, dataset, index, ni_list[index])
        process_output_gpu(source_package, dataset, index, ni_list[index])

        print('Finish', dataset, index, f'is_debug={debug}')