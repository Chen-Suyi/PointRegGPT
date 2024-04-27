import math
import os
from pathlib import Path
import pickle
import shutil
from random import random, shuffle, seed
from functools import partial
from collections import namedtuple
from multiprocessing import cpu_count
from typing import Optional, Union, Sequence
import gc

import cv2
import numpy as np
from scipy.spatial.transform import Rotation
import open3d as o3d
import matplotlib.pyplot as plt

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm
from ema_pytorch import EMA

from accelerate import Accelerator

from pytorch_fid.inception import InceptionV3
from pytorch_fid.fid_score import calculate_frechet_distance

# constants

ModelPrediction = namedtuple('ModelPrediction', ['pred_noise', 'pred_x_start'])

# helpers functions


def intrinsic_transform(
        intrinsic: np.ndarray,
        resize: Optional[Union[int, Sequence[int]]] = None,
        centercrop: Optional[Union[int, Sequence[int]]] = None) -> np.ndarray:
    """
    resize: (h, w)
    centercrop: (h, w)
    """

    old_fx = intrinsic[..., 0, 0]
    old_fy = intrinsic[..., 1, 1]
    old_cx = intrinsic[..., 0, 2]
    old_cy = intrinsic[..., 1, 2]

    old_size_x = np.int32(old_cx * 2)
    old_size_y = np.int32(old_cy * 2)

    new_fx = old_fx
    new_fy = old_fy
    new_cx = old_cx
    new_cy = old_cx

    new_size_x = old_size_x
    new_size_y = old_size_y

    # processing impact from resize
    if resize is not None:
        old_size_x = np.int32(old_cx * 2)
        old_size_y = np.int32(old_cy * 2)
        if type(resize) == int:
            if (old_size_x < old_size_y).all():
                new_size_x = int(resize)
                new_size_y = np.int32(
                    np.floor(resize * old_size_y / old_size_x))
            else:
                new_size_x = np.int32(
                    np.floor(resize * old_size_x / old_size_y))
                new_size_y = np.int32(resize)

        elif type(resize) == tuple:
            new_size_x = np.int32(resize[1])
            new_size_y = np.int32(resize[0])

        new_fx = np.float32(old_fx * new_size_x / old_size_x)
        new_fy = np.float32(old_fy * new_size_y / old_size_y)
        new_cx = np.float32(new_size_x / 2)
        new_cy = np.float32(new_size_y / 2)

    # processing impact from centercrop
    if centercrop is not None:
        if type(centercrop) == int:
            crop_width = centercrop
            crop_height = centercrop

        elif type(centercrop) == tuple:
            crop_width = centercrop[1]
            crop_height = centercrop[0]

        crop_left = np.int32(np.round((new_size_x - crop_width) / 2.0))
        crop_top = np.int32(np.round((new_size_y - crop_height) / 2.0))

        new_cx = new_cx - crop_left
        new_cy = new_cy - crop_top

    # make up new intrinsic
    new_intrinsic = np.zeros_like(intrinsic)
    new_intrinsic[..., 0, 0] = new_fx
    new_intrinsic[..., 1, 1] = new_fy
    new_intrinsic[..., 0, 2] = new_cx
    new_intrinsic[..., 1, 2] = new_cy
    new_intrinsic[..., 2, 2] = 1.0

    return new_intrinsic


def point_cloud(depth, intrinsic, clip=[0, 10]):
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]

    rows, cols = depth.shape
    r, c = np.meshgrid(np.arange(rows), np.arange(cols), indexing='ij')
    # Valid depths are defined by the camera clipping planes
    valid = (depth > clip[0]) & (depth < clip[1])
    # the unit of depth is 10m
    z = np.where(valid, depth, np.nan)
    # Center c and r relatively to the image size cols and rows
    x = np.where(valid, (c - cx) * z / fx, np.nan)
    y = np.where(valid, (r - cy) * z / fy, np.nan)

    pc = np.dstack((x, y, z))
    pc = pc.reshape(-1, 3)
    valid = valid.reshape(-1)
    pc = pc[valid]

    return pc


def depth_image(pc, intrinsic, image_size=[480, 640]):
    rows, cols = image_size

    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]

    x = pc[..., 0]
    y = pc[..., 1]
    z = pc[..., 2]

    c = np.round(x * fx / z + cx).astype(np.int32)
    r = np.round(y * fy / z + cy).astype(np.int32)

    valid_c = (c >= 0) & (c < cols)
    valid_r = (r >= 0) & (r < rows)
    valid_rc = valid_r & valid_c

    depth = np.zeros(image_size)
    depth[c[valid_rc], r[valid_rc]] = z[valid_rc]
    depth = depth.astype(np.float32)

    mask = np.zeros(image_size)
    mask[c[valid_rc], r[valid_rc]] = 1
    mask = mask.astype(bool)

    return depth, mask


def depth2pc_tensor(depth, intrinsic, *, clip=[0, 10], invalid_num=None):
    if invalid_num is None:
        invalid_num = torch.nan
    invalid_num = torch.tensor(invalid_num).to(dtype=depth.dtype,
                                               device=depth.device)

    fx = intrinsic[..., 0, 0]  # (b, )
    fy = intrinsic[..., 1, 1]  # (b, )
    cx = intrinsic[..., 0, 2]  # (b, )
    cy = intrinsic[..., 1, 2]  # (b, )

    batch, channels, rows, cols = depth.shape
    r, c = torch.meshgrid(torch.arange(rows),
                          torch.arange(cols),
                          indexing='ij')  # (h, w) and (h, w)
    r, c = r.to(depth.device), c.to(depth.device)
    # Valid depths are defined by the camera clipping planes
    if clip is None:
        valid = torch.ones_like(depth, dtype=bool)
    else:
        valid = (depth > clip[0]) & (depth < clip[1])  # （b, c, h, w）
    # the unit of depth is 10m
    z = torch.where(valid, depth, invalid_num)  # （b, c, h, w）
    # Center c and r relatively to the image size cols and rows
    x = torch.where(valid, (c[None, None, :, :] - cx[:, None, None, None]) *
                    z / fx[:, None, None, None], invalid_num)
    y = torch.where(valid, (r[None, None, :, :] - cy[:, None, None, None]) *
                    z / fy[:, None, None, None], invalid_num)

    pc = torch.stack((x, y, z), dim=-1)  # （b, c, h, w, 3）
    pc = pc.reshape(batch, -1, 3)  # （b, n, 3）
    valid = valid.reshape(batch, -1)  # （b, n)

    return pc, valid


def pc2depth_tensor(pc, valid, intrinsic, *, image_size=[480, 640]):
    batch_size, point_num, _ = pc.shape
    rows, cols = image_size

    fx = intrinsic[..., 0, 0]  # (b,)
    fy = intrinsic[..., 1, 1]  # (b,)
    cx = intrinsic[..., 0, 2]  # (b,)
    cy = intrinsic[..., 1, 2]  # (b,)

    x = pc[..., 0]  # (b, n)
    y = pc[..., 1]  # (b, n)
    z = pc[..., 2]  # (b, n)

    c = torch.round(x * fx[:, None] / z + cx[:, None]).to(torch.long)  # (b, n)
    r = torch.round(y * fy[:, None] / z + cy[:, None]).to(torch.long)  # (b, n)
    b = torch.arange(batch_size).to(device=pc.device,
                                    dtype=torch.long).reshape(
                                        (batch_size,
                                         1)).repeat(1, point_num)  # (b, n)

    valid_c = (c >= 0) & (c < cols)
    valid_r = (r >= 0) & (r < rows)
    valid_rc = valid_r & valid_c & valid & (z > 0)

    # get valid value
    b = b[valid_rc]  # (valid,)
    r = r[valid_rc]  # (valid,)
    c = c[valid_rc]  # (valid,)
    z = z[valid_rc]  # (valid,)

    # # sort, the front point should block the back
    # sorted_indices = torch.argsort(z, dim=-1, descending=True)
    # b = b[sorted_indices]
    # r = r[sorted_indices]
    # c = c[sorted_indices]
    # z = z[sorted_indices]

    # depth = torch.zeros((batch_size, rows, cols)).to(device=pc.device)
    # depth[b, r, c] = z

    linear_indices = b * rows * cols + r * cols + c
    depth = torch.zeros(batch_size * rows * cols).to(device=pc.device)
    depth = depth.scatter_reduce(dim=0,
                                 index=linear_indices,
                                 src=z,
                                 reduce="amin",
                                 include_self=False)
    depth = depth.reshape((batch_size, 1, rows, cols)).to(torch.float32)

    mask = torch.zeros((batch_size, rows, cols)).to(device=pc.device)
    mask[b, r, c] = 1
    mask = mask.reshape((batch_size, 1, rows, cols)).to(torch.bool)

    return depth, mask


def reproject_tensor(depth: torch.Tensor,
                     intrinsic: torch.Tensor,
                     relative_pose: torch.Tensor,
                     *,
                     clip=[0, 10],
                     invalid_num=None) -> torch.Tensor:
    b, c, h, w = depth.shape
    pc, valid = depth2pc_tensor(depth,
                                intrinsic,
                                clip=clip,
                                invalid_num=invalid_num)
    pc = torch.matmul(pc, relative_pose[:, :3, :3].transpose(
        -1, -2)) + relative_pose[:, None, :3, 3]
    depth_reproj, mask = pc2depth_tensor(pc,
                                         valid,
                                         intrinsic,
                                         image_size=[h, w])

    return depth_reproj, mask


def compute_overlap_region(src_xyz, tgt_xyz, voxel_size=0.025):
    o3d_tgt = o3d.geometry.PointCloud()
    o3d_tgt.points = o3d.utility.Vector3dVector(tgt_xyz)

    # voxel downsample
    overlap_radius = voxel_size * 1.5
    o3d_tgt_down = o3d_tgt.voxel_down_sample(voxel_size=voxel_size)
    tgt_tree = o3d.geometry.KDTreeFlann(o3d_tgt_down)

    # Find knn
    src_corr = np.full(src_xyz.shape[0], -1)
    for i, s in enumerate(src_xyz):
        num_knn, knn_indices, _ = tgt_tree.search_hybrid_vector_3d(
            s, radius=overlap_radius, max_nn=1)
        if num_knn > 0:
            src_corr[i] = knn_indices[0]

    # Compute overlapping ratio
    overlap_mask = src_corr >= 0
    return overlap_mask


def collate_func(data_dicts: list):
    collated_dict = {}
    for data_dict in data_dicts:
        for k, v in data_dict.items():
            if k not in collated_dict:
                collated_dict[k] = []
            collated_dict[k].append(v)

    for key, value in collated_dict.items():
        collated_dict[key] = torch.stack(value)

    return collated_dict


def data_to_device(data, device):
    if isinstance(data, list):
        for i, e in enumerate(data):
            data[i] = data_to_device(e, device)
    elif isinstance(data, dict):
        for k, v in data.items():
            data[k] = data_to_device(v, device)
    elif isinstance(data, tuple):
        for i, e in enumerate(data):
            data[i] = data_to_device(e, device)
    elif isinstance(data, torch.Tensor):
        data = data.to(device)
    else:
        pass

    return data


def param_vector(intrinsic):
    fx = intrinsic[..., 0, 0]
    fy = intrinsic[..., 1, 1]
    cx = intrinsic[..., 0, 2]
    cy = intrinsic[..., 1, 2]

    param_vec = torch.stack([fx, fy, cx, cy], dim=-1)

    return param_vec


def random_sample_intrinsic(batch_size) -> np.ndarray:
    """
    intrinsic matrices collected from 3DMatch
    """
    intrinsic_candidates = np.array(
        [[[585.0, 0.0, 320.0], [0.0, 585.0, 240.0], [0.0, 0.0, 1.0]],
         [[572.0, 0.0, 320.0], [0.0, 572.0, 240.0], [0.0, 0.0, 1.0]],
         [[583.0, 0.0, 320.0], [0.0, 583.0, 240.0], [0.0, 0.0, 1.0]],
         [[540.021232, 0.0, 320.0], [0.0, 540.021232, 240.0], [0.0, 0.0, 1.0]],
         [[570.342205, 0.0, 320.0], [0.0, 570.342205, 240.0], [0.0, 0.0, 1.0]],
         [[533.069214, 0.0, 320.0], [0.0, 533.069214, 240.0], [0.0, 0.0, 1.0]]
         ],
        dtype=np.float32)
    prob = np.array([7, 8, 18, 5, 47, 5])
    prob = prob / np.sum(prob)
    random_indices = np.random.choice(len(intrinsic_candidates),
                                      batch_size,
                                      replace=True,
                                      p=prob)

    return intrinsic_candidates[random_indices]


def random_sample_transform(intrinsic: np.ndarray, image_size=256):
    batch_size = intrinsic.shape[0]
    h, w = image_size, image_size

    fx = intrinsic[..., 0, 0]
    fy = intrinsic[..., 1, 1]
    cx = intrinsic[..., 0, 2]
    cy = intrinsic[..., 1, 2]

    lx = cx
    rx = w - cx
    ty = cy
    dy = h - cy

    theta_min, theta_max = -np.arctan(dy / fy), np.arctan(ty /
                                                          fy)  # pivot x-axis
    phi_min, phi_max = -np.arctan(lx / fx), np.arctan(rx / fx)  # pivot y-axis

    theta_seed = np.random.rand(intrinsic.shape[0])
    theta = (theta_seed * (theta_max - theta_min) + theta_min)

    phi_seed = np.random.rand(intrinsic.shape[0])
    phi = (phi_seed * (phi_max - phi_min) + phi_min)

    psi_seed = np.random.rand(intrinsic.shape[0])
    psi = (psi_seed * 2 * np.pi - np.pi)

    eular = np.stack((theta, phi, psi), axis=-1)

    rotation = Rotation.from_euler("XYZ", eular,
                                   degrees=False).as_matrix()  # (b, 3, 3)
    translation = np.random.randn(batch_size, 3) / 3 * 0  # (b, 3)
    transform = np.stack([np.eye(4) for _ in range(batch_size)])

    transform[..., :3, :3] = rotation[..., :3, :3]
    transform[..., :3, 3] = translation[..., :3]
    transform = transform.astype(dtype=np.float32)

    return transform

def random_sample_pose(batch_size, center=[0, 0, 3]):
    theta_min, theta_max = -np.pi/24, np.pi/24  # pivot x-axis
    phi_min, phi_max = -np.pi/12, np.pi/12  # pivot y-axis

    theta_seed = np.random.rand(batch_size)
    theta = (theta_seed * (theta_max - theta_min) + theta_min)

    phi_seed = np.random.rand(batch_size)
    phi = (phi_seed * (phi_max - phi_min) + phi_min)

    psi = np.zeros(batch_size)

    eular = np.stack((theta, phi, psi), axis=-1)

    rotation = Rotation.from_euler("XYZ", eular,
                                   degrees=False).as_matrix()  # (b, 3, 3)
    center = np.array(center)
    random_trans = np.random.randn(batch_size, 3) / 3
    random_trans[:, -1] = 0
    translation = center - rotation @ center + random_trans  # (b, 3)
    transform = np.stack([np.eye(4) for _ in range(batch_size)])

    transform[..., :3, :3] = rotation[..., :3, :3]
    transform[..., :3, 3] = translation[..., :3]
    transform = transform.astype(dtype=np.float32)

    return transform


def occlusion_filter(depth_rpj: torch.Tensor, mask_rpj: torch.Tensor):
    # preprocess
    depth_pre = depth_rpj.clone()
    depth_pre[~mask_rpj] = torch.inf

    # min pool
    min_neighbors = -F.max_pool2d(
        -depth_pre, kernel_size=3, stride=1, padding=1)

    assert min_neighbors.shape == depth_rpj.shape

    # filter
    mask = (depth_rpj - min_neighbors) < 0.0375
    # depth_rpj[~mask] = 0
    # mask_rpj = mask_rpj & mask
    depth_rpj = torch.where(mask, depth_rpj, min_neighbors)

    return depth_rpj, mask_rpj


def image_condition(depth,
                    intrinsic,
                    relative_pose,
                    depth_unit=10,
                    depth_clip=[0, 10],
                    use_occlusion_filter=False) -> torch.Tensor:
    """Get image condition

    Args:
        depth: (b, 1, h, w) value range from 0~1
        intrinsic: (b, 3, 3) intrinsic matrix for depth
        relative_pose: (b, 4, 4) objective pose
        depth_unit: the unit of depth will be interpreted depth_unit (m)
        depth_clip: clip range in (m)

    Returns:
        img_cond: (b, 2, h ,w) image condition normalized to [-1, 1] (the first channel: reprojected depth, the second channel: mask)
    """
    depth_rpj, mask_rpj = reproject_tensor(depth * depth_unit,
                                           intrinsic,
                                           relative_pose,
                                           clip=depth_clip)

    if use_occlusion_filter:
        depth_rpj, mask_rpj = occlusion_filter(depth_rpj, mask_rpj)

    depth_norm = depth_rpj / depth_unit
    img_cond = torch.cat([depth_norm, mask_rpj], dim=1)
    img_cond = normalize_to_neg_one_to_one(img_cond)

    return img_cond


def null_image_condition(batch_size,
                         image_size,
                         dtype=None,
                         device=None) -> torch.Tensor:
    return -torch.ones(
        (batch_size, 2, image_size, image_size)).to(dtype=dtype, device=device)


def get_mask_from_img_cond(img_cond):
    return unnormalize_to_zero_to_one(img_cond[:, 1, None, ...]) > 0.5


# helpers functions


def exists(x):
    return x is not None


def default(val, d):
    if exists(val):
        return val
    return d() if callable(d) else d


def identity(t, *args, **kwargs):
    return t


def cycle(dl):
    while True:
        for data in dl:
            yield data


def has_int_squareroot(num):
    return (math.sqrt(num)**2) == num


def num_to_groups(num, divisor):
    groups = num // divisor
    remainder = num % divisor
    arr = [divisor] * groups
    if remainder > 0:
        arr.append(remainder)
    return arr


def convert_image_to_fn(img_type, image):
    if image.mode != img_type:
        return image.convert(img_type)
    return image


# normalization functions


def normalize_to_neg_one_to_one(img):
    return img * 2 - 1


def unnormalize_to_zero_to_one(t):
    return (t + 1) * 0.5


# classifier free guidance functions


def uniform(shape, device):
    return torch.zeros(shape, device=device).float().uniform_(0, 1)


def prob_mask_like(shape, prob, device):
    if prob == 1:
        return torch.ones(shape, device=device, dtype=torch.bool)
    elif prob == 0:
        return torch.zeros(shape, device=device, dtype=torch.bool)
    else:
        return torch.zeros(shape, device=device).float().uniform_(0, 1) < prob


# small helper modules


class Residual(nn.Module):
    def __init__(self, fn):
        super().__init__()
        self.fn = fn

    def forward(self, x, *args, **kwargs):
        return self.fn(x, *args, **kwargs) + x


def Upsample(dim, dim_out=None):
    return nn.Sequential(nn.Upsample(scale_factor=2, mode='nearest'),
                         nn.Conv2d(dim, default(dim_out, dim), 3, padding=1))


def Downsample(dim, dim_out=None):
    return nn.Conv2d(dim, default(dim_out, dim), 4, 2, 1)


class WeightStandardizedConv2d(nn.Conv2d):
    """
    https://arxiv.org/abs/1903.10520
    weight standardization purportedly works synergistically with group normalization
    """
    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3

        weight = self.weight
        mean = reduce(weight, 'o ... -> o 1 1 1', 'mean')
        var = reduce(weight, 'o ... -> o 1 1 1',
                     partial(torch.var, unbiased=False))
        normalized_weight = (weight - mean) * (var + eps).rsqrt()

        return F.conv2d(x, normalized_weight, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)


class LayerNorm(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, dim, 1, 1))

    def forward(self, x):
        eps = 1e-5 if x.dtype == torch.float32 else 1e-3
        var = torch.var(x, dim=1, unbiased=False, keepdim=True)
        mean = torch.mean(x, dim=1, keepdim=True)
        return (x - mean) * (var + eps).rsqrt() * self.g


class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.fn = fn
        self.norm = LayerNorm(dim)

    def forward(self, x):
        x = self.norm(x)
        return self.fn(x)


# sinusoidal positional embeds


class SinusoidalPosEmb(nn.Module):
    def __init__(self, dim):
        super().__init__()
        self.dim = dim

    def forward(self, x):
        device = x.device
        half_dim = self.dim // 2
        emb = math.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, device=device) * -emb)
        emb = x[:, None] * emb[None, :]
        emb = torch.cat((emb.sin(), emb.cos()), dim=-1)
        return emb


class RandomOrLearnedSinusoidalPosEmb(nn.Module):
    """ following @crowsonkb 's lead with random (learned optional) sinusoidal pos emb """
    """ https://github.com/crowsonkb/v-diffusion-jax/blob/master/diffusion/models/danbooru_128.py#L8 """
    def __init__(self, dim, is_random=False):
        super().__init__()
        assert (dim % 2) == 0
        half_dim = dim // 2
        self.weights = nn.Parameter(torch.randn(half_dim),
                                    requires_grad=not is_random)

    def forward(self, x):
        x = rearrange(x, 'b -> b 1')
        freqs = x * rearrange(self.weights, 'd -> 1 d') * 2 * math.pi
        fouriered = torch.cat((freqs.sin(), freqs.cos()), dim=-1)
        fouriered = torch.cat((x, fouriered), dim=-1)
        return fouriered


# building block modules


class Block(nn.Module):
    def __init__(self, dim, dim_out, groups=8):
        super().__init__()
        self.proj = WeightStandardizedConv2d(dim, dim_out, 3, padding=1)
        self.norm = nn.GroupNorm(groups, dim_out)
        self.act = nn.SiLU()

    def forward(self, x, scale_shift=None):
        x = self.proj(x)
        x = self.norm(x)

        if exists(scale_shift):
            scale, shift = scale_shift
            x = x * (scale + 1) + shift

        x = self.act(x)
        return x


class ResnetBlock(nn.Module):
    def __init__(self,
                 dim,
                 dim_out,
                 *,
                 time_emb_dim=None,
                 param_emb_dim=None,
                 groups=8):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.SiLU(),
            nn.Linear(
                int(time_emb_dim) + int(param_emb_dim), dim_out *
                2)) if exists(time_emb_dim) or exists(param_emb_dim) else None

        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out,
                                  1) if dim != dim_out else nn.Identity()

    def forward(self, x, time_emb=None, param_emb=None):

        scale_shift = None
        if exists(self.mlp) and (exists(time_emb) or exists(param_emb)):
            cond_emb = tuple(filter(exists, (time_emb, param_emb)))
            cond_emb = torch.cat(cond_emb, dim=-1)
            cond_emb = self.mlp(cond_emb)
            cond_emb = rearrange(cond_emb, 'b c -> b c 1 1')
            scale_shift = cond_emb.chunk(2, dim=1)

        h = self.block1(x, scale_shift=scale_shift)

        h = self.block2(h)

        return h + self.res_conv(x)


class LinearAttention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads
        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)

        self.to_out = nn.Sequential(nn.Conv2d(hidden_dim, dim, 1),
                                    LayerNorm(dim))

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv)

        q = q.softmax(dim=-2)
        k = k.softmax(dim=-1)

        q = q * self.scale
        v = v / (h * w)

        context = torch.einsum('b h d n, b h e n -> b h d e', k, v)

        out = torch.einsum('b h d e, b h d n -> b h e n', context, q)
        out = rearrange(out,
                        'b h c (x y) -> b (h c) x y',
                        h=self.heads,
                        x=h,
                        y=w)
        return self.to_out(out)


class Attention(nn.Module):
    def __init__(self, dim, heads=4, dim_head=32):
        super().__init__()
        self.scale = dim_head**-0.5
        self.heads = heads
        hidden_dim = dim_head * heads

        self.to_qkv = nn.Conv2d(dim, hidden_dim * 3, 1, bias=False)
        self.to_out = nn.Conv2d(hidden_dim, dim, 1)

    def forward(self, x):
        b, c, h, w = x.shape
        qkv = self.to_qkv(x).chunk(3, dim=1)
        q, k, v = map(
            lambda t: rearrange(t, 'b (h c) x y -> b h c (x y)', h=self.heads),
            qkv)

        q = q * self.scale

        sim = einsum('b h d i, b h d j -> b h i j', q, k)
        attn = sim.softmax(dim=-1)
        out = einsum('b h i j, b h d j -> b h i d', attn, v)

        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)
        return self.to_out(out)


# model


class Unet(nn.Module):
    def __init__(
        self,
        dim,
        param_cond_dim,
        init_dim=None,
        out_dim=None,
        dim_mults=(1, 2, 4, 8),
        channels=1,
        resnet_block_groups=8,
        learned_variance=False,
        learned_sinusoidal_cond=False,
        random_fourier_features=False,
        learned_sinusoidal_dim=16,
    ):
        super().__init__()
        # determine dimensions

        self.channels = channels
        input_channels = channels

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(input_channels, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # time embeddings

        time_dim = dim * 4

        self.random_or_learned_sinusoidal_cond = learned_sinusoidal_cond or random_fourier_features

        if self.random_or_learned_sinusoidal_cond:
            sinu_pos_emb = RandomOrLearnedSinusoidalPosEmb(
                learned_sinusoidal_dim, random_fourier_features)
            fourier_dim = learned_sinusoidal_dim + 1
        else:
            sinu_pos_emb = SinusoidalPosEmb(dim)
            fourier_dim = dim

        self.time_mlp = nn.Sequential(sinu_pos_emb,
                                      nn.Linear(fourier_dim, time_dim),
                                      nn.GELU(), nn.Linear(time_dim, time_dim))

        # param embeddings
        self.param_cond_dim = param_cond_dim

        param_emb_dim = dim * 4

        self.param_mlp = nn.Sequential(
            nn.Linear(param_cond_dim, param_emb_dim), nn.GELU(),
            nn.Linear(param_emb_dim, param_emb_dim))

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in,
                                dim_in,
                                time_emb_dim=time_dim,
                                param_emb_dim=param_emb_dim),
                    block_klass(dim_in,
                                dim_in,
                                time_emb_dim=time_dim,
                                param_emb_dim=param_emb_dim),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                        dim_in, dim_out, 3, padding=1)
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim,
                                      mid_dim,
                                      time_emb_dim=time_dim,
                                      param_emb_dim=param_emb_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim,
                                      mid_dim,
                                      time_emb_dim=time_dim,
                                      param_emb_dim=param_emb_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in,
                                dim_out,
                                time_emb_dim=time_dim,
                                param_emb_dim=param_emb_dim),
                    block_klass(dim_out + dim_in,
                                dim_out,
                                time_emb_dim=time_dim,
                                param_emb_dim=param_emb_dim),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                        dim_out, dim_in, 3, padding=1)
                ]))

        default_out_dim = channels * (1 if not learned_variance else 2)
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2,
                                           dim,
                                           time_emb_dim=time_dim,
                                           param_emb_dim=param_emb_dim)
        self.final_conv = nn.Conv2d(dim, self.out_dim, 1)

    def forward(self, x, time, param_cond, img_cond=None):
        batch, image_size, dtype, device = x.shape[0], x.shape[
            -1], x.dtype, x.device

        # derive param condition
        p = self.param_mlp(param_cond)

        # unet

        x = self.init_conv(x)
        r = x.clone()

        t = self.time_mlp(time)

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x, t, p)
            h.append(x)

            x = block2(x, t, p)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x, t, p)
        x = self.mid_attn(x)
        x = self.mid_block2(x, t, p)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x, t, p)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x, t, p)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x, t, p)

        return self.final_conv(x)


# gaussian diffusion trainer class


def extract(a, t, x_shape):
    b, *_ = t.shape
    out = a.gather(-1, t)
    return out.reshape(b, *((1, ) * (len(x_shape) - 1)))


def linear_beta_schedule(timesteps):
    scale = 1000 / timesteps
    beta_start = scale * 0.0001
    beta_end = scale * 0.02
    return torch.linspace(beta_start, beta_end, timesteps, dtype=torch.float64)


def cosine_beta_schedule(timesteps, s=0.008):
    """
    cosine schedule
    as proposed in https://openreview.net/forum?id=-NEXDKk8gZ
    """
    steps = timesteps + 1
    x = torch.linspace(0, timesteps, steps, dtype=torch.float64)
    alphas_cumprod = torch.cos(
        ((x / timesteps) + s) / (1 + s) * math.pi * 0.5)**2
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


def sigmoid_beta_schedule(timesteps, start=-3, end=3, tau=1, clamp_min=1e-5):
    """
    sigmoid schedule
    proposed in https://arxiv.org/abs/2212.11972 - Figure 8
    better for images > 64x64, when used during training
    """
    steps = timesteps + 1
    t = torch.linspace(0, timesteps, steps, dtype=torch.float64) / timesteps
    v_start = torch.tensor(start / tau).sigmoid()
    v_end = torch.tensor(end / tau).sigmoid()
    alphas_cumprod = (-(
        (t *
         (end - start) + start) / tau).sigmoid() + v_end) / (v_end - v_start)
    alphas_cumprod = alphas_cumprod / alphas_cumprod[0]
    betas = 1 - (alphas_cumprod[1:] / alphas_cumprod[:-1])
    return torch.clip(betas, 0, 0.999)


class GaussianDiffusion(nn.Module):
    def __init__(self,
                 model,
                 *,
                 image_size,
                 timesteps=1000,
                 sampling_timesteps=None,
                 loss_type='l1',
                 objective='pred_noise',
                 beta_schedule='cosine',
                 ddim_sampling_eta=1.,
                 min_snr_loss_weight=False,
                 min_snr_gamma=5,
                 is_ddnm_sampling=True,
                 ddnm_sampling_dropout=0.,
                 ddnm_dropout_schedule='none'):
        super().__init__()
        assert not (type(self) == GaussianDiffusion
                    and model.channels != model.out_dim)
        assert not model.random_or_learned_sinusoidal_cond

        self.model = model
        self.channels = self.model.channels

        self.image_size = image_size

        self.objective = objective

        assert objective in {
            'pred_noise', 'pred_x0', 'pred_v'
        }, 'objective must be either pred_noise (predict noise) or pred_x0 (predict image start) or pred_v (predict v [v-parameterization as defined in appendix D of progressive distillation paper, used in imagen-video successfully])'

        if beta_schedule == 'linear':
            betas = linear_beta_schedule(timesteps)
        elif beta_schedule == 'cosine':
            betas = cosine_beta_schedule(timesteps)
        elif beta_schedule == 'sigmoid':
            betas = sigmoid_beta_schedule(timesteps)
        else:
            raise ValueError(f'unknown beta schedule {beta_schedule}')

        alphas = 1. - betas
        alphas_cumprod = torch.cumprod(alphas, dim=0)
        alphas_cumprod_prev = F.pad(alphas_cumprod[:-1], (1, 0), value=1.)

        timesteps, = betas.shape
        self.num_timesteps = int(timesteps)
        self.loss_type = loss_type

        # sampling related parameters

        self.sampling_timesteps = default(
            sampling_timesteps, timesteps
        )  # default num sampling timesteps to number of timesteps at training

        assert self.sampling_timesteps <= timesteps
        self.is_ddim_sampling = self.sampling_timesteps < timesteps
        self.ddim_sampling_eta = ddim_sampling_eta

        self.is_ddnm_sampling = is_ddnm_sampling
        self.ddnm_sampling_dropout = ddnm_sampling_dropout
        if ddnm_dropout_schedule == 'none':
            self.ddnm_dropouts = torch.linspace(self.ddnm_sampling_dropout,
                                                self.ddnm_sampling_dropout,
                                                timesteps,
                                                dtype=torch.float64)
        elif ddnm_dropout_schedule == 'linear':
            # t1000 -> 0 ~ t0 -> p
            self.ddnm_dropouts = torch.linspace(self.ddnm_sampling_dropout,
                                                0.,
                                                timesteps,
                                                dtype=torch.float64)
        else:
            raise ValueError(
                f'unknown ddnm dropout schedule {ddnm_dropout_schedule}')

        self.denoise_dropouts = torch.linspace(1.,
                                               0.,
                                               timesteps,
                                               dtype=torch.float64)**100

        # helper function to register buffer from float64 to float32

        register_buffer = lambda name, val: self.register_buffer(
            name, val.to(torch.float32))

        register_buffer('betas', betas)
        register_buffer('alphas_cumprod', alphas_cumprod)
        register_buffer('alphas_cumprod_prev', alphas_cumprod_prev)

        # calculations for diffusion q(x_t | x_{t-1}) and others

        register_buffer('sqrt_alphas_cumprod', torch.sqrt(alphas_cumprod))
        register_buffer('sqrt_one_minus_alphas_cumprod',
                        torch.sqrt(1. - alphas_cumprod))
        register_buffer('log_one_minus_alphas_cumprod',
                        torch.log(1. - alphas_cumprod))
        register_buffer('sqrt_recip_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod))
        register_buffer('sqrt_recipm1_alphas_cumprod',
                        torch.sqrt(1. / alphas_cumprod - 1))

        # calculations for posterior q(x_{t-1} | x_t, x_0)

        posterior_variance = betas * (1. - alphas_cumprod_prev) / (
            1. - alphas_cumprod)

        # above: equal to 1. / (1. / (1. - alpha_cumprod_tm1) + alpha_t / beta_t)

        register_buffer('posterior_variance', posterior_variance)

        # below: log calculation clipped because the posterior variance is 0 at the beginning of the diffusion chain

        register_buffer('posterior_log_variance_clipped',
                        torch.log(posterior_variance.clamp(min=1e-20)))
        register_buffer(
            'posterior_mean_coef1',
            betas * torch.sqrt(alphas_cumprod_prev) / (1. - alphas_cumprod))
        register_buffer('posterior_mean_coef2', (1. - alphas_cumprod_prev) *
                        torch.sqrt(alphas) / (1. - alphas_cumprod))

        # loss weight

        snr = alphas_cumprod / (1 - alphas_cumprod)

        maybe_clipped_snr = snr.clone()
        if min_snr_loss_weight:
            maybe_clipped_snr.clamp_(max=min_snr_gamma)

        if objective == 'pred_noise':
            loss_weight = maybe_clipped_snr / snr
        elif objective == 'pred_x0':
            loss_weight = maybe_clipped_snr
        elif objective == 'pred_v':
            loss_weight = maybe_clipped_snr / (snr + 1)

        register_buffer('loss_weight', loss_weight)

    def predict_start_from_noise(self, x_t, t, noise):
        return (
            extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t -
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape) * noise)

    def predict_noise_from_start(self, x_t, t, x0):
        return (
            (extract(self.sqrt_recip_alphas_cumprod, t, x_t.shape) * x_t - x0) / \
            extract(self.sqrt_recipm1_alphas_cumprod, t, x_t.shape)
        )

    def predict_v(self, x_start, t, noise):
        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * noise -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) *
                x_start)

    def predict_start_from_v(self, x_t, t, v):
        return (extract(self.sqrt_alphas_cumprod, t, x_t.shape) * x_t -
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_t.shape) * v)

    def q_posterior(self, x_start, x_t, t):
        posterior_mean = (
            extract(self.posterior_mean_coef1, t, x_t.shape) * x_start +
            extract(self.posterior_mean_coef2, t, x_t.shape) * x_t)
        posterior_variance = extract(self.posterior_variance, t, x_t.shape)
        posterior_log_variance_clipped = extract(
            self.posterior_log_variance_clipped, t, x_t.shape)
        return posterior_mean, posterior_variance, posterior_log_variance_clipped

    def model_predictions(self,
                          x,
                          t,
                          param_cond,
                          img_cond=None,
                          clip_x_start=False,
                          is_ban_ddnm=False,
                          is_denoise=False):
        model_output = self.model(x, t, param_cond)
        maybe_clip = partial(torch.clamp, min=-1.,
                             max=1.) if clip_x_start else identity

        if self.objective == 'pred_noise':
            pred_noise = model_output
            x_start = self.predict_start_from_noise(x, t, pred_noise)
            x_start = maybe_clip(x_start)

        elif self.objective == 'pred_x0':
            x_start = model_output
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        elif self.objective == 'pred_v':
            v = model_output
            x_start = self.predict_start_from_v(x, t, v)
            x_start = maybe_clip(x_start)
            pred_noise = self.predict_noise_from_start(x, t, x_start)

        if self.is_ddnm_sampling and img_cond is not None and not is_ban_ddnm:
            img_rpj = img_cond[:, 0, None, ...]
            mask_rpj = get_mask_from_img_cond(img_cond)
            if self.ddnm_dropouts[int(t[0])] > 0:
                keep_mask = torch.zeros_like(mask_rpj).float().uniform_(
                    0, 1) > self.ddnm_dropouts[int(t[0])]
                mask_rpj = keep_mask & mask_rpj

            x_start = torch.where(mask_rpj, img_rpj, x_start)

        elif is_denoise:
            img_rpj = img_cond[:, 0, None, ...]
            mask_rpj = get_mask_from_img_cond(img_cond)
            keep_mask = torch.zeros_like(mask_rpj).float().uniform_(
                0, 1) > self.denoise_dropouts[int(t[0])]
            mask_rpj = keep_mask & mask_rpj

            x_start = torch.where(mask_rpj, img_rpj, x_start)

        else:
            pass

        return ModelPrediction(pred_noise, x_start)

    def p_mean_variance(self,
                        x,
                        t,
                        param_cond,
                        img_cond,
                        clip_denoised=True,
                        is_ban_ddnm=False,
                        is_denoise=False):
        preds = self.model_predictions(x,
                                       t,
                                       param_cond,
                                       img_cond=img_cond,
                                       is_ban_ddnm=is_ban_ddnm,
                                       is_denoise=is_denoise)
        x_start = preds.pred_x_start

        if clip_denoised:
            x_start.clamp_(-1., 1.)

        model_mean, posterior_variance, posterior_log_variance = self.q_posterior(
            x_start=x_start, x_t=x, t=t)
        return model_mean, posterior_variance, posterior_log_variance, x_start

    @torch.no_grad()
    def p_sample(self,
                 x,
                 t: int,
                 param_cond,
                 img_cond,
                 clip_denoised=True,
                 is_ban_ddnm=False,
                 is_denoise=False):
        b, *_, device = *x.shape, x.device
        batched_times = torch.full((x.shape[0], ),
                                   t,
                                   device=x.device,
                                   dtype=torch.long)
        model_mean, _, model_log_variance, x_start = self.p_mean_variance(
            x=x,
            t=batched_times,
            param_cond=param_cond,
            img_cond=img_cond,
            clip_denoised=clip_denoised,
            is_ban_ddnm=is_ban_ddnm,
            is_denoise=is_denoise)
        noise = torch.randn_like(x) if t > 0 else 0.  # no noise if t == 0
        pred_img = model_mean + (0.5 * model_log_variance).exp() * noise
        return pred_img, x_start

    @torch.no_grad()
    def p_sample_loop(self,
                      param_cond,
                      img_cond,
                      shape,
                      disable_tqdm=False,
                      has_refine_step=False,
                      is_denoise=False):
        batch, device = shape[0], self.betas.device

        img = torch.randn(shape, device=device)

        x_start = None

        for t in tqdm(reversed(range(0, self.num_timesteps)),
                      desc='sampling loop time step',
                      total=self.num_timesteps,
                      disable=disable_tqdm):
            img, x_start = self.p_sample(img,
                                         t,
                                         param_cond,
                                         img_cond,
                                         is_denoise=is_denoise)

        if has_refine_step:
            img_refined, x_start = self.p_sample(img,
                                                 0,
                                                 param_cond,
                                                 img_cond,
                                                 is_ban_ddnm=True)
            mask_rpj = get_mask_from_img_cond(img_cond)
            img = torch.where(mask_rpj, img_refined, img)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def ddim_sample(self,
                    param_cond,
                    img_cond,
                    shape,
                    clip_denoised=True,
                    disable_tqdm=False,
                    has_refine_step=False,
                    is_denoise=False):
        batch, device, total_timesteps, sampling_timesteps, eta, objective = shape[
            0], self.betas.device, self.num_timesteps, self.sampling_timesteps, self.ddim_sampling_eta, self.objective

        times = torch.linspace(
            -1, total_timesteps - 1, steps=sampling_timesteps + 1
        )  # [-1, 0, 1, 2, ..., T-1] when sampling_timesteps == total_timesteps
        times = list(reversed(times.int().tolist()))
        time_pairs = list(
            zip(times[:-1],
                times[1:]))  # [(T-1, T-2), (T-2, T-3), ..., (1, 0), (0, -1)]

        img = torch.randn(shape, device=device)

        x_start = None

        for time, time_next in tqdm(time_pairs,
                                    desc='sampling loop time step',
                                    disable=disable_tqdm):
            time_cond = torch.full((batch, ),
                                   time,
                                   device=device,
                                   dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                param_cond,
                img_cond=img_cond,
                clip_x_start=clip_denoised,
                is_denoise=is_denoise)

            if time_next < 0:
                img = x_start
                continue

            alpha = self.alphas_cumprod[time]
            alpha_next = self.alphas_cumprod[time_next]

            sigma = eta * ((1 - alpha / alpha_next) * (1 - alpha_next) /
                           (1 - alpha)).sqrt()
            c = (1 - alpha_next - sigma**2).sqrt()

            noise = torch.randn_like(img)

            img = x_start * alpha_next.sqrt() + \
                  c * pred_noise + \
                  sigma * noise

        if has_refine_step:
            time_cond = torch.full((batch, ),
                                   0,
                                   device=device,
                                   dtype=torch.long)
            pred_noise, x_start, *_ = self.model_predictions(
                img,
                time_cond,
                param_cond,
                img_cond=img_cond,
                clip_x_start=clip_denoised,
                is_ban_ddnm=True)

            mask_rpj = get_mask_from_img_cond(img_cond)
            img = torch.where(mask_rpj, x_start, img)

        img = unnormalize_to_zero_to_one(img)
        return img

    @torch.no_grad()
    def sample(self,
               *,
               param_cond,
               img_cond=None,
               disable_tqdm=False,
               has_refine_step=False):
        batch_size, image_size, channels = param_cond.shape[
            0], self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        return sample_fn(param_cond,
                         img_cond,
                         shape=(batch_size, channels, image_size, image_size),
                         disable_tqdm=disable_tqdm,
                         has_refine_step=has_refine_step)

    @torch.no_grad()
    def denoise(self,
                *,
                param_cond,
                img_cond=None,
                disable_tqdm=False,
                has_refine_step=False):
        batch_size, image_size, channels = param_cond.shape[
            0], self.image_size, self.channels
        sample_fn = self.p_sample_loop if not self.is_ddim_sampling else self.ddim_sample

        return sample_fn(param_cond,
                         img_cond,
                         shape=(batch_size, channels, image_size, image_size),
                         disable_tqdm=disable_tqdm,
                         has_refine_step=has_refine_step,
                         is_denoise=True)

    @torch.no_grad()
    def interpolate(self, x1, x2, t=None, lam=0.5):
        b, *_, device = *x1.shape, x1.device
        t = default(t, self.num_timesteps - 1)

        assert x1.shape == x2.shape

        t_batched = torch.stack([torch.tensor(t, device=device)] * b)
        xt1, xt2 = map(lambda x: self.q_sample(x, t=t_batched), (x1, x2))

        img = (1 - lam) * xt1 + lam * xt2
        for i in tqdm(reversed(range(0, t)),
                      desc='interpolation sample time step',
                      total=t):
            img = self.p_sample(
                img, torch.full((b, ), i, device=device, dtype=torch.long))

        return img

    def q_sample(self, x_start, t, noise=None):
        noise = default(noise, lambda: torch.randn_like(x_start))

        return (extract(self.sqrt_alphas_cumprod, t, x_start.shape) * x_start +
                extract(self.sqrt_one_minus_alphas_cumprod, t, x_start.shape) *
                noise)

    @property
    def loss_fn(self):
        if self.loss_type == 'l1':
            return F.l1_loss
        elif self.loss_type == 'l2':
            return F.mse_loss
        else:
            raise ValueError(f'invalid loss type {self.loss_type}')

    def p_losses(self, x_start, t, param_cond, *, noise=None):
        b, c, h, w = x_start.shape
        noise = default(noise, lambda: torch.randn_like(x_start))

        # noise sample

        x = self.q_sample(x_start=x_start, t=t, noise=noise)

        # predict and take gradient step

        model_out = self.model(x, t, param_cond)

        if self.objective == 'pred_noise':
            target = noise
            pred_noise = model_out
            pred_x_start = self.predict_start_from_noise(x, t, pred_noise)
        elif self.objective == 'pred_x0':
            target = x_start
            pred_x_start = model_out
            pred_noise = self.predict_noise_from_start(x, t, pred_x_start)
        elif self.objective == 'pred_v':
            v = self.predict_v(x_start, t, noise)
            target = v
            pred_v = model_out
            pred_x_start = self.predict_start_from_v(x, t, pred_v)
            pred_noise = self.predict_noise_from_start(x, t, pred_x_start)
        else:
            raise ValueError(f'unknown objective {self.objective}')

        loss = self.loss_fn(model_out, target, reduction='none')
        loss = reduce(loss, 'b ... -> b (...)', 'mean')

        loss = loss * extract(self.loss_weight, t, loss.shape)
        return loss.mean()

    def forward(self, data_dict, *args, **kwargs):
        img = data_dict["img"]
        b, c, h, w, device, img_size, = *img.shape, img.device, self.image_size
        assert h == img_size and w == img_size, f'height and width of image must be {img_size}'
        t = torch.randint(0, self.num_timesteps, (b, ), device=device).long()

        intrinsic = data_dict["intrinsic"]
        param_cond = param_vector(intrinsic)

        img = normalize_to_neg_one_to_one(img)

        return self.p_losses(img, t, param_cond, *args, **kwargs)


# dataset classes
class DepthDataset(torch.utils.data.Dataset):
    def __init__(self,
                 folder,
                 image_size,
                 augment_horizontal_flip=False,
                 convert_image_to=None):
        super().__init__()
        self.folder = folder
        self.image_size = image_size
        self.paths = []
        with open("./dataset/3DMatch/metadata/gt.log", "r") as f:
            for line in f.readlines():
                line = line.removesuffix("\n")
                path = Path(os.path.join(folder, line.replace("\n", "")))
                self.paths.append(path)

        maybe_convert_fn = partial(
            convert_image_to_fn,
            convert_image_to) if exists(convert_image_to) else nn.Identity()

        # self.centercrop = T.CenterCrop(image_size)
        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.RandomHorizontalFlip()
            if augment_horizontal_flip else nn.Identity(),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, index):
        path = self.paths[index]

        # load img
        img = Image.open(path)
        img = self.transform(img)
        img = img * 1e-4  # original unit is mm, range from 0 ~ 10m
        img[img > 1] = 0  # clip

        # load intrinsic
        scene_path = path.parent.parent
        intrinsic_path = Path(scene_path, "camera-intrinsics.txt")
        intrinsic = np.loadtxt(intrinsic_path)
        intrinsic = intrinsic_transform(intrinsic,
                                        resize=self.image_size,
                                        centercrop=self.image_size)
        intrinsic = torch.Tensor(intrinsic)

        data_dict = {"img": img, "intrinsic": intrinsic}

        return data_dict


# trainer class


class Trainer(object):
    def __init__(self,
                 diffusion_model,
                 folder,
                 *,
                 train_batch_size=16,
                 gradient_accumulate_every=1,
                 augment_horizontal_flip=True,
                 train_lr=1e-4,
                 train_num_steps=100000,
                 ema_update_every=10,
                 ema_decay=0.995,
                 adam_betas=(0.9, 0.99),
                 save_and_sample_every=1000,
                 num_samples=25,
                 results_folder='./results',
                 samples_folder='./samples',
                 amp=False,
                 fp16=False,
                 split_batches=True,
                 convert_image_to=None,
                 calculate_fid=True,
                 inception_block_idx=2048):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no')

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters
        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples
        self.save_and_sample_every = save_and_sample_every

        self.batch_size = train_batch_size
        self.gradient_accumulate_every = gradient_accumulate_every

        self.train_num_steps = train_num_steps
        self.image_size = diffusion_model.image_size

        # dataset and dataloader

        self.ds = DepthDataset(folder,
                               self.image_size,
                               augment_horizontal_flip=augment_horizontal_flip,
                               convert_image_to=convert_image_to)
        dl = DataLoader(self.ds,
                        batch_size=train_batch_size,
                        shuffle=True,
                        pin_memory=True,
                        num_workers=cpu_count(),
                        collate_fn=collate_func)

        dl = self.accelerator.prepare(dl)
        self.dl = cycle(dl)

        # optimizer

        self.opt = Adam(diffusion_model.parameters(),
                        lr=train_lr,
                        betas=adam_betas)

        # for logging results in a folder periodically

        if self.accelerator.is_main_process:
            self.ema = EMA(diffusion_model,
                           beta=ema_decay,
                           update_every=ema_update_every)
            self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.samples_folder = Path(samples_folder)
        self.samples_folder.mkdir(exist_ok=True)

        # step counter state

        self.step = 0

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt = self.accelerator.prepare(self.model, self.opt)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_main_process:
            return

        data = {
            'step':
            self.step,
            'model':
            self.accelerator.get_state_dict(self.model),
            'opt':
            self.opt.state_dict(),
            'ema':
            self.ema.state_dict(),
            'scaler':
            self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler) else None
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),
                          map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.step = data['step']
        self.opt.load_state_dict(data['opt'])
        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def calculate_activation_statistics(self, samples):
        assert exists(self.inception_v3)

        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...').cpu().numpy()

        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def fid_score(self, real_samples, fake_samples):

        if self.channels != 3:
            real_samples, fake_samples = map(
                lambda t: repeat(t, 'b 1 ... -> b c ...', c=3),
                (real_samples, fake_samples))

        min_batch = min(real_samples.shape[0], fake_samples.shape[0])
        real_samples, fake_samples = map(lambda t: t[:min_batch],
                                         (real_samples, fake_samples))

        m1, s1 = self.calculate_activation_statistics(real_samples)
        m2, s2 = self.calculate_activation_statistics(fake_samples)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def train(self):
        accelerator = self.accelerator
        device = accelerator.device

        with tqdm(initial=self.step,
                  total=self.train_num_steps,
                  disable=not accelerator.is_main_process) as pbar:

            while self.step < self.train_num_steps:

                total_loss = 0.

                for _ in range(self.gradient_accumulate_every):
                    data = data_to_device(next(self.dl), device)

                    with self.accelerator.autocast():
                        loss = self.model(data)
                        loss = loss / self.gradient_accumulate_every
                        total_loss += loss.item()

                    self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                pbar.set_description(f'loss: {total_loss:.4f}')

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                self.step += 1
                if accelerator.is_main_process:
                    self.ema.update()

                    if self.step != 0 and self.step % self.save_and_sample_every == 0:
                        self.ema.ema_model.eval()

                        with torch.no_grad():
                            milestone = self.step // self.save_and_sample_every
                            batches = num_to_groups(
                                self.num_samples,
                                self.batch_size // torch.cuda.device_count())
                            all_images_list = []
                            for batch in batches:
                                torch.cuda.empty_cache()
                                intrinsic = intrinsic_transform(
                                    random_sample_intrinsic(batch_size=batch),
                                    resize=self.image_size,
                                    centercrop=self.image_size).astype(
                                        np.float32)

                                param_cond = param_vector(
                                    torch.tensor(intrinsic).to(self.device))

                                images = self.ema.ema_model.sample(
                                    param_cond=param_cond)
                                all_images_list.append(images)

                        all_images = torch.cat(all_images_list, dim=0)

                        utils.save_image(all_images,
                                         str(self.results_folder /
                                             f'sample-{milestone}.png'),
                                         nrow=int(math.sqrt(self.num_samples)))
                        self.save(milestone // 100 * 100)

                        # whether to calculate fid

                        if exists(self.inception_v3):
                            real_images = data["img"]
                            fid_score = self.fid_score(
                                real_samples=real_images,
                                fake_samples=all_images)
                            accelerator.print(f'fid_score: {fid_score}')

                pbar.update(1)

        accelerator.print('training complete')


class Tester(object):
    def __init__(self,
                 diffusion_model,
                 *,
                 batch_size=16,
                 ema_update_every=10,
                 ema_decay=0.995,
                 results_folder='./results',
                 samples_folder='./samples',
                 amp=False,
                 fp16=False,
                 split_batches=True,
                 calculate_fid=True,
                 inception_block_idx=2048):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no')

        self.accelerator.native_amp = amp

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # InceptionV3 for fid-score computation

        self.inception_v3 = None

        if calculate_fid:
            assert inception_block_idx in InceptionV3.BLOCK_INDEX_BY_DIM
            block_idx = InceptionV3.BLOCK_INDEX_BY_DIM[inception_block_idx]
            self.inception_v3 = InceptionV3([block_idx])
            self.inception_v3.to(self.device)

        # sampling and training hyperparameters
        self.batch_size = batch_size

        self.image_size = diffusion_model.image_size

        # for logging results in a folder periodically

        self.ema = EMA(diffusion_model,
                       beta=ema_decay,
                       update_every=ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.samples_folder = Path(samples_folder)
        if self.samples_folder.exists():
            shutil.rmtree(str(self.samples_folder))
        self.samples_folder.mkdir(exist_ok=True)

        # prepare model, dataloader, optimizer with accelerator

        self.model = self.accelerator.prepare(self.model)

    @property
    def device(self):
        return self.accelerator.device

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),
                          map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def compute_inception_features(self, samples):
        assert exists(self.inception_v3)

        features = self.inception_v3(samples)[0]
        features = rearrange(features, '... 1 1 -> ...').cpu().numpy()

        return features

    @torch.no_grad()
    def calculate_activation_statistics(self, features):
        mu = np.mean(features, axis=0)
        sigma = np.cov(features, rowvar=False)
        return mu, sigma

    def fid_score(self, real_features, fake_features):
        m1, s1 = self.calculate_activation_statistics(real_features)
        m2, s2 = self.calculate_activation_statistics(fake_features)

        fid_value = calculate_frechet_distance(m1, s1, m2, s2)
        return fid_value

    def sample_uncondition(self, num_samples=25):
        with torch.no_grad():
            batches = num_to_groups(num_samples, self.batch_size)
            all_images_list = []
            for batch in batches:
                torch.cuda.empty_cache()
                intrinsic = intrinsic_transform(
                    random_sample_intrinsic(batch_size=batch),
                    resize=self.image_size,
                    centercrop=self.image_size).astype(np.float32)

                param_cond = param_vector(
                    torch.tensor(intrinsic).to(self.device))

                images = self.ema.ema_model.sample(param_cond=param_cond)
                all_images_list.append(images)

        all_images = torch.cat(all_images_list, dim=0)

        utils.save_image(all_images,
                         str(self.samples_folder / f'unconditional.png'),
                         nrow=int(math.sqrt(num_samples)))

    @torch.no_grad()
    def sample(self, num_scenes, num_samples):
        self.ema.ema_model.eval()
        with torch.no_grad():
            batches = num_to_groups(num_scenes, self.batch_size)
            all_images_list = []
            for b_idx, batch in enumerate(batches):
                # unconditional generation
                intrinsic = intrinsic_transform(
                    random_sample_intrinsic(batch_size=batch),
                    resize=self.image_size,
                    centercrop=self.image_size).astype(np.float32)

                absolute_pose = np.stack([np.eye(4) for _ in range(batch)],
                                         axis=0).astype(np.float32)

                param_cond = param_vector(
                    torch.tensor(intrinsic).to(self.device))

                scene_images_list = []
                images = self.ema.ema_model.sample(param_cond=param_cond)
                scene_images_list.append(images)

                # save unconditional results
                for scene_idx, image in enumerate(images):
                    # save images
                    image = image.squeeze(0).cpu().numpy()
                    image_lst = np.zeros_like(image)
                    image_rpj = np.zeros_like(image)
                    image_vis = np.concatenate([image_lst, image_rpj, image],
                                               axis=-1)
                    plt.imsave(str(
                        self.samples_folder /
                        f'scene-{b_idx*self.batch_size + scene_idx}-sample-0.png'
                    ),
                               image_vis,
                               cmap="gray",
                               vmin=0,
                               vmax=1)

                    # save point clouds
                    pc = point_cloud(image * 10,
                                     intrinsic[scene_idx],
                                     clip=[0.5, 3.5])

                    o3d_pc = o3d.geometry.PointCloud()
                    o3d_pc.points = o3d.utility.Vector3dVector(pc)

                    pc_path = str(
                        self.samples_folder /
                        f'scene-{b_idx*self.batch_size + scene_idx}-sample-0.ply'
                    )
                    o3d.io.write_point_cloud(pc_path, o3d_pc)

                    # save intrinsic
                    intrinsic_path = str(
                        self.samples_folder /
                        f'scene-{b_idx*self.batch_size + scene_idx}-camera-intrinsics.txt'
                    )
                    np.savetxt(intrinsic_path, intrinsic[scene_idx])

                # successive conditional generation
                for sample_idx in range(1, num_samples):
                    # relative_pose = random_sample_transform(
                    #     intrinsic, image_size=self.image_size)
                    relative_pose = np.array(
                        [np.eye(4) for _ in range(len(intrinsic))])
                    relative_pose[..., :3, 3] = np.array([0, 0, 0.5])
                    relative_pose = relative_pose.astype(np.float32)
                    absolute_pose = relative_pose @ absolute_pose
                    images_rpj, mask_rpj = reproject_tensor(
                        images * 10,
                        torch.tensor(intrinsic).to(self.device),
                        torch.tensor(relative_pose).to(self.device))

                    if np.sum(absolute_pose[..., :3, 3]**2) != 0:
                        images_rpj, mask_rpj = occlusion_filter(
                            images_rpj, mask_rpj)

                    images_rpj = images_rpj * 0.1
                    img_cond = torch.cat([images_rpj, mask_rpj], dim=1)
                    img_cond = normalize_to_neg_one_to_one(img_cond)
                    # img_cond = image_condition(
                    #     images,
                    #     torch.tensor(intrinsic).to(self.device),
                    #     torch.tensor(relative_pose).to(self.device))
                    images_last = images
                    images = self.ema.ema_model.sample(param_cond=param_cond,
                                                       img_cond=img_cond)
                    scene_images_list.append(images)

                    # save conditional generation results
                    for scene_idx, image in enumerate(images):
                        image = image.squeeze(0).cpu().numpy()
                        image_lst = images_last[scene_idx].squeeze(
                            0).cpu().numpy()
                        image_rpj = images_rpj[scene_idx].squeeze(
                            0).cpu().numpy()
                        image_vis = np.concatenate(
                            [image_lst, image_rpj, image], axis=-1)
                        plt.imsave(str(
                            self.samples_folder /
                            f'scene-{b_idx*self.batch_size + scene_idx}-sample-{sample_idx}.png'
                        ),
                                   image_vis,
                                   cmap="gray",
                                   vmin=0,
                                   vmax=1)

                        pc = point_cloud(image * 10,
                                         intrinsic[scene_idx],
                                         clip=[0.5, 3.5])
                        pc = (pc - absolute_pose[scene_idx, :3, 3]
                              ) @ absolute_pose[scene_idx, :3, :3]
                        # pc = pc @ absolute_pose[
                        #     scene_idx, :3, :3].T + absolute_pose[scene_idx, :3,
                        #                                          3]

                        o3d_pc = o3d.geometry.PointCloud()
                        o3d_pc.points = o3d.utility.Vector3dVector(pc)

                        pc_path = str(
                            self.samples_folder /
                            f'scene-{b_idx*self.batch_size + scene_idx}-sample-{sample_idx}.ply'
                        )
                        o3d.io.write_point_cloud(pc_path, o3d_pc)

                scene_images = torch.cat(scene_images_list, dim=-1)

                all_images_list.append(scene_images)

            all_images = torch.cat(all_images_list, dim=0)
            grid_img = utils.make_grid(all_images, nrow=1)[0].cpu().numpy()
            plt.imsave(str(self.samples_folder / f'overview.png'),
                       grid_img,
                       cmap="gray",
                       vmin=0,
                       vmax=1)

    @torch.no_grad()
    def generate(self, num_scenes, num_samples, voxel_size=0.005):
        self.ema.ema_model.eval()
        with torch.no_grad():
            batches = num_to_groups(num_scenes, self.batch_size)
            all_images_list = []
            all_pointclouds_list = []
            for b_idx, batch in enumerate(batches):
                # unconditional generation
                intrinsic = intrinsic_transform(
                    random_sample_intrinsic(batch_size=batch),
                    resize=self.image_size,
                    centercrop=self.image_size).astype(np.float32)

                absolute_pose = np.stack([np.eye(4) for _ in range(batch)],
                                         axis=0).astype(np.float32)

                param_cond = param_vector(
                    torch.tensor(intrinsic).to(self.device))

                scene_images_list = []
                scene_pointclouds_list = []
                images = self.ema.ema_model.sample(param_cond=param_cond)
                scene_images_list.append(images)

                # save unconditional results
                for scene_idx, image in enumerate(images):
                    # save images
                    image = image.squeeze(0).cpu().numpy()
                    image_lst = np.zeros_like(image)
                    image_rpj = np.zeros_like(image)
                    image_vis = np.concatenate([image_lst, image_rpj, image],
                                               axis=-1)
                    plt.imsave(str(
                        self.samples_folder /
                        f'scene-{b_idx*self.batch_size + scene_idx}-sample-0.png'
                    ),
                               image_vis,
                               cmap="plasma",
                               vmin=0,
                               vmax=1)

                    # save point clouds
                    pc = point_cloud(image * 10,
                                     intrinsic[scene_idx],
                                     clip=[0.5, 3.5])

                    o3d_pc = o3d.geometry.PointCloud()
                    o3d_pc.points = o3d.utility.Vector3dVector(pc)
                    o3d_pc_down = o3d_pc.voxel_down_sample(
                        voxel_size=voxel_size)

                    scene_pointclouds_list.append(
                        np.asarray(o3d_pc_down.points).astype(np.float32))

                # successive conditional generation
                for sample_idx in range(1, num_samples):
                    relative_pose = random_sample_transform(
                        intrinsic, image_size=self.image_size)
                    absolute_pose = relative_pose @ absolute_pose

                    images_rpj_list = []
                    mask_rpj_list = []
                    for scene_idx, pointcloud in enumerate(
                            scene_pointclouds_list):
                        pointcloud = pointcloud @ absolute_pose[
                            scene_idx, :3, :3].T + absolute_pose[scene_idx, :3,
                                                                 3]
                        pointcloud = torch.tensor(
                            pointcloud[None, ...]).to(device=self.device)
                        img_rpj, msk_rpj = pc2depth_tensor(
                            pointcloud,
                            torch.ones(pointcloud.shape[:2],
                                       dtype=torch.bool,
                                       device=self.device),
                            torch.tensor(intrinsic).to(self.device),
                            image_size=[self.image_size, self.image_size])
                        images_rpj_list.append(img_rpj)
                        mask_rpj_list.append(msk_rpj)

                    images_rpj = torch.cat(images_rpj_list, dim=0)
                    mask_rpj = torch.cat(mask_rpj_list, dim=0)

                    images_rpj = images_rpj * 0.1
                    img_cond = torch.cat([images_rpj, mask_rpj], dim=1)
                    img_cond = normalize_to_neg_one_to_one(img_cond)

                    images_last = images
                    images = self.ema.ema_model.sample(param_cond=param_cond,
                                                       img_cond=img_cond)
                    scene_images_list.append(images)

                    # save conditional generation results
                    for scene_idx, image in enumerate(images):
                        image = image.squeeze(0).cpu().numpy()
                        image_lst = images_last[scene_idx].squeeze(
                            0).cpu().numpy()
                        image_rpj = images_rpj[scene_idx].squeeze(
                            0).cpu().numpy()
                        image_vis = np.concatenate(
                            [image_lst, image_rpj, image], axis=-1)
                        plt.imsave(str(
                            self.samples_folder /
                            f'scene-{b_idx*self.batch_size + scene_idx}-sample-{sample_idx}.png'
                        ),
                                   image_vis,
                                   cmap="plasma",
                                   vmin=0,
                                   vmax=1)

                        pc = point_cloud(image * 10,
                                         intrinsic[scene_idx],
                                         clip=[0.5, 3.5])
                        pc = (pc - absolute_pose[scene_idx, :3, 3]
                              ) @ absolute_pose[scene_idx, :3, :3]

                        pc = np.concatenate(
                            [scene_pointclouds_list[scene_idx], pc], axis=0)

                        o3d_pc = o3d.geometry.PointCloud()
                        o3d_pc.points = o3d.utility.Vector3dVector(pc)

                        o3d_pc_down = o3d_pc.voxel_down_sample(
                            voxel_size=voxel_size)

                        scene_pointclouds_list[scene_idx] = np.asarray(
                            o3d_pc_down.points).astype(np.float32)

                for scene_idx, pointcloud in enumerate(scene_pointclouds_list):
                    o3d_pc = o3d.geometry.PointCloud()
                    o3d_pc.points = o3d.utility.Vector3dVector(pointcloud)
                    o3d_pc_down = o3d_pc.voxel_down_sample(voxel_size=0.025)

                    pc_path = str(
                        self.samples_folder /
                        f'scene-{b_idx*self.batch_size + scene_idx}.ply')
                    o3d.io.write_point_cloud(pc_path, o3d_pc_down)

                scene_images = torch.cat(scene_images_list, dim=-1)

                all_images_list.append(scene_images)

            all_images = torch.cat(all_images_list, dim=0)
            grid_img = utils.make_grid(all_images, nrow=1)[0].cpu().numpy()
            plt.imsave(str(self.samples_folder / f'overview.png'),
                       grid_img,
                       cmap="plasma",
                       vmin=0,
                       vmax=1)


class Generator(object):
    def __init__(self,
                 diffusion_model,
                 folder,
                 *,
                 batch_size=16,
                 ema_update_every=10,
                 ema_decay=0.995,
                 results_folder='./results',
                 samples_folder='./samples',
                 amp=False,
                 fp16=False,
                 split_batches=True):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no')

        self.accelerator.native_amp = amp

        # reference folder
        self.folder = folder

        # model

        self.model = diffusion_model
        self.channels = diffusion_model.channels

        # sampling and training hyperparameters
        self.batch_size = batch_size

        self.image_size = diffusion_model.image_size

        # for logging results in a folder periodically

        self.ema = EMA(diffusion_model,
                       beta=ema_decay,
                       update_every=ema_update_every)
        self.ema.to(self.device)

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(parents=True, exist_ok=True)

        self.samples_folder = Path(samples_folder)
        self.samples_folder.mkdir(parents=True, exist_ok=True)

        # prepare model, dataloader, optimizer with accelerator

        self.model = self.accelerator.prepare(self.model)

    @property
    def device(self):
        return self.accelerator.device

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),
                          map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        if self.accelerator.is_main_process:
            self.ema.load_state_dict(data["ema"])

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])


    @torch.no_grad()
    def generate(self,
                start_scene_index,
                stop_scene_index,
                num_samples,
                memory_voxel_size=0.002,
                save_voxel_size=0.025,
                has_refine_step=True,
                depth_correction=None):
        self.ema.ema_model.eval()
        num_scenes = stop_scene_index - start_scene_index

        if depth_correction is not None:
            depth_correction: nn.Module
            self.depth_correction = depth_correction.to(
                self.accelerator.device)
            ckpt = torch.load(f'./depth_correction_results/model-best.pt',
                              map_location=self.accelerator.device)
            self.depth_correction.load_state_dict(ckpt['model'])
            self.depth_correction.eval()

        bbox = o3d.geometry.AxisAlignedBoundingBox(min_bound=(-1.5, -1.5, 0.5),
                                                   max_bound=(1.5, 1.5, 3.5))

        # real ground truth
        with open("./dataset/indoor/metadata/train_info.pkl",
                  'rb') as f:
            info_train = pickle.load(f)

        transform = T.Compose([
            T.Resize(self.image_size,
                     interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop(self.image_size),
            T.ToTensor()
        ])

        pbar = tqdm(total=num_scenes,
                    disable=not self.accelerator.is_main_process)
        with torch.no_grad():
            batches = num_to_groups(num_scenes, self.batch_size)
            all_images_list = []
            pbar.set_description("0/{}".format(num_samples))
            for b_idx, batch in enumerate(batches):
                # skip
                pc_path = str(self.samples_folder /
                              'scene-{:0>6d}/sample-{:0>6d}.cloud.ply'.format(
                                  start_scene_index + b_idx * self.batch_size +
                                  batch - 1, num_samples // 2))
                if os.path.isfile(pc_path):
                    print("Skip completed scene {:0>6d} - {:0>6d}.".format(
                        start_scene_index + b_idx * self.batch_size,
                        start_scene_index + b_idx * self.batch_size + batch -
                        1))
                    pbar.update(batch)
                    continue
                fragment_pointclouds_list = []
                fragment_poses_list = []
                scene_images_list = []
                scene_pointclouds_list = []

                intrinsic = np.zeros((batch, 3, 3), dtype=np.float32)

                # read ground-truth
                for scene_idx, abs_scene_idx in enumerate(
                        range(
                            start_scene_index + b_idx * self.batch_size,
                            start_scene_index + b_idx * self.batch_size +
                            batch)):

                    # switch src and tgt if needed
                    if (abs_scene_idx // 20642) % 2 == 0:
                        src_path = os.path.join(
                            "./dataset/indoor/data",
                            info_train['src'][abs_scene_idx % 20642])
                        tgt_path = os.path.join(
                            "./dataset/indoor/data",
                            info_train['tgt'][abs_scene_idx % 20642])
                    else:
                        src_path = os.path.join(
                            "./dataset/indoor/data",
                            info_train['tgt'][abs_scene_idx % 20642])
                        tgt_path = os.path.join(
                            "./dataset/indoor/data",
                            info_train['src'][abs_scene_idx % 20642])

                    src_info_path = src_path.replace(".pth", ".info.txt")
                    tgt_info_path = tgt_path.replace(".pth", ".info.txt")

                    # make scene dir
                    scene_path = Path(
                        str(self.samples_folder /
                            'scene-{:0>6d}'.format(start_scene_index +
                                                   b_idx * self.batch_size +
                                                   scene_idx)))
                    if scene_path.exists():
                        shutil.rmtree(str(scene_path), ignore_errors=True)
                    scene_path.mkdir(parents=True, exist_ok=True)

                    # source info
                    with open(src_info_path, "r") as f:
                        for idx, line in enumerate(f.readlines()):
                            if idx != 0:
                                break
                            scene_name, seq_name, frame_start_idx, frame_end_idx = line.removesuffix(
                                "\n").split()
                            scene_path = os.path.join(
                                self.folder, scene_name)

                            # get intrinsic
                            intrinsic[scene_idx, :, :] = intrinsic_transform(
                                np.loadtxt(
                                    os.path.join(scene_path,
                                                 "camera-intrinsics.txt")),
                                resize=self.image_size,
                                centercrop=self.image_size).astype(np.float32)

                            # save intrinsic
                            intrinsic_path = str(
                                self.samples_folder /
                                'scene-{:0>6d}/camera-intrinsics.txt'.format(
                                    start_scene_index +
                                    b_idx * self.batch_size + scene_idx))
                            np.savetxt(intrinsic_path, intrinsic[scene_idx])

                            frame_start_idx = int(frame_start_idx)
                            frame_end_idx = int(frame_end_idx)

                    # get frame point cloud for source
                    frame_path = os.path.join(
                        scene_path, seq_name,
                        "frame-{:0>6d}.depth.png".format(frame_start_idx))
                    frame_image = transform(Image.open(frame_path)) * 1e-4
                    frame_image[frame_image > 1] = 0

                    # save image
                    image_path = str(
                        self.samples_folder /
                        'scene-{:0>6d}/sample-{:0>6d}.image.png'.format(
                            start_scene_index + b_idx * self.batch_size +
                            scene_idx, 0))
                    utils.save_image(frame_image, image_path)

                    # # save depth
                    # depth_path = str(
                    #     self.samples_folder /
                    #     'scene-{:0>6d}/sample-{:0>6d}.depth.png'.format(
                    #         start_scene_index + b_idx * self.batch_size +
                    #         scene_idx, 0))
                    # frame_depth = frame_image.squeeze(0).cpu().numpy() * 1e4
                    # frame_depth = frame_depth.astype(np.uint16)
                    # cv2.imwrite(depth_path, frame_depth)

                    frame_pc = point_cloud(
                        frame_image.squeeze(0).cpu().numpy() * 10,
                        intrinsic[scene_idx],
                        clip=[0.5, 10]).astype(np.float32)

                    o3d_scene_pc = o3d.geometry.PointCloud()
                    o3d_scene_pc.points = o3d.utility.Vector3dVector(frame_pc)
                    o3d_scene_pc = o3d_scene_pc.crop(bbox)

                    scene_pc = np.asarray(o3d_scene_pc.points).astype(
                        np.float32)
                    scene_pointclouds_list.append(scene_pc)

                    o3d_scene_pc_down = o3d_scene_pc.voxel_down_sample(
                        voxel_size=save_voxel_size)

                    pc_path = str(
                        self.samples_folder /
                        'scene-{:0>6d}/sample-{:0>6d}.cloud.ply'.format(
                            start_scene_index + b_idx * self.batch_size +
                            scene_idx, 0))
                    o3d.io.write_point_cloud(pc_path, o3d_scene_pc_down)

                    # target info
                    with open(tgt_info_path, "r") as f:
                        for idx, line in enumerate(f.readlines()):
                            if idx != 0:
                                break
                            scene_name, seq_name, frame_start_idx, frame_end_idx = line.removesuffix(
                                "\n").split()
                            scene_path = os.path.join(
                                self.folder, scene_name)
                            frame_start_idx = int(frame_start_idx)
                            frame_end_idx = int(frame_end_idx)
                            for sample_idx, f_idx in enumerate(
                                    np.linspace(frame_start_idx, frame_end_idx,
                                                num_samples)):
                                frame_path = os.path.join(
                                    scene_path,
                                    seq_name, "frame-{:0>6d}.depth.png".format(
                                        int(f_idx)))

                # successive conditional generation
                param_cond = param_vector(
                    torch.tensor(intrinsic).to(self.device))

                for sample_idx in range(0, num_samples):
                    absolute_pose = random_sample_pose(batch).astype(
                        np.float32)

                    images_rpj_list = []
                    mask_rpj_list = []
                    for scene_idx, pointcloud in enumerate(
                            scene_pointclouds_list):
                        pointcloud = pointcloud @ absolute_pose[
                            scene_idx, :3, :3].T + absolute_pose[scene_idx, :3,
                                                                 3]
                        pointcloud = torch.tensor(
                            pointcloud[None, ...]).to(device=self.device)
                        img_rpj, msk_rpj = pc2depth_tensor(
                            pointcloud,
                            torch.ones(pointcloud.shape[:2],
                                       dtype=torch.bool,
                                       device=self.device),
                            torch.tensor(intrinsic[scene_idx, None,
                                                   ...]).to(self.device),
                            image_size=[self.image_size, self.image_size])
                        images_rpj_list.append(img_rpj)
                        mask_rpj_list.append(msk_rpj)

                    images_rpj = torch.cat(images_rpj_list, dim=0)
                    mask_rpj = torch.cat(mask_rpj_list, dim=0)

                    images_rpj = images_rpj * 0.1

                    # save reprojected image
                    for scene_idx, image_rpj in enumerate(images_rpj):
                        image_rpj_path = str(
                            self.samples_folder /
                            'scene-{:0>6d}/reprojected.image.png'.format(
                                start_scene_index + b_idx * self.batch_size +
                                scene_idx))
                        utils.save_image(image_rpj, image_rpj_path)

                    # use depth correction
                    mask_crt = self.depth_correction(images_rpj)
                    mask_crt = mask_crt > 0.99
                    images_rpj[~mask_crt] = 0
                    mask_rpj = mask_rpj & mask_crt

                    img_cond = torch.cat([images_rpj, mask_rpj], dim=1)
                    img_cond = normalize_to_neg_one_to_one(img_cond)

                    images = self.ema.ema_model.sample(
                        param_cond=param_cond,
                        img_cond=img_cond,
                        disable_tqdm=True,
                        has_refine_step=has_refine_step)

                    # use depth correction
                    mask_crt = self.depth_correction(images)
                    mask_crt = mask_crt > 0.99
                    images[~mask_crt] = 0

                    scene_images_list.append(images)

                    # save conditional generation results
                    for scene_idx, image in enumerate(images):
                        # save extrinsic
                        extrinsic_path = str(
                            self.samples_folder /
                            'scene-{:0>6d}/sample-{:0>6d}.pose.txt'.format(
                                start_scene_index + b_idx * self.batch_size +
                                scene_idx, sample_idx + 1))
                        np.savetxt(extrinsic_path,
                                   np.linalg.inv(absolute_pose[scene_idx]))

                        # save corrected image
                        image_crt_path = str(
                            self.samples_folder /
                            'scene-{:0>6d}/corrected.image.png'.format(
                                start_scene_index + b_idx * self.batch_size +
                                scene_idx))
                        utils.save_image(images_rpj[scene_idx], image_crt_path)

                        # save image
                        image_path = str(
                            self.samples_folder /
                            'scene-{:0>6d}/sample-{:0>6d}.image.png'.format(
                                start_scene_index + b_idx * self.batch_size +
                                scene_idx, sample_idx + 1))
                        utils.save_image(image, image_path)

                        # save depth image
                        depth_path = str(
                            self.samples_folder /
                            'scene-{:0>6d}/sample-{:0>6d}.depth.png'.format(
                                start_scene_index + b_idx * self.batch_size +
                                scene_idx, sample_idx + 1))
                        depth = image.squeeze(0).cpu().numpy() * 1e4
                        depth = depth.astype(np.uint16)
                        cv2.imwrite(depth_path, depth)

                        # save point cloud
                        pc = point_cloud(image.squeeze(0).cpu().numpy() * 10,
                                         intrinsic[scene_idx],
                                         clip=[0.5, 10])

                        pc = (pc - absolute_pose[scene_idx, :3, 3]
                              ) @ absolute_pose[scene_idx, :3, :3]

                        if sample_idx == 0:
                            fragment_pointclouds_list.append(pc)
                            fragment_poses_list.append(
                                absolute_pose[scene_idx])
                        else:
                            fragment_pointclouds_list[
                                scene_idx] = np.concatenate(
                                    [fragment_pointclouds_list[scene_idx], pc],
                                    axis=0)

                        if sample_idx == num_samples - 1:
                            o3d_pc = o3d.geometry.PointCloud()
                            o3d_pc.points = o3d.utility.Vector3dVector(
                                fragment_pointclouds_list[scene_idx])
                            o3d_pc = o3d_pc.transform(
                                fragment_poses_list[scene_idx])
                            o3d_pc = o3d_pc.crop(bbox)
                            o3d_pc_down = o3d_pc.voxel_down_sample(
                                voxel_size=save_voxel_size)
                            o3d_pc_down = o3d_pc_down.transform(
                                np.linalg.inv(fragment_poses_list[scene_idx]))

                            pc_path = str(
                                self.samples_folder /
                                'scene-{:0>6d}/sample-{:0>6d}.cloud.ply'.
                                format(
                                    start_scene_index +
                                    b_idx * self.batch_size + scene_idx, 1))
                            o3d.io.write_point_cloud(pc_path, o3d_pc_down)

                        # save scene memory
                        scene_pc = point_cloud(image.squeeze(0).cpu().numpy() *
                                               10,
                                               intrinsic[scene_idx],
                                               clip=[0.5, 10])
                        scene_pc = (scene_pc - absolute_pose[scene_idx, :3, 3]
                                    ) @ absolute_pose[scene_idx, :3, :3]

                        scene_pc = np.concatenate(
                            [scene_pointclouds_list[scene_idx], scene_pc],
                            axis=0)

                        o3d_scene_pc = o3d.geometry.PointCloud()
                        o3d_scene_pc.points = o3d.utility.Vector3dVector(
                            scene_pc)

                        o3d_scene_pc_down = o3d_scene_pc.voxel_down_sample(
                            voxel_size=memory_voxel_size)

                        scene_pointclouds_list[scene_idx] = np.asarray(
                            o3d_scene_pc_down.points).astype(np.float32)

                        # collect garbage
                        # del pc, o3d_pc, o3d_pc_down
                        del scene_pc, o3d_scene_pc, o3d_scene_pc_down
                        gc.collect()

                    pbar.set_description("{}/{}".format(
                        sample_idx + 1, num_samples))

                self.accelerator.wait_for_everyone()

                pbar.update(batch)

            pbar.close()

