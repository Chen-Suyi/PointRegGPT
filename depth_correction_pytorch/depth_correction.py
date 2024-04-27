import math
import time
import os
from pathlib import Path
import pickle
import json
import random
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
import matplotlib.cm as cm
import imageio

import logging
import coloredlogs

import torch
from torch import nn, einsum
import torch.nn.functional as F
from torch.nn.modules.utils import _pair, _quadruple
from torch.utils.data import Dataset, DataLoader

from torch.optim import Adam
from torch.optim.lr_scheduler import ExponentialLR

from torchvision import transforms as T, utils

from einops import rearrange, reduce, repeat
from einops.layers.torch import Rearrange

from PIL import Image
from tqdm.auto import tqdm

from accelerate import Accelerator

# helpers functions


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True


def create_logger(log_file=None):
    logger = logging.getLogger()
    logger.handlers.clear()
    logger.setLevel(level=logging.CRITICAL)
    logger.propagate = False

    format_str = '[%(asctime)s] [%(levelname).4s] %(message)s'
    if log_file is not None:
        file_handler = logging.FileHandler(log_file)
        formatter = logging.Formatter(format_str, datefmt='%Y-%m-%d %H:%M:%S')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        stream_handler = logging.StreamHandler()
        colored_formatter = coloredlogs.ColoredFormatter(format_str)
        stream_handler.setFormatter(colored_formatter)
        logger.addHandler(stream_handler)

    return logger


class Logger:
    def __init__(self, log_file=None, local_rank=-1):
        if local_rank == 0 or local_rank == -1:
            self.logger = create_logger(log_file=log_file)
        else:
            self.logger = None

    def debug(self, message):
        if self.logger is not None:
            self.logger.debug(message)

    def info(self, message):
        if self.logger is not None:
            self.logger.info(message)

    def warning(self, message):
        if self.logger is not None:
            self.logger.warning(message)

    def error(self, message):
        if self.logger is not None:
            self.logger.error(message)

    def critical(self, message):
        if self.logger is not None:
            self.logger.critical(message)


class AverageMeter():
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.val_previous = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def set(self, val):
        self.val = val
        self.avg = val

    def update(self, val, num=1):
        self.val_previous = self.val
        self.val = val
        self.sum += val * num
        self.count += num
        self.avg = self.sum / self.count

    def __float__(self):
        return float(self.avg)

    def __int__(self):
        return int(self.avg)


def make_gif(img_list, save_folder, name):
    os.makedirs(f"{save_folder}/", exist_ok=True)
    imageio.mimsave(f'{save_folder}/{name}.gif',
                    img_list,
                    duration=1000,
                    loop=0)


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

    # sort, the front point should block the back
    sorted_indices = torch.argsort(z, dim=-1, descending=True)
    b = b[sorted_indices]
    r = r[sorted_indices]
    c = c[sorted_indices]
    z = z[sorted_indices]

    depth = torch.zeros((batch_size, rows, cols)).to(device=pc.device)
    depth[b, r, c] = z
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


def compute_overlap_ratio(src_depth,
                          tgt_depth,
                          intrinsic,
                          relative_pose,
                          clip=[0, 10],
                          voxel_size=0.025):
    src_xyz = point_cloud(src_depth, intrinsic, clip=clip)
    tgt_xyz = point_cloud(tgt_depth, intrinsic, clip=clip)

    src_xyz = src_xyz @ relative_pose[:3, :3].T + relative_pose[:3, 3]

    o3d_src = o3d.geometry.PointCloud()
    o3d_src.points = o3d.utility.Vector3dVector(src_xyz)
    o3d_tgt = o3d.geometry.PointCloud()
    o3d_tgt.points = o3d.utility.Vector3dVector(tgt_xyz)

    # voxel downsample
    overlap_radius = voxel_size * 1.5
    o3d_src_down = o3d_src.voxel_down_sample(voxel_size=voxel_size)
    o3d_tgt_down = o3d_tgt.voxel_down_sample(voxel_size=voxel_size)
    src_down = np.asarray(o3d_src_down.points)
    tgt_down = np.asarray(o3d_tgt_down.points)
    src_tree = o3d.geometry.KDTreeFlann(o3d_src_down)
    tgt_tree = o3d.geometry.KDTreeFlann(o3d_tgt_down)

    # Find knn
    src_corr = np.full(src_down.shape[0], -1)
    for i, s in enumerate(src_down):
        num_knn, knn_indices, _ = tgt_tree.search_radius_vector_3d(
            s, overlap_radius)
        if num_knn > 0:
            src_corr[i] = knn_indices[0]
    tgt_corr = np.full(tgt_down.shape[0], -1)
    for i, t in enumerate(tgt_down):
        num_knn, knn_indices, _ = src_tree.search_radius_vector_3d(
            t, overlap_radius)
        if num_knn > 0:
            tgt_corr[i] = knn_indices[0]

    # Compute overlapping ratio
    src_overlap_ratio = np.sum(src_corr >= 0).astype(
        np.float32) / src_down.shape[0]
    tgt_overlap_ratio = np.sum(tgt_corr >= 0).astype(
        np.float32) / tgt_down.shape[0]
    return src_overlap_ratio, tgt_overlap_ratio


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


def occlusion_filter(depth_rpj: torch.Tensor, mask_rpj: torch.Tensor):
    # preprocess
    depth_pre = depth_rpj.clone()
    depth_pre[~mask_rpj] = torch.inf

    # min pool
    min_neighbors = -F.max_pool2d(
        -depth_pre, kernel_size=3, stride=1,
        padding=1)  # Implicit negative infinity padding

    assert min_neighbors.shape == depth_rpj.shape

    # filter
    mask = (depth_rpj - min_neighbors) < 0.0375
    # depth_rpj[~mask] = 0
    # mask_rpj = mask_rpj & mask
    depth_rpj = torch.where(mask, depth_rpj, min_neighbors)

    return depth_rpj, mask_rpj


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


def data_to_device(data_dict, device):
    for k, v in data_dict.items():
        if isinstance(v, torch.Tensor):
            data_dict[k] = v.to(device)

    return data_dict


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


class DepthAugment(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, depth, invalid_number=0):
        # preprocess
        depth_cln = depth.clone()
        depth_cln[depth_cln == invalid_number] = torch.inf

        # min pool
        min_neighbor = -F.max_pool2d(
            -depth_cln, kernel_size=3, stride=1,
            padding=1)  # Implicit negative infinity padding

        # keep grad graph
        min_neighbor_zero = -F.max_pool2d(
            -depth, kernel_size=3, stride=1,
            padding=1)  # Implicit negative infinity padding

        min_neighbor = torch.where(min_neighbor.isinf(), min_neighbor_zero,
                                   min_neighbor)

        residual = min_neighbor - depth

        output = torch.concat([depth, min_neighbor, residual], dim=-3)

        return output


class DepthDownsample(nn.Module):
    def __init__(self):
        super().__init__()
        pass

    def forward(self, depth, invalid_number=0):
        # preprocess
        depth_cln = depth.clone()
        depth_cln[depth_cln == invalid_number] = torch.inf

        # min pool
        depth_down = -F.max_pool2d(-depth_cln, kernel_size=2)
        # keep grad graph
        depth_down_zero = -F.max_pool2d(-depth, kernel_size=2)
        depth_down = torch.where(depth_down.isinf(), depth_down_zero,
                                 depth_down)

        return depth_down


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
    def __init__(self, dim, dim_out, *, groups=8):
        super().__init__()
        self.block1 = Block(dim, dim_out, groups=groups)
        self.block2 = Block(dim_out, dim_out, groups=groups)
        self.res_conv = nn.Conv2d(dim, dim_out,
                                  1) if dim != dim_out else nn.Identity()

    def forward(self, x):

        h = self.block1(x)

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

class MaskUnet(nn.Module):
    def __init__(self,
                 dim,
                 init_dim=None,
                 out_dim=None,
                 dim_mults=(1, 2, 4, 8),
                 resnet_block_groups=8,
                 learned_variance=False):
        super().__init__()

        # determine dimensions

        self.init_aug = DepthAugment()

        init_dim = default(init_dim, dim)
        self.init_conv = nn.Conv2d(3, init_dim, 7, padding=3)

        dims = [init_dim, *map(lambda m: dim * m, dim_mults)]
        in_out = list(zip(dims[:-1], dims[1:]))

        block_klass = partial(ResnetBlock, groups=resnet_block_groups)

        # layers

        self.downs = nn.ModuleList([])
        self.ups = nn.ModuleList([])
        num_resolutions = len(in_out)

        for ind, (dim_in, dim_out) in enumerate(in_out):
            is_last = ind >= (num_resolutions - 1)

            self.downs.append(
                nn.ModuleList([
                    block_klass(dim_in, dim_in),
                    block_klass(dim_in, dim_in),
                    Residual(PreNorm(dim_in, LinearAttention(dim_in))),
                    Downsample(dim_in, dim_out) if not is_last else nn.Conv2d(
                        dim_in, dim_out, 3, padding=1)
                ]))

        mid_dim = dims[-1]
        self.mid_block1 = block_klass(mid_dim, mid_dim)
        self.mid_attn = Residual(PreNorm(mid_dim, Attention(mid_dim)))
        self.mid_block2 = block_klass(mid_dim, mid_dim)

        for ind, (dim_in, dim_out) in enumerate(reversed(in_out)):
            is_last = ind == (len(in_out) - 1)

            self.ups.append(
                nn.ModuleList([
                    block_klass(dim_out + dim_in, dim_out),
                    block_klass(dim_out + dim_in, dim_out),
                    Residual(PreNorm(dim_out, LinearAttention(dim_out))),
                    Upsample(dim_out, dim_in) if not is_last else nn.Conv2d(
                        dim_out, dim_in, 3, padding=1)
                ]))

        default_out_dim = 1
        self.out_dim = default(out_dim, default_out_dim)

        self.final_res_block = block_klass(dim * 2, dim)
        self.final_conv = nn.Sequential(nn.Conv2d(dim, self.out_dim, 1),
                                        nn.Sigmoid())

    def forward(self, x):

        x = self.init_aug(x)
        x = self.init_conv(x)
        r = x.clone()

        h = []

        for block1, block2, attn, downsample in self.downs:
            x = block1(x)
            h.append(x)

            x = block2(x)
            x = attn(x)
            h.append(x)

            x = downsample(x)

        x = self.mid_block1(x)
        x = self.mid_attn(x)
        x = self.mid_block2(x)

        for block1, block2, attn, upsample in self.ups:
            x = torch.cat((x, h.pop()), dim=1)
            x = block1(x)

            x = torch.cat((x, h.pop()), dim=1)
            x = block2(x)
            x = attn(x)

            x = upsample(x)

        x = torch.cat((x, r), dim=1)

        x = self.final_res_block(x)
        return self.final_conv(x)


# dataset classes
class PairedDepthDataset(torch.utils.data.Dataset):
    def __init__(self, folder, subset, image_size, convert_image_to=None):
        super().__init__()
        self.image_size = image_size
        self.folder = folder
        self.subset = subset

        json_file = os.path.join(folder, "metadata/{}.json".format(subset))
        self.metadata = []
        with open(json_file, "r") as f:
            self.metadata.extend(json.load(f))

        self.transform = T.Compose([T.ToTensor()])

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, index):
        input_path = os.path.join(self.folder, "data", self.metadata[index]['input_path'])
        label_path = os.path.join(self.folder, "data", self.metadata[index]['label_path'])

        input_img = Image.open(input_path)
        input_img = self.transform(input_img)
        input_img = input_img * 1e-4  # original unit is mm, range from 0 ~ 10m
        input_img[input_img > 1] = 0  # clip

        label_img = Image.open(label_path)
        label_img = self.transform(label_img)
        label_img = label_img * 1e-4  # original unit is mm, range from 0 ~ 10m
        label_img[label_img > 1] = 0  # clip

        mask = (torch.abs(label_img - input_img) < 0.005).to(
            dtype=torch.float32)

        # data_dict
        data_dict = {
            "input_img": input_img,
            "label_img": label_img,
            "mask": mask
        }

        return data_dict


class TestDataset(torch.utils.data.Dataset):
    def __init__(self, info, folder, image_size, convert_image_to=None):
        super().__init__()
        self.info = info
        self.folder = folder
        self.image_size = image_size

        maybe_convert_fn = partial(
            convert_image_to_fn,
            convert_image_to) if exists(convert_image_to) else nn.Identity()

        self.transform = T.Compose([
            T.Lambda(maybe_convert_fn),
            T.Resize(image_size, interpolation=T.InterpolationMode.NEAREST),
            T.CenterCrop(image_size),
            T.ToTensor()
        ])

    def __len__(self):
        return len(self.info['src']) + len(self.info['tgt'])

    def __getitem__(self, index):
        # switch src and tgt if needed
        if (index // (self.__len__() // 2)) % 2 == 0:
            src_path = os.path.join(
                "./dataset/indoor/data",
                self.info['src'][index % (self.__len__() // 2)])
            tgt_path = os.path.join(
                "./dataset/indoor/data",
                self.info['tgt'][index % (self.__len__() // 2)])
        else:
            src_path = os.path.join(
                "./dataset/indoor/data",
                self.info['tgt'][index % (self.__len__() // 2)])
            tgt_path = os.path.join(
                "./dataset/indoor/data",
                self.info['src'][index % (self.__len__() // 2)])

        src_info_path = src_path.replace(".pth", ".info.txt")
        tgt_info_path = tgt_path.replace(".pth", ".info.txt")

        # source info
        with open(src_info_path, "r") as f:
            for idx, line in enumerate(f.readlines()):
                if idx != 0:
                    break
                scene_name, seq_name, frame_start_idx, frame_end_idx = line.removesuffix(
                    "\n").split()
                scene_path = os.path.join(self.folder,
                                          scene_name)

                # get intrinsic
                intrinsic = intrinsic_transform(
                    np.loadtxt(
                        os.path.join(scene_path, "camera-intrinsics.txt")),
                    resize=self.image_size,
                    centercrop=self.image_size).astype(np.float32)

                frame_start_idx = int(frame_start_idx)
                frame_end_idx = int(frame_end_idx)

                frame_path = os.path.join(
                    scene_path, seq_name,
                    "frame-{:0>6d}.depth.png".format(frame_start_idx))
                frame_image = self.transform(Image.open(frame_path)) * 1e-4
                frame_image[frame_image > 1] = 0

                src_image = frame_image

                src_pose = np.loadtxt(
                    frame_path.replace("depth.png", "pose.txt"))

        # target info
        with open(tgt_info_path, "r") as f:
            for idx, line in enumerate(f.readlines()):
                if idx != 0:
                    break
                scene_name, seq_name, frame_start_idx, frame_end_idx = line.removesuffix(
                    "\n").split()
                scene_path = os.path.join(self.folder,
                                          scene_name)
                frame_start_idx = int(frame_start_idx)
                frame_end_idx = int(frame_end_idx)

                frame_path = os.path.join(
                    scene_path, seq_name,
                    "frame-{:0>6d}.depth.png".format(frame_start_idx))
                frame_image = self.transform(Image.open(frame_path)) * 1e-4
                frame_image[frame_image > 1] = 0

                tgt_image = frame_image

                tgt_pose = np.loadtxt(
                    frame_path.replace("depth.png", "pose.txt"))

        relative_pose = np.linalg.inv(tgt_pose) @ src_pose

        input_img, input_mask = reproject_tensor(
            src_image[None, ...] * 10, torch.Tensor(intrinsic[None, ...]),
            torch.Tensor(relative_pose[None, ...]))
        input_img = input_img * 0.1
        input_img = input_img.reshape((1, self.image_size, self.image_size))
        input_mask = input_mask.reshape((1, self.image_size, self.image_size))

        label_img = tgt_image  # (c, h, w)
        label_mask = label_img > 0

        mutual_mask = input_mask & label_mask
        input_img[~mutual_mask] = 0
        label_img[~mutual_mask] = 0

        # data_dict
        data_dict = {"input_img": input_img, "label_img": label_img}
        gc.collect()

        return data_dict


# trainer class
class MaskTrainer(object):
    def __init__(self,
                 model,
                 folder,
                 *,
                 image_size=256,
                 train_batch_size=4,
                 train_lr=1e-4,
                 epochs=100,
                 adam_betas=(0.9, 0.99),
                 lr_gamma=0.95,
                 num_samples=25,
                 results_folder='./results',
                 samples_folder='./samples',
                 amp=False,
                 fp16=False,
                 split_batches=True,
                 convert_image_to=None):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no')

        self.accelerator.native_amp = amp

        # random seed
        setup_seed(self.accelerator.process_index)

        # model

        self.model = model

        # sampling and training hyperparameters
        assert has_int_squareroot(
            num_samples), 'number of samples must have an integer square root'
        self.num_samples = num_samples

        train_batch_size *= self.accelerator.num_processes
        self.batch_size = train_batch_size

        self.epochs = epochs
        self.image_size = image_size

        # epoch counter state
        self.epoch_last = -1
        self.epoch = 0

        # dataset and dataloader
        self.train_ds = PairedDepthDataset(folder,
                                           "train",
                                           self.image_size,
                                           convert_image_to=convert_image_to)
        self.train_dl = DataLoader(self.train_ds,
                                   batch_size=train_batch_size,
                                   shuffle=True,
                                   pin_memory=True,
                                   num_workers=cpu_count(),
                                   collate_fn=collate_func)
        self.train_dl = self.accelerator.prepare(self.train_dl)

        self.val_ds = PairedDepthDataset(folder,
                                         "val",
                                         self.image_size,
                                         convert_image_to=convert_image_to)
        self.val_dl = DataLoader(self.val_ds,
                                 batch_size=1,
                                 shuffle=False,
                                 pin_memory=True,
                                 num_workers=cpu_count(),
                                 collate_fn=collate_func)

        # loss
        self.loss_func = nn.BCELoss()
        self.loss_one_epoch = AverageMeter()
        self.loss_hist = []

        # metric
        if self.accelerator.is_main_process:
            self.metrics = {'best': {}, 'current': {}}

        # optimizer

        self.opt = Adam(self.model.parameters(), lr=train_lr, betas=adam_betas)

        # scheduler
        self.scheduler = ExponentialLR(self.opt, gamma=lr_gamma)

        # for logging results in a folder periodically

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.samples_folder = Path(samples_folder)
        self.samples_folder.mkdir(exist_ok=True)

        # logger
        if self.accelerator.is_main_process:
            log_file = self.results_folder / f'train.log'
            self.logger = Logger(log_file)

        # prepare model, dataloader, optimizer with accelerator

        self.model, self.opt, self.scheduler = self.accelerator.prepare(
            self.model, self.opt, self.scheduler)

    @property
    def device(self):
        return self.accelerator.device

    def save(self, milestone):
        if not self.accelerator.is_main_process:
            return

        data = {
            'epoch':
            self.epoch,
            'model':
            self.accelerator.get_state_dict(self.model),
            'opt':
            self.opt.state_dict(),
            'scheduler':
            self.scheduler.state_dict(),
            'scaler':
            self.accelerator.scaler.state_dict()
            if exists(self.accelerator.scaler) else None,
            'loss_hist':
            self.loss_hist,
            'metrics':
            self.metrics
        }

        torch.save(data, str(self.results_folder / f'model-{milestone}.pt'))

    def load(self, milestone):
        accelerator = self.accelerator
        device = accelerator.device

        data = torch.load(str(self.results_folder / f'model-{milestone}.pt'),
                          map_location=device)

        model = self.accelerator.unwrap_model(self.model)
        model.load_state_dict(data['model'])

        self.epoch_last = data['epoch']
        self.opt.load_state_dict(data['opt'])
        self.scheduler.load_state_dict(data['scheduler'])

        self.loss_hist = data['loss_hist']
        self.metrics = data['metrics']

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    def compute_metrics(self, batch_data, batch_output, mask_threshold=0.5):
        if self.accelerator.is_main_process:
            output_mask = batch_output > mask_threshold

            output_img = batch_data['input_img'].clone()
            output_img[~output_mask] = 0

            label_mask = batch_data['mask'] > mask_threshold
            label_img = batch_data['label_img'].clone()
            label_img[~label_mask] = 0

            mse = torch.mean((label_img - output_img)**2).item()
            mae = torch.mean(torch.abs(label_img - output_img)).item()
            sae = torch.sum(torch.abs(label_img - output_img))

            matrix = torch.bincount(2 * label_mask.flatten() + output_mask.flatten(), minlength=4).reshape((2,2))
            intersection = torch.diag(matrix)
            union = torch.sum(matrix, dim=1) + torch.sum(matrix, dim=0) - intersection
            iou = intersection / union
            miou = torch.nanmean(iou) # 
            pacc = torch.sum(intersection) / torch.sum(matrix) # pixel accuracy
            fp = matrix[0][1] # false positive

            # current metrics
            if not 'MSE' in self.metrics['current'].keys():
                self.metrics['current']['MSE'] = AverageMeter()
            self.metrics['current']['MSE'].update(mse)

            if not 'MAE' in self.metrics['current'].keys():
                self.metrics['current']['MAE'] = AverageMeter()
            self.metrics['current']['MAE'].update(mae)

            if not 'SAE' in self.metrics['current'].keys():
                self.metrics['current']['SAE'] = AverageMeter()
            self.metrics['current']['SAE'].update(sae)

            if not 'mIoU' in self.metrics['current'].keys():
                self.metrics['current']['mIoU'] = AverageMeter()
            self.metrics['current']['mIoU'].update(miou)

            if not 'PAcc' in self.metrics['current'].keys():
                self.metrics['current']['PAcc'] = AverageMeter()
            self.metrics['current']['PAcc'].update(pacc)

            if not 'FP' in self.metrics['current'].keys():
                self.metrics['current']['FP'] = AverageMeter()
            self.metrics['current']['FP'].update(fp)

    def reset_metrics(self):
        if self.accelerator.is_main_process:
            for key, value in self.metrics['current'].items():
                self.metrics['current'][key].reset()

    def better_than_best_metrics(self, metrics_name='SAE') -> bool:
        if self.accelerator.is_main_process:
            # best metrics
            if not metrics_name in self.metrics['best'].keys():
                self.metrics['best'][metrics_name] = float(
                    self.metrics['current'][metrics_name])
                return True
            else:
                if float(self.metrics['current']
                         [metrics_name]) <= self.metrics['best'][metrics_name]:
                    self.metrics['best'][metrics_name] = float(
                        self.metrics['current'][metrics_name])
                    return True
                else:
                    return False
        else:
            return False

    def train_one_epoch(self):
        accelerator = self.accelerator
        device = accelerator.device

        self.model.train()
        self.loss_one_epoch.reset()

        with tqdm(total=len(self.train_dl),
                  disable=not accelerator.is_main_process) as pbar:

            for batch_idx, batch_data in enumerate(self.train_dl):

                batch_data = data_to_device(batch_data, device)

                with self.accelerator.autocast():
                    batch_output = self.model(batch_data['input_img'])
                    loss = self.loss_func(batch_output, batch_data['mask'])
                    if accelerator.is_main_process:
                        self.loss_one_epoch.update(loss.item())

                self.accelerator.backward(loss)

                accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                massage = 'Epoch: {}/{}, loss: {:.4e}, batch size: {}, lr: {:.4e}'.format(
                    self.epoch + 1, self.epochs, loss.item(), self.batch_size,
                    self.opt.optimizer.param_groups[0]['lr'])
                pbar.set_description(massage)

                accelerator.wait_for_everyone()

                self.opt.step()
                self.opt.zero_grad()

                accelerator.wait_for_everyone()

                torch.cuda.empty_cache()
                pbar.update(1)

        if accelerator.is_main_process:
            massage = 'Epoch: {}/{}, loss: {:.4e}, batch size: {}, lr: {:.4e}'.format(
                self.epoch + 1, self.epochs, float(self.loss_one_epoch),
                self.batch_size, self.opt.optimizer.param_groups[0]['lr'])
            self.logger.critical(massage)
            self.loss_hist.append(float(self.loss_one_epoch))

        self.scheduler.step()

    @torch.no_grad()
    def eval_one_epoch(self):
        accelerator = self.accelerator
        device = accelerator.device

        self.model.eval()
        self.reset_metrics()

        if accelerator.is_main_process:
            with tqdm(total=len(self.val_dl),
                      disable=not accelerator.is_main_process) as pbar:

                for batch_idx, batch_data in enumerate(self.val_dl):

                    batch_data = data_to_device(batch_data, device)

                    with self.accelerator.autocast():
                        batch_output = self.model(batch_data['input_img'])
                        self.compute_metrics(batch_data, batch_output, mask_threshold=0.99)

                    massage = 'Epoch: {}/{}, mIoU: {:.4e}, PAcc: {:.4e}, FP: {:.4f}'.format(
                        self.epoch + 1, self.epochs,
                        float(self.metrics['current']['mIoU']),
                        float(self.metrics['current']['PAcc']),
                        float(self.metrics['current']['FP']))

                    pbar.set_description(massage)

                    torch.cuda.empty_cache()
                    pbar.update(1)

            self.logger.critical(massage)

    def train_and_eval(self):
        for epoch in range(self.epoch_last + 1, self.epochs):
            self.epoch = epoch
            self.train_one_epoch()
            self.eval_one_epoch()

            if self.better_than_best_metrics():
                self.save('best')

            self.save('latest')

    def test(self):
        self.epoch = self.epoch_last
        self.eval_one_epoch()



# Tester class
class MaskTester(object):
    def __init__(self,
                 model,
                 folder,
                 *,
                 image_size=256,
                 results_folder='./results',
                 samples_folder='./samples',
                 amp=False,
                 fp16=False,
                 split_batches=True,
                 convert_image_to=None):
        super().__init__()

        # accelerator

        self.accelerator = Accelerator(
            split_batches=split_batches,
            mixed_precision='fp16' if fp16 else 'no')

        self.accelerator.native_amp = amp

        # model

        self.model = model

        self.image_size = image_size

        # dataset and dataloader
        with open("./dataset/indoor/metadata/3DMatch.pkl",
                  'rb') as f:
            test_info = pickle.load(f)
        self.test_ds = TestDataset(test_info,
                                   folder,
                                   self.image_size,
                                   convert_image_to=convert_image_to)
        self.test_dl = DataLoader(self.test_ds,
                                  batch_size=1,
                                  shuffle=False,
                                  pin_memory=True,
                                  num_workers=cpu_count(),
                                  collate_fn=collate_func)

        # for logging results in a folder periodically

        self.results_folder = Path(results_folder)
        self.results_folder.mkdir(exist_ok=True)

        self.samples_folder = Path(samples_folder)
        self.samples_folder.mkdir(exist_ok=True)

        # logger
        if self.accelerator.is_main_process:
            log_file = self.results_folder / f'test.log'
            self.logger = Logger(log_file)

        # prepare model with accelerator

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

        if 'version' in data:
            print(f"loading from version {data['version']}")

        if exists(self.accelerator.scaler) and exists(data['scaler']):
            self.accelerator.scaler.load_state_dict(data['scaler'])

    @torch.no_grad()
    def eval_one_epoch(self):
        accelerator = self.accelerator
        device = accelerator.device

        self.model.eval()

        if accelerator.is_main_process:
            with tqdm(total=len(self.test_dl),
                      disable=not accelerator.is_main_process) as pbar:

                for batch_idx, batch_data in enumerate(self.test_dl):

                    batch_data = data_to_device(batch_data, device)

                    with self.accelerator.autocast():
                        batch_output = self.model(batch_data['input_img'])

                    input_img = batch_data['input_img']

                    output_mask = batch_output > 0.5

                    output_img = batch_data['input_img'].clone()
                    output_img[~output_mask] = 0

                    # label_mask = batch_data['mask'] > 0.5
                    # label_img = batch_data['label_img'].clone()
                    # label_img[label_mask] = 0

                    cmap = cm.get_cmap('gray')

                    image_list = [
                        cmap(input_img.squeeze().cpu().numpy(),
                             bytes=True)[..., :3],
                        cmap(output_img.squeeze().cpu().numpy(),
                             bytes=True)[..., :3]
                    ]
                    make_gif(image_list, self.samples_folder,
                             'sample-{:0>6d}'.format(batch_idx))

                    torch.cuda.empty_cache()
                    pbar.update(1)

    def test(self):
        self.eval_one_epoch()