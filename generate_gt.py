import os
from pathlib import Path
from typing import Optional, Union, Sequence
from itertools import combinations

import numpy as np
import open3d as o3d
import torch
from torchvision import transforms as T

from PIL import Image

from tqdm import tqdm

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--dataset_name',
                    default='generated_dataset',
                    type=str,
                    help='',
                    required=True)
parser.add_argument('--start_scene_index',
                    '-start',
                    default=0,
                    type=int,
                    help='scenes index to start')
parser.add_argument('--stop_scene_index',
                    '-stop',
                    default=1,
                    type=int,
                    help='scenes index to stop')
parser.add_argument('--num_samples',
                    default=2,
                    type=int,
                    help='sample numbers for each scene')
parser.add_argument('--disable_tqdm',
                    action="store_true",
                    help='disable tqdm')

args = parser.parse_args()


def point_cloud(depth, intrinsic, clip=[0.5, 9.5]):
    fx = intrinsic[0][0]
    fy = intrinsic[1][1]
    cx = intrinsic[0][2]
    cy = intrinsic[1][2]

    rows, cols = depth.shape
    c, r = np.meshgrid(np.arange(cols), np.arange(rows), sparse=True)
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


def compute_overlap_ratio(pc1,
                          pc2,
                          voxel_size=0.025,
                          overlap_factor=1.5,
                          is_down_sample=True):
    search_voxel_size = voxel_size * overlap_factor
    if is_down_sample:
        pc1_down = pc1.voxel_down_sample(voxel_size=voxel_size)
        pc2_down = pc2.voxel_down_sample(voxel_size=voxel_size)
    else:
        pc1_down = pc1
        pc2_down = pc2
    pc1_xyz = np.asarray(pc1_down.points)
    pc2_xyz = np.asarray(pc2_down.points)
    pc1_tree = o3d.geometry.KDTreeFlann(pc1_down)
    pc2_tree = o3d.geometry.KDTreeFlann(pc2_down)

    # Find knn
    pc1_corr = np.full(pc1_xyz.shape[0], -1)
    for i, s in enumerate(pc1_xyz):
        num_knn, knn_indices, _ = pc2_tree.search_radius_vector_3d(
            s, search_voxel_size)
        if num_knn > 0:
            pc1_corr[i] = knn_indices[0]
    pc2_corr = np.full(pc2_xyz.shape[0], -1)
    for i, t in enumerate(pc2_xyz):
        num_knn, knn_indices, _ = pc1_tree.search_radius_vector_3d(
            t, search_voxel_size)
        if num_knn > 0:
            pc2_corr[i] = knn_indices[0]

    # Compute overlapping ratio
    pc1_overlap_ratio = np.sum(pc1_corr >= 0) / pc1_xyz.shape[0]
    pc2_overlap_ratio = np.sum(pc2_corr >= 0) / pc2_xyz.shape[0]
    return pc1_overlap_ratio, pc2_overlap_ratio


def generate_gt(dataset_name, start_scene_index, stop_scene_index,
                num_samples):

    root_path = Path("./{}/data".format(dataset_name))
    depth_ext = "depth.png"
    pose_ext = "pose.txt"
    cloud_ext = "cloud.ply"
    scene_num = stop_scene_index - start_scene_index
    image_size = 256

    pbar = tqdm(range(start_scene_index, stop_scene_index),
                disable=args.disable_tqdm)

    for scene_idx in pbar:
        scene_name = "scene-{:0>6d}".format(scene_idx)
        scene_path = root_path.joinpath(scene_name)
        gt_infos = []
        gt_path = "./{}/data/scene-{:0>6d}/gt.log".format(
            dataset_name, scene_idx)
        gt_path = Path(gt_path)

        if gt_path.exists():
            print("scene gt log has existed, skip over it")
            continue

        for src_idx, tgt_idx in combinations(range(num_samples), 2):
            src_name = "sample-{:0>6d}.cloud.ply".format(src_idx)
            tgt_name = "sample-{:0>6d}.cloud.ply".format(tgt_idx)

            src_path = scene_path.joinpath(src_name)
            tgt_path = scene_path.joinpath(tgt_name)

            if (not src_path.exists()) or (not tgt_path.exists()):
                continue

            o3d_src = o3d.io.read_point_cloud(str(src_path))
            o3d_tgt = o3d.io.read_point_cloud(str(tgt_path))

            if np.asarray(o3d_src.points).shape[0] < 1000 or np.asarray(
                    o3d_tgt.points).shape[0] < 1000:
                continue

            overlap_src, overlap_tgt = compute_overlap_ratio(o3d_src, o3d_tgt)
            pbar.set_description(
                "src:{:0>2d}, tgt:{:0>2d}, ov_src:{:.3f}, ov_tgt:{:.3f}".
                format(src_idx, tgt_idx, overlap_src, overlap_tgt))

            if np.isnan(overlap_src) or np.isnan(overlap_tgt):
                continue
            if overlap_src < 0.1 and overlap_tgt < 0.1:
                continue

            gt_info = {
                "scene_name": scene_name,
                "src_idx": str(src_idx),
                "tgt_idx": str(tgt_idx),
                "overlap_src": "{:.4f}".format(overlap_src),
                "overlap_tgt": "{:.4f}".format(overlap_tgt)
            }
            gt_infos.append(gt_info)

        gt_path.parent.mkdir(exist_ok=True)

        with open(gt_path, "w") as f:
            for gt_info in gt_infos:
                lines = []
                lines.append(gt_info["scene_name"] + "\t" +
                             gt_info["src_idx"] + "\t" + gt_info["tgt_idx"] +
                             "\t" + gt_info["overlap_src"] + "\t" +
                             gt_info["overlap_tgt"] + "\n")
                f.writelines(lines)

def gather_gt(dataset_name, start_index, stop_index):
    print("gathering gt...")
    final_gt_path = Path("./{}/metadata/gt.log".format(dataset_name))
    final_gt_path.parent.mkdir(parents=True, exist_ok=True)
    if final_gt_path.exists():
        print("gt log exists, delete it")
        os.remove(str(final_gt_path))
    for scene_idx in range(start_index, stop_index):
        scene_gt_path = "./{}/data/scene-{:0>6d}/gt.log".format(
            dataset_name, scene_idx)
        print("cat {} >> {}".format(scene_gt_path, final_gt_path))
        os.system("cat {} >> {}".format(scene_gt_path, final_gt_path))


if __name__ == "__main__":
    generate_gt(args.dataset_name, args.start_scene_index,
                args.stop_scene_index, args.num_samples)
    gather_gt(args.dataset_name, args.start_scene_index, args.stop_scene_index)
