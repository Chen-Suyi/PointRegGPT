import os.path as osp
import pickle
import random
from typing import Dict

import numpy as np
import torch
import torch.utils.data
import open3d as o3d
from scipy.spatial import cKDTree

from geotransformer.utils.pointcloud import (
    random_sample_rotation,
    random_sample_rotation_v2,
    get_transform_from_rotation_translation,
)
from geotransformer.utils.registration import get_correspondences


class MixturePairDataset(torch.utils.data.Dataset):

    def __init__(
        self,
        dataset_root,
        subset,
        point_limit=None,
        use_augmentation=False,
        augmentation_noise=0.005,
        augmentation_rotation=1,
        overlap_threshold=None,
        return_corr_indices=False,
        matching_radius=None,
        rotated=False,
    ):
        """
        for loading both generated and real data
        """
        super(MixturePairDataset, self).__init__()

        self.dataset_root = dataset_root
        self.metadata_root = osp.join(self.dataset_root, 'metadata')
        self.data_root = osp.join(self.dataset_root, 'data')

        self.subset = subset
        self.point_limit = point_limit
        self.overlap_threshold = overlap_threshold
        self.rotated = rotated

        self.return_corr_indices = return_corr_indices
        self.matching_radius = matching_radius
        if self.return_corr_indices and self.matching_radius is None:
            raise ValueError(
                '"matching_radius" is None but "return_corr_indices" is set.')

        self.use_augmentation = use_augmentation
        self.aug_noise = augmentation_noise
        self.aug_rotation = augmentation_rotation

        with open(osp.join('../../data/3DMatch/metadata/train.pkl'),
                  'rb') as f:
            self.metadata_list = pickle.load(f)
            if self.overlap_threshold is not None:
                self.metadata_list = [
                    x for x in self.metadata_list
                    if x['overlap'] > self.overlap_threshold
                ]
            for idx, metadata in enumerate(self.metadata_list):
                src_path = osp.join('../../data/3DMatch/data',
                                    metadata['pcd0'])
                tgt_path = osp.join('../../data/3DMatch/data',
                                    metadata['pcd1'])
                self.metadata_list[idx]['pcd0'] = src_path
                self.metadata_list[idx]['pcd1'] = tgt_path

        # loading generated data
        with open(osp.join(self.metadata_root, "gt.log"), "r") as f:
            for line in f.readlines():
                scene_name, src_idx, tgt_idx, src_overlap, tgt_overlap = line.split(
                    "\t")
                _, scene_idx = scene_name.split("-")
                src_path = osp.join(
                    self.data_root, scene_name,
                    "sample-{:0>6d}.cloud.ply".format(int(src_idx)))
                tgt_path = osp.join(
                    self.data_root, scene_name,
                    "sample-{:0>6d}.cloud.ply".format(int(tgt_idx)))
                metadata = {}
                metadata['scene_name'] = scene_name
                metadata['frag_id0'] = src_idx
                metadata['frag_id1'] = tgt_idx
                metadata['overlap'] = float(src_overlap)
                metadata['pcd0'] = src_path
                metadata['pcd1'] = tgt_path
                self.metadata_list.append(metadata)

    def __len__(self):
        return len(self.metadata_list)

    def _load_point_cloud(self, file_path):
        if file_path.endswith('.ply'):
            pcd = o3d.io.read_point_cloud(file_path)
            points = np.asarray(pcd.points)
        elif file_path.endswith('.bin'):
            points = np.fromfile(file_path, dtype=np.float32).reshape(-1, 4)
        elif file_path.endswith('.pth'):
            points = torch.load(file_path)
            if not isinstance(points, np.ndarray):
                point = point.numpy()
        else:
            raise AssertionError('Cannot recognize point cloud format')

        if self.point_limit is not None and points.shape[0] > self.point_limit:
            indices = np.random.permutation(points.shape[0])[:self.point_limit]
            points = points[indices]
        return points

    def _augment_point_cloud(self, ref_points, src_points, rotation,
                             translation):
        r"""Augment point clouds.

        ref_points = src_points @ rotation.T + translation

        1. Random rotation to one point cloud.
        2. Random noise.
        """
        aug_rotation = random_sample_rotation(self.aug_rotation)
        if random.random() > 0.5:
            ref_points = np.matmul(ref_points, aug_rotation.T)
            rotation = np.matmul(aug_rotation, rotation)
            translation = np.matmul(aug_rotation, translation)
        else:
            src_points = np.matmul(src_points, aug_rotation.T)
            rotation = np.matmul(rotation, aug_rotation.T)

        ref_points += (np.random.rand(ref_points.shape[0], 3) -
                       0.5) * self.aug_noise
        src_points += (np.random.rand(src_points.shape[0], 3) -
                       0.5) * self.aug_noise

        return ref_points, src_points, rotation, translation

    def _cube_crop(self, ref_points, src_points, size=3):
        src_tree = cKDTree(src_points)
        _, indices_list = src_tree.query(
            ref_points,
            distance_upper_bound=0.05,
            workers=-1,
        )
        invalid_index = src_points.shape[0]
        ref_overlap = indices_list < invalid_index

        ref_points_overlap = ref_points[ref_overlap]

        ref_points_temp = ref_points.copy()
        src_points_temp = src_points.copy()

        c_idx = np.random.randint(ref_points_overlap.shape[0])
        center = ref_points_overlap[c_idx]
        min_square = center - size / 2
        max_square = center + size / 2

        ref_points_temp = ref_points_temp - center
        rotation = random_sample_rotation()
        ref_points_temp = ref_points_temp @ rotation.T
        ref_points_temp = ref_points_temp + center

        ref_mask = np.prod(
            (ref_points_temp - min_square) > 0, axis=1) * np.prod(
                (max_square - ref_points_temp) > 0, axis=1)
        ref_mask = ref_mask.astype(bool)
        ref_points = ref_points[ref_mask]

        src_points_temp = src_points_temp - center
        rotation = random_sample_rotation()
        src_points_temp = src_points_temp @ rotation.T
        src_points_temp = src_points_temp + center

        src_mask = np.prod(
            (src_points_temp - min_square) > 0, axis=1) * np.prod(
                (max_square - src_points_temp) > 0, axis=1)
        src_mask = src_mask.astype(bool)
        src_points = src_points[src_mask]

        return ref_points, src_points

    def __getitem__(self, index):
        data_dict = {}

        # metadata
        metadata: Dict = self.metadata_list[index]
        data_dict['scene_name'] = metadata['scene_name']
        data_dict['ref_frame'] = metadata['frag_id0']
        data_dict['src_frame'] = metadata['frag_id1']
        data_dict['overlap'] = metadata['overlap']

        # get transformation
        if 'rotation' in metadata.keys():
            rotation = metadata['rotation']
        else:
            rotation = np.eye(3)

        if 'translation' in metadata.keys():
            translation = metadata['translation']
        else:
            translation = np.zeros(3)

        # get point cloud
        ref_points = self._load_point_cloud(metadata['pcd0'])
        src_points = self._load_point_cloud(metadata['pcd1'])

        # augmentation
        if self.use_augmentation:
            ref_points, src_points, rotation, translation = self._augment_point_cloud(
                ref_points, src_points, rotation, translation)

        if self.rotated:
            ref_rotation = random_sample_rotation_v2()
            ref_points = np.matmul(ref_points, ref_rotation.T)
            rotation = np.matmul(ref_rotation, rotation)
            translation = np.matmul(ref_rotation, translation)

            src_rotation = random_sample_rotation_v2()
            src_points = np.matmul(src_points, src_rotation.T)
            rotation = np.matmul(rotation, src_rotation.T)

        transform = get_transform_from_rotation_translation(
            rotation, translation)

        # get correspondences
        if self.return_corr_indices:
            corr_indices = get_correspondences(ref_points, src_points,
                                               transform, self.matching_radius)
            data_dict['corr_indices'] = corr_indices

        data_dict['ref_points'] = ref_points.astype(np.float32)
        data_dict['src_points'] = src_points.astype(np.float32)
        data_dict['ref_feats'] = np.ones((ref_points.shape[0], 1),
                                         dtype=np.float32)
        data_dict['src_feats'] = np.ones((src_points.shape[0], 1),
                                         dtype=np.float32)
        data_dict['transform'] = transform.astype(np.float32)

        data_dict['index'] = index

        return data_dict