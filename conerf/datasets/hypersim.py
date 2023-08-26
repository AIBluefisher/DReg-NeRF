"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import os
import sys
from typing import Tuple

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import cv2

from conerf.datasets.dataset_base import DatasetBase
from conerf.datasets.utils import minify, visualize_block_poses
from conerf.register.cluster import clustering


_PATH = os.path.abspath(__file__)

sys.path.insert(
    0, os.path.join(os.path.dirname(_PATH), "..", "pycolmap", "pycolmap")
)
from scene_manager import SceneManager


def _collect_camera_names(directory: str) -> list:
    camera_names = []
    for _, dir_list, __ in os.walk(directory):
        for dir_name in dir_list:
            if not dir_name.find('cam_') < 0:
                camera_names.append(dir_name)
    
    return camera_names


def _get_all_filenames(directory: str) -> list:
    filenames = []
    for _, __, file_list in os.walk(directory):
        for filename in file_list:
            filenames.append(filename)
    
    return filenames


def _get_all_image_names(directory: str, image_type: str = 'tonemap') -> Tuple[list, list]:
    image_names, image_ids = [], []
    for _, __, file_list in os.walk(directory):
        for filename in file_list:
            if not filename.find(image_type) < 0:
                image_id = int(filename[6:10])
                image_names.append(os.path.join(directory, filename))
                image_ids.append(image_id)

    return sorted(image_names), image_ids


def get_filename_no_ext(filename):
    return os.path.splitext(filename)[0]


def get_file_extension(filename):
    return os.path.splitext(filename)[-1]


def get_all_image_names(dir, formats=[
    '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']) -> list:
    image_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if not get_file_extension(file) in formats:
                continue
            image_paths.append(os.path.join(dir, root, file))
    return sorted(image_paths)


def get_filename_from_abs_path(abs_path):
    return abs_path.split('/')[-1]



def _load_colmap(root_fp: str, subject_id: str, split: str, factor: int = 1, multi_blocks: bool = False, num_blocks: int = 1):
    assert factor in [1, 2, 4, 8]

    data_dir = os.path.join(root_fp, subject_id)
    colmap_dir = os.path.join(data_dir, "sparse/0")

    # Read bounding box.
    scene_bbox = torch.from_numpy(np.loadtxt(os.path.join(colmap_dir, 'bbox.txt'))).float()[:6]

    if factor != 1:
        minify(basedir=data_dir, factors=[factor])

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()

    # Assume shared intrinsics between all cameras.
    cam = manager.cameras[1]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= factor

    # Extract extrinsic matrices in world-to-camera format.
    imdata = manager.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    image_names = [imdata[k].name for k in imdata]

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]

    # Load images.
    colmap_image_dir = os.path.join(data_dir, "images")
    image_paths = [
        os.path.join(colmap_image_dir, f) for f in image_names
    ]
    print("loading images")
    images = [imageio.imread(x) for x in tqdm.tqdm(image_paths)]
    images = np.stack(images, axis=0)

    val_interval = 30
    if multi_blocks:
        labels = clustering(camtoworlds, num_clusters=num_blocks, method="KMeans")
        # visualize_block_poses(camtoworlds, labels)

        # Group images into blocks according to their cluster labels.
        block_image_ids = dict()
        for image_id, label in enumerate(labels):
            if label not in block_image_ids.keys():
                block_image_ids[label] = list()
            block_image_ids[label].append(image_id)

        num_blocks = len(block_image_ids)
        block_images, block_camtoworlds = [None] * num_blocks, [None] * num_blocks
        for block_id in block_image_ids.keys():
            image_ids = sorted(block_image_ids[block_id])
            image_ids = np.array(image_ids)
            # Select the split.
            all_indices = list(range(0, image_ids.shape[0]))
            test_indices = list(range(0, image_ids.shape[0], val_interval))
            train_indices = np.array([i for i in all_indices if i not in test_indices])
            split_indices = {
                "test": image_ids[np.array(test_indices)],
                "train": image_ids[train_indices]
            }
            indices = split_indices[split]
            block_images[block_id] = images[indices]
            block_camtoworlds[block_id] = camtoworlds[indices]

        return block_images, block_camtoworlds, K, scene_bbox
    
    # Select the split.
    all_indices = np.arange(images.shape[0])
    split_indices = {
        "test": all_indices[all_indices % val_interval == 0],
        "train": all_indices[all_indices % val_interval != 0],
    }
    indices = split_indices[split]
    # All per-image quantities must be re-indexed using the split indices.
    images = images[indices]
    camtoworlds = camtoworlds[indices]

    return images, camtoworlds, K, scene_bbox


class SubjectLoader(DatasetBase):
    SUBJECT_IDS = [
        "ai_001_001", "ai_001_002", "ai_001_003", "ai_001_004", "ai_001_005", 
        "ai_001_006", "ai_001_007", "ai_001_008", "ai_001_009", "ai_001_010", 
        "ai_002_001", "ai_002_002", "ai_002_003", "ai_002_004", "ai_002_005", 
        "ai_002_006", "ai_002_007", "ai_002_008", "ai_002_009", "ai_002_010", 
        "ai_003_001", "ai_003_002", "ai_003_003", "ai_003_004", "ai_003_005", 
        "ai_003_006", "ai_003_007", "ai_003_008", "ai_003_009", "ai_003_010", 
        "ai_004_001", "ai_004_002", "ai_004_003", "ai_004_004", "ai_004_005", 
        "ai_004_006", "ai_004_007", "ai_004_008", "ai_004_009", "ai_004_010", 
        "ai_005_001", "ai_005_002", "ai_005_003", "ai_005_004", "ai_005_005", 
        "ai_005_006", "ai_005_007", "ai_005_008", "ai_005_009", "ai_005_010", 
        "ai_006_001", "ai_006_002", "ai_006_003", "ai_006_004", "ai_006_005", 
        "ai_006_006", "ai_006_007", "ai_006_008", "ai_006_009", "ai_006_010", 
        "ai_007_001", "ai_007_002", "ai_007_003", "ai_007_004", "ai_007_005", 
        "ai_007_006", "ai_007_007", "ai_007_008", "ai_007_009", "ai_007_010", 
    ]

    OPENGL_CAMERA = False
    DATA_TYPE = "REAL_WORLD"

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        data_split_json: str = "",
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        factor: int = 1,
        multi_blocks: bool = False,
        num_blocks: int = 1
    ) -> None:
        super().__init__(
            subject_id,
            root_fp,
            split,
            data_split_json,
            color_bkgd_aug,
            num_rays,
            near,
            far,
            batch_over_images,
            factor,
            multi_blocks,
            num_blocks
        )

    def load_data(self, root_fp: str, subject_id: str, split: str, factor: int = 1, num_blocks: int = 1):
        images, camtoworlds, K, scene_bbox = _load_colmap(
            root_fp, subject_id, split, factor, self.multi_blocks, num_blocks
        )
        self.BBOX = scene_bbox.tolist()

        if not self.multi_blocks:
            images = torch.from_numpy(images).to(torch.uint8)
            camtoworlds = torch.from_numpy(camtoworlds).to(torch.float32)
        K = torch.tensor(K).to(torch.float32)

        return images, camtoworlds, K
