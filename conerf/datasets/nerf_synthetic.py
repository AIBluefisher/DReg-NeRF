"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import json
import os

import imageio.v2 as imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from conerf.datasets.dataset_base import DatasetBase
from conerf.datasets.utils import visualize_block_poses
from conerf.register.cluster import clustering


def _load_renderings(root_fp: str, subject_id: str, split: str, multi_blocks: bool = False, num_blocks: int = 1):
    """Load images from disk."""
    if not root_fp.startswith("/"):
        # allow relative path. e.g., "./data/nerf_synthetic/"
        root_fp = os.path.join(
            os.path.dirname(os.path.abspath(__file__)),
            "..",
            "..",
            root_fp,
        )

    data_dir = os.path.join(root_fp, subject_id)
    with open(
        os.path.join(data_dir, "transforms_{}.json".format(split)), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    pbar = tqdm.trange(len(meta["frames"]), desc=f"Loading {len(meta['frames'])} {split} set")
    for i in range(len(meta["frames"])):
        frame = meta["frames"][i]
        fname = os.path.join(data_dir, frame["file_path"] + ".png")
        rgba = imageio.imread(fname)
        camtoworlds.append(frame["transform_matrix"])
        images.append(rgba)
        pbar.update(1)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    val_interval = 20

    h, w = images.shape[1:3]
    camera_angle_x = float(meta["camera_angle_x"])
    focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    
    if multi_blocks:
        labels = clustering(camtoworlds, num_clusters=num_blocks, method="KMeans")
        # visualize_block_poses(self.camtoworlds, labels)

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

        return block_images, block_camtoworlds, focal

    # Select the split.
    all_indices = np.arange(images.shape[0])
    if split != 'train' or split != 'trainval':
        split_indices = {
            "test": all_indices[all_indices % val_interval == 0],
            "train": all_indices[all_indices % val_interval != 0],
        }
        indices = split_indices[split]
        # All per-image quantities must be re-indexed using the split indices.
        images = images[indices]
        camtoworlds = camtoworlds[indices]

    return images, camtoworlds, focal


class SubjectLoader(DatasetBase):
    SPLITS = ["train", "val", "trainval", "test"]
    SUBJECT_IDS = [
        "chair",
        "drums",
        "ficus",
        "hotdog",
        "lego",
        "materials",
        "mic",
        "ship",
    ]
    
    WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True
    DATA_TYPE = "SYNTHETIC"
    
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
        if split == "trainval":
            _images_train, _camtoworlds_train, _focal_train = _load_renderings(
                root_fp, subject_id, "train", self.multi_blocks, num_blocks
            )
            _images_val, _camtoworlds_val, _focal_val = _load_renderings(
                root_fp, subject_id, "val", self.multi_blocks, num_blocks
            )
            images = np.concatenate([_images_train, _images_val])
            camtoworlds = np.concatenate(
                [_camtoworlds_train, _camtoworlds_val]
            )
            focal = _focal_train
        else:
            images, camtoworlds, focal = _load_renderings(
                root_fp, subject_id, split, self.multi_blocks, num_blocks
            )
        
        if not self.multi_blocks:
            images = torch.from_numpy(images).to(torch.uint8)
            camtoworlds = torch.from_numpy(camtoworlds).to(torch.float32)
        K = torch.tensor(
            [
                [focal, 0, self.WIDTH / 2.0],
                [0, focal, self.HEIGHT / 2.0],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)

        return images, camtoworlds, K
