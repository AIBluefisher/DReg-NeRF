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
    data_dir = os.path.join(root_fp, subject_id)

    # Read bounding box.
    scene_bbox = torch.from_numpy(np.loadtxt(f'{data_dir}/bbox.txt')).float()[:6] #.view(2,3)

    # Read focal length.
    with open(os.path.join(data_dir, "intrinsics.txt")) as f:
        focal = float(f.readline().split()[0])

    # Load images and poses.
    pose_files = sorted(os.listdir(os.path.join(data_dir, 'pose')))
    image_files  = sorted(os.listdir(os.path.join(data_dir, 'rgb')))

    if split == 'train':
        pose_files = [x for x in pose_files if x.startswith('0_')]
        image_files = [x for x in image_files if x.startswith('0_')]
    elif split == 'val':
        pose_files = [x for x in pose_files if x.startswith('1_')]
        image_files = [x for x in image_files if x.startswith('1_')]
    elif split == 'test':
        test_pose_files = [x for x in pose_files if x.startswith('2_')]
        test_image_files = [x for x in image_files if x.startswith('2_')]
        if len(test_pose_files) == 0:
            test_pose_files = [x for x in pose_files if x.startswith('1_')]
            test_image_files = [x for x in image_files if x.startswith('1_')]
        pose_files = test_pose_files
        image_files = test_image_files

    images = []
    camtoworlds = []

    assert len(image_files) == len(pose_files)
    for image_name, pose_name in tqdm.tqdm(
        zip(image_files, pose_files), desc=f'Loading ({len(image_files)}) {split} set'
    ):
        image_path = os.path.join(data_dir, 'rgb', image_name)
        rgba = imageio.imread(image_path)
        c2w = np.loadtxt(os.path.join(data_dir, 'pose', pose_name))
        camtoworlds.append(c2w)
        images.append(rgba)

    images = np.stack(images, axis=0)
    camtoworlds = np.stack(camtoworlds, axis=0)
    val_interval = 20
    
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

        return block_images, block_camtoworlds, focal, scene_bbox

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

    return images, camtoworlds, focal, scene_bbox


class SubjectLoader(DatasetBase):
    SPLITS = ["train", "val", "test"]
    SUBJECT_IDS = [
        "Bike",
        "Lifestyle",
        "Palace",
        "Robot",
        "Spaceship",
        "Steamtrain",
        "Toad",
        "Wineholder",
    ]
    
    WIDTH, HEIGHT = 800, 800
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = False # True
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
        images, camtoworlds, focal, scene_bbox = _load_renderings(
            root_fp, subject_id, split, self.multi_blocks, num_blocks
        )
        self.BBOX = scene_bbox.tolist()

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
