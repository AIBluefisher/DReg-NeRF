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
    with open(
        os.path.join(data_dir, f"{split}.json"), "r"
    ) as fp:
        meta = json.load(fp)
    images = []
    camtoworlds = []

    if split.find('train') < 0:
        meta["frames"] = meta["frames"][0:len(meta["frames"]):10]

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

    # camera_angle_x = float(meta["camera_angle_x"])
    # focal = 0.5 * w / np.tan(0.5 * camera_angle_x)
    fx, fy = float(meta["fl_x"]), float(meta["fl_y"])
    cx, cy = float(meta["cx"]), float(meta["cy"])
    
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
            indices = image_ids[np.array(test_indices)] if split.find('train') < 0 else \
                    image_ids[np.array(all_indices)]
            block_images[block_id] = images[indices]
            block_camtoworlds[block_id] = camtoworlds[indices]

        return block_images, block_camtoworlds, fx, fy, cx, cy

    # Select the split.
    all_indices = np.arange(images.shape[0])
    if split.find('train') < 0:
        indices = all_indices[all_indices % val_interval == 0]
        # All per-image quantities must be re-indexed using the split indices.
        images = images[indices]
        camtoworlds = camtoworlds[indices]

    return images, camtoworlds, fx, fy, cx, cy


class SubjectLoader(DatasetBase):
    SPLITS = [
        # "train"
        "train_0", "train_1", "train_2", "train_3", "train_4", "train_5", "train_6", "train_7",
        "train_all", "train_all_100", "train_250", "train_500",
        # "val"
        "val_0", "val_1", "val_2", "val_3", "val_4", "val_5", "val_6", "val_7", "val_all",
        # "test"
        "test_0", "test_1", "test_2", "test_3", "test_4", "test_5", "test_6", "test_7", "test_all"
    ]
    SUBJECT_IDS = [
        "airplane1-005", "airplane2-023", "brontosaurus-030", "bulldozer1-022", "bulldozer2-016",
        "cheetah-017", "dump_truck1-034", "dump_truck2-035", "elephant-024", "excavator-004",
        "forklift-029", "giraffe-020", "helicopter1-015", "helicopter2-002", "lego-013",
        "lion-011", "plant1-033", "plant2-026", "plant3-006", "plant4-003", "plant5-019",
        "plant6-027", "plant7-001", "plant8-025", "plant9-031", "roadroller-014", "shark-010",
        "spinosaurus-008", "stegosaurus-007", "tiger-018", "tractor-012", "trex-009",
        "triceratops-021", "truck-032", "zebra-028"
    ]
    
    WIDTH, HEIGHT = 1440, 1080
    NEAR, FAR = 2.0, 6.0
    OPENGL_CAMERA = True # True
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
        if split == 'test' or split == 'val':
            # 'test_all' is not publicly available for scannerf.
            split = 'val_all'
        
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
        images, camtoworlds, fx, fy, cx, cy = _load_renderings(
            root_fp, subject_id, split, self.multi_blocks, num_blocks
        )
        
        if not self.multi_blocks:
            images = torch.from_numpy(images).to(torch.uint8)
            camtoworlds = torch.from_numpy(camtoworlds).to(torch.float32)
        K = torch.tensor(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)

        return images, camtoworlds, K
