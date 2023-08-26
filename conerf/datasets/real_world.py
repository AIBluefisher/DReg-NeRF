"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import collections
import os
import sys

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from conerf.datasets.dataset_base import DatasetBase
from conerf.datasets.utils import minify, visualize_poses
from conerf.register.cluster import clustering


_PATH = os.path.abspath(__file__)

sys.path.insert(
    0, os.path.join(os.path.dirname(_PATH), "..", "pycolmap", "pycolmap")
)
from scene_manager import SceneManager


def _load_colmap(root_fp: str, subject_id: str, split: str, factor: int = 1, multi_blocks: bool = False, num_blocks: int = 1):
    assert factor in [1, 2, 4, 8]

    data_dir = os.path.join(root_fp, subject_id)
    colmap_dir = os.path.join(data_dir, "sparse/0/")

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

    # # Switch from COLMAP (right, down, fwd) to Nerf (right, up, back) frame.
    # poses = poses @ np.diag([1, -1, -1, 1])

    # Get distortion parameters.
    type_ = cam.camera_type
    print(f'type: {type_}')

    if type_ == 0 or type_ == "SIMPLE_PINHOLE":
        params = None
        camtype = "perspective"

    elif type_ == 1 or type_ == "PINHOLE":
        params = None
        camtype = "perspective"

    if type_ == 2 or type_ == "SIMPLE_RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        camtype = "perspective"

    elif type_ == 3 or type_ == "RADIAL":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        camtype = "perspective"

    elif type_ == 4 or type_ == "OPENCV":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "p1", "p2"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["p1"] = cam.p1
        params["p2"] = cam.p2
        camtype = "perspective"

    elif type_ == 5 or type_ == "OPENCV_FISHEYE":
        params = {k: 0.0 for k in ["k1", "k2", "k3", "k4"]}
        params["k1"] = cam.k1
        params["k2"] = cam.k2
        params["k3"] = cam.k3
        params["k4"] = cam.k4
        camtype = "fisheye"

    # assert params is None, "Only support pinhole camera model."

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]

    # Load images.
    if factor > 1:
        image_dir_suffix = f"_{factor}"
    else:
        image_dir_suffix = ""
    colmap_image_dir = os.path.join(data_dir, "images")
    image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
    for d in [image_dir, colmap_image_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Image folder {d} does not exist.")
    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(os.listdir(colmap_image_dir))
    image_files = sorted(os.listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [
        os.path.join(image_dir, colmap_to_image[f]) for f in image_names
    ]
    print("loading images")
    images = [imageio.imread(x) for x in tqdm.tqdm(image_paths)]
    images = np.stack(images, axis=0)

    val_interval = 8
    if multi_blocks:
        labels = clustering(camtoworlds, num_clusters=num_blocks, method="KMeans")
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

        return block_images, block_camtoworlds, K
    
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

    return images, camtoworlds, K


class SubjectLoader(DatasetBase):
    SUBJECT_IDS = [
        # nerf_llff_data dataset
        "fern", "flower", "fortress", "horns", "leaves", "orchids", "room", "trex",
        # mipnerf_360 dataset
        "garden", "bicycle", "bonsai", "counter", "kitchen", "room", "stump",
        # small objects
        "elephant"
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
        images, camtoworlds, K = _load_colmap(
            root_fp, subject_id, split, factor, self.multi_blocks, num_blocks
        )

        if not self.multi_blocks:
            images = torch.from_numpy(images).to(torch.uint8)
            camtoworlds = torch.from_numpy(camtoworlds).to(torch.float32)
        K = torch.tensor(K).to(torch.float32)

        return images, camtoworlds, K
