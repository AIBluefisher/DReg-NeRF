import os
import json
from typing import Tuple

import torch
import torch.nn.functional as F
import imageio.v2 as imageio
import numpy as np

from conerf.datasets.utils import Rays
from conerf.geometry.pose_util import random_SE3


def _change_world_frame(camtoworlds: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
    # num_poses = camtoworlds.shape[0]
    SE3 = random_SE3(1)
    camtoworlds = SE3 @ camtoworlds
    
    return camtoworlds, SE3


def _read_transforms(data_dir: str) -> dict:
    json_file = os.path.join(data_dir, 'world_frame_transforms.json')
    if not os.path.exists(json_file):
        print(f'[WARNING] Transformation file {json_file} does not exist!')
        return None
    
    # print(f'Loading world frame transformations from {json_file}')
    with open(json_file, 'r') as f:
        data = json.load(f)

    transforms = dict()
    for block_id, datum in data.items():
        # print(f'block id: {block_id}, datum: {datum}')
        transforms[int(block_id)] = torch.Tensor(datum)

    assert len(transforms) > 0, f"Invalid transformation file: {json_file}"
    
    return transforms


def _save_transforms(data_dir: str, SE3s: dict) -> None:
    data = dict()
    for block_id, transform in SE3s.items():
        data[block_id] = transform.tolist()
    
    json_file = os.path.join(data_dir, 'world_frame_transforms.json')
    json_obj = json.dumps(data, indent=4)
    print(f'Saving world frame transformations to {json_file}')
    with open(json_file, 'w') as f:
        f.write(json_obj)


class DatasetBase(torch.utils.data.Dataset):
    """
    Single subject data loader for training and evaluation.
    """
    SPLITS = ["train", "test"]
    SUBJECT_IDS = [] # available scenes for training NeRF.
    
    BBOX = None
    WIDTH, HEIGHT = 0, 0
    NEAR, FAR = 0., 0.
    OPENGL_CAMERA = None
    DATA_TYPE = None # ["SYNTHETIC", "REAL_WORLD"]

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
        super().__init__()

        assert len(self.SUBJECT_IDS) > 0
        assert split in self.SPLITS, "%s" % split
        # assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]
        
        self.split = split
        self.num_rays = num_rays
        if near is not None:
            self.NEAR = near
        if far is not None:
            self.FAR = far
        self.training = (num_rays is not None) and (
            split in [
                "train", "trainval",
                # For ScanNeRF dataset
                "train_0", "train_1", "train_2", "train_3", "train_4",
                "train_5", "train_6", "train_7", "train_all", 
                "train_all_100", "train_250", "train_500"
            ]
        )
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images
        self.multi_blocks = multi_blocks

        # Logic to load specific dataset.
        if self.multi_blocks:
            data_dir = os.path.join(root_fp, subject_id)
            transforms_data = _read_transforms(data_dir)
            
            self.transforms = dict() if transforms_data is None else transforms_data
            num_blocks = num_blocks if transforms_data is None else len(self.transforms)
        
            self._block_images, self._block_camtoworlds, self.K = self.load_data(
                root_fp, subject_id, split, factor, num_blocks
            )
            self.num_blocks = len(self._block_images)
            self._current_block_id = 0

            for k in range(self.num_blocks):
                self._block_images[k] = torch.from_numpy(self._block_images[k]).to(torch.uint8)
                self._block_camtoworlds[k] = torch.from_numpy(self._block_camtoworlds[k]).to(torch.float32)
                print(f'block {k} has {len(self._block_images[k])} images.')
                # Change world frame.
                if transforms_data is None:
                    self._block_camtoworlds[k], SE3 = _change_world_frame(self._block_camtoworlds[k])
                    self.transforms[k] = SE3
                else:
                    SE3 = self.transforms[k]
                    self._block_camtoworlds[k] = SE3 @ self._block_camtoworlds[k]
            # Saving transformations.
            if transforms_data is None:
                _save_transforms(data_dir=data_dir, SE3s=self.transforms)
        else:
            self._images, self._camtoworlds, self.K = self.load_data(
                root_fp, subject_id, split, factor, num_blocks
            )
        
        if self.WIDTH > 0 and self.HEIGHT > 0:
            assert self.images.shape[1:3] == (self.HEIGHT, self.WIDTH)
        self.height, self.width = self.images.shape[1:3]

        # Check for parameter validity.
        self._check()

    def _check(self) -> None:
        if self.DATA_TYPE == "SYNTHETIC":
            assert self.WIDTH > 0 and self.HEIGHT > 0
        # assert self.NEAR > 0 and self.FAR > 0
        assert self.OPENGL_CAMERA is not None

    @property
    def current_block(self):
        assert self.multi_blocks == True
        return self._current_block_id

    @property
    def images(self):
        if self.multi_blocks == False:
            return self._images
        else:
            return self._block_images[self._current_block_id]
    
    @property
    def camtoworlds(self):
        if self.multi_blocks == False:
            return self._camtoworlds
        else:
            return self._block_camtoworlds[self._current_block_id]

    def to_device(self, device):
        if self.multi_blocks:
            self._block_images[self.current_block] = self._block_images[self.current_block].to(device)
            self._block_camtoworlds[self.current_block] = self._block_camtoworlds[self.current_block].to(device)
        else:
            self._images = self._images.to(device)
            self._camtoworlds = self._camtoworlds.to(device)

    def move_to_next_block(self):
        if self._current_block_id == self.num_blocks - 1:
            return
        self._current_block_id += 1

    def move_to_block(self, block_id):
        assert block_id < self.num_blocks, f"Invalid block id: {block_id}"
        self._current_block_id = block_id

    def load_data(self, root_fp: str, subject_id: str, split: str, factor: int = 1, num_blocks: int = 1):
        raise NotImplementedError

    def __len__(self):
        return len(self.images)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """
        Process the fetched / cached data with randomness.
        """
        pixels, rays = data["rgb"], data["rays"]

        if self.training:
            if self.color_bkgd_aug == "random":
                color_bkgd = torch.rand(3, device=self.images.device)
            elif self.color_bkgd_aug == "white":
                color_bkgd = torch.ones(3, device=self.images.device)
            elif self.color_bkgd_aug == "black":
                color_bkgd = torch.zeros(3, device=self.images.device)
        else:
            # just use white during inference
            color_bkgd = torch.ones(3, device=self.images.device)

        if self.DATA_TYPE == "SYNTHETIC":
            pixels, alpha = torch.split(pixels, [3, 1], dim=-1)
            pixels = pixels * alpha + color_bkgd * (1.0 - alpha)

        return {
            "pixels": pixels,  # [n_rays, 3] or [h, w, 3]
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rgb", "rays"]},
        }

    def update_num_rays(self, num_rays):
        self.num_rays = num_rays

    def fetch_data(self, index):
        """
        Fetch the data (it maybe cached for multiple batches).
        """
        num_rays = self.num_rays

        if self.training:
            if self.batch_over_images:
                image_id = torch.randint(
                    0,
                    len(self.images),
                    size=(num_rays,),
                    device=self.images.device,
                )
            else:
                image_id = [index]
            x = torch.randint(
                0, self.width, size=(num_rays,), device=self.images.device
            )
            y = torch.randint(
                0, self.height, size=(num_rays,), device=self.images.device
            )
        else:
            image_id = [index]
            x, y = torch.meshgrid(
                torch.arange(self.width, device=self.images.device),
                torch.arange(self.height, device=self.images.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()
        
        # generate rays
        rgb = self.images[image_id, y, x] / 255.0  # (num_rays, 3/4)
        image_channels = 4 if self.DATA_TYPE == "SYNTHETIC" else 3
        c2w = self.camtoworlds[image_id]  # (num_rays, 3, 4)
        camera_dirs = F.pad(
            torch.stack(
                [
                    (x - self.K[0, 2] + 0.5) / self.K[0, 0],
                    (y - self.K[1, 2] + 0.5)
                    / self.K[1, 1]
                    * (-1.0 if self.OPENGL_CAMERA else 1.0),
                ],
                dim=-1,
            ),
            (0, 1),
            value=(-1.0 if self.OPENGL_CAMERA else 1.0),
        )  # [num_rays, 3]

        # [n_cams, height, width, 3]
        directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
        origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
        viewdirs = directions / torch.linalg.norm(
            directions, dim=-1, keepdims=True
        )

        if self.training:
            origins = torch.reshape(origins, (num_rays, 3))
            viewdirs = torch.reshape(viewdirs, (num_rays, 3))
            rgb = torch.reshape(rgb, (num_rays, image_channels))
        else:
            origins = torch.reshape(origins, (self.height, self.width, 3))
            viewdirs = torch.reshape(viewdirs, (self.height, self.width, 3))
            rgb = torch.reshape(rgb, (self.height, self.width, image_channels))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rgb": rgb,  # [h, w, 3] or [num_rays, 3]
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }