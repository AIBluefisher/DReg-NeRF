import os
import json
from typing import Tuple

import torch
import torch.nn.functional as F
import numpy as np

from conerf.datasets.utils import Rays


class NeRFPoseOnlyDataset(torch.utils.data.Dataset):
    """
    Single subject data loader for training and evaluation.
    """
    SPLITS = ["train", "test"]
    SUBJECT_IDS = [] # available scenes for training NeRF.
    
    BBOX = None
    WIDTH, HEIGHT = 0, 0
    NEAR, FAR = 0., 0.
    OPENGL_CAMERA = None

    def __init__(
        self,
        dataset_name: str,
        camtoworlds: torch.Tensor = None,
        color_bkgd_aug: str = "white",
        batch_over_images: bool = True,
        factor: int = 1,
    ) -> None:
        super().__init__()

        # assert len(self.SUBJECT_IDS) > 0
        # assert subject_id in self.SUBJECT_IDS, "%s" % subject_id
        assert color_bkgd_aug in ["white", "black", "random"]

        self.dataset_name = dataset_name
        self.color_bkgd_aug = color_bkgd_aug
        self.batch_over_images = batch_over_images

        # Logic to load specific dataset.
        self.camtoworlds = camtoworlds
        self.K = self.load_data(factor)
        self.height, self.width = self.HEIGHT, self.WIDTH

        # Check for parameter validity.
        self._check()

    def _check(self) -> None:
        # assert self.NEAR > 0 and self.FAR > 0
        assert self.OPENGL_CAMERA is not None

    def to_device(self, device):
        self.camtoworlds = self.camtoworlds.to(device)

    def load_data(self, factor: int = 1):
        # raise NotImplementedError
        if self.dataset_name == 'objaverse' or self.dataset_name == 'nerf_synthetic':
            self.OPENGL_CAMERA = True
            self.WIDTH, self.HEIGHT = 800, 800
            camera_angle_x = 0.6911112070083618
            fx = 0.5 * self.WIDTH / np.tan(0.5 * camera_angle_x)
            fy = fx
            cx, cy = self.WIDTH / 2, self.HEIGHT / 2
            
        elif self.dataset_name == 'scannerf':
            self.OPENGL_CAMERA = True
            self.WIDTH, self.HEIGHT = 1440, 1080
            fx, fy = 1522.1201085541113, 1521.954743529035
            cx, cy = 727.9348613007779, 541.5426465751151,
            
        else:
            raise NotImplementedError
        
        K = torch.tensor(
            [
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1],
            ],
            dtype=torch.float32,
        )  # (3, 3)
        return K

    def __len__(self):
        return len(self.camtoworlds)

    @torch.no_grad()
    def __getitem__(self, index):
        data = self.fetch_data(index)
        data = self.preprocess(data)
        return data

    def preprocess(self, data):
        """
        Process the fetched / cached data with randomness.
        """
        rays = data["rays"]
        # just use white during inference
        color_bkgd = torch.ones(3, device=self.camtoworlds.device)

        return {
            "rays": rays,  # [n_rays,] or [h, w]
            "color_bkgd": color_bkgd,  # [3,]
            **{k: v for k, v in data.items() if k not in ["rays"]},
        }

    def fetch_data(self, index):
        """
        Fetch the data (it maybe cached for multiple batches).
        """
        image_id = [index]
        x, y = torch.meshgrid(
            torch.arange(self.width, device=self.camtoworlds.device),
            torch.arange(self.height, device=self.camtoworlds.device),
            indexing="xy",
        )
        x = x.flatten()
        y = y.flatten()
        
        # generate rays
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

        origins = torch.reshape(origins, (self.height, self.width, 3))
        viewdirs = torch.reshape(viewdirs, (self.height, self.width, 3))

        rays = Rays(origins=origins, viewdirs=viewdirs)

        return {
            "rays": rays,  # [h, w, 3] or [num_rays, 3]
        }