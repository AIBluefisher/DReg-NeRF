import os
import random
import json

import numpy as np
import torch
from torch.utils.data import Dataset

from conerf.datasets.dataset_base import _read_transforms
from conerf.register.se3 import se3_init, se3_cat, se3_inv, se3_transform


def uniform_2_sphere():
    """Uniform sampling on a 2-sphere

    Source: https://gist.github.com/andrewbolster/10274979

    Args:
        size: Number of vectors to sample

    Returns:
        Random Vector (np.ndarray) of size (size, 3) with norm 1.
        If size is None returned value will have size (3,)

    """

    phi = np.random.uniform(0.0, 2 * np.pi)
    cos_theta = np.random.uniform(-1.0, 1.0)

    theta = np.arccos(cos_theta)
    x = np.sin(theta) * np.cos(phi)
    y = np.sin(theta) * np.sin(phi)
    z = np.cos(theta)

    return np.stack((x, y, z), axis=-1)


def hat(v: np.ndarray) -> np.ndarray:
    """Maps a vector to a 3x3 skew symmetric matrix."""
    h = np.zeros((*v.shape, 3))
    h[..., 0, 1] = -v[..., 2]
    h[..., 0, 2] = v[..., 1]
    h[..., 1, 2] = -v[..., 0]
    h = h - h.swapaxes(-1, -2)
    return h


def exp_and_theta(omega: np.ndarray) -> np.ndarray:
    """Same as exp() but also returns theta (rotation angle in radians)
    """
    theta = np.linalg.norm(omega, axis=-1, keepdims=True)
    near_zero = np.isclose(theta, 0.)[..., None]

    # Near phi==0, use first order Taylor expansion
    rotmat_taylor = np.identity(3) + hat(omega)

    # Otherwise, use Rodrigues formulae
    with np.errstate(divide='ignore', invalid='ignore'):
        w = omega / theta  # axis, with norm = 1
    w_hat = hat(w)
    w_hat2 = w_hat @ w_hat
    s = np.sin(theta)[..., None]
    c = np.cos(theta)[..., None]
    rotmat_rodrigues = np.identity(3) + s * w_hat + (1 - c) * w_hat2

    rotmat = np.where(near_zero, rotmat_taylor, rotmat_rodrigues)

    return rotmat
        

def _sample_so3_small(std) -> np.ndarray:
    # First sample axis
    rand_dir = uniform_2_sphere()

    # Then samples angle magnitude
    theta = np.random.randn()
    theta *= std * np.pi / np.sqrt(3)

    return exp_and_theta(rand_dir * theta)


def _sample_se3_small(std) -> np.ndarray:
    rotmat = _sample_so3_small(std)
    trans = np.random.randn(3, 1) * std / np.sqrt(3)

    mat = np.concatenate([rotmat, trans], axis=-1)
    bottom_row = np.zeros_like(mat[..., :1, :])
    bottom_row[..., -1, -1] = 1.0
    mat = np.concatenate([mat, bottom_row], axis=-2)

    return mat


def _load_meta(
    root_fp: str,
    dataset: str,
    subject_id: str,
    model_dir: str
) -> dict:
    raw_data_dir = os.path.join(root_fp, 'images', subject_id)
    block_model_dir = os.path.join(root_fp, model_dir, subject_id)

    if not os.path.exists(block_model_dir):
        print(f'[WARNING] {block_model_dir} does not exist!')
        return None

    scene_meta = {'dataset': dataset, 'scene': subject_id}

    # transformations are applied to each block's camera poses before training 
    # NeRF.
    transforms = _read_transforms(raw_data_dir)
    num_blocks = len(transforms)

    for k in range(num_blocks):
        block_meta = dict()
        block_meta['transform'] = transforms[k]

        block_dir = os.path.join(root_fp, model_dir, subject_id, f'block_' + str(k))
        model_path = os.path.join(block_dir, 'model.pth')
        voxel_grid_path = os.path.join(block_dir, 'voxel_grid.pt')
        voxel_grid_mask_path = os.path.join(block_dir, 'voxel_mask.pt')
        voxel_grid_ply_path = os.path.join(block_dir, 'voxel_point_cloud.ply')

        assert os.path.exists(model_path), f"{model_path} does not exist!"
        assert os.path.exists(voxel_grid_path), f"{voxel_grid_path} does not exist!"
        assert os.path.exists(voxel_grid_mask_path), f"{voxel_grid_mask_path} does not exist!"
        assert os.path.exists(voxel_grid_ply_path), f"{voxel_grid_ply_path} does not exist!"

        block_meta['model_path'] = model_path
        block_meta['voxel_grid_path'] = voxel_grid_path
        block_meta['voxel_mask_path'] = voxel_grid_mask_path
        block_meta['voxel_grid_ply_path']  = voxel_grid_ply_path

        scene_meta[f'block_{k}'] = block_meta

    return scene_meta


class NeRFRegDataset(Dataset):
    SPLITS = ["train", "test"]
    SUBJECT_IDS = {}

    def __init__(
        self,
        root_fp: str,
        dataset: str = None,
        json_dir: str = "",
        subject_id: str = None,
        split: str = 'train',
        model_dir: str = 'nerf_models'
    ) -> None:
        super().__init__()
        
        self.load_split(json_dir)

        assert split in self.SPLITS, "%s" % split

        self.split = split
        self.mode = split

        # For points jittering
        self.scale = 0.005
        self.clip = 0.05

        # For rigid perturbation
        self.std = 0.1

        self.meta = dict()
        if dataset != None and subject_id != None:
            meta = _load_meta(
                root_fp=root_fp, subject_id=subject_id, model_dir=model_dir
            )
            if meta != None:
                self.meta[self.__len__()] = meta
        else:
            for key in self.SUBJECT_IDS.keys():
                if dataset is not None and key != dataset:
                    continue
                
                scenes = self.SUBJECT_IDS[key][split]
                dataset_dir = os.path.join(root_fp, key)

                assert os.path.exists(dataset_dir), f"Dataset dir: {dataset_dir} does not exit!"

                for scene in scenes:
                    meta = _load_meta(
                        root_fp=dataset_dir, dataset=key, subject_id=scene, model_dir=model_dir
                    )
                    if meta != None:
                        self.meta[self.__len__()] = meta
            
            print(f'Loaded {len(self.meta)} {self.split} scenes.')

    def load_split(self, json_dir: str):
        data_split_json = os.path.join(json_dir, 'objaverse.json')
        with open(data_split_json, "r") as fp:
            splits = json.load(fp)
        
        for idx, split in splits.items():
            if idx != 'objaverse':
                self.SUBJECT_IDS[idx] = split
            else:
                # Convert obj ids to names.
                obj_id_names_json = os.path.join(json_dir, 'obj_id_names.json')
                with open(obj_id_names_json, "r") as fp:
                    obj_id_to_name = json.load(fp)
                
                new_split = {}
                for sp, obj_ids in split.items():
                    obj_names = []
                    for obj_idx in obj_ids:
                        obj_name = obj_id_to_name[obj_idx]
                        obj_names.append(obj_name)
                    new_split[sp] = obj_names
                
                self.SUBJECT_IDS[idx] = new_split

    def __len__(self):
        return len(self.meta)

    @torch.no_grad()
    def __getitem__(self, index):
        scene_meta = self.meta[index]
        num_blocks = len(scene_meta) - 2 # exclude dataset name and scene name.
        block_list = [i for i in range(num_blocks)]
        random.shuffle(block_list)

        src_block = 'block_' + str(block_list[0])
        tgt_block = 'block_' + str(block_list[1])

        src_nerf_path = scene_meta[src_block]['model_path']
        src_voxel_grid_path = scene_meta[src_block]['voxel_grid_path']
        src_voxel_mask_path = scene_meta[src_block]['voxel_mask_path']

        tgt_nerf_path = scene_meta[tgt_block]['model_path']
        tgt_voxel_grid_path = scene_meta[tgt_block]['voxel_grid_path']
        tgt_voxel_mask_path = scene_meta[tgt_block]['voxel_mask_path']

        src_transform = scene_meta[src_block]['transform']
        tgt_transform = scene_meta[tgt_block]['transform']
        # ground truth relative pose from source to target.
        pose = tgt_transform @ torch.linalg.inv(src_transform)

        src_xyz_rgba = torch.load(src_voxel_grid_path).permute(3, 2, 0, 1).unsqueeze(dim=0)
        src_mask = torch.load(src_voxel_mask_path)

        tgt_xyz_rgba = torch.load(tgt_voxel_grid_path).permute(3, 2, 0, 1).unsqueeze(dim=0)
        tgt_mask = torch.load(tgt_voxel_mask_path)

        if self.mode == 'train':
            # print(f'src_xyz_rgba shape: {src_xyz_rgba.shape}')
            src_xyz_rgba[:, :3] = self.points_jitter(src_xyz_rgba[:, :3], src_mask)
            tgt_xyz_rgba[:, :3] = self.points_jitter(tgt_xyz_rgba[:, :3], tgt_mask)

        data = {
            'src_xyz_rgba': src_xyz_rgba,
            'tgt_xyz_rgba': tgt_xyz_rgba,
            'src_mask': src_mask,
            'tgt_mask': tgt_mask,
            'src_nerf_path': src_nerf_path,
            'tgt_nerf_path': tgt_nerf_path,
            'pose': pose,
            'scene': scene_meta['scene'],
            'dataset': scene_meta['dataset'],
            'index': index,
            'block_list': block_list[:2],
            'src_ply_path': scene_meta[src_block]['voxel_grid_ply_path'],
            'tgt_ply_path': scene_meta[tgt_block]['voxel_grid_ply_path'],
        }

        if self.mode == 'train':
            data = self.rigid_perturb(data)
            data = self.random_swap(data)

        return data

    def points_jitter(self, points, mask) -> torch.Tensor:
        batch_size, xyz_dim, z_res, x_res, y_res = points.shape
        xyz = points.permute(0, 3, 4, 2, 1).view(batch_size, -1, xyz_dim) # [B, N, XYZ_DIM]

        noise = (torch.randn(xyz[0, mask].shape) * self.scale).to(xyz.device)

        xyz[0, mask] = xyz[0, mask] + noise  # Add noise to xyz

        return points

    def rigid_perturb(self, data):
        perturb = _sample_se3_small(std=self.std)
        perturb = torch.from_numpy(perturb).float()

        perturb_source = random.random() > 0.5  # whether to perturb source or target

        batch_size, xyz_rgba_dim, z_res, x_res, y_res = data['src_xyz_rgba'].shape
        src_mask = data['src_mask']
        src_xyz = data['src_xyz_rgba'][:, :3].permute(0, 3, 4, 2, 1).view(batch_size, -1, 3) # [N, XYZ_DIM]
        tgt_mask = data['tgt_mask']
        tgt_xyz = data['tgt_xyz_rgba'][:, :3].permute(0, 3, 4, 2, 1).view(batch_size, -1, 3) # [N, XYZ_DIM]

        # Center perturbation around the point centroid (otherwise there's a large
        # induced translation as rotation is centered around origin)
        centroid = torch.mean(src_xyz, dim=1).unsqueeze(2) if perturb_source else \
                   torch.mean(tgt_xyz, dim=1).unsqueeze(2)
        center_transform = se3_init(rot=None, trans=-centroid).to('cpu')
        # print(f'center_transform shape: {center_transform.shape}', flush=True)
        center_transform = torch.cat([
            center_transform,
            torch.tensor(
                [[[0, 0, 0, 1]]],
                dtype=torch.float32,
                device=center_transform.device
            )], dim=1
        )
        perturb = torch.linalg.inv(center_transform) @ perturb @ center_transform

        if perturb_source:
            data['pose'] = data['pose'] @ torch.linalg.inv(perturb)
            # TODO(chenyu): visualize to see if transform takes effect.
            src_xyz[:, src_mask] = se3_transform(perturb.to(src_xyz.device), src_xyz[:, src_mask])
        else:
            data['pose'] = perturb @ data['pose']
            tgt_xyz[:, tgt_mask] = se3_transform(perturb.to(src_xyz.device), tgt_xyz[:, tgt_mask])

        return data

    def random_swap(self, data):
        if random.random() > 0.5:
            data['src_xyz_rgba'], data['tgt_xyz_rgba'] = data['tgt_xyz_rgba'], data['src_xyz_rgba']
            data['src_nerf_path'], data['tgt_nerf_path'] = data['tgt_nerf_path'], data['src_nerf_path']
            data['src_mask'], data['tgt_mask'] = data['tgt_mask'], data['src_mask']
            data['pose'] = torch.linalg.inv(data['pose'])
        return data
