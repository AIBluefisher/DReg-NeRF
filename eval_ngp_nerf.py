import os
import imageio
import tqdm
import json
import copy

import torch
import torch.nn.functional as F
import numpy as np
import open3d

import lpips
from nerfacc import ContractionType, OccupancyGrid

from conerf.base.checkpoint_manager import CheckPointManager
from conerf.datasets.utils import Rays
from conerf.loss.ssim_torch import ssim
from conerf.radiance_fields.ngp import NGPradianceField
from conerf.register.sample_grid import SampleGrid
from conerf.utils.config import config_parser
from conerf.utils.utils import render_image, colorize


def compute_psnr(gt_image, pred_image, eps=1e-6):
    mse = F.mse_loss(gt_image, pred_image)
    psnr = -10.0 * torch.log(mse + eps) / np.log(10.0)
    return psnr


def compute_ssim(gt_image, pred_image):
    return ssim(gt_image, pred_image).item()


def compute_lpips(lpips_loss, gt_image, pred_image):
    return lpips_loss(gt_image * 2 - 1, pred_image * 2 - 1).item()


class Evaluator():
    def __init__(self, config) -> None:
        self.config = config
        self.device = f"cuda:{self.config.local_rank}"
        
        self.output_dir = os.path.join(self.config.root_dir, 'eval', self.config.scene)
        os.makedirs(self.output_dir, exist_ok=True)

        self.model_dir = os.path.join(self.config.root_dir, 'out', self.config.scene)
        assert os.path.exists(self.model_dir)

        self.load_dataset()
        self.load_model()

        # if self.config.unbounded:
        if self.meta_data['unbounded']:
            self.scene_aabb = None
        else:
            self.scene_aabb = torch.tensor(
                self.meta_data['aabb'], dtype=torch.float32, device=self.device)

    def load_model(self):
        assert os.path.exists(self.config.ckpt_path), \
            f"Checkpoint path \'{self.config.ckpt_path}\' does not exist!"

        # Load meta data from checkpoint at first.
        meta_data = dict()
        meta_data['aabb'] = None
        meta_data['unbounded'] = None
        meta_data['grid_resolution'] = None
        meta_data['contraction_type'] = None
        meta_data['near_plane'] = None
        meta_data['far_plane'] = None
        meta_data['render_step_size'] = None
        meta_data['alpha_thre'] = None
        meta_data['cone_angle'] = None
        meta_data['camera_poses'] = None
        if self.config.multi_blocks:
            meta_data['block_id'] = None

        ckpt_manager = CheckPointManager()
        ckpt_manager.load(self.config, meta_data=meta_data)

        # Locate to the correct data block.
        if self.config.multi_blocks:
            current_block_id = meta_data['block_id']
            self.val_dataset.move_to_block(current_block_id)
            self.val_dataset.to_device(self.device)
            self.output_dir = os.path.join(self.output_dir, 'block_' + str(meta_data['block_id']))
            os.makedirs(self.output_dir, exist_ok=True)

            self.model_dir = os.path.join(self.model_dir, 'block_' + str(meta_data['block_id']))
            assert os.path.exists(self.model_dir)

        # Load models from checkpoint.
        self.meta_data = meta_data

        self.nerf = NGPradianceField(
            aabb=meta_data['aabb'],
            unbounded=meta_data['unbounded']
        ).to(self.device)

        self.occupancy_grid = OccupancyGrid(
            roi_aabb=meta_data['aabb'],
            resolution=meta_data['grid_resolution'],
            contraction_type=meta_data['contraction_type']
        ).to(self.device)

        self.sample_grid = SampleGrid(
            roi_aabb=meta_data['aabb'],
            resolution=meta_data['grid_resolution'],
            contraction_type=meta_data['contraction_type']
        ).to(self.device)

        models = dict()
        models['model'] = self.nerf
        models['occupancy_grid'] = self.occupancy_grid
        ckpt_manager.load(self.config, models=models)

        for param in self.nerf.parameters():
            param.requires_grad = False
        for param in self.occupancy_grid.parameters():
            param.requires_grad = False

        self.nerf.eval()
        self.occupancy_grid.eval()

    def load_dataset(self):
        test_dataset_kwargs = {"factor": self.config.factor}

        if self.config.multi_blocks:
            test_dataset_kwargs["multi_blocks"] = True
        
        if self.config.dataset == 'nerf_synthetic':
            from conerf.datasets.nerf_synthetic import SubjectLoader
        elif self.config.dataset == 'objaverse':
            from conerf.datasets.objaverse import SubjectLoader
        elif self.config.dataset == 'Synthetic_NSVF':
            from conerf.datasets.nsvf import SubjectLoader
        elif self.config.dataset == 'scannerf':
            from conerf.datasets.scan_nerf import SubjectLoader
        elif self.config.dataset == 'BlendedMVS' or self.config.dataset == 'dtu':
            from conerf.datasets.mvs import SubjectLoader
        elif self.config.dataset == 'Hypersim':
            from conerf.datasets.hypersim import SubjectLoader
        else:
            # self.config.dataset == 'nerf_llff_data'/'mipnerf_360':
            from conerf.datasets.real_world import SubjectLoader

        self.val_dataset = SubjectLoader(
            subject_id=self.config.scene,
            root_fp=self.config.root_dir,
            data_split_json=self.config.data_split_json,
            split="test",
            num_rays=None,
            **test_dataset_kwargs,
        )
        self.val_dataset.to_device(self.device)
        self.val_dataset.K = self.val_dataset.K.to(self.device)
        # print(f'[INFO] current block: {self.val_dataset.current_block}')

    @torch.no_grad()
    def evaluate(self):
        val_dir = os.path.join(self.output_dir, 'val')
        os.makedirs(val_dir, exist_ok=True)

        print(f'Evaluating model on scene: {self.config.scene} of ' +
              f'dataset: {self.config.dataset}' +
              f'Results will be saved to {val_dir}')
        lpips_loss = lpips.LPIPS(net="alex").cuda()
        pbar = tqdm.trange(len(self.val_dataset), desc=f"Validating {self.config.expname}", leave=False)
        PSNRs, LPIPSs, SSIMs = [], [], []
        scene_name = self.config.scene
        results_dict = {scene_name: {}}

        for i in range(len(self.val_dataset)):
            data = self.val_dataset[i]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            # rendering
            rgb, acc, depth, _ = render_image(
                self.nerf,
                self.occupancy_grid,
                rays,
                self.scene_aabb,
                # rendering options
                near_plane=self.meta_data['near_plane'],
                far_plane=self.meta_data['far_plane'],
                render_step_size=self.meta_data['render_step_size'],
                render_bkgd=render_bkgd,
                cone_angle=self.meta_data['cone_angle'],
                alpha_thre=self.meta_data['alpha_thre'],
                # test options
                test_chunk_size=self.config.test_chunk_size,
            )

            rgb = rgb.cpu().numpy()
            pixels = pixels.cpu().numpy()

            imageio.imwrite(
                os.path.join(val_dir, f"rgb_test_{i}.png"),
                (rgb * 255).astype(np.uint8),
            )
            imageio.imwrite(
                os.path.join(val_dir, f"rgb_gt_{i}.png"),
                (pixels * 255).astype(np.uint8),
            )
            inv_depth = 1. / depth.cpu()
            inv_depth = colorize(inv_depth.squeeze(-1), cmap_name='jet') #.permute(2, 0, 1)
            imageio.imwrite(
                os.path.join(val_dir, f"inv_depth_test_{i}.png"),
                (inv_depth.numpy() * 255).astype(np.uint8),
            )

            rgb = torch.from_numpy(rgb[None, ...]).cuda().permute(0, 3, 1, 2)
            pixels = torch.from_numpy(pixels[None, ...]).cuda().permute(0, 3, 1, 2)

            psnr = compute_psnr(rgb, pixels).item()
            PSNRs.append(psnr)

            m_ssim = compute_ssim(pixels, rgb)
            SSIMs.append(m_ssim)

            m_lpips = compute_lpips(lpips_loss, pixels, rgb)
            LPIPSs.append(m_lpips)

            results_dict[scene_name][i] = {'psnr': psnr,
                                            'ssim': m_ssim,
                                            'lpips': m_lpips,
                                          }
            pbar.update(1)
        
        psnr_avg = sum(PSNRs) / len(PSNRs)
        ssim_avg = sum(SSIMs) / len(SSIMs)
        lpips_avg = sum(LPIPSs) / len(LPIPSs)

        results_dict[scene_name]['psnr'] = psnr_avg
        results_dict[scene_name]['ssim'] = ssim_avg
        results_dict[scene_name]['lpips'] = lpips_avg

        json_file = os.path.join(self.output_dir, 'metrics.json')
        json_obj = json.dumps(results_dict, indent=4)
        print(f'Saving metrics to {json_file}')
        with open(json_file, 'w') as f:
            f.write(json_obj)

    @torch.no_grad()
    def generate_point_cloud(self):
        camera_poses = self.meta_data['camera_poses']
        num_camera_poses = camera_poses.shape[0]
        K = self.val_dataset.K
        width = self.val_dataset.width
        height = self.val_dataset.height

        pbar = tqdm.trange(num_camera_poses, desc=f"Generating Point Clouds", leave=False)
        point_cloud, rgbs = [], []
        min_depth = 2.0 # 1.0
        max_depth = 6.0

        for i in range(num_camera_poses):
            c2w = camera_poses[i].unsqueeze(0) # [1, 4, 4]
            x, y = torch.meshgrid(
                torch.arange(width, device=self.device),
                torch.arange(height, device=self.device),
                indexing="xy",
            )
            x = x.flatten()
            y = y.flatten()

            # Generate rays.
            camera_dirs = F.pad(
                torch.stack([
                        (x - K[0, 2] + 0.5) / K[0, 0],
                        (y - K[1, 2] + 0.5) / K[1, 1] * 
                        (-1.0 if self.val_dataset.OPENGL_CAMERA else 1.0)
                    ], dim=-1
                ), (0, 1), value=(-1.0 if self.val_dataset.OPENGL_CAMERA else 1.0),
            ) # [num_rays, 3]

            # [n_cams, height, width, 3]
            directions = (camera_dirs[:, None, :] * c2w[:, :3, :3]).sum(dim=-1)
            origins = torch.broadcast_to(c2w[:, :3, -1], directions.shape)
            viewdirs = directions / torch.linalg.norm(
                directions, dim=-1, keepdims=True
            )

            origins = torch.reshape(origins, (height, width, 3))
            viewdirs = torch.reshape(viewdirs, (height, width, 3))
            rays = Rays(origins=origins, viewdirs=viewdirs) # [h, w, 3]
            color_bkgd = torch.ones(3, device=self.device)

            # Volume rendering.
            rgb, _, depth, __ = render_image(
                self.nerf,
                self.occupancy_grid,
                rays,
                self.scene_aabb,
                # rendering options
                near_plane=self.meta_data['near_plane'],
                far_plane=self.meta_data['far_plane'],
                render_step_size=self.meta_data['render_step_size'],
                render_bkgd=color_bkgd,
                cone_angle=self.meta_data['cone_angle'],
                alpha_thre=self.meta_data['alpha_thre'],
                # test options
                test_chunk_size=self.config.test_chunk_size,
            )

            rgb = rgb.reshape(-1, 3)
            depth = depth.reshape(-1, 1)
            origins = origins.reshape(-1, 3)
            viewdirs = viewdirs.reshape(-1, 3)

            mask = torch.argwhere(
                (depth[:, 0] <= max_depth).cpu() & (depth[:, 0] >= min_depth).cpu()
            )
            points = origins[mask] + viewdirs[mask] * depth[mask]
            # points = origins + viewdirs * depth
            
            point_cloud.append(points.squeeze(dim=1))
            rgbs.append(rgb[mask].squeeze(dim=1))
            # rgbs.append(rgb.squeeze(dim=1))

            pbar.update(1)
        
        point_cloud = torch.cat(point_cloud, dim=0)
        rgbs = torch.cat(rgbs, dim=0)
        
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(point_cloud.float().cpu().numpy())
        pcd.colors = open3d.utility.Vector3dVector(rgbs.float().cpu().numpy())

        point_cloud_filename = os.path.join(self.model_dir, 'point_cloud.ply')
        open3d.io.write_point_cloud(point_cloud_filename, pcd)
        print(f'[INFO] Point Cloud Saved to {point_cloud_filename}.')

    @torch.no_grad()
    def sample_points(self):
        self.sample_grid.set_binary_fields(self.occupancy_grid.binary)

        points, rgb, alpha, indices, density_mask, surface_mask = \
            self.sample_grid.query_radiance_and_density_from_camera(
                radiance_field=self.nerf,
                occupancy_grid=self.occupancy_grid,
                meta_data=self.meta_data,
                device=self.device
            )
        
        x_res, y_res, z_res = self.sample_grid.binary.shape
        FEATURE_DIM = 7 # 3 for xyz, 3 for rgb, 1 for alpha

        ############################## Save points features by density field mask ##################
        df_points = points[density_mask]
        df_rgb = rgb[density_mask]
        df_alpha = alpha[density_mask]
        df_indices = indices[density_mask]

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(df_points.float().cpu().numpy())
        pcd.colors = open3d.utility.Vector3dVector(df_rgb.float().cpu().numpy())

        sampled_point_cloud_filename = os.path.join(self.model_dir, 'density_voxel_point_cloud.ply')
        open3d.io.write_point_cloud(sampled_point_cloud_filename, pcd)
        print(f'[INFO] {df_points.shape[0]} Sampled Density Point Cloud Saved to {sampled_point_cloud_filename}')

        grid_features = torch.zeros(
            (x_res, y_res, z_res, FEATURE_DIM),
            dtype=torch.float32,
            device=df_points.device
        ).reshape(-1, FEATURE_DIM)
        grid_features[df_indices, :3] = df_points
        grid_features[df_indices, 3:6] = df_rgb
        grid_features[df_indices, -1] = df_alpha.squeeze(dim=-1)
        grid_features = grid_features.reshape((x_res, y_res, z_res, FEATURE_DIM))

        grid_features_filename = os.path.join(self.model_dir, 'density_voxel_grid.pt')
        torch.save(grid_features, grid_features_filename)
        print(f'[INFO] Density Grid Features Saved to {grid_features_filename}')

        mask_filename = os.path.join(self.model_dir, 'density_voxel_mask.pt')
        torch.save(df_indices, mask_filename)

        ############################## Save points features by surface field mask ##################
        mask = surface_mask & density_mask
        points = points[mask]
        rgb = rgb[mask]
        alpha = alpha[mask]
        indices = indices[mask]

        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(points.float().cpu().numpy())
        pcd.colors = open3d.utility.Vector3dVector(rgb.float().cpu().numpy())

        sampled_point_cloud_filename = os.path.join(self.model_dir, 'voxel_point_cloud.ply')
        open3d.io.write_point_cloud(sampled_point_cloud_filename, pcd)
        print(f'[INFO] {points.shape[0]} Sampled Surface Point Cloud Saved to {sampled_point_cloud_filename}')

        grid_features = torch.zeros(
            (x_res, y_res, z_res, FEATURE_DIM),
            dtype=torch.float32,
            device=points.device
        ).reshape(-1, FEATURE_DIM)
        grid_features[indices, :3] = points
        grid_features[indices, 3:6] = rgb
        grid_features[indices, -1] = alpha.squeeze(dim=-1)
        grid_features = grid_features.reshape((x_res, y_res, z_res, FEATURE_DIM))

        grid_features_filename = os.path.join(self.model_dir, 'voxel_grid.pt')
        torch.save(grid_features, grid_features_filename)
        print(f'[INFO] Surface Grid Features Saved to {grid_features_filename}')

        mask_filename = os.path.join(self.model_dir, 'voxel_mask.pt')
        torch.save(indices, mask_filename)


if __name__ == '__main__':
    config = config_parser()

    assert config.data_split_json != "" or config.scene != ""

    if config.data_split_json != "" and config.scene == "":
        scenes = []
        with open(config.data_split_json, "r") as fp:
            obj_id_to_name = json.load(fp)
        
        for id, name in obj_id_to_name.items():
            scenes.append(name)

        for scene in scenes:
            data_dir = os.path.join(config.root_dir, scene)
            print(f'data dir: {data_dir}')
            if not os.path.exists(data_dir):
                continue

            local_config = copy.deepcopy(config)
            local_config.scene = scene
            local_config.expname = scene
            
            if local_config.multi_blocks:
                for k in range(0, 2):
                    local_config.ckpt_path = os.path.join(
                        local_config.root_dir, 'out', scene, f'block_{k}', 'model.pth'
                    )

                    evaluator = Evaluator(local_config)
                    evaluator.sample_points()
            else:
                local_config.ckpt_path = os.path.join(
                        local_config.root_dir, 'out', scene, 'model.pth'
                    )
                evaluator = Evaluator(local_config)
                evaluator.sample_points()
    else:
        # config.ckpt_path = os.path.join(
        #     config.root_dir, 'out', config.scene, 'model.pth'
        # )
        evaluator = Evaluator(config)
        evaluator.evaluate()
        # evaluator.generate_point_cloud()
        evaluator.sample_points()
