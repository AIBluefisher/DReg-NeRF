# import argparse
import math
import os
import time
import random
import sys
import json
import copy
from multiprocessing import Pool

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from nerfacc import ContractionType, OccupancyGrid

from conerf.radiance_fields.ngp import NGPradianceField
from conerf.utils.config import config_parser
from conerf.utils.utils import render_image, setup_seed, colorize
from conerf.base.trainer import BaseTrainer
from conerf.base.checkpoint_manager import CheckPointManager


class NGPTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        self.render_n_samples = 1024
        self.trainer_name = 'NGPTrainer'
        self.device = f"cuda:{config.local_rank}"
        self.config = config

        # super().__init__(config)
        self.output_path = os.path.join(config.root_dir, 'out', config.expname)
        if self.config.local_rank == 0:
            os.makedirs(self.output_path, exist_ok=True)
            print(f'[INFO] Outputs will be saved to {self.output_path}')

        self.log_file = open(os.path.join(self.output_path, 'log.txt'), 'w')
        self.scheduler = None
        self.model = None
        self.scalars_to_log = dict()
        self.ckpt_manager = CheckPointManager(
            save_path=self.output_path,
            max_to_keep=1000,
            keep_checkpoint_every_n_hours=0.5
        )
        
        self.train_done = False
        self._setup_visualizer()
        
        self.load_dataset()
        self.setup_bounding_box()

        # Functions need to be overwritten.
        self.build_networks()
        self.setup_optimizer()
        self.setup_loss_functions()
        self.compose_state_dicts()

    def setup_bounding_box(self):
        if self.config.auto_aabb and self.train_dataset.BBOX is None:
            camera_locs = torch.cat(
                [self.train_dataset.camtoworlds, self.val_dataset.camtoworlds]
            )[:, :3, -1]
            self.config.aabb = torch.cat(
                [camera_locs.min(dim=0).values, camera_locs.max(dim=0).values]
            ).tolist()
        elif self.train_dataset.BBOX is not None:
            self.config.aabb = self.train_dataset.BBOX
        print("Using aabb", self.config.aabb)

        # setup the scene bounding box.
        if self.config.unbounded:
            print("Using unbounded rendering")
            self.contraction_type = ContractionType.UN_BOUNDED_SPHERE # ContractionType.UN_BOUNDED_TANH
            self.scene_aabb = None
            self.near_plane = self.train_dataset.NEAR if self.train_dataset.NEAR > 0 else 0.2
            self.far_plane = self.train_dataset.NEAR if self.train_dataset.NEAR > 0 else 1e4
            self.render_step_size = 1e-2
            self.alpha_thre = 1e-2
        else:
            self.contraction_type = ContractionType.AABB
            self.scene_aabb = torch.tensor(self.config.aabb, dtype=torch.float32,
                                           device=self.device)
            self.near_plane = None
            self.far_plane = None
            self.render_step_size = (
                (self.scene_aabb[3:] - self.scene_aabb[:3]).max()
                * math.sqrt(3)
                / self.render_n_samples
            ).item()
            self.alpha_thre = 0.0

    def load_dataset(self):
        train_dataset_kwargs = {"factor": self.config.factor}
        test_dataset_kwargs = {"factor": self.config.factor}
        
        if self.config.multi_blocks:
            train_dataset_kwargs["multi_blocks"] = True
            train_dataset_kwargs["num_blocks"] = self.config.num_blocks
            test_dataset_kwargs["multi_blocks"] = True
            test_dataset_kwargs["num_blocks"] = self.config.num_blocks
        
        if self.config.dataset == 'nerf_synthetic':
            from conerf.datasets.nerf_synthetic import SubjectLoader
            self.target_sample_batch_size = 1 << 18
            self.grid_resolution = 128
        elif self.config.dataset == 'objaverse':
            from conerf.datasets.objaverse import SubjectLoader
            self.target_sample_batch_size = 1 << 18
            self.grid_resolution = 128
        elif self.config.dataset == 'Synthetic_NSVF':
            from conerf.datasets.nsvf import SubjectLoader
            self.target_sample_batch_size = 1 << 18
            self.grid_resolution = 128
        elif self.config.dataset == 'scannerf':
            from conerf.datasets.scan_nerf import SubjectLoader
            self.target_sample_batch_size = 1 << 18
            self.grid_resolution = 128
        elif self.config.dataset == 'BlendedMVS' or self.config.dataset == 'dtu':
            from conerf.datasets.mvs import SubjectLoader
            self.target_sample_batch_size = 1 << 18
            self.grid_resolution = 164
        elif self.config.dataset == 'Hypersim':
            from conerf.datasets.hypersim import SubjectLoader
            self.target_sample_batch_size = 1 << 20
            self.grid_resolution = 160
        else:
            # self.config.dataset == 'nerf_llff_data'/'mipnerf_360':
            from conerf.datasets.real_world import SubjectLoader
            self.target_sample_batch_size = 1 << 20
            train_dataset_kwargs["color_bkgd_aug"] = "random"
            self.grid_resolution = 128 # 256

        self.train_dataset = SubjectLoader(
            subject_id=self.config.scene,
            root_fp=self.config.root_dir,
            split=self.config.train_split,
            data_split_json=self.config.data_split_json,
            num_rays=self.target_sample_batch_size // self.render_n_samples,
            **train_dataset_kwargs,
        )
        self.train_dataset.to_device(self.device)
        self.train_dataset.K = self.train_dataset.K.to(self.device)

        self.val_dataset = SubjectLoader(
            subject_id=self.config.scene,
            root_fp=self.config.root_dir,
            split="test",
            data_split_json=self.config.data_split_json,
            num_rays=None,
            **test_dataset_kwargs,
        )
        self.val_dataset.to_device(self.device)
        self.val_dataset.K = self.val_dataset.K.to(self.device)

    def build_networks(self):
        self.model = NGPradianceField(
            aabb=self.config.aabb,
            unbounded=self.config.unbounded,
        ).to(self.device)

        self.occupancy_grid = OccupancyGrid(
            roi_aabb=self.config.aabb,
            resolution=self.grid_resolution,
            contraction_type=self.contraction_type,
        ).to(self.device)

        self.grad_scaler = torch.cuda.amp.GradScaler(2**10)

    def setup_optimizer(self):
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=1e-2, eps=1e-15
        )
        self.scheduler = torch.optim.lr_scheduler.MultiStepLR(
            self.optimizer,
            milestones=[self.config.max_iterations // 2,
                        self.config.max_iterations * 3 // 4,
                        self.config.max_iterations * 9 // 10],
            gamma=0.33,
        )

    def setup_loss_functions(self):
        pass

    def update_meta_data(self):
        if self.config.multi_blocks:
            self.state_dicts['meta_data']['block_id'] = self.train_dataset.current_block
        self.state_dicts['meta_data']['camera_poses'] = self.train_dataset.camtoworlds

    def compose_state_dicts(self) -> None:
        self.state_dicts = {'models': dict(), 'optimizers': dict(), 'schedulers': dict(), 'meta_data': dict()}
        
        self.state_dicts['models']['model'] = self.model
        self.state_dicts['models']['occupancy_grid'] = self.occupancy_grid
        self.state_dicts['optimizers']['optimizer'] = self.optimizer
        self.state_dicts['schedulers']['scheduler'] = self.scheduler
        
        # meta data for construction models.
        self.state_dicts['meta_data']['aabb'] = self.config.aabb
        self.state_dicts['meta_data']['unbounded'] = self.config.unbounded
        self.state_dicts['meta_data']['grid_resolution'] = self.grid_resolution
        self.state_dicts['meta_data']['contraction_type'] = self.contraction_type
        self.state_dicts['meta_data']['near_plane'] = self.near_plane
        self.state_dicts['meta_data']['far_plane'] = self.far_plane
        self.state_dicts['meta_data']['render_step_size'] = self.render_step_size
        self.state_dicts['meta_data']['alpha_thre'] = self.alpha_thre
        self.state_dicts['meta_data']['cone_angle'] = self.config.cone_angle

    def load_checkpoint(self, load_model=True, load_optimizer=True, load_scheduler=True, load_meta_data=False) -> int:
        return super().load_checkpoint(load_model, load_optimizer, load_scheduler, load_meta_data)

    def train(self):
        desc = f"Training {self.config.expname}" if not self.config.multi_blocks else \
               f"Training {self.config.expname} block_{self.train_dataset.current_block}"
        desc += f" ({len(self.train_dataset.images)} images)"
        pbar = tqdm.trange(self.config.max_iterations, desc=desc, leave=False)

        iter_start = self.load_checkpoint(load_optimizer=not self.config.no_load_opt,
                                          load_scheduler=not self.config.no_load_scheduler)
        if iter_start >= self.config.max_iterations:
            return

        self.epoch  = 0
        self.iteration = 0
        score = 0
        while self.iteration < iter_start:
            pbar.update(1)
            self.iteration += 1

        while self.iteration < self.config.max_iterations:
            for i in range(len(self.train_dataset)):
                self.model.train()
                data_batch = self.train_dataset[i]

                self.train_iteration(data_batch=data_batch)

                if self.iteration % self.config.n_validation == 0:
                    score = self.validate()

                # log to tensorboard.
                if self.iteration % self.config.n_tensorboard == 0:
                    self.log_info()

                if self.iteration % self.config.n_checkpoint == 0:
                    self.save_checkpoint(score=score)
                
                pbar.update(1)
                self.iteration += 1
                if self.iteration > self.config.max_iterations + 1:
                    break

            self.epoch += 1

        # if self.config.n_checkpoint % self.config.n_validation != 0:
        #     score = self.validate()
        #     self.save_checkpoint(score=score)

        self.train_done = True

    def train_iteration(self, data_batch) -> None:
        render_bkgd = data_batch["color_bkgd"]
        rays = data_batch["rays"]
        pixels = data_batch["pixels"]
        
        def occ_eval_fn(x):
            if self.config.cone_angle > 0.0:
                # randomly sample a camera for computing step size.
                camera_ids = torch.randint(
                    0, len(self.train_dataset), (x.shape[0],), device=self.device
                )
                origins = self.train_dataset.camtoworlds[camera_ids, :3, -1]
                t = (origins - x).norm(dim=-1, keepdim=True)
                # compute actual step size used in marching, based on the distance to the camera.
                step_size = torch.clamp(
                    t * self.config.cone_angle, min=self.render_step_size
                )
                # filter out the points that are not in the near far plane.
                if (self.near_plane is not None) and (self.far_plane is not None):
                    step_size = torch.where(
                        (t > self.near_plane) & (t < self.far_plane),
                        step_size,
                        torch.zeros_like(step_size),
                    )
            else:
                step_size = self.render_step_size
            # compute occupancy
            density = self.model.query_density(x)
            return density * step_size

        # update occupancy grid
        self.occupancy_grid.every_n_step(step=self.iteration, occ_eval_fn=occ_eval_fn)

        # render
        rgb, acc, depth, n_rendering_samples = render_image(
            self.model,
            self.occupancy_grid,
            rays,
            self.scene_aabb,
            # rendering options
            near_plane=self.near_plane,
            far_plane=self.far_plane,
            render_step_size=self.render_step_size,
            render_bkgd=render_bkgd,
            cone_angle=self.config.cone_angle,
            alpha_thre=self.alpha_thre,
        )
        if n_rendering_samples == 0:
            # continue
            return

        # dynamic batch size for rays to keep sample batch size constant.
        num_rays = len(pixels)
        num_rays = int(
            num_rays
            * (self.target_sample_batch_size / float(n_rendering_samples))
        )
        self.train_dataset.update_num_rays(num_rays)
        alive_ray_mask = acc.squeeze(-1) > 0

        # compute loss
        loss = F.smooth_l1_loss(rgb[alive_ray_mask], pixels[alive_ray_mask])

        self.optimizer.zero_grad()
        # do not unscale it because we are using Adam.
        self.grad_scaler.scale(loss).backward()
        self.optimizer.step()
        self.scheduler.step()

        mse = F.mse_loss(rgb, pixels)
        psnr = -10.0 * torch.log(mse) / np.log(10.0)
        prefix = f'block_{self.val_dataset.current_block}_' if self.config.multi_blocks else ''
        self.scalars_to_log[f'train/{prefix}psnr'] = psnr
        self.scalars_to_log[f'train/{prefix}loss'] = loss.detach().item()
        self.scalars_to_log[f'train/{prefix}alive_ray_mask'] = alive_ray_mask.long().sum()
        self.scalars_to_log[f'train/{prefix}n_rendering_samples'] = n_rendering_samples

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()
        
        val_dir = os.path.join(self.output_path, 'val')
        os.makedirs(val_dir, exist_ok=True)

        pbar = tqdm.trange(len(self.val_dataset), desc=f"Validating {self.config.expname}", leave=False)
        psnrs = []

        for i in range(len(self.val_dataset)):
            data = self.val_dataset[i]
            render_bkgd = data["color_bkgd"]
            rays = data["rays"]
            pixels = data["pixels"]

            # rendering
            rgb, acc, depth, _ = render_image(
                self.model,
                self.occupancy_grid,
                rays,
                self.scene_aabb,
                # rendering options
                near_plane=self.near_plane,
                far_plane=self.far_plane,
                render_step_size=self.render_step_size,
                render_bkgd=render_bkgd,
                cone_angle=self.config.cone_angle,
                alpha_thre=self.alpha_thre,
                # test options
                test_chunk_size=self.config.test_chunk_size,
            )
            mse = F.mse_loss(rgb, pixels)
            psnr = -10.0 * torch.log(mse) / np.log(10.0)
            psnrs.append(psnr.item())
            imageio.imwrite(
                os.path.join(val_dir, f"rgb_test_{i}.png"),
                (rgb.cpu().numpy() * 255).astype(np.uint8),
            )
            imageio.imwrite(
                os.path.join(val_dir, f"rgb_gt_{i}.png"),
                (pixels.cpu().numpy() * 255).astype(np.uint8),
            )
            inv_depth = 1. / depth.cpu()
            inv_depth = colorize(inv_depth.squeeze(-1), cmap_name='jet') #.permute(2, 0, 1)
            imageio.imwrite(
                os.path.join(val_dir, f"inv_depth_test_{i}.png"),
                (inv_depth.numpy() * 255).astype(np.uint8),
            )
            depth = colorize(depth.squeeze(-1), cmap_name='jet')
            imageio.imwrite(
                os.path.join(val_dir, f"depth_test_{i}.png"),
                (depth.cpu().numpy() * 255).astype(np.uint8),
            )

            pbar.update(1)
        
        psnr_avg = sum(psnrs) / len(psnrs)
        prefix = f'block_{self.val_dataset.current_block}_' if self.config.multi_blocks else ''
        self.writer.add_scalar(f'val/{prefix}psnr', psnr_avg, global_step=self.iteration)

        self.train_dataset.training = True
        torch.cuda.empty_cache()
        
        return psnr_avg


def train(config):
    if config.multi_blocks:
        assert config.min_num_blocks > 0
        assert config.max_num_blocks > 0
        assert config.min_num_blocks <= config.max_num_blocks
        
        num_blocks = random.randint(config.min_num_blocks, config.max_num_blocks)
        config.num_blocks = num_blocks
        print(f'Training {num_blocks} NeRFs with Multiple Blocks...')
        
        for k in range(num_blocks):
            trainer = NGPTrainer(config)

            trainer.train_dataset.move_to_block(k)
            trainer.train_dataset.to_device(trainer.device)

            trainer.val_dataset.move_to_block(k)
            trainer.val_dataset.to_device(trainer.device)

            # Modify trainer configurations.
            trainer.update_meta_data()
            trainer.output_path = os.path.join(config.root_dir, 'out', config.expname, f'block_{k}')

            os.makedirs(trainer.output_path, exist_ok=True)
            trainer.ckpt_manager = CheckPointManager(
                save_path=trainer.output_path,
                max_to_keep=100,
                keep_checkpoint_every_n_hours=0.5
            )

            trainer.train()

            del trainer
    else:
        trainer = NGPTrainer(config)
        trainer.update_meta_data()
        trainer.train()
        print(f'total iteration: {trainer.iteration}')


if __name__ == '__main__':
    config = config_parser()
    
    assert config.data_split_json != "" or config.scene != ""

    # setup_seed(config.seed)

    if config.data_split_json != "" and config.scene == "":
        scenes = []
        with open(config.data_split_json, "r") as fp:
            obj_id_to_name = json.load(fp)
        
        for id, name in obj_id_to_name.items():
            scenes.append(name)
        
        for scene in scenes:
            data_dir = os.path.join(config.root_dir, scene)
            print(data_dir)
            if not os.path.exists(data_dir):
                continue

            local_config = copy.deepcopy(config)
            local_config.scene = scene
            local_config.expname = scene
            train(local_config)
    else:
        train(config)
