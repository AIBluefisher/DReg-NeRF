import os
import random
import json
import time

import numpy as np
import torch
import tqdm
import open3d
import imageio

from conerf.base.checkpoint_manager import CheckPointManager
from conerf.datasets.register.dataset import NeRFRegDataset
from conerf.datasets.register.nerf_pose_only_dataset import NeRFPoseOnlyDataset
from conerf.geometry.global_registration import run_registration
from conerf.loss.feature_loss import InfoNCELoss
from conerf.loss.confidence_loss import load_radiance_fields
from conerf.register.nerf_regtr import NeRFRegTr
from conerf.register.se3 import se3_transform_list
from conerf.utils.config import config_parser
from conerf.utils.utils import all_to_device, setup_seed, render_image, colorize_np


def rotation_distance(R1, R2, eps=1e-7):
    """
    Args:
        R1: rotation matrix from camera 1 to world
        R2: rotation matrix from camera 2 to world
    Return:
        angle: the angular distance between camera 1 and camera 2.
    """
    # http://www.boris-belousov.net/2016/12/01/quat-dist/
    # R_diff = R1 @ R2.transpose(-2, -1)
    R_diff = R1.transpose(-2, -1) @ R2

    trace = R_diff[..., 0, 0] + R_diff[..., 1, 1] + R_diff[..., 2, 2]

    # numerical stability near -1/+1
    angle = ((trace - 1) / 2).clamp(-1 + eps, 1 - eps).acos_()
    angle = torch.rad2deg(angle)

    return angle


@torch.no_grad()
def evaluate_camera_alignment(pred_poses, poses_gt):
    """
    Args:
        pred_poses: [B, 3/4, 4]
        poses_gt: [B, 3/4, 4]
    """
    # measure errors in rotation and translation
    R_pred, t_pred = pred_poses.split([3, 1], dim=-1)
    R_gt, t_gt = poses_gt.split([3, 1], dim=-1)

    R_error = rotation_distance(R_pred[..., :3, :3], R_gt[..., :3, :3])
    t_error = (t_pred[..., :3, -1] - t_gt[..., :3, -1])[..., 0].norm(dim=-1)
    
    mean_rotation_error = R_error.mean().cpu()
    mean_position_error = t_error.mean()
    med_rotation_error = R_error.median().cpu()
    med_position_error = t_error.median()
    
    return {'R_error_mean': mean_rotation_error, "t_error_mean": mean_position_error,
            'R_error_med': med_rotation_error, 't_error_med': med_position_error}


@torch.no_grad()
def synthesize_novel_views(
    dataset,
    nerf,
    occupancy_grid,
    meta_data,
    device,
    test_chunk_size
):
    if meta_data['unbounded']:
        scene_aabb = None
    else:
        scene_aabb = torch.tensor(
            meta_data['aabb'], dtype=torch.float32, device=device)

    images, depths = [], []
    for i in range(len(dataset)):
        data = dataset[i]
        render_bkgd = data["color_bkgd"]
        rays = data["rays"]

        # rendering
        rgb, _, depth, _ = render_image(
            nerf,
            occupancy_grid,
            rays,
            scene_aabb,
            # rendering options
            near_plane=meta_data['near_plane'],
            far_plane=meta_data['far_plane'],
            render_step_size=meta_data['render_step_size'],
            render_bkgd=render_bkgd,
            cone_angle=meta_data['cone_angle'],
            alpha_thre=meta_data['alpha_thre'],
            # test options
            test_chunk_size=test_chunk_size,
        )

        images.append(rgb.cpu().numpy())
        depths.append(depth.cpu().numpy())
    
    return images, depths


@torch.no_grad()
def render_videos(
    nerf_dataset,
    src_nerf,
    src_occupancy_grid,
    src_meta,
    src_render_poses,
    tgt_nerf,
    tgt_occupancy_grid,
    tgt_meta,
    tgt_render_poses,
    device,
    test_chunk_size,
    output_dir,
    prefix="aligned"
):
    images_dir = os.path.join(output_dir, prefix + "_images")
    src_images_dir = os.path.join(output_dir, prefix + "_src_images")
    tgt_images_dir = os.path.join(output_dir, prefix + "_tgt_images")
    os.makedirs(images_dir, exist_ok=True)
    os.makedirs(src_images_dir, exist_ok=True)
    os.makedirs(tgt_images_dir, exist_ok=True)

    nerf_dataset.camtoworlds = src_render_poses
    src_rgbs, src_depths = synthesize_novel_views( # In source nerf's coordinate frame.
        nerf_dataset, src_nerf, src_occupancy_grid, src_meta, device,
        test_chunk_size
    )
    nerf_dataset.camtoworlds = tgt_render_poses
    tgt_rgbs, tgt_depths = synthesize_novel_views( # In target nerf's coordinate frame.
        nerf_dataset, tgt_nerf, tgt_occupancy_grid, tgt_meta, device,
        test_chunk_size
    )
    for i in range(len(src_rgbs)): # pylint: disable=C0200
        src_rgb, src_depth = src_rgbs[i], src_depths[i]
        src_color_depth = colorize_np(src_depth.squeeze(-1))
        imageio.imwrite(os.path.join(src_images_dir, f'rgb_{i}.png'),
                        (src_rgb * 255).astype(np.uint8)
        )
        # imageio.imwrite(os.path.join(src_images_dir, f'depth_{i}.png'),
        #                 (src_depth * 255).astype(np.uint8)
        # )
        tgt_rgb, tgt_depth = tgt_rgbs[i], tgt_depths[i]
        tgt_color_depth = colorize_np(tgt_depth.squeeze(-1))

        imageio.imwrite(os.path.join(tgt_images_dir, f'rgb_{i}.png'),
                        (tgt_rgb * 255).astype(np.uint8)
        )
        # imageio.imwrite(os.path.join(tgt_images_dir, f'depth_{i}.png'),
        #                 (tgt_depth * 255).astype(np.uint8)
        # )

        src_tgt_rgb_depth = np.concatenate([src_rgb, src_color_depth, tgt_rgb, tgt_color_depth], axis=1)
        imageio.imwrite(os.path.join(images_dir, f'src_tgt_rgb_depth_{i}.png'),
                        (src_tgt_rgb_depth * 255).astype(np.uint8)
        )
    # Compose images (half from source nerf, half from target nerf)
    print("generating videos...")
    src_tgt_rgb_depth_video = os.path.join(output_dir, prefix + "_src_tgt_rgb_depth.mp4")
    os.system(f"ffmpeg -framerate 2 -i '{images_dir}/src_tgt_rgb_depth_%d.png' " + \
              f"-vcodec libx264 -crf 25 -pix_fmt yuv420p {src_tgt_rgb_depth_video}")


class RegEvaluator():
    def __init__(self, config) -> None:
        self.config = config
        self.device = f"cuda:{config.local_rank}"

        self.output_dir = os.path.join(self.config.root_dir, 'eval', self.config.expname)
        os.makedirs(self.output_dir, exist_ok=True)

        self.model_dir = os.path.join(self.config.root_dir, 'out', self.config.expname)
        assert os.path.exists(self.model_dir)
        print(f'[INFO] Outputs will be saved to {self.output_dir}')

        self.model = None
        self.ckpt_manager = CheckPointManager(
            save_path=self.model_dir,
            max_to_keep=1000,
            keep_checkpoint_every_n_hours=0.5
        )
        
        self.load_dataset()
        self.build_networks()
        self.setup_loss_functions()
        self.compose_state_dicts()

    def load_dataset(self):
        self.val_dataset = NeRFRegDataset(
            root_fp=self.config.root_dir,
            json_dir=self.config.json_dir,
            dataset=self.config.dataset if self.config.dataset != "" else None,
            subject_id=self.config.scene if self.config.scene != "" else None,
            split=self.config.train_split, # 'train',
            model_dir='nerf_models'
        )
        self.val_dataset.mode = 'test'

        self.nerf_dataset = NeRFPoseOnlyDataset(self.config.dataset)

    def build_networks(self):
        self.model = NeRFRegTr(
            pos_emb_type=self.config.position_embedding_type,
            pos_emb_dim=self.config.position_embedding_dim,
            pos_emb_scaling=self.config.position_embedding_scaling,
            num_downsample=self.config.num_downsample
        ).to(self.device)

    def setup_loss_functions(self):
        # Feature Loss.
        self.feature_loss = InfoNCELoss(d_embed=256, r_p=0.2, r_n=0.4).to(self.device)

    def evaluate(self):
        desc = f"Evaluating {self.config.expname} NeRFRegTR"

        self.load_checkpoint(load_optimizer=not self.config.no_load_opt,
                             load_scheduler=not self.config.no_load_scheduler)

        train_ids = [i for i in range(len(self.val_dataset))]
        pbar = tqdm.trange(len(train_ids), desc=desc, leave=False)
        self.results_dict = dict()
        self.fgr_results_dict = dict()

        for i in train_ids:
            data_batch = self.val_dataset[i]
            self.eval_iteration(data_batch=data_batch)
            pbar.update(1)

        # Record metrics for our method.
        R_mean, t_mean = 0, 0
        for scene in self.results_dict.keys():
            R_mean += self.results_dict[scene]['R_mean']
            t_mean += self.results_dict[scene]['t_mean']
        R_mean /= len(self.results_dict)
        t_mean /= len(self.results_dict)
        self.results_dict['R_mean'] = R_mean
        self.results_dict['t_mean'] = t_mean

        metrics_dir = os.path.join(self.output_dir, self.config.dataset)
        os.makedirs(metrics_dir, exist_ok=True)

        json_file = os.path.join(self.output_dir, self.config.dataset, f'metrics_{self.config.train_split}.json')
        json_obj = json.dumps(self.results_dict, indent=4)
        print(f'Saving metrics to {json_file}')
        with open(json_file, 'w') as f:
            f.write(json_obj)

        # Record metrics for fast global registration.
        R_mean, t_mean = 0, 0
        for scene in self.fgr_results_dict.keys():
            R_mean += self.fgr_results_dict[scene]['R_mean']
            t_mean += self.fgr_results_dict[scene]['t_mean']
        R_mean /= len(self.fgr_results_dict)
        t_mean /= len(self.fgr_results_dict)
        self.fgr_results_dict['R_mean'] = R_mean
        self.fgr_results_dict['t_mean'] = t_mean

        fgr_json_file = os.path.join(self.output_dir, self.config.dataset, f'fgr_metrics_{self.config.train_split}.json')
        fgr_json_obj = json.dumps(self.fgr_results_dict, indent=4)
        print(f'Saving metrics to {fgr_json_file}')
        with open(fgr_json_file, 'w') as f:
            f.write(fgr_json_obj)

    @torch.no_grad()
    def eval_iteration(self, data_batch) -> None:
        data_batch = all_to_device(data=data_batch, device=self.device)
        start_time = time.time()
        pred = self.model(data_batch)
        end_time = time.time()
        time_took = end_time - start_time

        scene_name = data_batch['scene']
        self.results_dict[scene_name] = {}

        pose_gt = data_batch['pose']  # [B, 4, 4]
        self.pose_gt = pose_gt
        pred_poses = pred['pose'][-1] # [B, 3, 4]
        pred_pose_4x4 = torch.cat([
            pred_poses[0],
            torch.tensor([[0, 0, 0, 1]], dtype=torch.float32, device=pred_poses.device)],
        )

        pose_error = evaluate_camera_alignment(pred_poses, pose_gt)
        print(f'pose_error: {pose_error}')
        print(f'data batch: {data_batch["scene"]}')
        self.results_dict[scene_name]['R_mean'] = float(pose_error['R_error_mean'].cpu())
        self.results_dict[scene_name]['t_mean'] = float(pose_error['t_error_mean'].cpu())
        self.results_dict[scene_name]['R_med'] = float(pose_error['R_error_med'].cpu())
        self.results_dict[scene_name]['t_med'] = float(pose_error['t_error_med'].cpu())
        self.results_dict[scene_name]['time'] = time_took

        self.fgr_results_dict[scene_name] = {}
        fgr_pose, fgr_time = run_registration(data_batch['src_ply_path'], data_batch['tgt_ply_path'])
        fgr_pose = torch.from_numpy(fgr_pose).to(self.device).unsqueeze(0).float()
        fgr_pose_error = evaluate_camera_alignment(fgr_pose, pose_gt)
        self.fgr_results_dict[scene_name]['R_mean'] = float(fgr_pose_error['R_error_mean'].cpu())
        self.fgr_results_dict[scene_name]['t_mean'] = float(fgr_pose_error['t_error_mean'].cpu())
        self.fgr_results_dict[scene_name]['R_med'] = float(fgr_pose_error['R_error_med'].cpu())
        self.fgr_results_dict[scene_name]['t_med'] = float(fgr_pose_error['t_error_med'].cpu())
        self.fgr_results_dict[scene_name]['time'] = fgr_time
        
        ################################## Output Visualization result ##########################
        output_dir = os.path.join(self.output_dir, data_batch['dataset'], scene_name)
        os.makedirs(output_dir, exist_ok=True)
        pred_transform_json = os.path.join(output_dir, 'transformation_est.json')
        pred_transform_json_obj = json.dumps({'transformation': pred_pose_4x4.cpu().numpy().tolist()}, indent=4)
        print(f'Saving metrics to {pred_transform_json}')
        with open(pred_transform_json, 'w') as f:
            f.write(pred_transform_json_obj)

        # (1) For aligned camera poses.
        src_nerf_path = data_batch['src_nerf_path']
        tgt_nerf_path = data_batch['tgt_nerf_path']

        src_nerf, src_occupancy_grid, src_meta = load_radiance_fields(src_nerf_path, self.device)
        tgt_nerf, tgt_occupancy_grid, tgt_meta = load_radiance_fields(tgt_nerf_path, self.device)
        src_nerf.eval(), src_occupancy_grid.eval()
        tgt_nerf.eval(), tgt_occupancy_grid.eval()
            
        trans_src_camera_poses_gt = pose_gt[0] @ src_meta['camera_poses'] # [N, 4, 4]
        trans_src_camera_poses_pred = pred_pose_4x4 @ src_meta['camera_poses']
        
        unaligned_poses = torch.cat([src_meta['camera_poses'], tgt_meta['camera_poses']], dim=0)
        aligned_poses_gt = torch.cat([trans_src_camera_poses_gt, tgt_meta['camera_poses']], dim=0)
        aligned_poses_pred = torch.cat([trans_src_camera_poses_pred, tgt_meta['camera_poses']], dim=0)

        torch.save(unaligned_poses.detach().cpu(),
                   os.path.join(output_dir, f'unaligned_poses.pt'))
        torch.save(aligned_poses_gt.detach().cpu(),
                   os.path.join(output_dir, f'aligned_poses_gt.pt'))
        torch.save(aligned_poses_pred.detach().cpu(),
                   os.path.join(output_dir, f'aligned_poses_pred.pt'))
        
        # (2) For novel view synthesis.
        trans_tgt_camera_poses_gt = torch.inverse(pose_gt[0]) @ tgt_meta['camera_poses']
        src_render_poses_gt = torch.cat([src_meta['camera_poses'], trans_tgt_camera_poses_gt], dim=0)
        render_videos(
            self.nerf_dataset,
            src_nerf, src_occupancy_grid, src_meta, src_render_poses_gt,
            tgt_nerf, tgt_occupancy_grid, tgt_meta, aligned_poses_gt,
            self.device, self.config.test_chunk_size, output_dir, "gt"
        )

        trans_tgt_camera_poses_pred = torch.inverse(pred_pose_4x4) @ tgt_meta['camera_poses']
        src_render_poses_pred = torch.cat([src_meta['camera_poses'], trans_tgt_camera_poses_pred], dim=0)
        render_videos(
            self.nerf_dataset,
            src_nerf, src_occupancy_grid, src_meta, src_render_poses_pred,
            tgt_nerf, tgt_occupancy_grid, tgt_meta, aligned_poses_pred,
            self.device, self.config.test_chunk_size, output_dir, "aligned"
        )

        render_videos(
            self.nerf_dataset,
            src_nerf, src_occupancy_grid, src_meta, unaligned_poses,
            tgt_nerf, tgt_occupancy_grid, tgt_meta, unaligned_poses,
            self.device, self.config.test_chunk_size, output_dir, "unaligned"
        )

        # (3) For aligned point clouds.
        red = np.array([[1.0, 0, 0]])
        green = np.array([[0, 1.0, 0]])
        pcd = open3d.geometry.PointCloud()
        pcd.points = open3d.utility.Vector3dVector(pred['src_kp'][0].detach().cpu().numpy())
        open3d.io.write_point_cloud(os.path.join(output_dir, f'src_xyz.ply'), pcd)

        pcd.points = open3d.utility.Vector3dVector(pred['tgt_kp'][0].detach().cpu().numpy())
        open3d.io.write_point_cloud(os.path.join(output_dir, f'tgt_xyz.ply'), pcd)

        pcd.points = open3d.utility.Vector3dVector(pred['src_kp_warped'][0][-1].detach().cpu().numpy())
        open3d.io.write_point_cloud(os.path.join(output_dir, f'src_kp_warped.ply'), pcd)

        pcd.points = open3d.utility.Vector3dVector(pred['tgt_kp_warped'][0][-1].detach().cpu().numpy())
        open3d.io.write_point_cloud(os.path.join(output_dir, f'tgt_kp_warped.ply'), pcd)

        all_src_xyz = torch.cat([pred['src_kp'][0], pred['tgt_kp_warped'][0][-1]], dim=0)
        all_src_xyz_numpy = all_src_xyz.detach().cpu().numpy()
        pcd.points = open3d.utility.Vector3dVector(all_src_xyz_numpy)
        color = np.concatenate([
            np.repeat(red, pred['src_kp'][0].shape[0], axis=0),
            np.repeat(green, pred['tgt_kp_warped'][0][-1].shape[0], axis=0)
        ], axis=0)
        pcd.colors = open3d.utility.Vector3dVector(color)
        open3d.io.write_point_cloud(os.path.join(output_dir, f'all_src_xyz.ply'), pcd)

        all_tgt_xyz = torch.cat([pred['src_kp_warped'][0][-1], pred['tgt_kp'][0]], dim=0)
        all_tgt_xyz_numpy = all_tgt_xyz.detach().cpu().numpy()
        pcd.points = open3d.utility.Vector3dVector(all_tgt_xyz_numpy)
        color = np.concatenate([
            np.repeat(red, pred['src_kp_warped'][0][-1].shape[0], axis=0),
            np.repeat(green, pred['tgt_kp'][0].shape[0], axis=0)
        ], axis=0)
        pcd.colors = open3d.utility.Vector3dVector(color)
        open3d.io.write_point_cloud(os.path.join(output_dir, f'all_tgt_xyz.ply'), pcd)

        all_overlap_pred = torch.cat(pred['src_overlap'] + pred['tgt_overlap'], dim=-2)[-1] # [N_src+N_tgt, 1]
        all_overlap_pred = (all_overlap_pred >= 0.5).squeeze(-1)

        src_xyz_trans_pred = se3_transform_list(pred_poses, pred['src_kp'])
        tgt_xyz = pred['tgt_kp']

        xyz_pred = torch.cat(src_xyz_trans_pred + tgt_xyz, dim=0)
        xyz_pred_numpy = xyz_pred.detach().cpu().numpy()
        pcd.points = open3d.utility.Vector3dVector(xyz_pred_numpy)
        color = np.concatenate(
            [np.repeat(red, src_xyz_trans_pred[0].shape[0], axis=0),
            np.repeat(green, tgt_xyz[0].shape[0], axis=0)], axis=0
        )
        pcd.colors = open3d.utility.Vector3dVector(color)
        open3d.io.write_point_cloud(os.path.join(output_dir, f'noisy_point_cloud_pred.ply'), pcd)

        xyz_pred = xyz_pred[all_overlap_pred].detach().cpu().numpy()
        pcd.points = open3d.utility.Vector3dVector(xyz_pred)
        pcd.colors = open3d.utility.Vector3dVector(np.repeat(green, xyz_pred.shape[0], axis=0))
        open3d.io.write_point_cloud(os.path.join(output_dir, f'point_cloud_pred.ply'), pcd)

        src_xyz_trans_gt = se3_transform_list(pose_gt, pred['src_kp'])
        xyz_gt = torch.cat(src_xyz_trans_gt + tgt_xyz, dim=0)
        xyz_gt_numpy = xyz_gt.detach().cpu().numpy()
        pcd.points = open3d.utility.Vector3dVector(xyz_gt_numpy)
        pcd.colors = open3d.utility.Vector3dVector(np.repeat(red, xyz_gt_numpy.shape[0], axis=0))
        open3d.io.write_point_cloud(os.path.join(output_dir, f'noisy_point_cloud_gt.ply'), pcd)

        xyz_gt = xyz_gt[all_overlap_pred].detach().cpu().numpy()
        pcd.points = open3d.utility.Vector3dVector(xyz_gt)
        pcd.colors = open3d.utility.Vector3dVector(np.repeat(red, xyz_gt.shape[0], axis=0))
        open3d.io.write_point_cloud(os.path.join(output_dir, f'point_cloud_gt.ply'), pcd)

    def compose_state_dicts(self) -> None:
        self.state_dicts = {'models': dict(), 'optimizers': dict(), 'schedulers': dict(), 'meta_data': None}

        self.state_dicts['models']['model'] = self.model
        self.state_dicts['models']['feature_loss'] = self.feature_loss
    
    def load_checkpoint(self, load_model=True, load_optimizer=True, load_scheduler=True, load_meta_data=False) -> int:
        iter_start = self.ckpt_manager.load(
            config=self.config,
            models=self.state_dicts['models'] if load_model else None,
            optimizers=self.state_dicts['optimizers'] if load_optimizer else None,
            schedulers=self.state_dicts['schedulers'] if load_scheduler else None,
            meta_data=self.state_dicts['meta_data'] if load_meta_data else None
        )

        return iter_start


if __name__ == '__main__':
    config = config_parser()
    torch.multiprocessing.set_start_method('spawn')

    setup_seed(config.seed)

    evaluator = RegEvaluator(config)
    evaluator.evaluate()
