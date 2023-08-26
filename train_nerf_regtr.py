import os
import random

import numpy as np
import torch
import tqdm

from conerf.base.checkpoint_manager import CheckPointManager
from conerf.base.trainer import BaseTrainer
from conerf.datasets.register.dataset import NeRFRegDataset
from conerf.loss.feature_loss import InfoNCELoss
from conerf.loss.correspondence_loss import CorrespondenceLoss
from conerf.register.nerf_regtr import NeRFRegTr
from conerf.register.se3 import se3_transform_list, se3_inv
from conerf.utils.config import config_parser
from conerf.utils.utils import all_to_device, setup_seed
from conerf.loss.confidence_loss import compute_visibility_score


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


class RegTrainer(BaseTrainer):
    def __init__(self, config) -> None:
        super().__init__(config)

        self.trainer_name = 'RegTrainer'
        self.grad_clip = 0.1

    def load_dataset(self):
        self.train_dataset = NeRFRegDataset(
            root_fp=self.config.root_dir,
            json_dir=self.config.json_dir,
            subject_id=self.config.scene if self.config.scene != "" else None,
            split='train',
            model_dir='nerf_models'
        )

        self.val_dataset = NeRFRegDataset(
            root_fp=self.config.root_dir,
            json_dir=self.config.json_dir,
            subject_id=self.config.scene if self.config.scene != "" else None,
            split='test',
            model_dir='nerf_models'
        )

    def build_networks(self):
        self.model = NeRFRegTr(
            pos_emb_type=self.config.position_embedding_type,
            pos_emb_dim=self.config.position_embedding_dim,
            pos_emb_scaling=self.config.position_embedding_scaling,
            num_downsample=self.config.num_downsample
        ).to(self.device)

    def setup_optimizer(self):
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(), lr=self.config.lr, weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.StepLR(
            self.optimizer, step_size=34000, gamma=0.5
        )

    def setup_loss_functions(self):
        # loss weights.
        self.weight_dict = {}
        self.weight_dict['overlap'] = 1.0
        self.weight_dict['nerf_cont'] = 1.0
        self.weight_dict['feature'] = 0.1
        self.weight_dict['corr'] = 1.0

        # overlapping loss.
        self.overlap_loss = torch.nn.BCEWithLogitsLoss()

        # Feature Loss.
        self.feature_loss = InfoNCELoss(d_embed=256, r_p=0.2, r_n=0.4).to(self.device)

        # Correspondence loss.
        self.corr_loss = CorrespondenceLoss(
            metric='mae',
            robust_loss=self.config.robust_loss
        )

    def train(self):
        desc = f"Training {self.config.expname} NeRFRegTR"
        max_iterations = self.config.epochs * len(self.train_dataset)
        pbar = tqdm.trange(max_iterations, desc=desc, leave=False)

        iter_start = self.load_checkpoint(load_optimizer=not self.config.no_load_opt,
                                          load_scheduler=not self.config.no_load_scheduler)
        self.epoch  = 0
        self.iteration = 0
        if not self.config.finetune:
            while self.iteration < iter_start:
                pbar.update(1)
                self.iteration += 1
        score = 0
        train_ids = [i for i in range(len(self.train_dataset))]

        while self.epoch < self.config.epochs:
            random.shuffle(train_ids)
            for i in train_ids:
                data_batch = self.train_dataset[i]

                self.train_iteration(data_batch=data_batch)

                if self.iteration % self.config.n_validation == 0:
                    score = self.validate()
                    self.model.train()

                # log to tensorboard.
                if self.iteration % self.config.n_tensorboard == 0:
                    self.log_info()

                if self.iteration % self.config.n_checkpoint == 0:
                    self.save_checkpoint(score=score)
                
                pbar.update(1)
                self.iteration += 1
                if self.iteration > max_iterations + 1:
                    break

            self.epoch += 1

        if self.config.n_checkpoint % self.config.n_validation != 0:
            score = self.validate()
            self.save_checkpoint(score=score)

        self.train_done = True

    def train_iteration(self, data_batch) -> None:
        data_batch = all_to_device(data=data_batch, device=self.device)

        pred = self.model(data_batch)

        pose_gt = data_batch['pose']  # [B, 4, 4]
        self.pose_gt = pose_gt
        pred_poses = pred['pose'][-1] # [B, 3, 4]

        losses = dict()

        self.optimizer.zero_grad()

        # compute overlap loss.
        batch_size = len(pred['src_kp'])
        num_layers = pred['src_kp_warped'][0].shape[0]
        src_kp_list = [pred['src_kp'][b].expand(num_layers, -1, -1) for b in range(batch_size)]
        tgt_kp_list = [pred['tgt_kp'][b].expand(num_layers, -1, -1) for b in range(batch_size)]

        # the overlap score is either the density field or the surface field.
        self.src_overlap_gt = compute_visibility_score(src_kp_list, data_batch['src_nerf_path']) # list of [nl, N, 1]
        self.tgt_overlap_gt = compute_visibility_score(tgt_kp_list, data_batch['tgt_nerf_path']) # list of [nl, N, 1]
        all_overlap_gt = torch.cat(self.src_overlap_gt + self.tgt_overlap_gt, dim=-2)
        all_overlap_pred = torch.cat(pred['src_overlap'] + pred['tgt_overlap'], dim=-2)
        losses['overlap'] = self.overlap_loss(all_overlap_gt[-1], all_overlap_pred[-1])

        # nerf consistency loss.
        src_overlap_tilde = compute_visibility_score(pred['src_kp_warped'], data_batch['src_nerf_path'])
        tgt_overlap_tilde = compute_visibility_score(pred['tgt_kp_warped'], data_batch['tgt_nerf_path'])
        all_overlap_tilde = torch.cat(src_overlap_tilde + tgt_overlap_tilde, dim=-2)
        losses['nerf_cont'] = torch.nn.functional.smooth_l1_loss(all_overlap_gt, all_overlap_tilde)

        # compute feature loss.
        losses['feature'] = self.feature_loss(
            [src_feat[-1] for src_feat in pred['src_feats']],
            [tgt_feat[-1] for tgt_feat in pred['tgt_feats']],
            se3_transform_list(pose_gt, pred['src_kp']), pred['tgt_kp']
        )
        assert not losses['feature'].isnan().any()

        # compute correspondence loss.
        src_corr_loss = self.corr_loss(
            pred['src_kp'],
            [w[-1] for w in pred['src_kp_warped']],
            pose_gt,
            overlap_weights=self.src_overlap_gt
        )
        tgt_corr_loss = self.corr_loss(
            pred['tgt_kp'],
            [w[-1] for w in pred['tgt_kp_warped']],
            torch.stack([se3_inv(p) for p in pose_gt]),
            overlap_weights=self.tgt_overlap_gt
        )
        losses['corr'] = src_corr_loss + tgt_corr_loss

        losses['total'] = torch.sum(
            torch.stack([(losses[k] * self.weight_dict[k]) for k in losses])
        )
        losses['total'].backward()

        # Clip gradient.
        if self.grad_clip > 0:
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(), max_norm=self.grad_clip
            )

        self.optimizer.step()
        if not self.config.finetune:
            self.scheduler.step()

        self.scalars_to_log['train/overlap'] = losses['overlap']
        self.scalars_to_log['train/nerf_cont'] = losses['nerf_cont']
        self.scalars_to_log['train/feature'] = losses['feature']
        self.scalars_to_log['train/corr'] = losses['corr']
        self.scalars_to_log['train/total'] = losses['total']

        pose_error = evaluate_camera_alignment(pred_poses, pose_gt)
        self.scalars_to_log['train/R_mean'] = pose_error['R_error_mean']
        self.scalars_to_log['train/t_mean'] = pose_error['t_error_mean']

        src_visibility_score = torch.mean(pred['src_overlap'][0][-1])
        tgt_visibility_score = torch.mean(pred['tgt_overlap'][0][-1])
        self.scalars_to_log['train/src_overlap'] = src_visibility_score
        self.scalars_to_log['train/tgt_overlap'] = tgt_visibility_score

        self.scalars_to_log['lr'] = self.scheduler.get_last_lr()[0]

    @torch.no_grad()
    def validate(self) -> float:
        self.model.eval()

        R_mean, t_mean = 0.0, 0.0
        R_med, t_med = 0.0, 0.0
        score = 0
        num_val_scenes = len(self.val_dataset)
        val_ids = [i for i in range(num_val_scenes)]
        random.shuffle(val_ids)
        num_val_scenes = int(num_val_scenes * 0.2)
        print(f'Validating...')

        for i in range(num_val_scenes):
            data_batch = all_to_device(self.val_dataset[i], device=self.device)
            pred = self.model(data_batch)

            pred_poses = pred['pose'][-1] # [B, 3, 4]
            pose_gt = data_batch['pose']  # [B, 4, 4]

            pose_error = evaluate_camera_alignment(pred_poses, pose_gt)
            R_mean += pose_error['R_error_mean']
            t_mean += pose_error['t_error_mean']
            R_med += pose_error['R_error_med']
            t_med += pose_error['t_error_med']
            score += R_mean

        self.scalars_to_log['val/R_mean'] = R_mean / num_val_scenes
        self.scalars_to_log['val/t_mean'] = t_mean / num_val_scenes
        self.scalars_to_log['val/R_med'] = R_med / num_val_scenes
        self.scalars_to_log['val/t_med'] = t_med / num_val_scenes

        score = float(num_val_scenes) / score
        return score

    def compose_state_dicts(self) -> None:
        self.state_dicts = {'models': dict(), 'optimizers': dict(), 'schedulers': dict(), 'meta_data': None}

        self.state_dicts['models']['model'] = self.model
        self.state_dicts['models']['feature_loss'] = self.feature_loss
        self.state_dicts['optimizers']['optimizer'] = self.optimizer
        self.state_dicts['schedulers']['scheduler'] = self.scheduler


if __name__ == '__main__':
    config = config_parser()
    torch.multiprocessing.set_start_method('spawn')

    setup_seed(config.seed)

    trainer = RegTrainer(config)
    trainer.train()
