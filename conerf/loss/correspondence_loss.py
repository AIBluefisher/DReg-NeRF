import torch
import robust_loss_pytorch.general as robust_loss

from conerf.register.se3 import se3_transform_list


class CorrespondenceLoss(torch.nn.Module):
    def __init__(self, metric: str = 'mae', robust_loss: bool = True) -> None:
        super().__init__()

        assert metric in ['mse', 'mae']

        self.metric = metric
        self.robust_loss_func = robust_loss

    def forward(self, kp_before, kp_warped_pred, pose_gt, overlap_weights=None, eps=1e-6):
        kp_warped_gt = se3_transform_list(pose_gt, kp_before)
        corr_err = torch.cat(kp_warped_pred, dim=0) - torch.cat(kp_warped_gt, dim=0)
        # alpha: For smooth interpolation between a number of discrete robust losses:
        #   alpha=-Infinity: Welsch/Leclerc Loss.
        #   alpha=-2: Geman-McClure loss.
        #   alpha=0: Cauchy/Lortentzian loss.
        #   alpha=1: Charbonnier/pseudo-Huber loss.
        #   alpha=2: L2 loss.
        #
        # scale: The scale parameter of the loss. When |x| < scale, the loss is an
        # L2-like quadratic bowl, and when |x| > scale the loss function takes on a
        # different shape according to alpha. Must be a tensor of single-precision
        # floats.
        if self.robust_loss_func:
            corr_err = robust_loss.lossfun(
                corr_err,
                alpha=torch.tensor([1.], dtype=corr_err.dtype, device=corr_err.device),
                scale=torch.tensor([0.5], dtype=torch.float32, device=corr_err.device)
            )

        if self.metric == 'mae':
            corr_err = torch.sum(torch.abs(corr_err), dim=-1)
        elif self.metric == 'mse':
            corr_err = torch.sum(torch.square(corr_err), dim=-1)
        else:
            raise NotImplementedError

        if overlap_weights is not None:
            overlap_weights = torch.cat(overlap_weights)
            mean_err = torch.sum(overlap_weights * corr_err) / \
                       torch.clamp_min(torch.sum(overlap_weights), eps)
        else:
            mean_err = torch.mean(corr_err, dim=1)
        
        return mean_err
