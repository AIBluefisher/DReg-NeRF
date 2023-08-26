from typing import Union, List

import torch
import torch.nn.functional as F


def se3_init(
    rot: torch.Tensor = None, trans: torch.Tensor = None
) -> torch.Tensor:

    assert rot is not None or trans is not None

    if rot is not None and trans is not None:
        pose = torch.cat([rot, trans], dim=-1)
    elif rot is None:  # rotation not provided: will set to identity
        pose = F.pad(trans, (3, 0))
        pose[..., 0, 0] = pose[..., 1, 1] = pose[..., 2, 2] = 1.0
    elif trans is None:  # translation not provided: will set to zero
        pose = F.pad(rot, (0, 1))

    return pose


def se3_cat(a, b):
    """Concatenates two SE3 transforms"""
    rot_a, trans_a = a[..., :3, :3], a[..., :3, 3:4]
    rot_b, trans_b = b[..., :3, :3], b[..., :3, 3:4]

    rot = rot_a @ rot_b
    trans = rot_a @ trans_b + trans_a
    dst = se3_init(rot, trans)
    return dst


def se3_inv(pose: torch.Tensor) -> torch.Tensor:
    """Inverts the SE3 transform"""
    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    irot = rot.transpose(-1, -2)
    itrans = -irot @ trans
    return se3_init(irot, itrans)


def se3_transform(pose, xyz):
    """Apply rigid transformation to points

    Args:
        pose: ([B,] 3, 4)
        xyz: ([B,] N, 3)

    Returns:

    """

    assert xyz.shape[-1] == 3 and pose.shape[:-2] == xyz.shape[:-2]

    rot, trans = pose[..., :3, :3], pose[..., :3, 3:4]
    transformed = torch.einsum('...ij,...bj->...bi', rot, xyz) + trans.transpose(-1, -2)  # Rx + t

    return transformed


def se3_transform_list(
    pose: Union[List[torch.Tensor], torch.Tensor],
    xyz: List[torch.Tensor]
) -> list:
    """Similar to se3_transform, but processes lists of tensors instead

    Args:
        pose: List of (3, 4)
        xyz: List of (N, 3)

    Returns:
        List of transformed xyz
    """

    B = len(xyz)
    assert all([xyz[b].shape[-1] == 3 and pose[b].shape[:-2] == xyz[b].shape[:-2] for b in range(B)])

    transformed_all = []
    for b in range(B):
        R, t = pose[b][..., :3, :3], pose[b][..., :3, 3:4]
        Np = xyz[b].shape[0]
        transformed = R @ xyz[b].transpose(-1, -2) + t.expand(-1, Np)
        transformed_all.append(transformed.transpose(-1, -2))

    return transformed_all


def compute_rigid_transform(
    a: torch.Tensor,
    b: torch.Tensor,
    weights: torch.Tensor = None,
    eps: float = 1e-6
) -> torch.Tensor:
    """Compute rigid transforms between two point sets

    Args:
        a (torch.Tensor): ([*,] N, 3) points
        b (torch.Tensor): ([*,] N, 3) points
        weights (torch.Tensor): ([*, ] N)

    Returns:
        Transform T ([*,] 3, 4) to get from a to b, i.e. T*a = b
    """

    assert a.shape == b.shape
    assert a.shape[-1] == 3

    if weights is not None:
        assert a.shape[:-1] == weights.shape
        assert weights.min() >= 0 and weights.max() <= 1

        weights_normalized = weights[..., None] / \
                             torch.clamp_min(torch.sum(weights, dim=-1, keepdim=True)[..., None], eps)
        centroid_a = torch.sum(a * weights_normalized, dim=-2)
        centroid_b = torch.sum(b * weights_normalized, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ (b_centered * weights_normalized)
    else:
        centroid_a = torch.mean(a, dim=-2)
        centroid_b = torch.mean(b, dim=-2)
        a_centered = a - centroid_a[..., None, :]
        b_centered = b - centroid_b[..., None, :]
        cov = a_centered.transpose(-2, -1) @ b_centered

    # Compute rotation using Kabsch algorithm. Will compute two copies with +/-V[:,:3]
    # and choose based on determinant to avoid flips
    u, s, v = torch.svd(cov, some=False, compute_uv=True)
    rot_mat_pos = v @ u.transpose(-1, -2)
    v_neg = v.clone()
    v_neg[..., 2] *= -1
    rot_mat_neg = v_neg @ u.transpose(-1, -2)
    rot_mat = torch.where(torch.det(rot_mat_pos)[..., None, None] > 0, rot_mat_pos, rot_mat_neg)

    # Compute translation (uncenter centroid)
    translation = -rot_mat @ centroid_a[..., :, None] + centroid_b[..., :, None]

    transform = torch.cat((rot_mat, translation), dim=-1)
    return transform
