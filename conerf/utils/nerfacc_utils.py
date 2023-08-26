from typing import Callable, Optional, Tuple

import torch

import nerfacc.cuda as _C

from nerfacc.contraction import ContractionType
from nerfacc.grid import Grid
from nerfacc.intersection import ray_aabb_intersect
# from nerfacc.vol_rendering import render_visibility
from nerfacc.vol_rendering import _RenderingTransmittanceFromAlphaCUB, _RenderingTransmittanceFromAlphaNaive
from nerfacc.pack import pack_info


@torch.no_grad()
def render_transmittance(
    alphas: torch.Tensor,
    *,
    ray_indices: Optional[torch.Tensor] = None,
    packed_info: Optional[torch.Tensor] = None,
    n_rays: Optional[int] = None,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
) -> torch.Tensor:
    """Filter out transparent and occluded samples.

    In this function, we first compute the transmittance from the sample opacity. The
    transmittance is then used to filter out occluded samples. And opacity is used to
    filter out transparent samples. The function returns a boolean tensor indicating
    which samples are visible (`transmittance > early_stop_eps` and `opacity > alpha_thre`).

    Note:
        Either `ray_indices` or `packed_info` should be provided. If `ray_indices` is 
        provided, CUB acceleration will be used if available (CUDA >= 11.6). Otherwise,
        we will use the naive implementation with `packed_info`.

    Args:
        alphas: The opacity values of the samples. Tensor with shape (n_samples, 1).
        packed_info: Optional. Stores information on which samples belong to the same ray. \
            See :func:`nerfacc.ray_marching` for details. LongTensor with shape (n_rays, 2).
        ray_indices: Optional. Ray index of each sample. LongTensor with shape (n_sample).
        n_rays: Optional. Number of rays. Only useful when `ray_indices` is provided yet \
            CUB acceleration is not available. We will implicitly convert `ray_indices` to \
            `packed_info` and use the naive implementation. If not provided, we will infer \
            it from `ray_indices` but it will be slower.
        early_stop_eps: The early stopping threshold on transmittance.
        alpha_thre: The threshold on opacity.
    
    Returns:
        The visibility of each sample. Tensor with shape (n_samples, 1).

    Examples:

    .. code-block:: python

        >>> alphas = torch.tensor( 
        >>>     [[0.4], [0.8], [0.1], [0.8], [0.1], [0.0], [0.9]], device="cuda")
        >>> ray_indices = torch.tensor([0, 0, 0, 1, 1, 2, 2], device="cuda")
        >>> transmittance = render_transmittance_from_alpha(alphas, ray_indices=ray_indices)
        tensor([[1.0], [0.6], [0.12], [1.0], [0.2], [1.0], [1.0]])
        >>> visibility = render_visibility(
        >>>     alphas, ray_indices=ray_indices, early_stop_eps=0.3, alpha_thre=0.2)
        tensor([True,  True, False,  True, False, False,  True])

    """
    assert (
        ray_indices is not None or packed_info is not None
    ), "Either ray_indices or packed_info should be provided."
    if ray_indices is not None and _C.is_cub_available():
        transmittance = _RenderingTransmittanceFromAlphaCUB.apply(
            ray_indices, alphas
        )
    else:
        if packed_info is None:
            packed_info = pack_info(ray_indices, n_rays=n_rays)
        transmittance = _RenderingTransmittanceFromAlphaNaive.apply(
            packed_info, alphas
        )

    return transmittance


@torch.no_grad()
def ray_marching(
    # rays
    rays_o: torch.Tensor,
    rays_d: torch.Tensor,
    t_max: torch.Tensor,
    # bounding box of the scene
    scene_aabb: torch.Tensor,
    # binarized grid for skipping empty space
    grid: Grid,
    # sigma/alpha function for skipping invisible space
    sigma_fn: Callable,
    early_stop_eps: float = 1e-4,
    alpha_thre: float = 0.0,
    # rendering options
    render_step_size: float = 1e-3,
    cone_angle: float = 0.0,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Ray marching with space skipping.

    Warning:
        This function is not differentiable to any inputs.

    Args:
        rays_o: Ray origins of shape (n_rays, 3).
        rays_d: Normalized ray directions of shape (n_rays, 3).
        scene_aabb: Scene bounding box for computing t_min and t_max.
            A tensor with shape (6,) {xmin, ymin, zmin, xmax, ymax, zmax}.
            `scene_aabb` will be ignored if both `t_min` and `t_max` are provided.
        grid: Grid that indicates where to skip during marching.
            See :class:`nerfacc.Grid` for details.
        sigma_fn: If provided, the marching will skip the invisible space
            by evaluating the density along the ray with `sigma_fn`. It should be a 
            function that takes in samples {t_starts (N, 1), t_ends (N, 1),
            ray indices (N,)} and returns the post-activation density values (N, 1).
            You should only provide either `sigma_fn` or `alpha_fn`.
        early_stop_eps: Early stop threshold for skipping invisible space. Default: 1e-4.
        alpha_thre: Alpha threshold for skipping empty space. Default: 0.0.
        render_step_size: Step size for marching. Default: 1e-3.
        cone_angle: Cone angle for linearly-increased step size. 0. means
            constant step size. Default: 0.0.

    Returns:
        A tuple of tensors.

            - **ray_indices**: Ray index of each sample. IntTensor with shape (n_samples).
            - **t_starts**: Per-sample start distance. Tensor with shape (n_samples, 1).
            - **t_ends**: Per-sample end distance. Tensor with shape (n_samples, 1).

    Examples:

    .. code-block:: python

        import torch
        from nerfacc import OccupancyGrid, ray_marching, unpack_info

        device = "cuda:0"
        batch_size = 128
        rays_o = torch.rand((batch_size, 3), device=device)
        rays_d = torch.randn((batch_size, 3), device=device)
        rays_d = rays_d / rays_d.norm(dim=-1, keepdim=True)

        # Ray marching with aabb.
        scene_aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d, scene_aabb=scene_aabb, render_step_size=1e-3
        )

        # Ray marching with aabb and skip areas based on occupancy grid.
        scene_aabb = torch.tensor([0.0, 0.0, 0.0, 1.0, 1.0, 1.0], device=device)
        grid = OccupancyGrid(roi_aabb=[0.0, 0.0, 0.0, 0.5, 0.5, 0.5]).to(device)
        ray_indices, t_starts, t_ends = ray_marching(
            rays_o, rays_d, scene_aabb=scene_aabb, grid=grid, render_step_size=1e-3
        )

        # Convert t_starts and t_ends to sample locations.
        t_mid = (t_starts + t_ends) / 2.0
        sample_locs = rays_o[ray_indices] + t_mid * rays_d[ray_indices]

    """
    if not rays_o.is_cuda:
        raise NotImplementedError("Only support cuda inputs.")

    # logic for t_min and t_max:
    # 1. scene_aabb is given, use ray_aabb_intersect to compute t_min and t_max.
    t_min, _ = ray_aabb_intersect(rays_o, rays_d, scene_aabb)

    # use grid for skipping
    grid_roi_aabb = grid.roi_aabb
    grid_binary = grid.binary
    contraction_type = grid.contraction_type.to_cpp_version()

    # marching with grid-based skipping
    packed_info, ray_indices, t_starts, t_ends = _C.ray_marching(
        # rays
        rays_o.contiguous(),
        rays_d.contiguous(),
        t_min.contiguous(),
        t_max.contiguous(),
        # contraction and grid
        grid_roi_aabb.contiguous(),
        grid_binary.contiguous(),
        contraction_type,
        # sampling
        render_step_size,
        cone_angle,
    )

    # Query sigma without gradients to skip invisible space.
    sigmas = sigma_fn(t_starts, t_ends, ray_indices)

    assert (
        sigmas.shape == t_starts.shape
    ), "sigmas must have shape of (N, 1)! Got {}".format(sigmas.shape)
    alphas = 1.0 - torch.exp(-sigmas * (t_ends - t_starts))

    # Compute visibility of the samples, and filter out invisible samples
    transmittance = render_transmittance(
        alphas,
        ray_indices=ray_indices,
        packed_info=packed_info,
        early_stop_eps=early_stop_eps,
        alpha_thre=alpha_thre,
        n_rays=rays_o.shape[0],
    ) # [n_pts, 1]
        
    visibility = transmittance >= early_stop_eps
    if alpha_thre > 0:
        visibility = visibility & (alphas >= alpha_thre)
    masks = visibility.squeeze(-1)

    alphas, transmittance, ray_indices, t_starts, t_ends = (
        alphas[masks],
        transmittance[masks],
        ray_indices[masks],
        t_starts[masks],
        t_ends[masks],
    )

    return alphas, transmittance, ray_indices, t_starts, t_ends
