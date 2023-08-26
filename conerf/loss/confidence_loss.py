# import argparse
from typing import List, Tuple

import torch
from torch_scatter import scatter_max

from nerfacc import ContractionType, OccupancyGrid, rendering

from conerf.base.checkpoint_manager import CheckPointManager
from conerf.datasets.utils import Rays, namedtuple_map
from conerf.radiance_fields.ngp import NGPradianceField
from conerf.utils.nerfacc_utils import ray_marching


@torch.no_grad()
def load_radiance_fields(
    nerf_model_path: str,
    device: str,
) -> Tuple[NGPradianceField, OccupancyGrid, dict]:
    # Load meta data from checkpoint at first.
    meta_data = {
        'aabb': None,
        'near_plane': None,
        'far_plane': None,
        'unbounded': None,
        'grid_resolution': None,
        'contraction_type': None,
        'render_step_size': None,
        'alpha_thre': None,
        'cone_angle': None,
        'camera_poses': None,
    }

    ckpt_manager = CheckPointManager(verbose=False)
    ckpt_manager.load_no_config(ckpt_path=nerf_model_path, meta_data=meta_data)

    nerf = NGPradianceField(
        aabb=meta_data['aabb'],
        unbounded=meta_data['unbounded']
    ).to(device)
    occupancy_grid = OccupancyGrid(
        roi_aabb=meta_data['aabb'],
        resolution=meta_data['grid_resolution'],
        contraction_type=meta_data['contraction_type']
    ).to(device)

    models = dict()
    models['model'] = nerf
    models['occupancy_grid'] = occupancy_grid
    ckpt_manager.load_no_config(ckpt_path=nerf_model_path, models=models)

    return nerf, occupancy_grid, meta_data


# @torch.no_grad()
def compute_visibility_score(
    xyz_list: List[torch.Tensor],
    nerf_model_path: str,
    delta: float = 1e-2,
    cut_off: float = 0.5,
    score_type: str = 'surface_field' # ['density_field', 'surface_field']
) -> List[torch.Tensor]:
    """
    Args:
        xyz_list: list of point coordinates, [num_layers, N, 3],
        nerf_model_path:
        delta:
    Return:
        list of visibility score.
    """

    device = xyz_list[0].device
    radiance_field, occupancy_grid, meta_data = load_radiance_fields(nerf_model_path, device)
    scene_aabb = torch.tensor(meta_data['aabb'], dtype=torch.float32, device=device)
    render_step_size = meta_data['render_step_size']
    cone_angle = meta_data['cone_angle']
    alpha_thre = meta_data['alpha_thre']

    # Using density field as overlapping score.
    if score_type == 'density_field':
        visibility_list = []
        for xyz in xyz_list:
            num_layers, num_points = xyz.shape[0], xyz.shape[1]
            density = radiance_field.query_density(xyz.reshape(-1, 3))
            alpha = torch.clip(1 - torch.exp(-delta * density), 0, 1)
            alpha = alpha.reshape(num_layers, num_points, 1)
            visibility_list.append(alpha)
        
        return visibility_list

    scores = []
    # Using surface field as overlapping score.
    for xyz in xyz_list:
        num_layers, num_points = xyz.shape[0], xyz.shape[1]
        
        camtoworlds = meta_data['camera_poses'].unsqueeze(dim=1) # [Nc, 1, 4, 4]
        Nc = camtoworlds.shape[0]
        origins = camtoworlds[..., :3, -1].squeeze(dim=-1) # [Nc, 1, 3]

        queried_points_world = xyz.reshape(-1, 3) # [Np, 3], queried points in world frame.
        Np = queried_points_world.shape[0]
        
        origins = origins.expand(-1, Np, -1) # [Nc, Np, 3]
        directions = queried_points_world - origins # [Nc, Np, 3]
        
        origins = origins.reshape(-1, 3)        # [Nc*Np, 3]
        directions = directions.reshape(-1, 3)  # [Nc*Np, 3]

        dirs_norm = torch.linalg.norm(directions, dim=-1, keepdims=True)
        viewdirs = directions / dirs_norm # [Nc*Np, 3]
        rays = Rays(origins=origins, viewdirs=viewdirs)

        # query points' volume densities and transmittance.
        rays_shape = rays.origins.shape
        num_rays, _ = rays_shape

        def sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0
            
            return radiance_field.query_density(positions)

        chunk_size = 60000 # 8192
        surface_field_list = []

        for i in range(0, num_rays, chunk_size):
            chunk_rays = namedtuple_map(lambda r: r[i : i + chunk_size], rays)
            n_rays = chunk_rays.origins.shape[0]
            t_max = dirs_norm[i : i + chunk_size].squeeze(dim=-1)
            alpha, transmittance, ray_indices, t_starts, t_ends = ray_marching(
                chunk_rays.origins,
                chunk_rays.viewdirs,
                t_max=t_max,
                scene_aabb=scene_aabb,
                grid=occupancy_grid,
                sigma_fn=sigma_fn,
                render_step_size=render_step_size,
                cone_angle=cone_angle,
                alpha_thre=alpha_thre,
            )

            # Approximation of surface field: S(t) = T(t) * (1 - exp(-2 * sigma * delta))
            surface_field = alpha * transmittance
            # n_rays = int(ray_indices.max()) + 1
            con_surface_field = torch.zeros(
                (n_rays, 1), device=device, dtype=torch.float32
            )

            con_surface_field, _ = scatter_max(surface_field, ray_indices, out=con_surface_field, dim=0)
            surface_field_list.append(con_surface_field)
        
        surface_fields = torch.cat(surface_field_list, dim=0)
        surface_fields = (surface_fields >= cut_off).float() # make binary
        surface_fields, _ = surface_fields.reshape(Nc, Np, 1).max(dim=0)
        surface_fields = surface_fields.reshape(num_layers, num_points, 1)

        scores.append(surface_fields)

    return scores
