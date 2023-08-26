import math
from typing import Union, List

import torch
import tqdm
from torch_scatter import scatter_max

from nerfacc import rendering
from nerfacc.contraction import ContractionType, contract_inv

from conerf.utils.nerfacc_utils import ray_marching
from conerf.datasets.utils import Rays, namedtuple_map


def _meshgrid3d(
    res: torch.Tensor, device: Union[torch.device, str] = "cpu"
) -> torch.Tensor:
    """Create 3D grid coordinates."""
    assert len(res) == 3
    res = res.tolist()
    return torch.stack(
        torch.meshgrid(
            [
                torch.arange(res[0], dtype=torch.long),
                torch.arange(res[1], dtype=torch.long),
                torch.arange(res[2], dtype=torch.long),
            ],
            indexing="ij",
        ),
        dim=-1,
    ).to(device)


def occ_eval_fn(x, model, camera_poses, render_step_size, device='cpu', cone_angle=0., near_plane=None, far_plane=None):
    if cone_angle > 0.0:
        camera_ids = torch.randint(
            0, camera_poses.shape[0], (x.shape[0],), device=device
        )
        origins = camera_poses[camera_ids, :3, -1]
        t = (origins - x).norm(dim=-1, keepdim=True)
        # compute actual step size used in marching, based on the distance to the camera.
        step_size = torch.clamp(
            t * cone_angle, min=render_step_size
        )
        # filter out the points that are not in the near far plane.
        if (near_plane is not None) and (far_plane is not None):
            step_size = torch.where(
                (t > near_plane) & (t < far_plane),
                step_size,
                torch.zeros_like(step_size),
            )
    else:
        step_size = render_step_size
    # compute occupancy
    density = model.query_density(x)
    return density * step_size


class SampleGrid(torch.nn.Module):
    NUM_DIM: int = 3
    
    def __init__(
        self,
        roi_aabb: Union[List[int], torch.Tensor],
        resolution: Union[int, List[int], torch.Tensor] = 128,
        contraction_type: ContractionType = ContractionType.AABB,
    ) -> None:
        super().__init__()

        if isinstance(resolution, int):
            resolution = [resolution] * self.NUM_DIM
        if isinstance(resolution, (list, tuple)):
            resolution = torch.tensor(resolution, dtype=torch.int32)
        assert isinstance(
            resolution, torch.Tensor
        ), f"Invalid type: {type(resolution)}"
        assert resolution.shape == (
            self.NUM_DIM,
        ), f"Invalid shape: {resolution.shape}"

        if isinstance(roi_aabb, (list, tuple)):
            roi_aabb = torch.tensor(roi_aabb, dtype=torch.float32)
        assert isinstance(
            roi_aabb, torch.Tensor
        ), f"Invalid type: {type(roi_aabb)}"
        assert roi_aabb.shape == torch.Size(
            [self.NUM_DIM * 2]
        ), f"Invalid shape: {roi_aabb.shape}"

        # Total number of voxels.
        self.num_voxels = int(resolution.prod().item())

        # binary occupancy field.
        # TODO(chenyu): make it a multi-scale binary occupancy field.
        self.register_buffer(
            "_binary", torch.zeros(resolution.tolist(), dtype=torch.bool)
        )
        self.register_buffer("resolution", resolution)
        self.register_buffer("_roi_aabb", roi_aabb)
        self._contraction_type = contraction_type
        print(f'roi aabb: {roi_aabb}')
        print(f'resolution: {resolution}')

        # Grid coordinates and indices.
        grid_coords = _meshgrid3d(resolution).reshape(
            self.num_voxels, self.NUM_DIM
        )
        grid_indices = torch.arange(self.num_voxels)
        self.register_buffer("grid_coords", grid_coords, persistent=False)
        self.register_buffer("grid_indices", grid_indices, persistent=False)

        self._delta = 1e-2 # a chosen distance according to NeRF-RPN.
        self._viewdirs = self._generate_fixed_viewing_directions()

    @property
    def binary(self) -> torch.Tensor:
        """A 3D binarized tensor with torch.bool data type.

        The tensor is of shape (res_x, res_y, res_z), in which each boolean value
        represents whether the corresponding voxel should be kept or not.
        """
        if hasattr(self, "_binary"):
            return getattr(self, "_binary")
        else:
            raise NotImplementedError("please set an attribute named _binary!")

    @torch.no_grad()
    def set_binary_fields(self, binary):
        self._binary = binary

    @torch.no_grad()
    def _generate_fixed_viewing_directions(self) -> torch.Tensor:
        phis = [math.pi / 3, 0, -math.pi]
        thetas = [k * math.pi / 3 for k in range(0, 6)]
        viewdirs = []

        for phi in phis:
            for theta in thetas:
                viewdirs.append(torch.Tensor([
                    math.cos(phi) * math.sin(theta),
                    math.cos(phi) * math.sin(theta),
                    math.sin(theta)
                ]))
        viewdirs = torch.stack(viewdirs, dim=0)
        return viewdirs

    @torch.no_grad()
    def uniform_sample_occupied_voxels(self) -> torch.Tensor:
        """
        Sample occupied voxels.
        """
        occupied_indices = torch.nonzero(self._binary.flatten())[:, 0]
        return occupied_indices

    @torch.no_grad()
    def query_radiance_and_density(self, model, device, density_thre=0.7):
        indices = self.uniform_sample_occupied_voxels()

        # infer occupancy.
        grid_coords = self.grid_coords[indices]
        x = (
            grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)
        ) / self.resolution
        if self._contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            # only the points inside the sphere are valid.
            mask = (x - 0.5).norm(dim=1) < 0.5
            x = x[mask]
            indices = indices[mask]
        
        # voxel coordinates [0, 1]^3 -> world
        x = contract_inv(
            x,
            roi=self._roi_aabb,
            type=self._contraction_type,
        ) # [n, 3]

        x = x.to(device)
        self._viewdirs = self._viewdirs.to(device)
        
        num_viewdirs = self._viewdirs.shape[0]
        
        # Query density and radiance.
        pbar = tqdm.trange(num_viewdirs, desc=f"Querying Density and Radiance", leave=False)
        density, embedding = model.query_density(x, return_feat=True)

        density_mask = (density > density_thre).squeeze(-1)
        density = density[density_mask]
        embedding = embedding[density_mask]
        x = x[density_mask]
        indices = indices[density_mask]

        color = []
        num_points = x.shape[0]

        for k in range(num_viewdirs):
            viewdirs = self._viewdirs[k].repeat((num_points, 1))
            rgb = model.query_rgb(viewdirs, embedding) # [N, 3]
            color.append(rgb)
            
            pbar.update(1)

        color = torch.stack(color, dim=0) # [M, N, 3]
        color = torch.mean(color, dim=0) # [N, 3]
        alpha = torch.clip(1 - torch.exp(-self._delta * density), 0, 1)

        return x, color, density, alpha, indices

    @torch.no_grad()
    def query_radiance_and_density_from_camera(
        self,
        radiance_field,
        occupancy_grid,
        meta_data,
        device,
        density_thre=0.7,
        cut_off: float = 0.5,
    ):
        scene_aabb = torch.tensor(meta_data['aabb'], dtype=torch.float32, device=device)
        render_step_size = meta_data['render_step_size']
        cone_angle = meta_data['cone_angle']
        alpha_thre = meta_data['alpha_thre']

        indices = self.uniform_sample_occupied_voxels()

        # infer occupancy.
        grid_coords = self.grid_coords[indices]
        x = (
            grid_coords + torch.rand_like(grid_coords, dtype=torch.float32)
        ) / self.resolution
        if self._contraction_type == ContractionType.UN_BOUNDED_SPHERE:
            # only the points inside the sphere are valid.
            mask = (x - 0.5).norm(dim=1) < 0.5
            x = x[mask]
            indices = indices[mask]
        
        # voxel coordinates [0, 1]^3 -> world
        queried_points_world = contract_inv(
            x,
            roi=self._roi_aabb,
            type=self._contraction_type,
        ) # [Np, 3], queried points in world frame.
        queried_points_world = queried_points_world.to(device)
        Np = queried_points_world.shape[0]

        ################################### Query for surface field mask #######################
        color_bkgd = torch.ones(3, device=device)
        camtoworlds = meta_data['camera_poses'].unsqueeze(dim=1) # [Nc, 1, 4, 4]
        Nc = camtoworlds.shape[0]
        origins = camtoworlds[..., :3, -1].squeeze(dim=-1) # [Nc, 1, 3]
        
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

        def rgb_sigma_fn(t_starts, t_ends, ray_indices):
            t_origins = chunk_rays.origins[ray_indices]
            t_dirs = chunk_rays.viewdirs[ray_indices]
            positions = t_origins + t_dirs * (t_starts + t_ends) / 2.0

            return radiance_field(positions, t_dirs)

        chunk_size = 8192
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
            rgb, opacity, depth = rendering(
                t_starts,
                t_ends,
                ray_indices,
                n_rays=chunk_rays.origins.shape[0],
                rgb_sigma_fn=rgb_sigma_fn,
                render_bkgd=color_bkgd,
            )

            # Approximation of surface field: S(t) = T(t) * (1 - exp(-2 * sigma * delta))
            surface_field = alpha * transmittance
            # n_rays = int(ray_indices.max()) + 1
            con_surface_field = torch.zeros(
                (n_rays, 1), device=device, dtype=torch.float32
            )

            con_surface_field, _ = scatter_max(surface_field, ray_indices, out=con_surface_field, dim=0)
            surface_field_list.append(con_surface_field)
        
        surface_mask = torch.cat(surface_field_list, dim=0)
        surface_mask = surface_mask >= cut_off # make binary
        surface_mask, _ = surface_mask.reshape(Nc, Np).max(dim=0) # [Np, 1]

        ################################### Query for radiance and density #################################
        self._viewdirs = self._viewdirs.to(device)
        num_viewdirs = self._viewdirs.shape[0]
        
        # Query density and radiance.
        pbar = tqdm.trange(num_viewdirs, desc=f"Querying Density and Radiance", leave=False)
        density, embedding = radiance_field.query_density(queried_points_world, return_feat=True)

        density_mask = (density > density_thre).squeeze(-1)

        color = []

        for k in range(num_viewdirs):
            viewdirs = self._viewdirs[k].repeat((Np, 1))
            rgb = radiance_field.query_rgb(viewdirs, embedding) # [N, 3]
            color.append(rgb)
            
            pbar.update(1)

        color = torch.stack(color, dim=0) # [M, N, 3]
        color = torch.mean(color, dim=0) # [N, 3]
        alpha = torch.clip(1 - torch.exp(-self._delta * density), 0, 1)

        return queried_points_world, color, alpha, indices, density_mask, surface_mask
