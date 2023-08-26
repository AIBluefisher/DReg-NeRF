import MinkowskiEngine as ME

import torch


def batched_grid_subsample(points, features, batched_lengths, sample_dl: float = 0.1):
    """
    Note: This function is not deterministic and may return downsampled points
        in a different ordering, which will cause the subsequent steps to differ
        slightly.
    
    Args:
        points: point clouds with shape [[N0+N1+...], XYZ_DIM=3]
        features: associated features of point clouds with shape [[N0+N1+...], FEAT_DIM=256]
    """
    batch_size = len(batched_lengths)
    device = points[0].device
    
    batch_start_end = torch.nn.functional.pad(
        torch.cumsum(batched_lengths, 0), (1, 0)
    )

    featured_points = torch.cat([points, features], dim=-1) # [batch_size, N, XYZ_DIM + FEAT_DIM]
    coordinates = ME.utils.batched_coordinates(
        [   # coordinates must be defined in a integer grid
            points[batch_start_end[b]:batch_start_end[b + 1]] / sample_dl \
                for b in range(batch_size)
        ], device=device
    )

    sparse_tensor = ME.SparseTensor(
        features=featured_points,
        coordinates=coordinates,
        # when used with continuous coordinates, average features in the same coordinate.
        quantization_mode=ME.SparseTensorQuantizationMode.UNWEIGHTED_AVERAGE
    )

    ds_featured_points = sparse_tensor.features
    ds_points_length = torch.tensor([
        feat.shape[0] for feat in sparse_tensor.decomposed_features
        ], device=device
    )

    return ds_featured_points, ds_points_length


def hierarchical_grid_subsample(
    points: torch.Tensor,
    features: torch.Tensor,
    point_lengths: torch.Tensor,
    num_hierarchical: int = 4,
    init_subsample_dl: float = 0.025,
    subsample_radius: float = 2.75
):
    """
    Args:
        points: point clouds with shape [[N0+N1+...], XYZ_DIM=3]
        features: associated features of point clouds with shape [[N0+N1+...], FEAT_DIM=256]
        point_lengths: number of points in source point cloud and target point cloud
        num_hierarchical: how many layers to subsample
        init_subsample_dl: set smaller to have a higher resolution
        subsample_radius: sampling radius
    Return:
        ds_points: downsample points
        ds_features: features corresponds to the downsample points
        ds_points_length: downsampled points length in source and target
    """
    radius_normal = init_subsample_dl * subsample_radius
    ds_points, ds_features = points, features
    max_num_points = 1500 # TODO(chenyu): set up a threshold for the number of point clouds.

    for k in range(num_hierarchical):
        # print(f'downsampled point clouds shape: {ds_points.shape}', flush=True)
        # print(f'downsampled features shape: {ds_features.shape}', flush=True)
        
        # New subsample length.
        dl = 2 * radius_normal / subsample_radius
        # print(f'dl: {dl}', flush=True)

        # subsample points.
        ds_featured_points, ds_points_length = batched_grid_subsample(
            ds_points, ds_features, point_lengths, dl
        )

        # Update input by the latest downsampled points.
        ds_points, ds_features = ds_featured_points[..., :3], ds_featured_points[..., 3:]
        point_lengths = ds_points_length
        radius_normal *= 2

        # Return immediately when the point cloud size is almost the threshold (e.g. 1000).
        if ds_points.shape[0] <= 2 * max_num_points:
            break

    return ds_points, ds_features, ds_points_length
