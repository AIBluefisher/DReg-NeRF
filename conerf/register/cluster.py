import torch
import numpy as np

from sklearn.cluster import SpectralClustering, KMeans


def clustering(poses: np.ndarray, num_clusters: int = 2, method: str = "KMeans") -> np.array:
    """
    Args:
        params poses: camera poses from camera frame to world frame, [N, 4, 4]
        params num_cluster: number of clusters to partition
        params method: use 'KMeans' or 'Spectral'
    Return:
        cluster labels for corresponding camera poses.
    """
    centers = poses[..., :3, -1]

    if method == 'KMeans':
        clustering = KMeans(
            n_clusters=num_clusters,
            random_state=0,
            n_init="auto"
        ).fit(centers)
    elif method == 'Spectral':
        clustering = SpectralClustering(
            n_clusters=num_clusters,
            assign_labels='discretize',
            random_state=0
        ).fit(centers)
    else:
        raise NotImplementedError
    
    return clustering.labels_
