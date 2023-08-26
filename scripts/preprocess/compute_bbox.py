import os
import sys
import argparse

import numpy as np

_PATH = os.path.abspath(__file__)

sys.path.insert(
    0, os.path.join(os.path.dirname(_PATH), "../../conerf", "pycolmap", "pycolmap")
)
from scene_manager import SceneManager


def config_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--colmap_dir",
                        type=str,
                        required=True,
                        help="directory that store colmap output")
    parser.add_argument("--output",
                        type=str,
                        default="",
                        help="directory to store bounding box file")

    return parser.parse_args()


def compute_bounding_box(colmap_dir, output):
    manager = SceneManager(colmap_dir)
    manager.load_points3D()
    points3D = manager.points3D # [N, 3]
    num_points = points3D.shape[0]

    sorted_points = np.sort(points3D, axis=0)
    p0, p1 = 0.02, 0.98
    P0, P1 = int(p0 * (num_points - 1)), int(p1 * (num_points - 1))
    # P0, P1 = int(p0 * (num_points - 1)), int(p1 * (num_points - 1))
    aabb = np.array([sorted_points[P0, 0], sorted_points[P0, 1], sorted_points[P0, 2],
                     sorted_points[P1, 0], sorted_points[P1, 1], sorted_points[P1, 2]])

    # enlarge aabb
    scale_factor = 1.4
    A, B = aabb[:3], aabb[3:]
    C = (A + B) / 2.0 # AABB center
    half_diagonal_len = np.linalg.norm(B - A) / 2
    ca_ray_dir = (A - C) / np.linalg.norm(A - C)
    cb_ray_dir = (B - C) / np.linalg.norm(B - C)
    A = C + ca_ray_dir * scale_factor * half_diagonal_len
    B = C + cb_ray_dir * scale_factor * half_diagonal_len
    aabb = np.concatenate([A, B])

    aabb_file = os.path.join(output, 'bbox.txt')
    with open(aabb_file, 'w') as f:
        f.write(f'{aabb[0]} {aabb[1]} {aabb[2]} {aabb[3]} {aabb[4]} {aabb[5]}')
    
    print(f'Bounding box file saved to {aabb_file}')


if __name__ == '__main__':
    args = config_parser()

    if args.output == "":
        args.output = args.colmap_dir
    
    compute_bounding_box(args.colmap_dir, args.output)
