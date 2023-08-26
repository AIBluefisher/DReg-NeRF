import os
import sys

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm
import cv2

from conerf.datasets.dataset_base import DatasetBase
from conerf.datasets.utils import minify, visualize_block_poses
from conerf.register.cluster import clustering


_PATH = os.path.abspath(__file__)

sys.path.insert(
    0, os.path.join(os.path.dirname(_PATH), "..", "pycolmap", "pycolmap")
)
from scene_manager import SceneManager


def read_pfm(filename):
    file = open(filename, "rb")
    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().decode("utf-8").rstrip()
    if header == "PF":
        color = True
    elif header == "Pf":
        color = False
    else:
        raise Exception("Not a PFM file.")

    dim_match = re.match(r"^(\d+)\s(\d+)\s$", file.readline().decode("utf-8"))
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception("Malformed PFM header.")

    scale = float(file.readline().rstrip())
    if scale < 0:  # little-endian
        endian = "<"
        scale = -scale
    else:
        endian = ">"  # big-endian

    data = np.fromfile(file, endian + "f")
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    file.close()
    return data, scale


def get_filename_no_ext(filename):
    return os.path.splitext(filename)[0]


def get_file_extension(filename):
    return os.path.splitext(filename)[-1]


def get_all_image_names(dir, formats=[
    '.png', '.PNG', '.jpg', '.JPG', '.jpeg', '.JPEG']) -> list:
    image_paths = []
    for root, dirs, files in os.walk(dir):
        for file in files:
            if not get_file_extension(file) in formats:
                continue
            image_paths.append(os.path.join(dir, root, file))
    return sorted(image_paths)


def get_filename_from_abs_path(abs_path):
    return abs_path.split('/')[-1]



def _load_colmap(root_fp: str, subject_id: str, split: str, factor: int = 1, multi_blocks: bool = False, num_blocks: int = 1):
    assert factor in [1, 2, 4, 8]

    data_dir = os.path.join(root_fp, subject_id)
    colmap_dir = os.path.join(data_dir, "sparse/0")

    # Read bounding box.
    scene_bbox = torch.from_numpy(np.loadtxt(os.path.join(colmap_dir, 'bbox.txt'))).float()[:6]

    if factor != 1:
        minify(basedir=data_dir, factors=[factor])

    manager = SceneManager(colmap_dir)
    manager.load_cameras()
    manager.load_images()

    # Assume shared intrinsics between all cameras.
    cam = manager.cameras[1]
    fx, fy, cx, cy = cam.fx, cam.fy, cam.cx, cam.cy
    K = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]])
    K[:2, :] /= factor

    # Extract extrinsic matrices in world-to-camera format.
    imdata = manager.images
    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in imdata:
        im = imdata[k]
        rot = im.R()
        trans = im.tvec.reshape(3, 1)
        w2c = np.concatenate([np.concatenate([rot, trans], 1), bottom], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    # Image names from COLMAP. No need for permuting the poses according to
    # image names anymore.
    image_names = [imdata[k].name for k in imdata]

    # # Switch from COLMAP (right, down, fwd) to Nerf (right, up, back) frame.
    # poses = poses @ np.diag([1, -1, -1, 1])

    # Get distortion parameters.
    type_ = cam.camera_type
    print(f'type: {type_}')

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]

    # Load images.
    if factor > 1:
        image_dir_suffix = f"_{factor}"
    else:
        image_dir_suffix = ""
    colmap_image_dir = os.path.join(data_dir, "images")
    # image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
    image_dir = os.path.join(data_dir, "images" + image_dir_suffix)
    for d in [image_dir, colmap_image_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Image folder {d} does not exist.")
    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = sorted(os.listdir(colmap_image_dir))
    image_files = sorted(os.listdir(image_dir))
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [
        os.path.join(image_dir, colmap_to_image[f]) for f in image_names
    ]
    print("loading images")
    images = [imageio.imread(x) for x in tqdm.tqdm(image_paths)]
    images = np.stack(images, axis=0)

    val_interval = 30
    if multi_blocks:
        labels = clustering(camtoworlds, num_clusters=num_blocks, method="KMeans")
        # visualize_block_poses(camtoworlds, labels)

        # Group images into blocks according to their cluster labels.
        block_image_ids = dict()
        for image_id, label in enumerate(labels):
            if label not in block_image_ids.keys():
                block_image_ids[label] = list()
            block_image_ids[label].append(image_id)

        num_blocks = len(block_image_ids)
        block_images, block_camtoworlds = [None] * num_blocks, [None] * num_blocks
        for block_id in block_image_ids.keys():
            image_ids = sorted(block_image_ids[block_id])
            image_ids = np.array(image_ids)
            # Select the split.
            all_indices = list(range(0, image_ids.shape[0]))
            test_indices = list(range(0, image_ids.shape[0], val_interval))
            train_indices = np.array([i for i in all_indices if i not in test_indices])
            split_indices = {
                "test": image_ids[np.array(test_indices)],
                "train": image_ids[train_indices]
            }
            indices = split_indices[split]
            block_images[block_id] = images[indices]
            block_camtoworlds[block_id] = camtoworlds[indices]

        return block_images, block_camtoworlds, K, scene_bbox
    
    # Select the split.
    all_indices = np.arange(images.shape[0])
    split_indices = {
        "test": all_indices[all_indices % val_interval == 0],
        "train": all_indices[all_indices % val_interval != 0],
    }
    indices = split_indices[split]
    # All per-image quantities must be re-indexed using the split indices.
    images = images[indices]
    camtoworlds = camtoworlds[indices]

    return images, camtoworlds, K, scene_bbox


def build_proj_mats(pose_files):
    all_intrinsics, c2w_mats = [], []
    scale_factor = None

    for vid, pose_file in enumerate(pose_files):
        intrinsics, extrinsics, depth_min, depth_max, scale_factor = read_cam_file(
            pose_file, scale_factor
        )

        all_intrinsics.append(intrinsics)
        c2w_mats.append(np.linalg.inv(extrinsics))
        
    all_intrinsics = np.stack(all_intrinsics)
    c2w_mats = np.stack(c2w_mats)
    # print(f'c2w_mats shape: {c2w_mats.shape}')
    # print(f'all_intrinsics shape: {all_intrinsics.shape}')
        
    return all_intrinsics, c2w_mats, depth_min, depth_max


def read_cam_file(filename, scale_factor):
    with open(filename) as f:
        lines = [line.rstrip() for line in f.readlines()]
    
    # extrinsics: line [1,5), 4x4 matrix
    extrinsics = np.fromstring(' '.join(lines[1:5]), dtype=np.float32, sep=' ')
    extrinsics = extrinsics.reshape((4, 4))
    
    # intrinsics: line [7-10), 3x3 matrix
    intrinsics = np.fromstring(' '.join(lines[7:10]), dtype=np.float32, sep=' ')
    intrinsics = intrinsics.reshape((3, 3))

    # # depth_min & depth_interval: line 11
    depth_min = float(lines[11].split()[0])
    depth_max = float(lines[11].split()[-1])

    # use the first cam to determine scale factor
    if scale_factor is None:
        scale_factor = 5 / depth_min

    depth_min *= scale_factor
    depth_max *= scale_factor
    extrinsics[:3, 3] *= scale_factor
    
    return intrinsics, extrinsics, depth_min, depth_max, scale_factor


def _load_mvs(
    root_fp: str,
    subject_id: str,
    split: str,
    factor: int = 1,
    multi_blocks: bool = False,
    num_blocks: int = 1,
):
    assert factor in [1, 2, 4, 8]

    data_dir = os.path.join(root_fp, subject_id)

    image_dir = os.path.join(data_dir, 'images')
    camera_dir = os.path.join(data_dir, 'cams')
    depth_dir = os.path.join(data_dir, 'rendered_depth_maps')

    image_files = get_all_image_names(image_dir)
    pose_files, depth_files = [], []
    for image_file in image_files:
        image_name = get_filename_no_ext(get_filename_from_abs_path(image_file))
        pose_file = os.path.join(camera_dir, image_name + '_cam.txt')
        depth_file = os.path.join(depth_dir, image_name + '.pfm')
        pose_files.append(pose_file)
        depth_files.append(depth_file)

    all_intrinsics, camtoworlds, depth_min, depth_max = build_proj_mats(pose_files)
    K = all_intrinsics[0]
    near_depth, far_depth = depth_min, depth_max
    K[:2, :] /= factor

    # if factor != 1:
    #     minify(basedir=data_dir, factors=[factor])

    print("loading images")
    images = [imageio.imread(x) for x in tqdm.tqdm(image_files)]
    images = np.stack(images, axis=0)

    val_interval = 30
    if multi_blocks:
        labels = clustering(camtoworlds, num_clusters=num_blocks, method="KMeans")
        # visualize_block_poses(camtoworlds, labels)

        # Group images into blocks according to their cluster labels.
        block_image_ids = dict()
        for image_id, label in enumerate(labels):
            if label not in block_image_ids.keys():
                block_image_ids[label] = list()
            block_image_ids[label].append(image_id)

        num_blocks = len(block_image_ids)
        block_images, block_camtoworlds = [None] * num_blocks, [None] * num_blocks
        for block_id in block_image_ids.keys():
            image_ids = sorted(block_image_ids[block_id])
            image_ids = np.array(image_ids)
            # Select the split.
            all_indices = list(range(0, image_ids.shape[0]))
            test_indices = list(range(0, image_ids.shape[0], val_interval))
            train_indices = np.array([i for i in all_indices if i not in test_indices])
            split_indices = {
                "test": image_ids[np.array(test_indices)],
                "train": image_ids[train_indices]
            }
            indices = split_indices[split]
            block_images[block_id] = images[indices]
            block_camtoworlds[block_id] = camtoworlds[indices]

        return block_images, block_camtoworlds, K, near_depth, far_depth
    
    # Select the split.
    all_indices = np.arange(images.shape[0])
    split_indices = {
        "test": all_indices[all_indices % val_interval == 0],
        "train": all_indices[all_indices % val_interval != 0],
    }
    indices = split_indices[split]
    # All per-image quantities must be re-indexed using the split indices.
    images = images[indices]
    camtoworlds = camtoworlds[indices]

    return images, camtoworlds, K, near_depth, far_depth


class SubjectLoader(DatasetBase):
    NEAR = 0.02
    FAR = 500
    SUBJECT_IDS = [
        "scan3", "scan4", "scan5", "scan6", "scan9", "scan10", "scan12",
        "scan13", "scan14", "scan15", "scan16", "scan17", "scan18", "scan19",
        "scan20", "scan22", "scan23", "scan24", "scan28", "scan32", "scan33",
        "scan37", "scan42", "scan43", "scan44", "scan46", "scan49", "scan50",
        "scan52", "scan59", "scan60", "scan61", "scan62", "scan64", "scan66",
        "scan70", "scan71", "scan72", "scan74", "scan75", "scan76", "scan84",
        "scan94", "scan95", "scan96", "scan97", "scan98", "scan99", "scan100",
        "scan104", "scan105", "scan106", "scan107", "scan108", "scan109", "scan119",
        "scan120", "scan121", "scan122", "scan123", "scan124", "scan125", "scan128",
        "clay0", "clay1", "stone0", "clay2", "stone1", "stone2", "stone3", "stone4",
        "big_ben", "clay3", "ancient_clock", "stone5", "clay4", "clay5", "stone6",
        "stone7", "stone8", "sculpture0", "jar", "sculpture1", "sculpture2",
        "sculpture3", "clay6", "shoes1", "sculpture4", "sculpture5", "sculpture6",
        "sculpture7", "sculpture8", "sculpture9", "santa_clay", "egg_clay", "food",
        "doll_bear", "sculpture10", "clay7", "gundam", "sculpture11", "bread",
        "sculpture12", "dull_dog", "camera", "shoes2", "basketball", "sculpture13",
        "sculpture14",
    ]

    OPENGL_CAMERA = False
    DATA_TYPE = "REAL_WORLD"

    def __init__(
        self,
        subject_id: str,
        root_fp: str,
        split: str,
        data_split_json: str = "",
        color_bkgd_aug: str = "white",
        num_rays: int = None,
        near: float = None,
        far: float = None,
        batch_over_images: bool = True,
        factor: int = 1,
        multi_blocks: bool = False,
        num_blocks: int = 1
    ) -> None:
        super().__init__(
            subject_id,
            root_fp,
            split,
            data_split_json,
            color_bkgd_aug,
            num_rays,
            near,
            far,
            batch_over_images,
            factor,
            multi_blocks,
            num_blocks
        )

    def load_data(self, root_fp: str, subject_id: str, split: str, factor: int = 1, num_blocks: int = 1):
        images, camtoworlds, K, scene_bbox = _load_colmap(
            root_fp, subject_id, split, factor, self.multi_blocks, num_blocks
        )
        # images, camtoworlds, K, _, __ = _load_mvs(
        #     root_fp, subject_id, split, factor, self.multi_blocks, num_blocks
        # )
        self.BBOX = scene_bbox.tolist()

        if not self.multi_blocks:
            images = torch.from_numpy(images).to(torch.uint8)
            camtoworlds = torch.from_numpy(camtoworlds).to(torch.float32)
        K = torch.tensor(K).to(torch.float32)

        return images, camtoworlds, K
