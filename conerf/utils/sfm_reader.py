import os
import collections
import numpy as np
import struct

from conerf.geometry.rotation import (Rotation, Quaternion)


CameraModel = collections.namedtuple(
    "CameraModel", ["model_id", "model_name", "num_params"])
Camera = collections.namedtuple(
    "Camera", ["id", "model", "width", "height", "params"])
BaseImage = collections.namedtuple(
    "Image", ["id", "qvec", "tvec", "camera_id", "name", "xys", "point3D_ids"])
Point3D = collections.namedtuple(
    "Point3D", ["id", "xyz", "rgb", "error", "image_ids", "point2D_idxs"])


class Image(BaseImage):
    def qvec2rotmat(self):
        return Quaternion.to_rotation_matrix(self.qvec)


CAMERA_MODELS = {
    CameraModel(model_id=0, model_name="SIMPLE_PINHOLE", num_params=3),
    CameraModel(model_id=1, model_name="PINHOLE", num_params=4),
    CameraModel(model_id=2, model_name="SIMPLE_RADIAL", num_params=4),
    CameraModel(model_id=3, model_name="RADIAL", num_params=5),
    CameraModel(model_id=4, model_name="OPENCV", num_params=8),
    CameraModel(model_id=5, model_name="OPENCV_FISHEYE", num_params=8),
    CameraModel(model_id=6, model_name="FULL_OPENCV", num_params=12),
    CameraModel(model_id=7, model_name="FOV", num_params=5),
    CameraModel(model_id=8, model_name="SIMPLE_RADIAL_FISHEYE", num_params=4),
    CameraModel(model_id=9, model_name="RADIAL_FISHEYE", num_params=5),
    CameraModel(model_id=10, model_name="THIN_PRISM_FISHEYE", num_params=12)
}
CAMERA_MODEL_IDS = dict([(camera_model.model_id, camera_model) \
                         for camera_model in CAMERA_MODELS])


def read_next_bytes(fid, num_bytes, format_char_sequence, endian_character="<"):
    """Read and unpack the next bytes from a binary file.
    :param fid:
    :param num_bytes: Sum of combination of {2, 4, 8}, e.g. 2, 6, 16, 30, etc.
    :param format_char_sequence: List of {c, e, f, d, h, H, i, I, l, L, q, Q}.
    :param endian_character: Any of {@, =, <, >, !}
    :return: Tuple of read and unpacked values.
    """
    data = fid.read(num_bytes)
    return struct.unpack(endian_character + format_char_sequence, data)


class SfMReader(object):
    def __init__(self, dataset_path, sfm_folder="", data_type="COLMAP") -> None:
        """
        :param dataset_path: absolute path of the dataset
        :param sfm_folder: folder which saves the sfm result
        :param data_type: determines whether loading 'COLMAP' format or 'DAGSFM' format
        """
        
        self.dataset_path = dataset_path
        self.data_type = data_type # ["COLMAP", "DAGSFM"]
        self.image_dir = os.path.join(dataset_path)
        self.sfm_dir = os.path.join(dataset_path, sfm_folder)
        self.sfm_model_dir = os.path.join(self.sfm_dir, '0')
        self.block_file_path = os.path.join(self.sfm_dir, 'partition_info.txt')

    def read_model(self, ext=".bin"):
        bounding_boxes = None
        image_blocks = None

        if ext == ".txt":
            cameras = self.read_cameras_text(os.path.join(self.sfm_model_dir, "cameras" + ext))
            images = self.read_images_text(os.path.join(self.sfm_model_dir, "images" + ext))
            points3D = self.read_points3D_text(os.path.join(self.sfm_model_dir, "points3D") + ext)
            if self.data_type == "DAGSFM":
                bounding_boxes = self.read_bounding_boxes_text(
                    os.path.join(self.sfm_model_dir, "bounding_boxes") + ext)
                image_blocks = self.read_block_info(self.block_file_path)
        else:
            cameras = self.read_cameras_binary(os.path.join(self.sfm_model_dir, "cameras" + ext))
            images = self.read_images_binary(os.path.join(self.sfm_model_dir, "images" + ext))
            points3D = self.read_points3d_binary(os.path.join(self.sfm_model_dir, "points3D") + ext)
            if self.data_type == "DAGSFM":
                bounding_boxes = self.read_bounding_boxes_binary(
                    os.path.join(self.sfm_model_dir, "bounding_boxes") + ext)
                image_blocks = self.read_block_info(self.block_file_path)
        
        return cameras, images, points3D, bounding_boxes, image_blocks

    def read_images_text(self, path) -> dict:
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadImagesText(const std::string& path)
            void Reconstruction::WriteImagesText(const std::string& path)
        """
        images = {}
        block_offset = 0 if self.data_type == 'COLMAP' else 1
        with open(path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    image_id = int(elems[0])
                    # cluster_id = int(elems[1])
                    qvec = np.array(tuple(map(float, elems[1+block_offset:5+block_offset])))
                    tvec = np.array(tuple(map(float, elems[5+block_offset:8+block_offset])))
                    camera_id = int(elems[8+block_offset])
                    image_name = elems[9+block_offset]
                    elems = fid.readline().split()
                    xys = np.column_stack([tuple(map(float, elems[0::3])),
                                           tuple(map(float, elems[1::3]))])
                    point3D_ids = np.array(tuple(map(int, elems[2::3])))
                    images[image_id] = Image(
                        id=image_id, qvec=qvec, tvec=tvec,
                        camera_id=camera_id, name=image_name,
                        xys=xys, point3D_ids=point3D_ids)
        return images

    def read_images_binary(self, path_to_model_file) -> dict:
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadImagesBinary(const std::string& path)
            void Reconstruction::WriteImagesBinary(const std::string& path)
        """
        images = {}
        block_offset = 0 if self.data_type == 'COLMAP' else 1
        with open(path_to_model_file, "rb") as fid:
            num_reg_images = read_next_bytes(fid, 8, "Q")[0]
            for image_index in range(num_reg_images):
                if self.data_type == 'DAGSFM':
                    binary_image_properties = read_next_bytes(
                        fid, num_bytes=72, format_char_sequence="iddddddddi")
                else:
                    binary_image_properties = read_next_bytes(
                        fid, num_bytes=64, format_char_sequence="idddddddi")
                
                image_id = binary_image_properties[0]
                # cluster_id = binary_image_properties[1]
                qvec = np.array(binary_image_properties[1+block_offset:5+block_offset])
                tvec = np.array(binary_image_properties[5+block_offset:8+block_offset])
                camera_id = binary_image_properties[8+block_offset]
                image_name = ""
                current_char = read_next_bytes(fid, 1, "c")[0]
                while current_char != b"\x00":   # look for the ASCII 0 entry
                    image_name += current_char.decode("utf-8")
                    current_char = read_next_bytes(fid, 1, "c")[0]
                num_points2D = read_next_bytes(fid, num_bytes=8,
                                               format_char_sequence="Q")[0]
                x_y_id_s = read_next_bytes(fid, num_bytes=24*num_points2D,
                                           format_char_sequence="ddq"*num_points2D)
                xys = np.column_stack([tuple(map(float, x_y_id_s[0::3])),
                                       tuple(map(float, x_y_id_s[1::3]))])
                point3D_ids = np.array(tuple(map(int, x_y_id_s[2::3])))
                images[image_id] = Image(
                    id=image_id, qvec=qvec, tvec=tvec,
                    camera_id=camera_id, name=image_name,
                    xys=xys, point3D_ids=point3D_ids)
        return images

    def read_cameras_text(self, path):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::WriteCamerasText(const std::string& path)
            void Reconstruction::ReadCamerasText(const std::string& path)
        """
        cameras = {}
        with open(path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    camera_id = int(elems[0])
                    model = elems[1]
                    width = int(elems[2])
                    height = int(elems[3])
                    params = np.array(tuple(map(float, elems[4:])))
                    cameras[camera_id] = Camera(id=camera_id, model=model,
                                                width=width, height=height,
                                                params=params)
        return cameras

    def read_cameras_binary(self, path_to_model_file):
        """
        see: src/base/reconstruction.cc
            void Reconstruction::WriteCamerasBinary(const std::string& path)
            void Reconstruction::ReadCamerasBinary(const std::string& path)
        """
        cameras = {}
        with open(path_to_model_file, "rb") as fid:
            num_cameras = read_next_bytes(fid, 8, "Q")[0]
            for camera_line_index in range(num_cameras):
                camera_properties = read_next_bytes(
                    fid, num_bytes=24, format_char_sequence="iiQQ")
                camera_id = camera_properties[0]
                model_id = camera_properties[1]
                model_name = CAMERA_MODEL_IDS[camera_properties[1]].model_name
                width = camera_properties[2]
                height = camera_properties[3]
                num_params = CAMERA_MODEL_IDS[model_id].num_params
                params = read_next_bytes(fid, num_bytes=8*num_params,
                                         format_char_sequence="d"*num_params)
                cameras[camera_id] = Camera(id=camera_id,
                                            model=model_name,
                                            width=width,
                                            height=height,
                                            params=np.array(params))
            assert len(cameras) == num_cameras
        return cameras

    def read_points3D_text(self, path) -> dict:
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DText(const std::string& path)
            void Reconstruction::WritePoints3DText(const std::string& path)
        """
        points3D = {}
        with open(path, "r") as fid:
            while True:
                line = fid.readline()
                if not line:
                    break
                line = line.strip()
                if len(line) > 0 and line[0] != "#":
                    elems = line.split()
                    point3D_id = int(elems[0])
                    xyz = np.array(tuple(map(float, elems[1:4])))
                    rgb = np.array(tuple(map(int, elems[4:7])))
                    error = float(elems[7])
                    image_ids = np.array(tuple(map(int, elems[8::2])))
                    point2D_idxs = np.array(tuple(map(int, elems[9::2])))
                    points3D[point3D_id] = Point3D(id=point3D_id, xyz=xyz, rgb=rgb,
                                                   error=error, image_ids=image_ids,
                                                   point2D_idxs=point2D_idxs)
        return points3D

    def read_points3d_binary(self, path_to_model_file) -> dict:
        """
        see: src/base/reconstruction.cc
            void Reconstruction::ReadPoints3DBinary(const std::string& path)
            void Reconstruction::WritePoints3DBinary(const std::string& path)
        """
        points3D = {}
        with open(path_to_model_file, "rb") as fid:
            num_points = read_next_bytes(fid, 8, "Q")[0]
            for point_line_index in range(num_points):
                binary_point_line_properties = read_next_bytes(
                    fid, num_bytes=43, format_char_sequence="QdddBBBd")
                point3D_id = binary_point_line_properties[0]
                xyz = np.array(binary_point_line_properties[1:4])
                rgb = np.array(binary_point_line_properties[4:7])
                error = np.array(binary_point_line_properties[7])
                track_length = read_next_bytes(
                    fid, num_bytes=8, format_char_sequence="Q")[0]
                track_elems = read_next_bytes(
                    fid, num_bytes=8*track_length,
                    format_char_sequence="ii"*track_length)
                image_ids = np.array(tuple(map(int, track_elems[0::2])))
                point2D_idxs = np.array(tuple(map(int, track_elems[1::2])))
                points3D[point3D_id] = Point3D(
                    id=point3D_id, xyz=xyz, rgb=rgb,
                    error=error, image_ids=image_ids,
                    point2D_idxs=point2D_idxs)
        return points3D

    def read_bounding_boxes_text(self, path) -> list:
        with open(path, "r") as fid:
            line = fid.readline().strip()
            reference_image_id = int(line) # never used.
            line = fid.readline().strip()
            num_bounding_boxes = int(line)
            bounding_boxes = [None] * num_bounding_boxes
            for i in range(num_bounding_boxes):
                line = fid.readline().strip()
                elems = line.split()
                bbox = []
                for elem in elems:
                    bbox.append(float(elem))
                bounding_boxes[i] = bbox

        return bounding_boxes

    def read_bounding_boxes_binary(self, path) -> list:
        with open(path, "rb") as fid:
            reference_image_id = read_next_bytes(fid, 8, "Q")[0]
            num_bounding_boxes = read_next_bytes(fid, 8, "Q")[0]
            bounding_boxes = [None] * num_bounding_boxes
            for i in range(num_bounding_boxes):
                bbox_coordinate = read_next_bytes(fid, 48, "dddddd")
                bbox = []
                for j in range(6):
                    bbox.append(bbox_coordinate[j])
                bounding_boxes[i] = bbox
        
        return bounding_boxes

    def read_block_info(self, path) -> list:
        block_file = open(path, "r")
        line = block_file.readline()
        num_blocks = int(line[0])
        image_blocks = [[]] * num_blocks
        line = block_file.readline()

        while line:
            data = line.split(' ')
            image_name, block_id, image_dir = data[0], int(data[1]), data[2].strip()
            item = [image_name, block_id, image_dir]
            image_blocks[block_id].append(item)
            line = block_file.readline()

        block_file.close()

        return image_blocks


if __name__ == "__main__":
    sfm_reader = SfMReader(dataset_path='/home/chenyu/Datasets/mipnerf_360/bonsai',
                           sfm_folder='colmap_output',
                           data_type='DAGSFM')
    cameras, images, points3D, bounding_boxes, image_blocks = sfm_reader.read_model('.bin')
    print(f'Num of cameras: {len(cameras)}')
    print(f'Num of images: {len(images)}')
    print(f'Num of points: {len(points3D)}')
    print(f'Num of bounding boxes: {len(bounding_boxes)}')
    print(f'Num of blocks: {len(image_blocks)}')
