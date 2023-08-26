"""
Adapted from `360_view.py` & `360_view_test.py` in the original NeRF synthetic
Blender dataset blend-files and objaverse:
https://github.com/allenai/objaverse-rendering/blob/970731404ae2dd091bb36150e04c4bd6ff59f0a0/scripts/blender_script.py#L151
"""
import argparse
import os
import json
from math import radians
import bpy
import numpy as np
import math
import random
from multiprocessing import Pool

from mathutils import Vector


COLOR_SPACES = [ "display", "linear" ]
DEVICES = [ "cpu", "cuda", "optix" ]

CIRCLE_FIXED_START = ( 0, 0, 0 )
CIRCLE_FIXED_END = ( .7, 0, 0 )


def add_lighting() -> None:
    # delete the default light
    if "Light" in bpy.data.objects.keys():
        # return
        bpy.data.objects["Light"].select_set(True)
        bpy.ops.object.delete()
    
    # bpy.data.objects["Light"].select_set(True)
    # bpy.ops.object.delete()
    # add a new light
    bpy.ops.object.light_add(type="AREA")
    light2 = bpy.data.lights["Area"]
    light2.energy = 30000
    bpy.data.objects["Area"].location[2] = 0.5
    bpy.data.objects["Area"].scale[0] = 120
    bpy.data.objects["Area"].scale[1] = 120
    bpy.data.objects["Area"].scale[2] = 120


def reset_scene() -> None:
    """Resets the scene to a clean state."""
    # delete everything that isn't part of a camera or a light
    for obj in bpy.data.objects:
        if obj.type not in {"CAMERA", "LIGHT"}:
            bpy.data.objects.remove(obj, do_unlink=True)
    # delete all the materials
    for material in bpy.data.materials:
        bpy.data.materials.remove(material, do_unlink=True)
    # delete all the textures
    for texture in bpy.data.textures:
        bpy.data.textures.remove(texture, do_unlink=True)
    # delete all the images
    for image in bpy.data.images:
        bpy.data.images.remove(image, do_unlink=True)


def scene_root_objects():
    for obj in bpy.context.scene.objects.values():
        if not obj.parent:
            yield obj


def scene_meshes():
    for obj in bpy.context.scene.objects.values():
        if isinstance(obj.data, (bpy.types.Mesh)):
            yield obj


def scene_bbox(single_obj=None, ignore_matrix=False):
    bbox_min = (math.inf,) * 3
    bbox_max = (-math.inf,) * 3
    found = False
    for obj in scene_meshes() if single_obj is None else [single_obj]:
        found = True
        for coord in obj.bound_box:
            coord = Vector(coord)
            if not ignore_matrix:
                coord = obj.matrix_world @ coord
            bbox_min = tuple(min(x, y) for x, y in zip(bbox_min, coord))
            bbox_max = tuple(max(x, y) for x, y in zip(bbox_max, coord))
    if not found:
        raise RuntimeError("no objects in scene to compute bounding box for")
    return Vector(bbox_min), Vector(bbox_max)


def normalize_scene():
    bbox_min, bbox_max = scene_bbox()
    # scale = 1 / max(bbox_max - bbox_min)
    scale = 2.5 / max(bbox_max - bbox_min)
    for obj in scene_root_objects():
        obj.scale = obj.scale * scale
    # Apply scale to matrix_world.
    bpy.context.view_layer.update()
    bbox_min, bbox_max = scene_bbox()
    offset = -(bbox_min + bbox_max) / 2
    for obj in scene_root_objects():
        obj.matrix_world.translation += offset
    bpy.ops.object.select_all(action="DESELECT")


def listify_matrix(matrix):
    matrix_list = []
    for row in matrix:
        matrix_list.append(list(row))
    return matrix_list


def parent_obj_to_camera(b_camera):
    origin = (0, 0, 0)
    b_empty = bpy.data.objects.new("Empty", None)
    b_empty.location = origin
    b_camera.parent = b_empty  # setup parenting

    scn = bpy.context.scene
    scn.collection.objects.link(b_empty)
    bpy.context.view_layer.objects.active = b_empty

    return b_empty


def main(args, blend_path, blend_name):
    # open the scene blend-file
    print('open file')
    reset_scene()
    # bpy.ops.wm.open_mainfile(filepath=args.blend_path)
    bpy.ops.import_scene.gltf(filepath=blend_path, merge_vertices=True)
    print('file opened!')

    # initialize render settings
    # scene = bpy.data.scenes["Scene"]
    scene = bpy.context.scene
    scene.render.engine = "CYCLES"
    scene.render.use_persistent_data = True

    if args.device == "cpu":
        bpy.context.preferences.addons['cycles'].preferences \
           .compute_device_type = "NONE"
        bpy.context.scene.cycles.device = "CPU"
    elif args.device == "cuda":
        bpy.context.preferences.addons['cycles'].preferences \
           .compute_device_type = "CUDA"
        bpy.context.scene.cycles.device = "GPU"
    elif args.device == "optix":
        bpy.context.preferences.addons['cycles'].preferences \
           .compute_device_type = "OPTIX"
        bpy.context.scene.cycles.device = "GPU"
    bpy.context.preferences.addons['cycles'].preferences.get_devices()

    # initialize compositing nodes
    scene.view_layers[0].use_pass_combined = True
    scene.use_nodes = True

    # initialize RGB render image output settings
    # scene.render.filepath = args.renders_path
    scene.render.filepath =  os.path.join(args.output_path, blend_name, 'train')
    image_output_path =  os.path.join(args.output_path, blend_name, 'train')
    print(f'filename: {scene.render.filepath}')
    scene.render.use_file_extension = True
    scene.render.use_overwrite = True
    scene.render.image_settings.color_mode = 'RGBA'

    if args.color_space == "display":
        scene.render.image_settings.file_format = "PNG"
        scene.render.image_settings.color_depth = "8"
        scene.render.image_settings.color_management = "FOLLOW_SCENE"
    elif args.color_space == "linear":
        scene.render.image_settings.file_format = "OPEN_EXR"
        scene.render.image_settings.color_depth = "32"
        scene.render.image_settings.use_zbuffer = False

    # initialize camera settings
    scene.render.dither_intensity = 0.0
    scene.render.film_transparent = True
    scene.render.resolution_percentage = 100
    scene.render.resolution_x = args.resolution[0]
    scene.render.resolution_y = args.resolution[1]

    normalize_scene()
    add_lighting()

    cam = bpy.data.objects["Camera"]
    cam.location = (0, 4.0, 0.5)
    cam.rotation_mode = "XYZ"
    cam_constraint = cam.constraints.new(type="TRACK_TO")
    cam_constraint.track_axis = "TRACK_NEGATIVE_Z"
    cam_constraint.up_axis = "UP_Y"
    b_empty = parent_obj_to_camera(cam)
    cam_constraint.target = b_empty

    # preprocess & derive paths
    args.output_path = os.path.normpath(args.output_path)  # remove trailing slashes
    folder_name = os.path.basename(args.output_path)
    renders_parent_path = os.path.join(args.output_path, blend_name) # os.path.dirname(args.output_path)  
    transforms_path = os.path.join(
        # renders_parent_path, f"transforms_{folder_name}.json"
        args.output_path, blend_name, f"transforms.json"
    )

    # render novel views
    stepsize = 360.0 / args.num_views
    if not args.random_views:
        vertical_diff = CIRCLE_FIXED_END[0] - CIRCLE_FIXED_START[0]
        b_empty.rotation_euler = CIRCLE_FIXED_START
        b_empty.rotation_euler[0] = CIRCLE_FIXED_START[0] + vertical_diff

    out_data = {
        "camera_angle_x": cam.data.angle_x,
        'frames': []
    }
    print(f'rendering')
    for i in range(0, args.num_views):
        if args.random_views:
            if args.upper_views:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
        else:
            print("Rotation {}, {}".format((stepsize * i), radians(stepsize * i)))
            
        # scene.render.filepath = os.path.join(args.output_path, f"r_{i}")
        scene.render.filepath = os.path.join(image_output_path, f"r_{i}")
        bpy.ops.render.render(write_still=True)

        frame_data = {
            "file_path": os.path.join(".", os.path.relpath(
                            scene.render.filepath, start=renders_parent_path
                         )),
            "rotation": radians(stepsize),
            "transform_matrix": listify_matrix(cam.matrix_world)
        }
        print(f'framedata: {frame_data["file_path"]}')
        out_data["frames"].append(frame_data)

        if args.random_views:
            if args.upper_views:
                rot = np.random.uniform(0, 1, size=3) * (1,0,2*np.pi)
                rot[0] = np.abs(np.arccos(1 - 2 * rot[0]) - np.pi/2)
                b_empty.rotation_euler = rot
            else:
                b_empty.rotation_euler = np.random.uniform(0, 2*np.pi, size=3)
        else:
            b_empty.rotation_euler[0] = (
                CIRCLE_FIXED_START[0]
                + (np.cos(radians(stepsize*i))+1)/2 * vertical_diff
            )
            b_empty.rotation_euler[2] += radians(2*stepsize)

    with open(transforms_path, "w") as out_file:
        json.dump(out_data, out_file, indent=4)
    print(f'Write transformation into {transforms_path}')

    bbox_min, bbox_max = scene_bbox()
    bbox_filepath = os.path.join(args.output_path, blend_name, 'bbox.txt')
    with open(bbox_filepath, "w") as f:
        f.write(f"{bbox_min[0]} {bbox_min[1]} {bbox_min[2]} {bbox_max[0]} {bbox_max[1]} {bbox_max[2]}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description=("Script for rendering novel views of"
                     " synthetic Blender scenes.")
    )
    parser.add_argument(
        "--json_path", type=str,
        help="Path to the blend-file of the synthetic Blender scene."
    )
    parser.add_argument(
        "--output_path", type=str,
        help="Desired path to the novel view renders."
    )
    parser.add_argument(
        "--num_views", type=int,
        help="Number of novel view renders."
    )
    parser.add_argument(
        "--resolution", type=int, nargs=2,
        help="Image resolution of the novel view renders."
    )
    parser.add_argument(
        "--color_space", type=str, choices=COLOR_SPACES, default="display",
        help="Color space of the output novel view images."
    )
    parser.add_argument(
        "--device", type=str, choices=DEVICES, default="cpu",
        help="Compute device type for rendering."
    )
    parser.add_argument(
        "--random_views", action="store_true",
        help="Randomly sample novel views."
    )
    parser.add_argument(
        "--upper_views", action="store_true",
        help="Only sample novel views from the upper hemisphere."
    )
    args = parser.parse_args()
    args.upper_views = True if random.randint(0, 2) == 1 else False

    json_path = args.json_path
    with open(json_path, "r") as fp:
        obj_name_to_filepath = json.load(fp)
    
    num_process = 6
    process_pool = Pool(num_process)
    obj_names = list(obj_name_to_filepath.keys())

    i = 0
    while i < len(obj_names):
        k = 0
        while k < num_process:
            if i + k >= len(obj_names):
                break
            obj_name = obj_names[i + k]
            obj_path = obj_name_to_filepath[obj_name]
            tmp_path = os.path.join(os.path.normpath(args.output_path), obj_name, "transforms.json")
            transforms_path = os.path.join(args.output_path, obj_name, "transforms.json")
            if not os.path.exists(tmp_path) and not os.path.exists(transforms_path):
                process_pool.apply_async(main, args=(args, obj_path, obj_name))
                k += 1
            else:
                i += 1
        process_pool.close()
        process_pool.join()
        i += num_process
        process_pool = Pool(num_process)
