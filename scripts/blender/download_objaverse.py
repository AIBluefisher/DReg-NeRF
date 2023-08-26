import os
import argparse
import multiprocessing
import json

import tqdm
import objaverse


def get_filename_from_abs_path(abs_path):
    return abs_path.split('/')[-1]


def get_filename_no_ext(filename):
    return os.path.splitext(filename)[0]


def load_obj_uids(json_file):
    selected_uids = []
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    for key, item in data.items():
        selected_uids += item
    
    return selected_uids


def download_objaverse(args):
    processes = multiprocessing.cpu_count()
    output_dir = args.output_dir

    selected_uids = load_obj_uids('../../conerf/datasets/register/obj_id_names.json')
    start_index = args.start_index
    end_index = len(selected_uids) if args.end_index == -1 else args.end_index
    selected_uids = selected_uids[start_index:end_index]

    objects = objaverse.load_objects(
        uids=selected_uids,
        download_processes=processes
    )
    annotations = objaverse.load_annotations(selected_uids)
    obj_id_to_name = {}
    obj_name_paths = {}

    objects = list(objects.values())
    pbar = tqdm.trange(len(objects), desc="Copying objects file", leave=False)
    for obj_path in objects:
        # print(obj)
        obj_filename_ext = get_filename_from_abs_path(obj_path)
        obj_filename = get_filename_no_ext(obj_filename_ext)
        # print(obj_filename)
        # print(obj_path)
        obj_name = annotations[obj_filename]['name'].replace(" ", "_").replace("#", "").replace(",", "").replace("*", "")
        obj_name = obj_name.replace(":", "").replace("\"", "").replace(".", "").replace("/", "").replace("'", "").replace("|", "")
        obj_name = obj_name + obj_filename[:2] + obj_filename[-2:] # including obj ids as there may exist duplicate objname 
        # print(f'obj_name: {obj_name}')
        obj_id_to_name[obj_filename] = obj_name
        new_obj_path = os.path.join(output_dir, obj_filename_ext)
        obj_name_paths[obj_name] = new_obj_path
        # print(f'obj_path: {obj_path}')
        # print(f'new_obj_path: {new_obj_path}')

        os.system(f'cp {obj_path} {new_obj_path}')
        pbar.update(1)

    json_file = os.path.join(output_dir, f'obj_id_names_{start_index}_{end_index}.json')
    json_obj = json.dumps(obj_id_to_name, indent=2)
    with open(json_file, 'w') as f:
        f.write(json_obj)

    json_file = os.path.join(output_dir, f'obj_name_path_{start_index}_{end_index}.json')
    json_obj = json.dumps(obj_name_paths, indent=2)
    with open(json_file, 'w') as f:
        f.write(json_obj)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description=("Script for rendering novel views of"
                     " synthetic Blender scenes.")
    )
    parser.add_argument(
        "--output_dir", type=str,
        help="Output directory to save the downloaded glb files"
    )
    parser.add_argument(
        "--start_index", type=int, default=0,
        help="start index to download the objects"
    )
    parser.add_argument(
        "--end_index", type=int, default=-1,
        help="end index to download the objects"
    )
    args = parser.parse_args()

    download_objaverse(args)
