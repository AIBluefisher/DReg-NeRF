"""
Copyright (c) 2022 Ruilong Li, UC Berkeley.
"""

import os
import collections
import trimesh

import numpy as np


Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


def namedtuple_map(fn, tup):
    """Apply `fn` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else fn(x) for x in tup))


def visualize_poses(poses, size=0.1):
    """
    Visualize camera poses in the axis-aligned bounding box, which
    can be utilized to tune the aabb size.
    """
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(bounds=[[-2, -2, -2], [2, 2, 2]]).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)
        o = pos + dir * 3

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def visualize_block_poses(poses, labels, size=0.1):
    """
    Visualize camera poses in the axis-aligned bounding box, which
    can be utilized to tune the aabb size.
    """
    # poses: [B, 4, 4]
    colors = [[255, 0, 0, 255], [0, 255, 0, 255], [0, 0, 255, 255],
              [255, 255, 0, 255], [255, 0, 255, 255], [0, 255, 255, 255]]
    axes = trimesh.creation.axis(axis_length=4)
    box = trimesh.primitives.Box(bounds=[[-2, -2, -2], [2, 2, 2]]).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for i, pose in enumerate(poses):
        label = labels[i]

        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        dir = (a + b + c + d) / 4 - pos
        dir = dir / (np.linalg.norm(dir) + 1e-8)

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs) # type: trimesh.path.path.Path3D
        for i in range(len(segs.entities)):
            segs.entities[i].color = colors[label]
        objects.append(segs)

    trimesh.Scene(objects).show()


def minify(basedir, factors=[], resolutions=[]):
    need_to_load = False
    
    for r in factors:
        image_dir = os.path.join(basedir, 'images_{}'.format(r))
        if not os.path.exists(image_dir):
            need_to_load = True
    
    for r in resolutions:
        image_dir = os.path.join(basedir, 'images_{}x{}'.format(r[1], r[0]))
        if not os.path.exists(image_dir):
            need_to_load = True
    
    if not need_to_load:
        return

    from subprocess import check_output

    image_dir = os.path.join(basedir, 'images')
    imgs = [os.path.join(image_dir, f) for f in sorted(os.listdir(image_dir))]
    imgs = [f for f in imgs if any([f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']])]
    image_dir_orig = image_dir

    wd = os.getcwd()

    for r in factors + resolutions:
        if isinstance(r, int):
            name = 'images_{}'.format(r)
            resize_arg = '{}%'.format(100. / r)
        else:
            name = 'images_{}x{}'.format(r[1], r[0])
            resize_arg = '{}x{}'.format(r[1], r[0])
        image_dir = os.path.join(basedir, name)
        if os.path.exists(image_dir):
            continue

        print('Minifying', r, basedir)

        os.makedirs(image_dir)
        check_output('cp {}/* {}'.format(image_dir_orig, image_dir), shell=True)

        ext = imgs[0].split('.')[-1]
        args = ' '.join(['mogrify', '-resize', resize_arg, '-format', 'png', '*.{}'.format(ext)])
        print(args)
        os.chdir(image_dir)
        check_output(args, shell=True)
        os.chdir(wd)

        if ext != 'png':
            check_output('rm {}/*.{}'.format(image_dir, ext), shell=True)
            print('Removed duplicates')
        print('Done')
