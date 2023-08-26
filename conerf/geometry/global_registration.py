# http://www.open3d.org/docs/0.16.0/tutorial/pipelines/global_registration.html?highlight=registration

import os
import copy
import time

import open3d as o3d
import numpy as np


def draw_registration_result(source, target, transformation):
    source_temp = copy.deepcopy(source)
    target_temp = copy.deepcopy(target)
    source_temp.paint_uniform_color([1, 0.706, 0])
    target_temp.paint_uniform_color([0, 0.651, 0.929])
    source_temp.transform(transformation)
    o3d.visualization.draw_geometries([source_temp, target_temp])


def preprocess_point_cloud(pcd, voxel_size = 0.05):
    # print(":: Downsample with a voxel size %.3f." % voxel_size)
    pcd_down = pcd.voxel_down_sample(voxel_size)

    radius_normal = voxel_size * 2
    # print(":: Estimate normal with search radius %.3f." % radius_normal)
    pcd_down.estimate_normals(
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_normal, max_nn=30))

    radius_feature = voxel_size * 5
    # print(":: Compute FPFH feature with search radius %.3f." % radius_feature)
    pcd_fpfh = o3d.pipelines.registration.compute_fpfh_feature(
        pcd_down,
        o3d.geometry.KDTreeSearchParamHybrid(radius=radius_feature, max_nn=100))
    return pcd_down, pcd_fpfh


def prepare_dataset(source_ply_file, target_ply_file, voxel_size = 0.05):
    source = o3d.io.read_point_cloud(source_ply_file)
    target = o3d.io.read_point_cloud(target_ply_file)

    source_down, source_fpfh = preprocess_point_cloud(source, voxel_size)
    target_down, target_fpfh = preprocess_point_cloud(target, voxel_size)
    return source, target, source_down, target_down, source_fpfh, target_fpfh


def execute_global_registration(source_down, target_down, source_fpfh,
                                target_fpfh, voxel_size = 0.05):
    distance_threshold = voxel_size * 1.5
    # print(":: RANSAC registration on downsampled point clouds.")
    # print("   Since the downsampling voxel size is %.3f," % voxel_size)
    # print("   we use a liberal distance threshold %.3f." % distance_threshold)
    start = time.time()
    result = o3d.pipelines.registration.registration_ransac_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh, True,
        distance_threshold,
        o3d.pipelines.registration.TransformationEstimationPointToPoint(False),
        3, [
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnEdgeLength(
                0.9),
            o3d.pipelines.registration.CorrespondenceCheckerBasedOnDistance(
                distance_threshold)
        ], o3d.pipelines.registration.RANSACConvergenceCriteria(100000, 0.999))
    end = time.time()
    time_took = end - start

    return result.transformation, time_took


def execute_fast_global_registration(source_down, target_down, source_fpfh,
                                     target_fpfh, voxel_size = 0.05):
    distance_threshold = voxel_size * 10 # 0.5
    # print(":: Apply fast global registration with distance threshold %.3f" \
    #         % distance_threshold)
    start = time.time()
    result = o3d.pipelines.registration.registration_fgr_based_on_feature_matching(
        source_down, target_down, source_fpfh, target_fpfh,
        o3d.pipelines.registration.FastGlobalRegistrationOption(
            maximum_correspondence_distance=distance_threshold))
    end = time.time()
    time_took = end - start

    return result.transformation, time_took


def refine_registration(source, target, init_transformation, voxel_size = 0.05):
    distance_threshold = voxel_size * 0.4
    print(":: Point-to-plane ICP registration is applied on original point")
    print("   clouds to refine the alignment. This time we use a strict")
    print("   distance threshold %.3f." % distance_threshold)
    result = o3d.pipelines.registration.registration_icp(
        source, target, distance_threshold, init_transformation,
        o3d.pipelines.registration.TransformationEstimationPointToPlane())
    return result.transformation


def run_registration(source_ply_file, target_ply_file, voxel_size = 0.05, method="fast"):
    source, target, source_down, target_down, source_fpfh, target_fpfh = \
        prepare_dataset(source_ply_file, target_ply_file, voxel_size)
    
    if method != "fast":
        transformation, time = execute_global_registration(
            source_down, target_down,
            source_fpfh, target_fpfh,
            voxel_size
        )
    else:
        transformation, time = execute_fast_global_registration(
            source_down, target_down,
            source_fpfh, target_fpfh,
            voxel_size
        )
        
    # result_icp = refine_registration(source, target, transformation, voxel_size)
    # print(result_icp)
    # draw_registration_result(source, target, result_icp.transformation)
    return transformation, time


# if __name__ == "__main__":
#     src_ply_file = os.path.join(PLY_DIR, SCENE_NAME, 'block_0/voxel_point_cloud.ply')
#     tgt_ply_file = os.path.join(PLY_DIR, SCENE_NAME, 'block_1/voxel_point_cloud.ply')

#     transformation, time_took = run_registration(src_ply_file, tgt_ply_file, method="fast")
#     print("Fast global registration took %.3f sec.\n" % time_took)
#     print(f'fast global registration result: {transformation}')

#     # voxel_size = 0.05  # means 5cm for the dataset
