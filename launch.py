from imports import bag_to_numpy
from visualization import visualize_animation
from analysis import (
    get_cluster_centers_per_frame,
    filter_clusters_1,
    # select_N_brightest_per_frame,
    # select_with_threshold,
)
from filter_clusters_2 import filter_clusters_2
import numpy as np
from pprint import pprint
from tkinter_ui import create_gui

# MAIN
# TODO make filenames/topics configurable via console/UI
filename1 = (
    "/home/max/UNI/Job_IRT/LIDAR/temporal_reflector/temporal_reflector_disturbance.bag",
    "rslidar_points_ref",  # topic
)
filename2 = (
    "/home/max/UNI/Job_IRT/LIDAR/Calibration_Target_Deneb/calibration_target_1.bag",
    "/rslidar_points_vr_v",  # topic
)
# get topics using `rosbag info filename.bag`


frames = bag_to_numpy(*filename2)


def calc(params):
    print("calculating using new params")
    centers, visualization = get_cluster_centers_per_frame(
        frames,
        rel_intensity_threshold=params["relative intensity threshold"],
        DBSCAN_epsilon=params["DBSCAN epsilon"],
        DBSCAN_min_samples=int(params["DBSCAN min samples"]),
    )

    selection_indices = filter_clusters_2(
        centers,
        max_distance=params["maximum neighbor distance"],
        min_velocity=params["minimum velocity"],
        velocity_lookahead=int(params["velocity lookahead"]),
    )
    chosen_centers = []
    for frame_i in range(len(selection_indices)):
        if selection_indices[frame_i] is None:
            chosen_centers.append(None)
        else:
            chosen_centers.append(centers[frame_i][selection_indices[frame_i]])
        # currently, visualization contains indices of clusters
        # for o3d visualization, we want to convert this to codes meaning "any cluster" or "chosen cluster"
        chosen_cluster_selection = (
            visualization[frame_i, :, 3] == selection_indices[frame_i]
        )
        any_cluster_selection = visualization[frame_i, :, 3] >= 0  # any cluster
        visualization[frame_i, any_cluster_selection, 3] = 1
        visualization[frame_i, chosen_cluster_selection, 3] = 2

    print("showing open3d visualization, this will block the settings UI")
    print("press escape to close 3d view, then enter new values")
    visualize_animation(visualization, chosen_centers)
    print("returning to settings UI")


create_gui(
    params={
        "relative intensity threshold": 0.7,
        "DBSCAN epsilon": 0.4,
        "DBSCAN min samples": 9,
        "maximum neighbor distance": 0.4,
        "minimum velocity": 0.2,
        "velocity lookahead": 8,
    },
    callback=calc,
)
