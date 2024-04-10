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

# MAIN
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

centers, visualization = get_cluster_centers_per_frame(
    frames, rel_intensity_threshold=0.7, DBSCAN_epsilon=0.3, DBSCAN_min_samples=4
)

# chosen_centers = filter_clusters_1(centers, max_distance=0.3)
chosen_centers = filter_clusters_2(
    centers, max_distance=0.3, min_velocity=0.15, velocity_lookahead=5
)

visualize_animation(visualization, chosen_centers)
