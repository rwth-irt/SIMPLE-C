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
filename = (
    "/home/max/UNI/Job_IRT/LIDAR/temporal_reflector/temporal_reflector_disturbance.bag"
)
# filename = "labor_lidar.bag"
frames = bag_to_numpy(filename)

centers, visualization = get_cluster_centers_per_frame(
    frames, rel_intensity_threshold=0.5, DBSCAN_epsilon=0.3, DBSCAN_min_samples=4
)

# chosen_centers = filter_clusters_1(centers, max_distance=0.3)
chosen_centers = filter_clusters_2(
    centers, max_distance=0.3, min_velocity=0.15, velocity_lookahead=5
)

visualize_animation(visualization, chosen_centers)
