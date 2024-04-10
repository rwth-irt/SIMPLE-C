from imports import bag_to_numpy
from visualization import visualize_animation
from analysis import get_cluster_centers_per_frame, filter_clusters_1
from filter_clusters_2 import filter_clusters_2
import numpy as np
from pprint import pprint

# MAIN
filename = (
    "/home/max/UNI/Job_IRT/LIDAR/temporal_reflector/temporal_reflector_disturbance.bag"
)
# filename = "labor_lidar.bag"
frames = bag_to_numpy(filename)
centers, visualization = get_cluster_centers_per_frame(frames)

# chosen_centers = filter_clusters_1(centers, max_distance=0.3)
chosen_centers = filter_clusters_2(centers, max_distance=0.15, min_velocity=0.08)

visualize_animation(visualization, chosen_centers)
