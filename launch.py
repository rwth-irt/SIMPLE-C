from imports import bag_to_numpy
from visualization import visualize_animation
from analysis import get_cluster_centers_per_frame, filter_clusters_1
import numpy as np
from pprint import pprint

# MAIN
filename = (
    "/home/max/UNI/Job_IRT/LIDAR/temporal_reflector/temporal_reflector_disturbance.bag"
)
# filename = "labor_lidar.bag"
frames = bag_to_numpy(filename)
centers, visualization = get_cluster_centers_per_frame(frames)
chosen_centers = filter_clusters_1(centers)

visualize_animation(visualization, chosen_centers)