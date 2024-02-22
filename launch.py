from imports import bag_to_numpy
from visualization import visualize_animation

filename = (
    "/home/max/UNI/Job_IRT/LIDAR/temporal_reflector/temporal_reflector_disturbance.bag"
)
# filename = "labor_lidar.bag"
frames = bag_to_numpy(filename)
# visualize_frame(frames[0])
visualize_animation(frames)
