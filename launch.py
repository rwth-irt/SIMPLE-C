from imports import bag_to_numpy
from visualization import visualize_animation
import numpy as np

filename = (
    "/home/max/UNI/Job_IRT/LIDAR/temporal_reflector/temporal_reflector_disturbance.bag"
)
# filename = "labor_lidar.bag"
frames = bag_to_numpy(filename)


def process(frames):
    # per frame, get the indices of the N points with highest intensity
    N = 6
    out = np.zeros((len(frames), N, 4))
    selection = np.argsort(-frames[:, :, 3], axis=1)[:, :N]
    for i, (sel, frame) in enumerate(zip(selection, frames)):
        out[i] = frame[sel]
        # don't know how to do this with one of those billion np functions, select, take, take_along_axis
    return out


# visualize_frame(frames[0])
visualize_animation([process(frames), frames])
