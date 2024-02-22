from imports import bag_to_numpy
from visualization import visualize_animation
import numpy as np
from sklearn.cluster import DBSCAN


filename = (
    "/home/max/UNI/Job_IRT/LIDAR/temporal_reflector/temporal_reflector_disturbance.bag"
)
# filename = "labor_lidar.bag"
frames = bag_to_numpy(filename)


def process(frames):
    # per frame, get the indices of the N points with highest intensity
    N = 30 # TODO params
    brightest_indices = np.argsort(-frames[:, :, 3], axis=1)[:, :N]

    # output classes
    frames[:, :, 3] = 0  # 0: standard
    for i in range(len(frames)):
        frames[i, brightest_indices[i], 3] = 1 # 1: N with highest intensity

    # get selected points in extra array
    brightest_points = np.zeros((len(frames), N, 4))
    for i, (sel, frame) in enumerate(zip(brightest_indices, frames)):
        brightest_points[i] = frame[sel]
        # don't know how to do this with one of those billion np functions, select, take, take_along_axis
    
    for i in range(len(frames)):
        # perform DBSCAN per frame
        # eps is in meter, as lidar output is
        clusterlabels = DBSCAN(eps=.1, min_samples=4).fit_predict(brightest_points[i, :, :3]) # TODO params!
        values, counts = np.unique(clusterlabels, return_counts=True)
        biggest_cluster = values[np.argmax(counts)]
        if biggest_cluster >= 0:
            # mark points for visualization
            frames[i, brightest_indices[i, clusterlabels == biggest_cluster], 3] = 2

    # return out


process(frames)
visualize_animation(frames)
