from imports import bag_to_numpy
from visualization import visualize_animation
import numpy as np
from sklearn.cluster import DBSCAN


filename = (
    "/home/max/UNI/Job_IRT/LIDAR/temporal_reflector/temporal_reflector_disturbance.bag"
)
# filename = "labor_lidar.bag"
frames = bag_to_numpy(filename)


def get_cluster_centers_per_frame(
    frames,  # input points: frames[frame_index, point_index] = [x,y,z,intensity]
    N_highest_intensity=30,  # the N points with highest intens. used for DBSCAN clustering
    DBSCAN_epsilon=0.1,  # in meters, as is LIDAR output (TODO: is that correct?)
    DBSCAN_min_samples=4,  # obvious
):
    # per frame, get the indices of the N points with highest intensity
    brightest_indices = np.argsort(-frames[:, :, 3], axis=1)[:, :N_highest_intensity]

    # mark points for visualization, rewriting the intensity channel in `frames`
    frames[:, :, 3] = 0  # 0 -> any point
    for i in range(len(frames)):
        frames[i, brightest_indices[i], 3] = 1  # 1 -> N with highest intensity

    # get selected points in extra array
    brightest_points = np.zeros((len(frames), N_highest_intensity, 4))
    for i, (sel, frame) in enumerate(zip(brightest_indices, frames)):
        brightest_points[i] = frame[sel]
        # TODO don't know how to do this with one of those billion np functions, select, take, take_along_axis

    cluster_centers_per_frame = []
    for i in range(len(frames)):
        # perform DBSCAN per frame
        clusterlabels = DBSCAN(
            eps=DBSCAN_epsilon, min_samples=DBSCAN_min_samples
        ).fit_predict(
            brightest_points[i, :, :3]  # :3, without intensity
        )

        # get centroids of clusters and mark them for visualization
        centers = []
        for clusterlabel in np.unique(clusterlabels):
            if clusterlabel == -1:
                continue  # ignore noise
            centers.append(
                np.mean(
                    frames[
                        i, brightest_indices[i, clusterlabels == clusterlabel], :3
                    ],  # :3, without intensity
                    axis=0,
                )
            )
            # mark points as 2 -> "in cluster" for visualization
            frames[i, brightest_indices[i, clusterlabels == clusterlabel], 3] = 2
        cluster_centers_per_frame.append(centers)

    return cluster_centers_per_frame


centers = get_cluster_centers_per_frame(frames)

visualize_animation(frames)
