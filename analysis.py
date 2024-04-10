import numpy as np
from sklearn.cluster import DBSCAN
from pprint import pprint


def get_cluster_centers_per_frame(
    frames,  # input points: frames[frame_index, point_index] = [x,y,z,intensity]
    rel_intensity_threshold,
    DBSCAN_epsilon=0.15,  # in meters, as is LIDAR output (TODO: is that correct?)
    DBSCAN_min_samples=3,
):

    intensity_threshold = np.max(frames[..., 3]) * rel_intensity_threshold
    point_selection = frames[..., 3] >= intensity_threshold

    visualization = np.copy(frames)
    visualization[:, :, 3] = -2 # any point
    visualization[point_selection, 3] = -1  # 1 for selected/bright points

    cluster_centers_per_frame = []
    for i in range(len(frames)):
        frame_points = frames[i, point_selection[i]]
        # perform DBSCAN per frame
        clusterlabels = DBSCAN(
            eps=DBSCAN_epsilon, min_samples=DBSCAN_min_samples
        ).fit_predict(frame_points[:, :3])

        # get centroids of clusters and mark them for visualization
        centers = []
        for clusterlabel in np.unique(clusterlabels):
            if clusterlabel == -1:
                continue  # ignore noise
            cluster_points = frame_points[clusterlabels == clusterlabel, :]
            centroid = np.mean(cluster_points, axis=0)  # also calculate avg intensity
            centers.append(np.append(centroid, len(cluster_points)))

            # mark cluster points for visualization
            in_frame_indices = np.nonzero(point_selection[i])[0]
            in_cluster_indices = in_frame_indices[clusterlabels == clusterlabel]
            visualization[i, in_cluster_indices, 3] = (
                clusterlabel # starting at 0
            )  # -> values 0..N are for clusters with respective index

        cluster_centers_per_frame.append(np.array(centers))
        # returns list of np array with one row per centroid, containing:
        # [
        #  x, y, z     the cluster's centroid
        #  intensity   the mean intensity
        #  points      the number of points
        # ]

    return cluster_centers_per_frame, visualization


def filter_clusters_1(centers, max_distance):
    # per frame, get cluster with most points
    biggest_centers = []
    for i in range(len(centers)):
        biggest_index = np.argmax(centers[i][:, 4], axis=0)
        biggest_centers.append(centers[i][biggest_index])

    # drop centers where the next one is too far away
    for i in range(len(biggest_centers) - 1):
        dist_to_next = np.linalg.norm(
            biggest_centers[i][:3] - biggest_centers[i + 1][:3]
        )
        if dist_to_next > max_distance:
            biggest_centers[i] = None

    return biggest_centers
    # TODO version 2 now returns the indices of the clusters instead of the point itself
