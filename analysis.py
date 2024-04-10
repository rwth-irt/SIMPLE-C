import numpy as np
from sklearn.cluster import DBSCAN
from pprint import pprint


def get_cluster_centers_per_frame(
    frames,  # input points: frames[frame_index, point_index] = [x,y,z,intensity]
    N_highest_intensity=30,  # the N points with highest intens. used for DBSCAN clustering
    DBSCAN_epsilon=0.15,  # in meters, as is LIDAR output (TODO: is that correct?)
    DBSCAN_min_samples=3,
):
    # per frame, get the indices of the N points with highest intensity
    brightest_indices = np.argsort(-frames[:, :, 3], axis=1)[:, :N_highest_intensity]
    # TODO maybe use a threshold instead

    # mark points for visualization, rewriting the intensity channel in `frames`
    # 0: any point
    # 1: for the N points with highest intensity
    # 2: part of a cluster

    visualization = np.copy(frames)
    visualization[:, :, 3] = 0  # reset intensity data
    for i in range(len(frames)):
        visualization[i, brightest_indices[i], 3] = 1  # 1 for brightest points

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
        ).fit_predict(brightest_points[i, :, :3])

        # get centroids of clusters and mark them for visualization
        centers = []
        for clusterlabel in np.unique(clusterlabels):
            if clusterlabel == -1:
                continue  # ignore noise
            points_in_cluster = frames[
                i, brightest_indices[i, clusterlabels == clusterlabel]
            ]
            centroid = np.mean(points_in_cluster, axis=0)
            centers.append(np.append(centroid, len(points_in_cluster)))

            visualization[i, brightest_indices[i, clusterlabels == clusterlabel], 3] = 2

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
