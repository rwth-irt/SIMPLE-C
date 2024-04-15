import numpy as np
from sklearn.cluster import DBSCAN


def get_cluster_centers_per_frame(
        frames,  # input points: frames[frame_index, point_index] = [x,y,z,intensity]
        rel_intensity_threshold,
        DBSCAN_epsilon=0.15,  # in meters, as is LIDAR output (TODO: is that correct?)
        DBSCAN_min_samples=3,
        create_visualization=False,
):
    """
    TODO

    :param frames:
    :param rel_intensity_threshold:
    :param DBSCAN_epsilon:
    :param DBSCAN_min_samples:
    :param create_visualization:
    :return: List with (possibly empty, meaning no clusters) numpy array
        containing mean values of points in respective cluster (x_mean, y_mean, z_mean, intensity_mean).
        If `create_visualization`, then also a visualization array is returned in which clusters have been marked
        with their index replacing the original value in the intensity channel (see `prepare_tracking_visualization` in
        tracking_visualization.py)
    """
    intensity_threshold = np.max(frames[..., 3]) * rel_intensity_threshold
    point_selection = frames[..., 3] >= intensity_threshold

    if create_visualization:
        visualization = np.copy(frames)
        visualization[:, :, 3] = -2  # any point
        visualization[point_selection, 3] = -1  # 1 for selected/bright points

    cluster_centers_per_frame = []
    for i in range(len(frames)):
        frame_points = frames[i, point_selection[i]]
        if len(frame_points) == 0:
            # no bright points in this frame
            cluster_centers_per_frame.append(np.array([]))  # empty array: no centers
            continue
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
            if create_visualization:
                visualization[i, in_cluster_indices, 3] = (
                    clusterlabel  # starting at 0  # -> values 0..N are for clusters with respective index
                )

        cluster_centers_per_frame.append(np.array(centers))
        # returns list of np array with one row per centroid, containing:
        # [
        #  x, y, z     the cluster's centroid
        #  intensity   the mean intensity
        #  points      the number of points
        # ]

    if create_visualization:
        return cluster_centers_per_frame, visualization
    return cluster_centers_per_frame
