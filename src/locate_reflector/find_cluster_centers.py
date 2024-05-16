import numpy as np
from sklearn.cluster import DBSCAN


def get_cluster_centers_single_frame(
        frame: np.ndarray,
        rel_intensity_threshold: float,
        DBSCAN_epsilon: float,
        DBSCAN_min_samples: int
) -> tuple[np.ndarray, np.ndarray]:
    """
    Searches for bright (high intensity) clusters in a given Lidar Frame. Performs the following steps:

    1. Find the brightest N points in the given frame, where `N = len(frame) * rel_intensity_threshold`.
    2. On the xyz-coordinates of those points, apply the DBSCAN algorithm to find clusters in space.
       See corresponding arguments for DBSCAN configuration. Points considered noise by DBSCAN are ignored.
    3. For each cluster determined by DBSCAN, calculate the following values:
        - mean of x, y, z coordinates
        - mean intensity
        - number of points

    A numpy array is then returned which consists of those values per point.

    In addition, a copy of `frame` is returned, in which the 4th entry of each point (the former "intensity" channel)
    holding the following values:
    - If the point is assigned to a cluster, the index of that cluster.
    - If the point's intensity is below the intensity threshold: -2
    - If the point's intensity is above the intensity threshold: -1

    :param frame: lidar frame as numpy array: [ x y z intensity ] per point
    :param rel_intensity_threshold: value between 0 and 1 to determine the ratio of points considered "bright".
    :param DBSCAN_epsilon: epsilon parameter for DBSCAN. Should correspond to distances in meters, as euclidian
      distance and xyz values are used for DBSCAN
    :param DBSCAN_min_samples: min_samples parameter for DBSCAN
    :return:
        - A tuple `(cluster_means, clustering)`, where
        - cluster_means is a numpy array with one column per cluster containing [ x y z intensity #points ]
          (mean values!). If no clusters are found, an empty numpy array is returned.
        - clustering is a copy of `frame`, where the intensity data (`frame[..., 3]` is replaced with the
          cluster index of the respective point or -1 if the point is in no cluster.
    """
    clustering = frame.copy()
    clustering[:, 3] = -2

    intensity_threshold = np.max(frame[:, 3]) * rel_intensity_threshold
    point_selection = frame[:, 3] >= intensity_threshold
    clustering[point_selection, 3] = -1
    bright = frame[point_selection]
    if len(bright) == 0:
        # no bright points in this frame
        return np.array([]), clustering  # empty centers array: no centers

    # perform DBSCAN
    clusterlabels = DBSCAN(
        eps=DBSCAN_epsilon, min_samples=DBSCAN_min_samples
    ).fit_predict(bright[:, :3])  # only based on xyz

    # get centroids of clusters
    centers = []
    for clusterlabel in np.unique(clusterlabels):
        if clusterlabel == -1:
            continue  # ignore noise
        cluster_points = bright[clusterlabels == clusterlabel, :]
        centroid = np.mean(cluster_points, axis=0)  # also calculate avg intensity
        centroid = np.append(centroid, len(cluster_points))
        centers.append(centroid)

        # mark cluster points in clustering array
        # 1. get indices of all bright points in frame which have been passed to DBSCAN
        in_frame_indices = np.nonzero(point_selection)[0]
        # 2. from these, select the ones which belong to this cluster
        in_cluster_indices = in_frame_indices[clusterlabels == clusterlabel]
        # 3. we can use these indices now in the original `frame` array containing *all* points
        clustering[in_cluster_indices, 3] = clusterlabel

    return np.array(centers), clustering


def get_cluster_centers_multiple_frames(
        frames,
        rel_intensity_threshold: float,
        DBSCAN_epsilon: float,
        DBSCAN_min_samples: int,
        create_visualization=False,
):
    """
    Wrapper for :func:`get_cluster_centers_single_frame`, which is called successively for multiple frames. In addition,
    visualization data can be returned together with the found cluster centers.

    TODO **The usage of this function is deprecated. Use a list of `Frame` objects instead.**

    :param frames: numpy array containing multiple lidar frames.
    :param rel_intensity_threshold: see :func:`get_cluster_centers_single_frame`
    :param DBSCAN_epsilon: see :func:`get_cluster_centers_single_frame`
    :param DBSCAN_min_samples: see :func:`get_cluster_centers_single_frame`
    :param create_visualization: whether to return a visualization array for open3d result visualization.
    :return: List with (possibly empty, meaning no clusters) numpy array
        containing mean values of points in respective cluster:
        (x_mean, y_mean, z_mean, intensity_mean, number_of_points).
        If `create_visualization`, then also a visualization array is returned in which clusters have been marked
        with their index replacing the original value in the intensity channel (see `prepare_tracking_visualization` in
        tracking_visualization.py)
    """
    visualization_list = []

    cluster_centers_per_frame = []
    for i, frame in enumerate(frames):
        args = [frame, rel_intensity_threshold, DBSCAN_epsilon, DBSCAN_min_samples]
        centers, clustering = get_cluster_centers_single_frame(*args)
        cluster_centers_per_frame.append(centers)
        visualization_list.append(clustering)

    if create_visualization:
        return cluster_centers_per_frame, np.array(visualization_list)
    return cluster_centers_per_frame
