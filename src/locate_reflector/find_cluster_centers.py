import numpy as np
from sklearn.cluster import DBSCAN
from scipy.spatial.distance import mahalanobis


def mahalanobis_distance_metric(X):
    """
    Calculates the Mahablonis distance matrix of a given matrix dataset X
    https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.distance.mahalanobis.html

    :param X: Data matrix, for example n x 4 with n points each described by x, y, z, intensity
    :return mahalanobis_dist: Distance matrix with n x n describing the distance/similarity between each point of X
    """

    covariance = np.cov(X, rowvar=False)
    covariance_inverse = np.linalg.inv(covariance)

    # Compute the Mahalanobis distance matrix
    mahalanobis_dist = np.zeros((len(X), len(X)))
    for i in range(len(X)):
        for j in range(len(X)):
            mahalanobis_dist[i, j] = mahalanobis(X[i], X[j], covariance_inverse)
    
    return mahalanobis_dist


def get_cluster_centers_single_frame(
        frame,
        rel_intensity_threshold,
        DBSCAN_epsilon,
        DBSCAN_min_samples,
        metric="euclidean",
        # visualization-only parameters
        visualization=None,
        i=None,
):
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

    If `visualization` is passed, this function writes cluster centers in it's intensity channel!
    For an explanation of the visualization array, see documentation in `tracking_visualization.py`.

    :param frame: lidar frame as numpy array: [ x y z intensity ] per point
    :param rel_intensity_threshold: value between 0 and 1 to determine the ratio of points considered "bright".
    :param DBSCAN_epsilon: epsilon parameter for DBSCAN. Should correspond to distances in meters, as euclidian
      distance and xyz values are used for DBSCAN
    :param DBSCAN_min_samples: min_samples parameter for DBSCAN
    :param visualization: optional visualization array to write cluster indices to.
    :param i: Required only if visualization is given, this specifies the frame index to write to.
    :return: numpy array with one column per cluster containing [ x y z intensity #points ] (mean values!). If
      no clusters are found, an empty numpy array is returned.
    """
    intensity_threshold = np.max(frame[:, 3]) * rel_intensity_threshold
    point_selection = frame[:, 3] >= intensity_threshold
    bright = frame[point_selection]
    if len(bright) == 0:
        # no bright points in this frame
        return np.array([])  # empty array: no centers
    
    # Normalize the data by radial distance
    radial_distances = np.sqrt(np.sum(bright[:, :3] ** 2, axis=1))
    normalized_points = bright[:, :3] / np.expand_dims(radial_distances, axis=1)
    normalized_data = np.concatenate((normalized_points, np.expand_dims(bright[:, 3]/255.0, axis=1)), axis=1)

    if metric == "Mahalanobis":
        mahalanobis_points = mahalanobis_distance_metric(normalized_data)

        # perform DBSCAN
        clusterlabels = DBSCAN(
            eps=DBSCAN_epsilon, metric="precomputed", min_samples=DBSCAN_min_samples
        ).fit_predict(mahalanobis_points)  # based on normalized x, y, z, i
    else: # euclidean based clustering
        # perform DBSCAN
        clusterlabels = DBSCAN(
            eps=DBSCAN_epsilon, min_samples=DBSCAN_min_samples
        ).fit_predict(normalized_data[:, :3])  # only based on xyz

    # get centroids of clusters
    centers = []
    for clusterlabel in np.unique(clusterlabels):
        if clusterlabel == -1:
            continue  # ignore noise
        cluster_points = bright[clusterlabels == clusterlabel, :]
        centroid = np.mean(cluster_points, axis=0)  # also calculate avg intensity
        centroid = np.append(centroid, len(cluster_points))
        centers.append(centroid)

        if visualization is not None and i is not None:
            # mark cluster points in visualization array if given
            in_frame_indices = np.nonzero(point_selection)[0]  # np.nonzero returns a tuple
            in_cluster_indices = in_frame_indices[clusterlabels == clusterlabel]
            visualization[i, in_cluster_indices, 3] = clusterlabel
            # starting at 0  # -> values 0..N are for clusters with respective index

    return np.array(centers)


def get_cluster_centers_multiple_frames(
        frames,
        rel_intensity_threshold,
        DBSCAN_epsilon,
        DBSCAN_min_samples,
        create_visualization=False,
        metric="euclidean"
):
    """
    Wrapper for `get_get_cluster_centers`, which is called successively for multiple frames. In addition,
    visualization data can be returned together with the found cluster centers.

    :param frames: numpy array containing multiple lidar frames.
    :param rel_intensity_threshold: see `get_cluster_centers`
    :param DBSCAN_epsilon: see `get_cluster_centers`
    :param DBSCAN_min_samples: see `get_cluster_centers`
    :param create_visualization: whether to return a visualization array for open3d result visualization.
    :return: List with (possibly empty, meaning no clusters) numpy array
        containing mean values of points in respective cluster:
        (x_mean, y_mean, z_mean, intensity_mean, number_of_points).
        If `create_visualization`, then also a visualization array is returned in which clusters have been marked
        with their index replacing the original value in the intensity channel (see `prepare_tracking_visualization` in
        tracking_visualization.py)
    """
    intensity_threshold = np.max(frames[..., 3]) * rel_intensity_threshold
    point_selection = frames[..., 3] >= intensity_threshold

    if create_visualization:
        # prepare visualization data
        visualization = np.copy(frames)
        visualization[:, :, 3] = -2  # any point
        visualization[point_selection, 3] = -1  # 1 for selected/bright points

    cluster_centers_per_frame = []
    for i, frame in enumerate(frames):
        args = [frame, rel_intensity_threshold, DBSCAN_epsilon, DBSCAN_min_samples, metric]
        if create_visualization:
            args += [visualization, i]

        centers = get_cluster_centers_single_frame(*args)
        cluster_centers_per_frame.append(centers)

    if create_visualization:
        return cluster_centers_per_frame, visualization
    return cluster_centers_per_frame
