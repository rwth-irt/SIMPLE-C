import numpy as np

from filter_clusters_2 import filter_clusters_2
from find_cluster_centers import get_cluster_centers_per_frame
from visualization import visualize_animation


def prepare_visualization(selection_indices, visualization):
    """
    Expects visualization to contain cluster indices in intensity channel of points.
    The intensity channel is used as indicator for rendering color in the o3d UI, see visualization.py.
    For given index of the selected cluster per frame (`selection_indices`), set the
    respectively selected cluster's value to 2 for highlighting and all others to 1.

    **Alters visualization**, which can then be passed to the UI.

    :param selection_indices: list with index of selected cluster per frame
    :param visualization: lidar data numpy array, but with visualization info in intensity, as obtained \
    from `get_cluster_centers_per_frame(create_visualization=True)`
    :return: nothing, writes into visualization array
    """
    for frame_i in range(len(selection_indices)):
        # currently, visualization contains indices of clusters
        # for o3d visualization, we want to convert this to codes meaning "any cluster" or "chosen cluster"
        chosen_cluster_selection = visualization[frame_i, :, 3] == selection_indices[frame_i]
        any_cluster_selection = visualization[frame_i, :, 3] >= 0  # any cluster
        visualization[frame_i, any_cluster_selection, 3] = 1
        visualization[frame_i, chosen_cluster_selection, 3] = 2


def get_centers(frames, params, visualize: bool):
    """
    Try to locate the marker in given lidar data (multiple frames) from a single sensor and return its positions.

    Applies `get_cluster_centers_per_frame` to locate clusters and obtain their centers, and then `filter_clusters_2` to
    try to find the cluster originating from the calibration reflector. See docs of these functions for further
    information.

    :param frames: numpy array with multiple lidar frames.
        1st axis: frame index,
        2nd axis: point index,
        3rd axis: x, y, z, intensity

    :param params: dict with parameters, e.g. obtained by reading a parameter JSON file.
    :param visualize: bool whether to show the open3d visualization of the analysis results
    :return: a list of the found cluster centers as numpy arrays
    """
    print("calculating cluster centers")
    centers = get_cluster_centers_per_frame(
        frames,
        rel_intensity_threshold=params["relative intensity threshold"],
        DBSCAN_epsilon=params["DBSCAN epsilon"],
        DBSCAN_min_samples=int(params["DBSCAN min samples"]),
        create_visualization=visualize,
    )
    if visualize:
        # in this case, get_cluster_centers_per_frame returned a tuple, so unpack it
        centers, visualization = centers

    selection_indices = filter_clusters_2(
        centers,
        max_distance=params["maximum neighbor distance"],
        min_velocity=params["minimum velocity"],
        velocity_lookahead=int(params["velocity lookahead"]),
        max_vector_angle_rad=2 * np.pi * params["max. vector angle [deg]"] / 360,
    )

    # extract chosen centers by indices
    chosen_centers = []
    for frame_i in range(len(selection_indices)):
        if selection_indices[frame_i] is None:
            chosen_centers.append(None)
        else:
            chosen_centers.append(centers[frame_i][selection_indices[frame_i]])

    if visualize:
        prepare_visualization(selection_indices, visualization)
        print("showing open3d visualization, this will block the settings UI")
        print("press escape to close 3d view, then enter new values")
        visualize_animation(visualization, chosen_centers)
        print("returning to settings UI")

    return chosen_centers
