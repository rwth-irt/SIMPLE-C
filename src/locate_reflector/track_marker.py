from typing import Union, List, Dict, Tuple
import numpy as np


def get_nearest_neighbor(origin, candidates):
    # only considering xyz coordinates
    assert len(candidates) > 0
    dist = np.linalg.norm(candidates[:, :3] - origin[:3], axis=1)
    best_index = np.argmin(dist)
    return candidates[best_index]


# TODO note somewhere that "center" and "cluster" is treated as synonym in most docstrings!

def get_nearest_neighbor_trace(clusters: List[np.ndarray], start_i) -> np.ndarray:
    """
    Get the trace of nearest neighbors, going **backwards** from the "start cluster" `clusters[-1][start_i]`.
    The start cluster is contained in the returned trace.
    If a frame contains no clusters, the search is cancelled and the trace up to this point is returned.
    """
    if len(clusters) == 0:
        return np.array([])
    clusters = clusters.copy()
    out = [clusters.pop()[start_i]]
    while clusters:
        next_candidates = clusters.pop()
        if len(next_candidates) == 0:
            # no candidates in next frame, stop search
            return np.array(out)
        nn = get_nearest_neighbor(out[0], next_candidates)
        out.insert(0, nn)
    return np.array(out)


def find_marker_single_frame(clusters, max_distance, min_velocity, max_vector_angle_rad) \
        -> Tuple[Union[None, Tuple[np.ndarray, int]], str]:
    """
    Decide which cluster center in the **last** frame of `clusters` is most likely the reflector.
    The decision is **not** based on previous decisions!

    The trace (backwards in time) with the nearest neighbors in the previous frame is calculated for each cluster.
    These traces are then filtered by multiple criteria.
    See comments in the code below for the individual filters.

    If a single cluster passes all filters, it is chosen. Otherwise no choice is made.

    :param clusters: list with a numpy array per frame, containing all found cluster centers in this frame.
    :param max_distance: maximum distance between adjacent clusters in a trace
    :param min_velocity: minimum average movement distance for clusters between two frames in a trace
    :param max_vector_angle_rad: maximum angle in radians between two movement vectors for a cluster
    :return: A tuple (result, status).
        `result` is None (no unique solution) or a tuple (cluster, cluster_index_in_frame).
        `status` is a string indicating whether there was a match ("UNIQUE_MATCH"), no cluster or no
        match found ("NO_MATCH"), or multiple matches ("MULTIPLE_MATCHES").
    """
    if len(clusters[-1]) == 0:
        return None, "NO_MATCH"

    # calculate a trace of nearest neighbors for each cluster, backwards in time.
    traces: list[np.ndarray] = [
        get_nearest_neighbor_trace(clusters, start_i) for start_i in range(len(clusters[-1]))
    ]

    # Drop all traces which are too short
    length_filter = np.array(list(map(
        lambda t: len(t) == len(clusters),
        traces
    )))

    # Drop all traces where the distance between two points is greater than `max_distance`.
    # (The reflector is expected to not move too quickly/jump.)
    distance_filter = np.array(list(map(
        lambda t: np.all(np.linalg.norm(np.diff(t[..., :3], axis=0), axis=1) <= max_distance),
        traces
    )))

    # Drop all traces where the average cluster velocity is too low.
    # (The reflector is expected to move at a consistent speed. This filters out bright static objects.)
    velocity_filter = np.array(list(map(
        lambda t: np.mean(np.linalg.norm(np.diff(t[..., :3], axis=0), axis=1)) > min_velocity,
        traces
    )))

    # Drop all traces where the angle between two update vectors (deltas) is too big.
    # (The reflector is expected to move at a consistent speed.)
    def check_angle(trace: np.ndarray):
        deltas = np.diff(trace[..., :3], axis=0)
        a = deltas[1:]
        b = deltas[:-1]
        angle_rad = np.arccos(
            np.diag(a @ b.T)  # scalar products between vectors in a and b (axis 1)
            / (np.linalg.norm(a, axis=1) * np.linalg.norm(b, axis=1))
        )
        # return whether this trace passes the filter
        return np.all(angle_rad <= max_vector_angle_rad)

    angle_filter = np.array(list(map(check_angle, traces)))

    combined_filter = length_filter & distance_filter & velocity_filter & angle_filter
    if sum(combined_filter) == 0:
        # did not find unique solution
        return None, "NO_MATCH"
    elif sum(combined_filter) > 1:
        return None, "MULTIPLE_MATCHES"
    i = np.argmax(combined_filter)  # get index of the chosen cluster
    return (clusters[-1][i, :3], i), "UNIQUE_MATCH"  # only return xyz of cluster
    # TODO do we even need the intensity mean and number of clusters in this analysis? If not, don't pass
    #  them in here and remove all the [:3] and [..., :3] etc.!


def track_marker_multiple_frames(
        clusters, max_distance, min_velocity, window_size, max_vector_angle_rad
) -> Tuple[List[np.ndarray], List[int]]:
    """
    Applies `find_marker` successively for multiple frames.
    Therefore, tracks the reflector in the data of a single sensor.
    Passes slices of size `window_size` to `find_marker`.
    Returns array of selected cluster centers and array of indices.

    :param clusters: list with a numpy array per frame, containing all found cluster centers in this frame.
      Slices of this list will be passed to `find_marker`.
    :param max_distance: See `find_marker`.
    :param min_velocity: See `find_marker`.
    :param window_size: Length of the slices to pass to `find_marker`.
    :param max_vector_angle_rad: See `find_marker`.
    :return: (array of selected clusters, array holding indices of selected cluster per frame)
    """
    indices = [None] * (window_size - 1)  # can not calculate for early ones
    centers = [None] * (window_size - 1)

    for frame_i in range(len(clusters) - window_size + 1):
        choice, _ = find_marker_single_frame(clusters[frame_i:frame_i + window_size],
                                             max_distance, min_velocity, max_vector_angle_rad)
        if choice is None:
            centers.append(None)
            indices.append(None)
        else:
            centers.append(choice[0])
            indices.append(choice[1])

    return centers, indices
