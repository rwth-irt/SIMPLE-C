import numpy as np


def filter_locations(marker_locations, topics):
    """
    Returns a filtered copy of `marker_locations` which only contains the points from frames where a location is
    present for each sensor/topic.

    NOTE: Be aware to specify `topics` such that the tracked object is known to be the same! Later usage of the filtered
    locations will produce garbage if the tracked objects are actually uncorrelated and only appear in the same frames
    randomly!

    :param marker_locations:
    :param topics: A list of topics to consider.
    :return: filtered copy of `marker_locations`
    """
    chosen_frames = []
    out = {}
    for frame_i in range(min(map(len, marker_locations.values()))):
        all_present = True
        for t in topics:
            if marker_locations[t][frame_i] is None:
                all_present = False
                break
        if all_present:
            chosen_frames.append(frame_i)
    for t in topics:
        # convert to numpy array and filter
        out[t] = np.array([marker_locations[t][i] for i in chosen_frames], dtype=np.float64)
    return out


def calc_transformation(P: np.array, Q: np.array):
    """
    Kabsch Algorithm https://en.wikipedia.org/wiki/Kabsch_algorithm

    Directly following https://igl.ethz.ch/projects/ARAP/svd_rot.pdf
    but without weights (no weighted centroids, no weight matrix W)

    Calculates the optimal transformation which transforms the
    points P to resemble Q with the least squared error.
    """

    assert len(P) == len(Q)

    p_bar = np.mean(P, axis=0)  # currently without weights
    q_bar = np.mean(Q, axis=0)

    X = (P - p_bar).T
    Y = (Q - q_bar).T

    S = X @ Y.T  # again, without weights (W)

    U, Sigma, VT = np.linalg.svd(S)
    V = VT.T

    detdiag = np.eye(len(V))
    detdiag[-1, -1] = np.linalg.det(V @ U.T)
    R = V @ detdiag @ U.T
    t = q_bar - R @ p_bar

    return R, t

# TODO
#  use scipy implementation of the Kabsch Algorithm instead, as it offers a sensitivity matrix for the transformation!
#  https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html
