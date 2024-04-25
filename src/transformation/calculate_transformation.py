import numpy as np
from scipy.spatial.transform import Rotation


def filter_locations(marker_locations: dict[str, list[np.ndarray]], topics: list[str]):
    """
    Returns a filtered copy of `marker_locations` which only contains the points from frames where a location is
    present for each sensor/topic.

    NOTE: Be aware to specify `topics` such that the tracked object is known to be the same! Later usage of the filtered
    locations will produce garbage if the tracked objects are actually uncorrelated and only appear in the same frames
    randomly!

    :param marker_locations: Dictionary mapping sensor topic to a list of marker locations (individual numpy arrays
      or None if not found in this frame)
    :param topics: A list of topics/sensors to consider.
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


def calc_transformation_scipy(P: np.ndarray, Q: np.ndarray, weights: np.ndarray = None):
    """
    Apply the Kabsch algorithm [1] using the implementation in SciPy [2].

    Optionally, weights for the given points can be provided.

    Calculates the optimal transformation which transforms the
    points P to resemble Q with the least squared error.

    [1] https://en.wikipedia.org/wiki/Kabsch_algorithm \n
    [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html

    :param P: One set of points which are transformed using the resulting transformation to then resemble Q.
        Numpy array of shape (N, 3).
    :param Q: The other set of points. Numpy array of shape (N, 3).
    :param weights: Optional weights (if None, all points are weighted equally). Numpy array of shape (N).
    :return: Tuple containing: (Rotation matrix, translation vector, Rotation quaternion, sensitivity matrix
        of rotation).
    """
    assert len(P) == len(Q)
    R, rssd, sensitivity = Rotation.align_vectors(Q, P, weights=weights, return_sensitivity=True)
    # !!! Q, P flipped compared to own implementation!

    # Returns rotation as generic scipy rotation object
    Rq = R.as_quat()  # Quaternion
    Rm = R.as_matrix()  # Rotation matrix

    # See https://igl.ethz.ch/projects/ARAP/svd_rot.pdf on how to calculate the corresponding translation vector.
    p_bar = np.average(P, axis=0, weights=weights)
    q_bar = np.average(Q, axis=0, weights=weights)
    t = q_bar - Rm @ p_bar

    return Rm, t, Rq, sensitivity
