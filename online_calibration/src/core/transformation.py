from dataclasses import dataclass

import numpy as np
from scipy.spatial.transform import Rotation


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


@dataclass
class Transformation:
    R: np.ndarray  # Rotation matrix
    t: np.ndarray  # Translation vector
    R_quat: np.ndarray  # Rotation quaternion TODO add property for it instead of specifying explicitly!
    R_sensitivity: np.ndarray  # Sensitivity matrix for R

    # TODO add calculation of (un)certainty

    @property
    def inverse(self):
        """Warning: loses R_sensitivity!"""
        return Transformation(
            R=self.R.T,
            t=-self.t,
            R_quat=Rotation.from_quat(self.R_quat).inv().as_quat(),
            R_sensitivity=np.zeros((3, 3))
            # TODO set sensitivity to None so that nothing calculates with it by accident.
            # (This is no problem as we only use the inverse when calculating chained transformations,
            # which are for visualization purposes only.)
        )

    @property
    def matrix(self) -> np.ndarray:
        out = np.zeros((4, 4))
        out[3, 3] = 1
        out[:3, :3] = self.R
        out[:3, 3] = self.t
        return out

    @staticmethod
    def from_matrix(matrix: np.ndarray):
        """Warning: loses R_sensitivity!"""
        return Transformation(
            R=matrix[:3, :3],
            t=matrix[:3, 3],
            R_quat=Rotation.from_matrix(matrix[:3, :3]).as_quat(),
            R_sensitivity=np.zeros((3, 3))  # TODO see above in self.inverse(..)!
        )

    @staticmethod
    def unity():
        return Transformation(
            R=np.eye(3),
            t=np.zeros((3)),
            R_quat=np.array([0, 0, 0, 1]),
            R_sensitivity=np.zeros((3, 3))  # TODO see above in self.inverse(..)!
        )


def calc_transformation_scipy(P: np.ndarray, Q: np.ndarray, weights: np.ndarray = None) -> Transformation:
    """
    Apply the Kabsch algorithm [1]_ using the implementation in SciPy [2]_.

    Optionally, weights for the given points can be provided.

    Calculates the optimal transformation (translation and rotation) which transforms the
    points P to resemble Q with the least squared error.

    .. [1] https://en.wikipedia.org/wiki/Kabsch_algorithm \n
    .. [2] https://docs.scipy.org/doc/scipy/reference/generated/scipy.spatial.transform.Rotation.align_vectors.html

    :param P: One set of points which are transformed using the resulting transformation to then resemble Q.
        Numpy array of shape (N, 3).
    :param Q: The other set of points. Numpy array of shape (N, 3).
    :param weights: Optional weights (if None, all points are weighted equally). Numpy array of shape (N).
    :return: An instance of the Transformation dataclass.
    """
    assert len(P) == len(Q) == len(weights)
    # !!! Q, P flipped compared to own implementation!

    # calculate weighted mean of each set of points
    p_bar = np.average(P, axis=0, weights=weights)
    q_bar = np.average(Q, axis=0, weights=weights)

    # Scipy expects points to be centered, substract the data point's centroids.
    R, rssd, sensitivity = Rotation.align_vectors(Q - q_bar, P - p_bar, weights=weights, return_sensitivity=True)

    # Returns rotation as generic scipy rotation object
    Rq = R.as_quat()  # Quaternion
    Rm = R.as_matrix()  # Rotation matrix

    # See https://igl.ethz.ch/projects/ARAP/svd_rot.pdf on how to calculate the corresponding translation vector.
    t = q_bar - Rm @ p_bar
    # TODO sensitivity for translation?

    return Transformation(Rm, t, Rq, sensitivity)


def apply_transformation(points: np.ndarray, trafo: Transformation):
    """
    Apply the transformation on multiple points.

    :param points: a numpy array containing multiple 3d points to transform
    :param trafo: a Transformation dataclass instance
    :return: `points`, transformed
    """
    return (trafo.R @ points.T).T + trafo.t
