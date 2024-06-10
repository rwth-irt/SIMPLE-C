import numpy as np


class ReflectorLocation:
    def __init__(self, cluster_mean: np.ndarray, cluster_points: np.ndarray, cluster_index_in_frame: int):
        # Note that this object does intentionally not have any reference to the frame object it is based on,
        # so that it (including the large point cloud array) can be garbage collected.

        # deconstruct cluster_mean
        self.centroid = cluster_mean[:3]
        self.avg_intensity = cluster_mean[3]
        self.number_of_points_in_cluster = cluster_mean[4]

        self.cluster_points = cluster_points
        self.cluster_index_in_frame = cluster_index_in_frame

        self.normal_vector = self.fit_plane()

    def fit_plane(self) -> np.ndarray:
        centered_points = self.cluster_points - self.centroid
        U, S, Vt = np.linalg.svd(centered_points)
        normal = Vt[-1]  # The normal of the plane is the last singular vector
        normal /= np.linalg.norm(normal)
        return normal

    @property
    def weight(self):
        # cosine similarity of normal of points in cluster and vector from origin (sensor location) to centroid
        normal_weight = np.arccos(
            self.normal_vector @ self.centroid / (np.linalg.norm(self.centroid) * np.linalg.norm(self.normal_vector))
        )
        # TODO implement weighting of weights
        return normal_weight
