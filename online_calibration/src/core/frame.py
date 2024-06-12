import numpy as np

from . import parameters
from .locate_reflector.find_cluster_centers import get_cluster_centers_single_frame


class Frame:

    def __init__(self, data: np.ndarray, timestamp_sec: float, topic: str):
        """
        An object representing Lidar data in one frame of a single sensor.
        Lazily computes and caches clustering results.

        :param data: Raw point data from the sensor, each row is [x, y, z, intensity]
        :param timestamp_sec: The timestamp when the frame was recorded, which approximately is the internal timestamp
            when the last point was recorded.
        :param topic: The ROS topic/sensor name the frame belongs to.
        """
        self.topic = topic
        self.data = data
        self.timestamp_sec = timestamp_sec
        self._cluster_centers = None
        self._clustering = None

    @property
    def cluster_centers(self) -> np.ndarray:
        """
        A numpy array containing the following rows for each cluster found in this Frame:
        (x_mean, y_mean, z_mean, intensity_mean, number_of_points_in_cluster)

        Output is the result of find_cluster_centers.get_cluster_centers_single_frame, lazily calculated and cached.
        """
        if self._cluster_centers is None:
            self._calc_clusters()
        return self._cluster_centers

    @property
    def clustering(self) -> np.ndarray:
        """
        A numpy array containing the cluster assignment (index) for each point in `self.data`.

        Output is the result of find_cluster_centers.get_cluster_centers_single_frame, lazily calculated and cached.
        """
        if self._clustering is None:
            self._calc_clusters()
        return self._clustering

    def _calc_clusters(self):
        self._cluster_centers, self._clustering = get_cluster_centers_single_frame(
            self.data,
            rel_intensity_threshold=parameters.get_param("relative intensity threshold"),
            DBSCAN_epsilon=parameters.get_param("DBSCAN epsilon"),
            DBSCAN_min_samples=int(parameters.get_param("DBSCAN min samples")),
        )

    def get_cluster_points(self, index: int) -> np.ndarray:
        """
        Returns a numpy array with all points from `self.data` which belong to the cluster with a given index.

        :param index: The index of the cluster, should occur in `self.clustering`.
        :return: The points of the referenced cluster. Is a *new* numpy array, so no references to the underlying
            data array are required to allow for garbage collection.
        """
        return self.data[self.clustering == index, :3].copy()
