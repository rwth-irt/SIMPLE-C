import numpy as np

from . import parameters
from .locate_reflector.find_cluster_centers import get_cluster_centers_single_frame


# TODO document, especially shape and content of numpy arrays!
# currently, the documentation of the cluster_centers and clustering arrays
# can be found in the docs of find_cluster_centers.get_cluster_centers_single_frame

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
        # Calculate lazily, will not be required in dropped frames and is expensive.
        if self._cluster_centers is None:
            self._calc_clusters()
        return self._cluster_centers

    @property
    def clustering(self) -> np.ndarray:
        # Calculate lazily, will not be required in dropped frames and is expensive.
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

    def get_cluster_points(self, index: int):
        return self.data[self.clustering == index, :3].copy()
        # call copy() to decouple from big frame array to allow for garbage collection
