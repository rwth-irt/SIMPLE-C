from datetime import datetime

import numpy as np

from . import parameters
from .locate_reflector.find_cluster_centers import get_cluster_centers_single_frame


# TODO document, especially shape and content of numpy arrays!
class Frame:
    def __init__(self, data: np.ndarray, timestamp: datetime):
        self.data: np.ndarray = data
        self.timestamp: datetime = timestamp
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
            rel_intensity_threshold=parameters.params["relative intensity threshold"],
            DBSCAN_epsilon=parameters.params["DBSCAN epsilon"],
            DBSCAN_min_samples=int(parameters.params["DBSCAN min samples"]),
        )

    def get_cluster_points(self, index: int):
        return self.clustering[self.clustering[:, 3] == index].copy()
        # call copy() to decouple from big frame array to allow for garbage collection
