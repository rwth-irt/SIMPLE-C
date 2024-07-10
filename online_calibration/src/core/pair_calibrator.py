import itertools
import logging
from collections import deque
from typing import Callable, Iterable

import numpy as np

from . import parameters
from .frame import Frame
from .locate_reflector.track_marker import find_marker_single_frame
from .reflector_location import ReflectorLocation
from .transformation import Transformation, calc_transformation_scipy, apply_transformation
from .websocket_server import broadcast_pair_metadata

logger = logging.getLogger(__name__)


class PairCalibrator:
    """
    An object which receives frame data from two sensors. If up-to-date frame data exists for both sensors,
    a new transformation is calculated if possible. Caches the found ReflectorLocations.
    """

    def __init__(self, topic1: str, topic2: str, trafo_callback: Callable[[Transformation, str, str], None] | None):
        """
        Initialize a new PairCalibrator.

        :param topic1: topic name of detector 1
        :param topic2: topic name of detector 2
        :param trafo_callback: function to call when a new transformation is found. It is called with
            (transformation, topic1, topic2).
        """
        # Maximum age for a frame before it expires
        self._expiry_duration_sec = 1.0 / float(parameters.get_param("sample_rate_Hz")) * 0.6
        # Do not use 50% because if detectors are just offset equally, dropping one frame does not help.
        # If dropping at 60%, the next offset will only be 40%. This avoids many frames to be dropped in sequence.

        self._frame_buffer_1: deque[Frame] = deque(maxlen=int(parameters.get_param("window size")))
        self._frame_buffer_2: deque[Frame] = deque(maxlen=int(parameters.get_param("window size")))
        self.topic1 = topic1
        self.topic2 = topic2
        self._last1: Frame | None = None
        self._last2: Frame | None = None
        self.reflector_locations_1: list[ReflectorLocation] = []
        self.reflector_locations_2: list[ReflectorLocation] = []
        self.transformation: Transformation | None = None
        self._trafo_callback = trafo_callback

    def new_frame(self, f: Frame):
        """
        Call with new frame data for either topic1 or topic2. If a new transformation can be calculated using the
        passed frame, self.trafo_callback will be called in turn.

        :param f: The new frame.
        """
        # store temporarily
        if f.topic == self.topic1:
            self._last1 = f
        else:
            assert f.topic == self.topic2
            self._last2 = f
        if self._last1 is None or self._last2 is None:
            return

        # check if temporary frames are expired
        if self._last1.timestamp_sec - self._last2.timestamp_sec > self._expiry_duration_sec:
            logger.info(f"Frame for {self.topic2} expired.")
            self._last2 = None
            return
        if self._last2.timestamp_sec - self._last1.timestamp_sec > self._expiry_duration_sec:
            logger.info(f"Frame for {self.topic1} expired.")
            self._last1 = None
            return

        # if we have frames for both sensors which are not expired, add them to buffer and calculate transformation
        self._frame_buffer_1.append(self._last1)
        self._frame_buffer_2.append(self._last2)
        self._last1 = None
        self._last2 = None

        self._new_frame_pair()

    @staticmethod
    def calc_marker_location(buffer: deque[Frame]) -> tuple[ReflectorLocation | None, str]:
        """
        Obtain a ReflectorLocation (if found) object from a time series of Frame objects.
        Wraps track_marker.find_marker_single_frame and constructs ReflectorLocation objects.

        :param buffer: a time series of Frame objects
        :return: Tuple (result, status) where result is a ReflectorLocation or None, depending whether the reflector
            was found in the given Frame objects. Status is the status obtained by find_marker_single_frame.
        """
        centers = [f.cluster_centers for f in buffer]
        result, status = find_marker_single_frame(
            centers,
            max_distance=parameters.get_param("maximum neighbor distance"),
            min_velocity=parameters.get_param("minimum velocity"),
            max_vector_angle_rad=2 * np.pi * parameters.get_param("max. vector angle [deg]") / 360,
            max_point_number_change_ratio=parameters.get_param("max_point_number_change_ratio")
        )
        if not result:
            return None, status
        cluster_mean, cluster_index_in_frame = result
        cluster_points = buffer[-1].get_cluster_points(cluster_index_in_frame)
        return ReflectorLocation(cluster_mean, cluster_points, cluster_index_in_frame), status

    def _new_frame_pair(self):
        # first call calculate_marker_location of latest frames
        reflector1, status1 = PairCalibrator.calc_marker_location(self._frame_buffer_1)
        reflector2, status2 = PairCalibrator.calc_marker_location(self._frame_buffer_2)
        # TODO do something with the status field...

        # TODO use some logging system, remove those debug prints
        # logger.info(f"{' ' * 20} status1: {str(status1).ljust(20)} status2: {str(status2).ljust(20)}")

        if reflector1 is None or reflector2 is None:
            # Only continue if reflector is found in both new frames
            logger.debug("New frame pair, reflector NOT found in both frames")
            return

        logger.info("New frame pair, reflector found in both frames")
        # Save the obtained reflector locations
        self.reflector_locations_1.append(reflector1)
        self.reflector_locations_2.append(reflector2)

        if len(self.reflector_locations_1) < 3:
            # we need at least 3 point pairs
            logger.info("Not enough point pairs yet")
            return

        # Recalculate and publish transformation with new data
        weights = self._calculate_weights()
        P, Q, location_filter = None, None, None
        if self.transformation:
            location_filter = self._get_location_filter()
            if sum(location_filter) > 3:
                # only filter if at least 3 points will remain after filtering
                P = np.array([rl.centroid for rl in _filter_list(self.reflector_locations_1, location_filter)])
                Q = np.array([rl.centroid for rl in _filter_list(self.reflector_locations_2, location_filter)])
                weights = list(_filter_list(weights, location_filter))
        if P is None:
            # use unfiltered reflector locations until we have enough data
            P = np.array([rl.centroid for rl in self.reflector_locations_1])
            Q = np.array([rl.centroid for rl in self.reflector_locations_2])
            # weights remains unfiltered as well.

        logger.info("Calculating new transformation (using {0} / {1} point pairs)".format(
            str(len(Q)).rjust(3),
            str(len(self.reflector_locations_1)).rjust(3)
        ))

        self.transformation = calc_transformation_scipy(P, Q, weights)
        if self._trafo_callback:
            self._trafo_callback(self.transformation, self.topic1, self.topic2)

        # broadcast to websocket
        broadcast_pair_metadata(
            self.topic1,
            self.topic2,
            self.transformation,
            len(Q),
            len(self.reflector_locations_1)
        )

    def _calculate_weights(self):
        normal_weight_share = parameters.get_param("normal_cosine_weight_share")
        number_weight_share = parameters.get_param("point_number_weight_share")
        if normal_weight_share == 0 and number_weight_share == 0:
            # both weights were disabled, use unity weights
            return np.ones((len(self.reflector_locations_1)))

        # normal cosine weight for each point pair (choose smaller value per pair)
        normal_cosine_weights = np.min(
            np.stack((
                [rl.normal_cosine_weight for rl in self.reflector_locations_1],
                [rl.normal_cosine_weight for rl in self.reflector_locations_2]
            )),
            axis=0
        )

        # weight from number of points in cluster
        max_points_in_cluster = np.max(
            [rl.number_of_points_in_cluster for rl in self.reflector_locations_1] +
            [rl.number_of_points_in_cluster for rl in self.reflector_locations_2]
        )  # number of points in biggest cluster of all sensor's frames combined

        point_number_weights = np.min(
            np.stack((
                [rl.number_of_points_in_cluster / max_points_in_cluster for rl in self.reflector_locations_1],
                [rl.number_of_points_in_cluster / max_points_in_cluster for rl in self.reflector_locations_2]
            )),
            axis=0
        )  # weight number in each cluster relative to maximum, choose smaller value per pair

        return (
                + normal_weight_share * normal_cosine_weights
                + number_weight_share * point_number_weights
        )

    def _get_location_filter(self) -> np.ndarray:
        """
        Returns a filter boolean array of filters for which an initial transformation must be present.
        Currently, this is only whether the distance between two transformed reflector locations is very large.
        """
        # drop point pairs whose points are comparably far apart from each other
        points1 = np.array([p.centroid for p in self.reflector_locations_1])
        points2 = np.array([p.centroid for p in self.reflector_locations_2])
        points1_transformed = apply_transformation(points1, self.transformation)
        distance = np.linalg.norm(points1_transformed - points2, axis=1)
        filter1 = distance < (np.mean(distance) * float(parameters.get_param("outlier_mean_factor")))

        return filter1


def _filter_list(to_filter, boolean_array) -> Iterable:
    return itertools.compress(to_filter, boolean_array)
