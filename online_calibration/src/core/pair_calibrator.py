import itertools
from collections import deque
from typing import Callable

import numpy as np

from . import parameters
from .frame import Frame
from .locate_reflector.track_marker import find_marker_single_frame
from .reflector_location import ReflectorLocation
from .transformation import Transformation, calc_transformation_scipy, apply_transformation


class PairCalibrator:

    def __init__(self, topic1: str, topic2: str, trafo_callback: Callable[[Transformation, str, str], None] | None):
        # Maximum age for a frame before it expires
        self._expiry_duration_sec = 1.0 / float(parameters.get_param("sample_rate_Hz")) / 2
        self._frame_buffer_1: deque[Frame] = deque(maxlen=int(parameters.get_param("window size")))
        self._frame_buffer_2: deque[Frame] = deque(maxlen=int(parameters.get_param("window size")))
        self._topic1 = topic1
        self._topic2 = topic2
        self._last1: Frame | None = None
        self._last2: Frame | None = None
        self.reflector_locations_1: list[ReflectorLocation] = []
        self.reflector_locations_2: list[ReflectorLocation] = []
        self.transformation: Transformation | None = None
        self._trafo_callback = trafo_callback

    def new_frame(self, f: Frame):
        # store temporarily
        if f.topic == self._topic1:
            self._last1 = f
        else:
            assert f.topic == self._topic2
            self._last2 = f
        if self._last1 is None or self._last2 is None:
            return

        # check if temporary frames are expired
        if self._last1.timestamp_sec - self._last2.timestamp_sec > self._expiry_duration_sec:
            print(f"Frame for {self._topic2} expired.")
            self._last2 = None
            return
        if self._last2.timestamp_sec - self._last1.timestamp_sec > self._expiry_duration_sec:
            print(f"Frame for {self._topic1} expired.")
            self._last1 = None
            return

        # if we have frames for both sensors which are not expired, add them to buffer and calculate transformation
        self._frame_buffer_1.append(self._last1)
        self._frame_buffer_2.append(self._last2)
        self._last1 = None
        self._last2 = None

        self.new_frame_pair()

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

    def new_frame_pair(self):
        print("New frame pair")
        # first call calculate_marker_location of latest frames
        reflector1, status1 = PairCalibrator.calc_marker_location(self._frame_buffer_1)
        reflector2, status2 = PairCalibrator.calc_marker_location(self._frame_buffer_2)
        # TODO do something with the status field...

        # TODO use some logging system, remove those debug prints
        # print(f"{' ' * 20} status1: {str(status1).ljust(20)} status2: {str(status2).ljust(20)}")

        if reflector1 is None or reflector2 is None:
            # Only continue if reflector is found in both new frames
            return

        print("Reflector found in both frames")
        # Save the obtained reflector locations
        self.reflector_locations_1.append(reflector1)
        self.reflector_locations_2.append(reflector2)

        if len(self.reflector_locations_1) < 3:
            # we need at least 3 point pairs
            print("Not enough point pairs yet")
            return

        print(f"Calculating new transformation (using {str(len(self.reflector_locations_1)).rjust(3)} points)")
        # Recalculate and publish transformation with new data

        P, Q = None, None
        if self.transformation:
            filtered1, filtered2 = self.get_locations_filtered()
            if len(filtered1) > 3:
                P = np.array([rl.centroid for rl in filtered1])
                Q = np.array([rl.centroid for rl in filtered1])
                print("using filtered reflector locations")
        if not P:
            # use unfiltered reflector locations until we have enough data
            P = np.array([rl.centroid for rl in self.reflector_locations_1])
            Q = np.array([rl.centroid for rl in self.reflector_locations_2])
            print("using UNfiltered reflector locations")

        weights = np.array([
            min(rl1.weight, rl2.weight)
            for rl1, rl2 in zip(self.reflector_locations_1, self.reflector_locations_2)
        ])
        self.transformation = calc_transformation_scipy(P, Q, weights)
        if self._trafo_callback:
            self._trafo_callback(self.transformation, self._topic1, self._topic2)

    def get_locations_filtered(self):
        """
        Applies filters based on all reflector locations found yet.

        :return: Filtered version of
        """
        # drop point pairs whose points are comparably far apart from each other
        points1 = np.array([p.centroid for p in self.reflector_locations_1])
        points2 = np.array([p.centroid for p in self.reflector_locations_2])
        points1_transformed = apply_transformation(points1, self.transformation)
        distance = np.linalg.norm(points1_transformed - points2, axis=1)
        filter1 = distance > np.mean(distance) * float(parameters.get_param("outlier_mean_factor"))

        return (
            list(itertools.compress(self.reflector_locations_1, filter1)),
            list(itertools.compress(self.reflector_locations_2, filter1))
        )
