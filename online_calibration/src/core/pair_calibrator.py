import itertools
import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
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

        # initialize values for convergence check
        self.std_xyz = None
        self.rmse = None
        self.min_eigenvalue  = None
        self.condition_number = None
        self.std_threshold = parameters.get_param("std_threshold")
        self.rmse_threshold = parameters.get_param("rmse_threshold")
        self.min_eigenvalue_threshold = parameters.get_param("min_eigenvalue_threshold")
        self.condition_number_threshold = parameters.get_param("condition_number_threshold")

        # logging for evaluation
        self._log = None
        self._logfile = None
        logdir = parameters.get_param("eval_log_dir")
        if logdir.lower() != "none":
            # logging enabled, prepare filename (which will stay the same for this session)
            logpath = Path(logdir)
            logpath.mkdir(parents=True, exist_ok=True)
            self._log = {"transformations": []}  # more metadata may be added later on
            timestamp = datetime.now().strftime('%Y-%m-%d_%H-%M-%S')
            filename = f"calib_log_{timestamp}_{self.topic1}_{self.topic2}.json"
            filename = filename.replace("/", "")  # ROS topics often contain slashes, bad for filenames.
            # (ROS topics often contain slashes, bad for filenames.)

            self._logfile = logpath / filename
            logger.info(f"Logging to file: {self._logfile}")

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

        # Output reflector "tracking"/detection status (one of: "UNIQUE_MATCH", "NO_MATCH", "MULTIPLE_MATCHES")
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

        # Get calibration point cloud based on current reflector_locations
        P = np.array([rl.centroid for rl in self.reflector_locations_1])
        Q = np.array([rl.centroid for rl in self.reflector_locations_2])
        weights = self._calculate_weights()

        # Apply filters that depend on an existing transformation
        # i.e. "adaptive outlier rejection"
        if parameters.get_param("disable_outlier_rejection"):
            logger.warning("Skipping outlier rejection! Only use this for tests!")
        else:
            if self.transformation:
                outlier_filter = self._get_calib_pointcloud_outlier_filter()
                if sum(outlier_filter) > 3:
                    # only filter if at least 3 points will remain after filtering, otherwise leave unchanged
                    P = P[outlier_filter]
                    Q = Q[outlier_filter]
                    weights = weights[outlier_filter]  # (need to filter weights as well!)


        # Calculate transformation based on calibration point cloud
        logger.info("Calculating new transformation (using {0} / {1} point pairs)".format(
            str(len(Q)).rjust(3),
            str(len(self.reflector_locations_1)).rjust(3)
        ))

        self.transformation = calc_transformation_scipy(P, Q, weights)
        if self._trafo_callback:
            self._trafo_callback(self.transformation, self.topic1, self.topic2)

        # Apply the transformation to P
        P_transformed_to_Q = np.dot(P, self.transformation.R.T) + self.transformation.t

        # Calculate pairwise distances in x, y, z dimensions
        pointwise_distances = np.abs(Q - P_transformed_to_Q)

        # Calculate mean and standard deviation and rmse of distances
        mean_xyz = np.mean(pointwise_distances, axis=0)
        std_xyz = np.std(pointwise_distances, axis=0)
        self.std_xyz = std_xyz
        
        squared_euclidean_distances = np.sum(np.square(pointwise_distances), axis=1)
        mse = np.mean(squared_euclidean_distances)
        rmse = np.sqrt(mse)
        self.rmse = rmse
        
        cov_P = np.cov(P.T)
        eigenvalues_P = np.linalg.eigvalsh(cov_P) # eigenvectors of covariance matrix give direction of maximum/minumum variance of point cloud
        
        if self.eigenvalues_P is not None and len(self.eigenvalues_P) == 3:
            min_eigenvalue = eigenvalues_P[0] # smallest eigenvalue
            max_eigenvalue = eigenvalues_P[2] # largest eigenvalue
            
            self.min_eigenvalue = min_eigenvalue

            # avoid division by zero
            if max_eigenvalue > 1e-9:
                 condition_number = min_eigenvalue / max_eigenvalue # should be > threshold (closer to 1)
            else:
                 condition_number = 0.0 # degenerate case
            self.condition_number = condition_number

        # broadcast to websocket
        broadcast_pair_metadata(
            self.topic1,
            self.topic2,
            self.transformation,
            len(Q),
            len(self.reflector_locations_1),
            std_xyz,
            rmse,
            min_eigenvalue,
            condition_number,
        )

        if self._log is not None:

            # calculate maximum spread in x, y, z dimensions for P
            max_extent_P = np.ptp(P, axis=0)

            # calculate maximum spread in x, y, z dimensions for Q
            max_extent_Q = np.ptp(Q, axis=0)

            # append logging information
            self._log["transformations"].append({
                "R": self.transformation.R,
                "R_sensitivity": self.transformation.R_sensitivity,
                "t": self.transformation.t,
                "topic_from": self.topic1,
                "topic_to": self.topic2,
                "point_pairs_used": len(Q),
                "point_pairs_total": len(self.reflector_locations_1),
                "max_extent_P": max_extent_P,
                "max_extent_Q": max_extent_Q,
                "mean_distances": mean_xyz,
                "std_distances": std_xyz,
                "rmse": rmse,
                "condition_number": condition_number,
                "min_eigenvalue": min_eigenvalue,
                "parameters": {
                    "rel_intensity_threshold": parameters.get_param("relative intensity threshold"),
                    "DBSCAN_epsilon": parameters.get_param("DBSCAN epsilon"),
                    "DBSCAN_min_samples": parameters.get_param("DBSCAN min samples"),
                    "max_neighbor_distance": parameters.get_param("maximum neighbor distance"),
                    "min_velocity": parameters.get_param("minimum velocity"),
                    "window_size": parameters.get_param("window size"),
                    "max_vector_angle_deg": parameters.get_param("max. vector angle [deg]"),
                    "outlier_mean_factor": parameters.get_param("outlier_mean_factor"),
                    "max_point_number_change_ratio": parameters.get_param("max_point_number_change_ratio"),
                    "normal_cosine_weight": parameters.get_param("normal_cosine_weight"),
                    "point_number_weight": parameters.get_param("point_number_weight"),
                    "gaussian_range_weight": parameters.get_param("gaussian_range_weight"),
                    "std_threshold": str(parameters.get_param("std_threshold")),
                    "minimum_iterations_until_convergence": parameters.get_param("minimum_iterations_until_convergence"),
                    "rmse_threshold": parameters.get_param("rmse_threshold"),
                    "min_eigenvalue_threshold": parameters.get_param("min_eigenvalue_threshold"),
                    "condition_number_threshold": parameters.get_param("condition_number_threshold")
                }
            })
            # write to log file
            with open(self._logfile, "w") as lf:
                json.dump(self._log, lf, cls=_NumpyEncoder, indent=2, allow_nan=False)

    def _calculate_weights(self):
        normal_weight = parameters.get_param("normal_cosine_weight")
        number_weight = parameters.get_param("point_number_weight")
        gaussian_weight = parameters.get_param("gaussian_range_weight")

        if normal_weight == 0 and number_weight == 0 and gaussian_weight == 0:
            # use unity weights if no subweights are used
            return np.ones((len(self.reflector_locations_1)))

        if normal_weight == 0:
            # use unity weights if normal weight is not used
            normal_cosine_weights = np.ones((len(self.reflector_locations_1)))
        else:
            # normal cosine weight for each point pair (choose smaller value per pair)
            normal_cosine_weights = np.min(
                np.stack((
                    [rl.normal_cosine_weight for rl in self.reflector_locations_1],
                    [rl.normal_cosine_weight for rl in self.reflector_locations_2]
                )),
                axis=0
            )

        if number_weight == 0:
            # use unity weights if number weight is not used
            point_number_weights = np.ones((len(self.reflector_locations_1)))
        else:
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

        if gaussian_weight == 0:
            # use unity weights if gaussian range weight is not used
            gaussian_range_weights = np.ones((len(self.reflector_locations_1)))
        else:
            # maximum squared range for normalization
            max_range_squared_inv = 1 / (200 ** 2)
            min_range_squared_inv = 1 / (0.5 ** 2)

            # weight for gaussian range uncertainty, take lowest weight (maximum range)
            gaussian_range_weights = np.min(
                np.stack((
                    [(rl.range_squared_inv - max_range_squared_inv) / (min_range_squared_inv - max_range_squared_inv)
                     for rl in self.reflector_locations_1],
                    [(rl.range_squared_inv - max_range_squared_inv) / (min_range_squared_inv - max_range_squared_inv)
                     for rl in self.reflector_locations_2]
                )),
                axis=0
            )

        # link the subweights via multiplication
        return normal_cosine_weights * point_number_weights * gaussian_range_weights


    def _get_calib_pointcloud_outlier_filter(self) -> np.ndarray:
        """
        Returns a filter boolean array to filter the current calibration point cloud with.
        This step is also referred to as "adaptive outlier rejection".
        The resulting filter array is true for all point pairs in which the distance of the two
        points **after applying the transformation** is less than `outlier_mean_factor * mean_distance`
        with `mean_distance` being the mean distance of all point pairs in the current calibration point cloud.
        """
        assert self.transformation is not None
        # drop point pairs whose points are comparably far apart from each other
        points1 = np.array([p.centroid for p in self.reflector_locations_1])
        points2 = np.array([p.centroid for p in self.reflector_locations_2])
        points1_transformed = apply_transformation(points1, self.transformation)
        distance = np.linalg.norm(points1_transformed - points2, axis=1)
        filter1 = distance < (np.mean(distance) * float(parameters.get_param("outlier_mean_factor")))

        return filter1
    
    def check_convergence(self) -> bool:
        """
        Check if the convergence criteria are met for this PairCalibrator.
        :return: True if convergence is achieved, False otherwise.
        """

        if self.std_xyz is None:
            logger.debug(f"ConvCheck ({self.topic1}<->{self.topic2}): Failed - std_xyz is None.")
            return False
        if self.rmse is None:
            logger.debug(f"ConvCheck ({self.topic1}<->{self.topic2}): Failed - rmse is None.")
            return False
        if self.eigenvalues_P is None:
            logger.debug(f"ConvCheck ({self.topic1}<->{self.topic2}): Failed - eigenvalues_P is None.")
            return False
        if self.condition_number is None:
             logger.debug(f"ConvCheck ({self.topic1}<->{self.topic2}): Failed - condition_number is None.")
             return False
        if len(self.std_xyz) != 3 or len(self.std_threshold) != 3:
             logger.error(f"ConvCheck ({self.topic1}<->{self.topic2}): std_xyz or threshold length mismatch.")
             return False
        if len(self.eigenvalues_P) != 3:
             logger.error(f"ConvCheck ({self.topic1}<->{self.topic2}): eigenvalues_P length mismatch.")
             return False
         
        std_converged = (self.std_xyz[0] < self.std_threshold[0] and
                         self.std_xyz[1] < self.std_threshold[1] and
                         self.std_xyz[2] < self.std_threshold[2])

        rmse_converged = self.rmse < self.rmse_threshold
        
        min_eigenvalue_converged = self.min_eigenvalue > self.min_eigenvalue_threshold
        
        cond_condition_number_converged = self.condition_number > self.condition_number_threshold

        all_converged = std_converged and rmse_converged and min_eigenvalue_converged and cond_condition_number_converged
        
        if all_converged:
            logger.info(f"ConvCheck ({self.topic1}<->{self.topic2}): Converged!")
            return True
        
        return False

def _filter_list(to_filter, boolean_array) -> Iterable:
    return itertools.compress(to_filter, boolean_array)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
