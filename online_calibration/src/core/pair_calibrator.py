import itertools
import json
import logging
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable

import numpy as np
from scipy.spatial import cKDTree
from scipy.linalg import block_diag

from . import parameters
from .frame import Frame
from .locate_reflector.track_marker import find_marker_single_frame
from .reflector_location import ReflectorLocation
from .transformation import Transformation, calc_transformation_scipy, apply_transformation
from .websocket_server import broadcast_pair_metadata
from .observability import (
    compute_observability_metrics,
    build_measurement_covariance,
    parameter_covariance,
    compute_whitened_jacobian_metrics,
)

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
        self.observability_nullspace_vector = None
        self.observability_largest_sv = None
        self.observability_smallest_sv = None
        self.observability_condition_number = None
        self.eigenvalues_P = None  # eigenvalues of covariance matrix of P
        self.uncertainty_propagation_rotation_rad = None
        self.uncertainty_propagation_rotation_deg = None
        self.uncertainty_propagation_translation = None
        self.std_xyz = None
        self.rmse = None
        self.min_eigenvalue  = None
        self.condition_number = None

        # convergence and weighting parameters
        self.minimum_numbers_of_iterations_until_convergence = parameters.get_param("minimum_iterations_until_convergence")
        self.use_covariance_trace_weight = parameters.get_param("use_covariance_trace_weight")
        self.shape_similarity_weight = parameters.get_param("shape_similarity_weight")
        self.shape_similarity_k_factor = parameters.get_param("shape_similarity_k_factor")

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

        # ICP refinement parameters and state
        self.enable_icp_refinement = bool(parameters.get_param("enable_icp_refinement"))
        self.icp_max_iterations = int(parameters.get_param("icp_max_iterations"))
        self.icp_inlier_threshold = float(parameters.get_param("icp_inlier_threshold"))
        self.icp_update_transformation = bool(parameters.get_param("icp_update_transformation"))
        self.icp_fitness_score = None
        self.just_calibrated = False

    def new_frame(self, f: Frame):
        """
        Call with new frame data for either topic1 or topic2. If a new transformation can be calculated using the
        passed frame, self.trafo_callback will be called in turn.

        :param f: The new frame.
        """
        self.just_calibrated = False
        
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

        # Assemble raw cluster points for each pair to build centroid covariances for weighting
        raw_clusters_1_for_weighting = [rl.cluster_points for rl in self.reflector_locations_1]
        raw_clusters_2_for_weighting = [rl.cluster_points for rl in self.reflector_locations_2]

        # Estimate the centroid covariance for each cluster of each sensor separately
        from .observability import centroid_covariance
        centroid_covariances_1 = [centroid_covariance(c) for c in raw_clusters_1_for_weighting]
        centroid_covariances_2 = [centroid_covariance(c) for c in raw_clusters_2_for_weighting]

        weights = self._calculate_weights(centroid_covariances_1, centroid_covariances_2)

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
                    # Also filter the covariance lists that will be used for Sigma_e assembly
                    centroid_covariances_1 = [cov for cov, keep in zip(centroid_covariances_1, outlier_filter) if keep]
                    centroid_covariances_2 = [cov for cov, keep in zip(centroid_covariances_2, outlier_filter) if keep]


        # Optional subsampling of calibration pairs: use every n-th pair
        subsample_step = int(parameters.get_param("calibration_subsample_step"))
        if subsample_step and subsample_step > 1:
            P = P[::subsample_step]
            Q = Q[::subsample_step]
            if isinstance(weights, np.ndarray):
                weights = weights[::subsample_step]
            else:
                weights = np.asarray(weights)[::subsample_step]
            centroid_covariances_1 = centroid_covariances_1[::subsample_step]
            centroid_covariances_2 = centroid_covariances_2[::subsample_step]

        # Ensure enough pairs remain
        if len(Q) < 3:
            logger.info("Not enough point pairs after subsampling/filtering")
            return

        # Calculate transformation based on (optionally subsampled) calibration point cloud
        logger.info("Calculating new transformation (using {0} / {1} point pairs)".format(
            str(len(Q)).rjust(3),
            str(len(self.reflector_locations_1)).rjust(3)
        ))

        self.transformation = calc_transformation_scipy(P, Q, weights)
        
        # The transformation is no longer immediately sent to the callback.
        # Instead, it is processed by the new gating logic below.

        # Apply the transformation to P
        P_transformed_to_Q = np.dot(P, self.transformation.R.T) + self.transformation.t        
        
        ################# CALCULATE STATISTICS #################
        
        # Compute Jacobian for observability analysis
        largest_sv, smallest_sv, condition_number_obs, nullspace_vector, J = compute_observability_metrics(P_transformed_to_Q)
        
        dof_names = ['roll', 'pitch', 'yaw', 'tx', 'ty', 'tz']
        dominant_dof = dof_names[int(np.argmax(np.abs(nullspace_vector)))]

        # Store all three values
        self.observability_nullspace_vector = nullspace_vector
        self.observability_largest_sv   = largest_sv
        self.observability_smallest_sv  = smallest_sv
        self.observability_condition_number = condition_number_obs
        self.dominant_dof = dominant_dof
                
        # --- Raw Clouds, ICP, Raw RQE, RMSE ---
        if self.transformation:
            latest_frame_1 = self._frame_buffer_1[-1]
            latest_frame_2 = self._frame_buffer_2[-1]

            raw_point_cloud_P = latest_frame_1.get_points
            raw_point_cloud_Q = latest_frame_2.get_points

            # raw_P_transformed_to_Q = np.dot(raw_point_cloud_P, self.transformation.R.T) + self.transformation.t

            # optional: ICP refinement on raw point clouds
            if self.enable_icp_refinement and raw_point_cloud_P.shape[0] > 0 and raw_point_cloud_Q.shape[0] > 0:
                R_icp, t_icp, fitness = self._run_icp_point_to_point(
                    src=raw_point_cloud_P,
                    dst=raw_point_cloud_Q,
                    R_init=self.transformation.R,
                    t_init=self.transformation.t,
                    max_iters=self.icp_max_iterations,
                    inlier_thresh=self.icp_inlier_threshold
                )
                # log fitness score and optionally update transformation
                self.icp_fitness_score = fitness
                logger.info(f"ICP refinement: fitness={fitness:.3f}, iters={self.icp_max_iterations}")
                if self.icp_update_transformation:
                    self.transformation.R = np.array(R_icp, copy=True)
                    self.transformation.t = np.array(t_icp, copy=True)
                    
        if self._trafo_callback:
            self._trafo_callback(self.transformation, self.topic1, self.topic2)
        self.just_calibrated = True

        # variances for the parameters
        # rot_var = np.diag(Sigma_params[:3, :3])   # roll, pitch, yaw
        # trans_var = np.diag(Sigma_params[3:, 3:]) # tx, ty, tz
        
        # self.uncertainty_propagation_rotation_rad = np.sqrt(rot_var)
        # self.uncertainty_propagation_rotation_deg = self.uncertainty_propagation_rotation_rad * 180.0 / np.pi
        # self.uncertainty_propagation_translation = np.sqrt(trans_var)

        # calculate pairwise distances in x, y, z dimensions
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
        self.eigenvalues_P = eigenvalues_P
        
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
            self.std_xyz,
            self.rmse,
            self.min_eigenvalue,
            self.condition_number,
            self.icp_fitness_score
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
                "observability_nullspace_vector": self.observability_nullspace_vector,
                "observability_largest_sv": self.observability_largest_sv,
                "observability_smallest_sv": self.observability_smallest_sv,
                "observability_condition_number": self.observability_condition_number,
                "dominant_dof": self.dominant_dof,
                "max_extent_P": max_extent_P,
                "max_extent_Q": max_extent_Q,
                "mean_distances": mean_xyz,
                "std_distances": std_xyz,
                "rmse": rmse,
                "condition_number": condition_number,
                "min_eigenvalue": min_eigenvalue,
                "icp_fitness_score": self.icp_fitness_score,
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
                    "minimum_iterations_until_convergence": parameters.get_param("minimum_iterations_until_convergence"),
                    "use_covariance_trace_weight": parameters.get_param("use_covariance_trace_weight"),
                    "shape_similarity_weight": parameters.get_param("shape_similarity_weight"),
                    "shape_similarity_k_factor": parameters.get_param("shape_similarity_k_factor"),
                    "enable_icefinement": parameters.get_param("enable_icp_refinement"),
                    "icp_max_iterations": parameters.get_param("icp_max_iterations"),
                    "icp_inlier_threshold": parameters.get_param("icp_inlier_threshold"),
                    "icp_update_transformation": parameters.get_param("icp_update_transformation"),
                    "calibration_subsample_step": parameters.get_param("calibration_subsample_step"),
                }
            })
            # write to log file
            with open(self._logfile, "w") as lf:
                json.dump(self._log, lf, cls=_NumpyEncoder, indent=2, allow_nan=False)

    def add_duration_to_log(self, duration_ms: float):
        """Adds the processing duration to the last entry in the log."""
        if self._log and self._log["transformations"]:
            self._log["transformations"][-1]["processing_time_ms"] = duration_ms
            # Re-write the log file with the new data
            with open(self._logfile, "w") as lf:
                json.dump(self._log, lf, cls=_NumpyEncoder, indent=2, allow_nan=False)

    def _calculate_weights(self, centroid_covariances_1: list[np.ndarray], centroid_covariances_2: list[np.ndarray]):
        """
        Calculates the weights for the point pairs by multiplicatively combining different quality metrics.
        Each metric can be enabled or disabled via parameters. The final weight for each pair is the product
        of all enabled component weights.
        """
        num_pairs = len(self.reflector_locations_1)
        final_weights = np.ones(num_pairs)

        # --- 1. Heuristic: Normal Cosine Similarity ---
        normal_weight_factor = parameters.get_param("normal_cosine_weight")
        if normal_weight_factor > 0:
            normal_cosine_scores = np.min(
                np.stack((
                    [rl.normal_cosine_weight for rl in self.reflector_locations_1],
                    [rl.normal_cosine_weight for rl in self.reflector_locations_2]
                )),
                axis=0
            )
            # A score of 1 is good, 0 is bad. Weight = 1 - factor * (1 - score)
            component_weights = 1 - normal_weight_factor * (1 - normal_cosine_scores)
            final_weights *= component_weights

        # --- 2. Heuristic: Point Number ---
        number_weight_factor = parameters.get_param("point_number_weight")
        if number_weight_factor > 0:
            all_points = [rl.number_of_points_in_cluster for rl in self.reflector_locations_1] + \
                         [rl.number_of_points_in_cluster for rl in self.reflector_locations_2]
            max_points_in_cluster = np.max(all_points) if all_points else 1
            
            point_number_scores = np.min(
                np.stack((
                    [rl.number_of_points_in_cluster / max_points_in_cluster for rl in self.reflector_locations_1],
                    [rl.number_of_points_in_cluster / max_points_in_cluster for rl in self.reflector_locations_2]
                )),
                axis=0
            )
            component_weights = 1 - number_weight_factor * (1 - point_number_scores)
            final_weights *= component_weights

        # --- 3. Heuristic: Gaussian Range ---
        gaussian_weight_factor = parameters.get_param("gaussian_range_weight")
        if gaussian_weight_factor > 0:
            max_range_squared_inv = 1 / (200 ** 2)
            min_range_squared_inv = 1 / (0.5 ** 2)
            range_diff = min_range_squared_inv - max_range_squared_inv
            
            if range_diff > 1e-9:
                gaussian_range_scores = np.min(
                    np.stack((
                        np.clip([(rl.range_squared_inv - max_range_squared_inv) / range_diff for rl in self.reflector_locations_1], 0, 1),
                        np.clip([(rl.range_squared_inv - max_range_squared_inv) / range_diff for rl in self.reflector_locations_2], 0, 1)
                    )),
                    axis=0
                )
                component_weights = 1 - gaussian_weight_factor * (1 - gaussian_range_scores)
                final_weights *= component_weights

        # --- 4. Covariance Trace-based Weight ---
        if self.use_covariance_trace_weight:
            with np.errstate(divide='ignore'):
                weights1 = 1.0 / np.array([np.trace(c) for c in centroid_covariances_1]) # sensor 1
                weights2 = 1.0 / np.array([np.trace(c) for c in centroid_covariances_2]) # sensor 2
            
            weights1[np.isinf(weights1)] = 1e12
            weights2[np.isinf(weights2)] = 1e12

            component_weights = np.minimum(weights1, weights2)

            if component_weights.size > 0 and np.mean(component_weights) > 1e-9:
                component_weights /= np.mean(component_weights) # Normalize to mean 1
            
            final_weights *= component_weights

        # --- 5. Shape Similarity Weight ---
        shape_weight_factor = self.shape_similarity_weight
        if shape_weight_factor > 0:
            k = self.shape_similarity_k_factor
            shape_distances = []
            for rl1, rl2 in zip(self.reflector_locations_1, self.reflector_locations_2):
                cluster1 = rl1.cluster_points
                cluster2 = rl2.cluster_points

                if cluster1.shape[0] < 3 or cluster2.shape[0] < 3:
                    shape_distances.append(np.inf)
                    continue

                cov1 = np.cov(cluster1.T)
                cov2 = np.cov(cluster2.T)
                eig1 = np.linalg.eigvalsh(cov1)
                eig2 = np.linalg.eigvalsh(cov2)
                trace1 = np.sum(eig1)
                trace2 = np.sum(eig2)

                if trace1 < 1e-9 or trace2 < 1e-9:
                    shape_distances.append(np.inf)
                    continue

                form_vector1 = np.sort(eig1) / trace1
                form_vector2 = np.sort(eig2) / trace2
                
                dist = np.linalg.norm(form_vector1 - form_vector2)
                shape_distances.append(dist)
            
            shape_distances = np.array(shape_distances)
            # An exp-based score: 1 for perfect match (dist=0), -> 0 for large dist
            exp_scores = np.exp(-k * shape_distances)
            # Final component weight
            component_weights = 1 - shape_weight_factor * (1 - exp_scores)
            final_weights *= component_weights

        return final_weights


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
    
    def _run_icp_point_to_point(self, src: np.ndarray, dst: np.ndarray, R_init: np.ndarray, t_init: np.ndarray,
                                max_iters: int, inlier_thresh: float) -> tuple[np.ndarray, np.ndarray, float]:
        """
        Point-to-point ICP using Open3D's implementation.
        - Aligns `src` (Nx3) to `dst` (Mx3)
        - Uses `inlier_thresh` as the max correspondence distance
        - Returns refined (R, t) and fitness in [0,1]
        """
        if src.shape[0] == 0 or dst.shape[0] == 0:
            logger.warning("ICP refinement skipped due to empty source or destination point cloud.")
            return R_init, t_init, 0.0

        try:
            import open3d as o3d
        except (ImportError, OSError) as e:
            # ImportError: package not installed
            # OSError: missing native libs (e.g. libGL.so.1) in headless/container environment
            logger.warning(f"Open3D unavailable (reason: {e}). Skipping ICP refinement.")
            return R_init, t_init, 0.0

        src_pc = o3d.geometry.PointCloud()
        dst_pc = o3d.geometry.PointCloud()
        src_pc.points = o3d.utility.Vector3dVector(np.asarray(src, dtype=np.float64))
        dst_pc.points = o3d.utility.Vector3dVector(np.asarray(dst, dtype=np.float64))

        T_init = np.eye(4, dtype=np.float64)
        T_init[:3, :3] = np.asarray(R_init, dtype=np.float64)
        T_init[:3, 3] = np.asarray(t_init, dtype=np.float64)

        estimation = o3d.pipelines.registration.TransformationEstimationPointToPoint()
        criteria = o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=int(max_iters))
        
        result = o3d.pipelines.registration.registration_icp(
            src_pc,
            dst_pc,
            float(inlier_thresh),
            T_init,
            estimation,
            criteria
        )
        
        T = np.asarray(result.transformation, dtype=np.float64)
        R_ref = T[:3, :3]
        t_ref = T[:3, 3]
        fitness = float(result.fitness)

        return R_ref, t_ref, fitness
    

    def check_convergence(self) -> bool:
        """
        Check if this pair calibrator has converged.
        Currently based on minimum number of iterations.
        """
        return len(self.reflector_locations_1) >= self.minimum_numbers_of_iterations_until_convergence

def _filter_list(to_filter, boolean_array) -> Iterable:
    return itertools.compress(to_filter, boolean_array)


class _NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return json.JSONEncoder.default(self, obj)
