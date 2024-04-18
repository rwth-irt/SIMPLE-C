#!/usr/bin/env python
import collections
import json
import pathlib

import numpy as np
import rospy
from sensor_msgs import point_cloud2

from locate_reflector.find_cluster_centers import get_cluster_centers
from locate_reflector.track_marker import track_marker, positions_from_indices
from transformation.calculate_transformation import filter_locations, calc_transformation


class OnlineCalibrator:
    def __init__(self, params: dict, sensor_pairs: list[tuple[str, str]]):
        """
        Create an object to manage online calibration for multiple detectors.
        Will subscribe to all relevant sensors and publish current calibration info
        in a ROS topic as soon as it is available. TODO do that!

        :param params: Parameter dict (as obtained from parameter JSON file)
        :param sensor_pairs: list with 2-tuples of sensor topics. One sensor may appear in multiple tuples.
          Will subscribe to all specified sensor's ROS topics and calculate transformations for all those
          sensor pairs if enough data is present.
        """
        # extract unique topics from pairs
        self.sensor_pairs = sensor_pairs
        topics = set()
        for a, b in sensor_pairs:
            topics.add(a)
            topics.add(b)

        # create circular buffers
        self.reflector_position_buffer: dict[str, collections.deque[np.ndarray]] = {
            t: collections.deque(maxlen=int(params["velocity lookahead"]))
            for t in topics
        }
        self.frame_buffer: dict[str, collections.deque[np.ndarray]] = {
            t: collections.deque(maxlen=int(params["velocity lookahead"]))
            for t in topics
        }

        # initialize ROS node and subscribe to sensors
        rospy.init_node("online_calibration", anonymous=True)
        for t in topics:
            # subscribe to all sensors
            rospy.Subscriber(t, point_cloud2, lambda pc2: self.on_message(t, pc2))

    def on_message(self, topic: str, pc2: point_cloud2):
        data = np.array(point_cloud2.read_points_list(pc2, skip_nans=False))
        # TODO: fix performance problem. See `rosbag_to_numpy.py` as well.
        self.frame_buffer[topic].append(data)

        centers = get_cluster_centers(
            data,
            rel_intensity_threshold=params["relative intensity threshold"],
            DBSCAN_epsilon=params["DBSCAN epsilon"],
            DBSCAN_min_samples=int(params["DBSCAN min samples"])
        )
        indices = track_marker(
            centers,
            max_distance=params["maximum neighbor distance"],
            min_velocity=params["minimum velocity"],
            velocity_lookahead=int(params["velocity lookahead"]),
            max_vector_angle_rad=2 * np.pi * params["max. vector angle [deg]"] / 360,
        )  # TODO this function will be changed!
        # TODO somehow published a "target locked" state per sensor for live validation.
        self.reflector_position_buffer[topic].append(positions_from_indices(indices, centers))
        # TODO
        #  deal with timing somehow: Currently, if one detector stops to send frames, its old frames
        #  will be used forever without throwing an error!
        self.update_transformation()

    def update_transformation(self):
        for sensors in self.sensor_pairs:
            filtered = filter_locations(self.reflector_position_buffer, sensors)
            if len(filtered[sensors[0]]) >= 3:
                # trafo calculation is possible
                R, t = calc_transformation(filtered[sensors[0]], filtered[sensors[1]])
                # TODO
                #  use scipy instead
                #  publish trafo + sensitivity


if __name__ == '__main__':
    # load reflector tracking parameters
    paramfile = pathlib.Path(__file__).parent.parent.absolute() / "default_params.json"
    # TODO look for replacement?
    with open(paramfile, "r") as f:
        params = json.load(f)
    sensor_pairs = []  # TODO
    OnlineCalibrator(params, sensor_pairs)
    rospy.spin()
