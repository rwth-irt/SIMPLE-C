#!/usr/bin/env python
import collections
import json
import pathlib
from typing import Union

import numpy as np
import rospy
from geometry_msgs.msg import Transform, Vector3, Quaternion
from sensor_msgs import point_cloud2
from std_msgs.msg import String

from locate_reflector.find_cluster_centers import get_cluster_centers_single_frame
from locate_reflector.track_marker import find_marker_single_frame
from transformation.calculate_transformation import filter_locations, calc_transformation_scipy


class OnlineCalibrator:
    def __init__(self, params: dict, sensor_pairs: list[tuple[str, str]]):
        """
        Create an object to manage online calibration for multiple detectors.
        Will subscribe to all relevant sensors and publish current calibration info
        in a ROS topic as soon as it is available.

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

        self.reflector_positions: dict[str, list[Union[None, np.ndarray]]] = {
            t: [] for t in topics
        }
        # circular buffer for frames
        self.frame_buffer: dict[str, collections.deque[np.ndarray]] = {
            t: collections.deque(maxlen=int(params["window size"]))
            for t in topics
        }
        # counts frame indices per sensor to avoid calculating transformation with different
        # amount of frames received per sensor
        self.frame_index: dict[str, int] = {
            t: 0 for t in topics
        }

        # initialize ROS node, add publishers and subscribe to sensors
        self.trafo_pub = rospy.Publisher("transformation", Transform, queue_size=10)
        self.state_pub = rospy.Publisher("calibration_state", String, queue_size=10)
        # TODO get topic names from config

        for t in topics:
            # subscribe to all sensors
            rospy.Subscriber(t, point_cloud2, lambda pc2: self.on_message(t, pc2))
        rospy.init_node("online_calibration", anonymous=True)

    def on_message(self, topic: str, pc2: point_cloud2):
        data = np.array(point_cloud2.read_points_list(pc2, skip_nans=False))
        # TODO: fix performance problem. See `rosbag_to_numpy.py` as well.
        self.frame_buffer[topic].append(data)

        centers = get_cluster_centers_single_frame(
            data,
            rel_intensity_threshold=params["relative intensity threshold"],
            DBSCAN_epsilon=params["DBSCAN epsilon"],
            DBSCAN_min_samples=int(params["DBSCAN min samples"])
        )
        search_result, search_state = find_marker_single_frame(
            centers,
            max_distance=params["maximum neighbor distance"],
            min_velocity=params["minimum velocity"],
            max_vector_angle_rad=2 * np.pi * params["max. vector angle [deg]"] / 360,
        )
        self.frame_index[topic] += 1  # TODO somehow use timestamps
        if search_result:
            reflector_pos, index = search_result
            self.reflector_positions[topic].append(reflector_pos)
            self.update_transformation()
            # a new transformation can not be calculated without new data, so only try if a new data point is added
        else:
            self.reflector_positions[topic].append(None)
        self.state_pub.publish(String(search_state))

    def update_transformation(self):
        index = None
        for i in self.frame_index.values():
            if index is not None and i != index:
                # wait until number of frames is the same for each sensor
                if abs(i - index) > 1:
                    # TODO how to deal with this error?
                    print("One sensor seems to drop frames!")
                return
            index = i

        del index

        for sensors in self.sensor_pairs:
            filtered = filter_locations(self.reflector_positions, sensors)
            if len(filtered[sensors[0]]) >= 3:
                # trafo calculation is possible
                a = filtered[sensors[0]]
                b = filtered[sensors[1]]
                R, t, Rq, sensitivity = calc_transformation_scipy(a, b)

                # publish
                self.trafo_pub.publish(Transform(
                    translation=Vector3(x=t[0], y=t[1], z=t[2]),
                    rotation=Quaternion(x=Rq[0], y=Rq[1], z=Rq[2], w=Rq[3])
                ))


if __name__ == '__main__':
    # load reflector tracking parameters
    paramfile = pathlib.Path(__file__).parent.parent.absolute() / "default_params.json"
    # TODO look for manually passed file/get from somewhere?
    with open(paramfile, "r") as f:
        params = json.load(f)
    sensor_pairs = []  # TODO get from somewhere
    OnlineCalibrator(params, sensor_pairs)
    rospy.spin()
