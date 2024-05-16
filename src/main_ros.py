#!/usr/bin/env python

from collections import deque
from datetime import datetime, timedelta

import numpy as np
import rclpy
from geometry_msgs.msg import TransformStamped
from rclpy.node import Node
from sensor_msgs import point_cloud2

import parameters
from frame import Frame
from locate_reflector.track_marker import find_marker_single_frame
from parameters import params
from reflector_location import ReflectorLocation
from transformation import calc_transformation_scipy, Transformation

# Maximum age for a frame before it expires
expiry_duration = timedelta(seconds=1 / float(params["sample_rate_Hz"]) / 2)


class OnlineCalibrator(Node):
    def __init__(self, sensor_pairs: list[tuple[str, str]]):
        """
        Create an object to manage online calibration for multiple Lidar sensors/sensor pairs.
        Will subscribe to all relevant sensors and pass frames down to respective PairCalibrator instances.
        These will then publish the current transformations as soon as they are available.

        :param sensor_pairs: list with 2-tuples of sensor topics. One sensor may appear in multiple tuples.
        """

        # init ROS
        super().__init__("online_calibration")
        trafo_publisher = self.create_publisher(TransformStamped, "transformations", queue_size=10)

        topics = set()  # collect the topics we have to subscribe to
        for a, b in sensor_pairs:
            topics.add(a)
            topics.add(b)

        # a dict which holds a list of interested paircalibrators per topic
        self.pair_calibrators: dict[str, list[PairCalibrator]] = {
            topic: [] for topic in topics
        }

        # initialize ROS node, add publishers and subscribe to sensors
        for topic in topics:
            # subscribe to all sensors
            self.create_subscription(
                point_cloud2.PointCloud2,
                topic,
                lambda pc2: self.on_message(topic, pc2)
            )

        # create pair calibrators
        for a, b in sensor_pairs:
            pc = PairCalibrator(self, a, b, trafo_publisher)
            self.pair_calibrators[a].append(pc)
            self.pair_calibrators[b].append(pc)

    def on_message(self, topic: str, pc2: point_cloud2):
        data = np.array(point_cloud2.read_points_list_numpy(pc2, skip_nans=True))
        frame = Frame(data, datetime.now()) # TODO the frame should get the original timestamp from the sensor not from system
        # pass the new frame to all interested PairCalibrators, which will perform
        # buffering and calculate a transformation if possible
        for pc in self.pair_calibrators[topic]:
            pc.new_frame(frame, topic)


class PairCalibrator:

    def __init__(self, node, topic1: str, topic2: str, trafo_publisher):
        self.node = node
        self.frame_buffer_1: deque[Frame] = deque(maxlen=int(params["window size"]))
        self.frame_buffer_2: deque[Frame] = deque(maxlen=int(params["window size"]))
        self.topic1 = topic1
        self.topic2 = topic2
        self.last1: Frame | None = None
        self.last2: Frame | None = None
        self.reflector_locations_1: list[ReflectorLocation] = []
        self.reflector_locations_2: list[ReflectorLocation] = []
        self.transformation: Transformation | None = None
        self.trafo_publisher = trafo_publisher

    def new_frame(self, f: Frame, topic: str):
        if topic == self.topic1:
            self.last1 = f
        else:
            assert topic == self.topic2
            self.last2 = f
        # TODO this checks temporal difference between two measurements in the same sensor not across two sensors
        # TODO do not use current system time but timestamps of the measurements
        # check if last frames are expired
        if datetime.now() - self.last1.timestamp > expiry_duration:
            self.last1 = None
        if datetime.now() - self.last2.timestamp > expiry_duration:
            self.last2 = None
        if self.last1 is None or self.last2 is None:
            return
        # if we have frames for both sensors which are not expired, add them to buffer and calculate transformation
        self.frame_buffer_1.append(self.last1)
        self.frame_buffer_2.append(self.last2)
        self.last1 = None
        self.last2 = None

        # TODO transformation shall only be called if the buffer contain > 3 points
        self.update_transformation()

    @staticmethod
    def calc_marker_location(buffer: deque[Frame]):
        centers = [f.cluster_centers for f in buffer]
        return find_marker_single_frame(
            centers,
            max_distance=params["maximum neighbor distance"],
            min_velocity=params["minimum velocity"],
            max_vector_angle_rad=2 * np.pi * params["max. vector angle [deg]"] / 360,
        )

    def update_transformation(self):
        # first call calculate_marker_location of latest frames
        result1, status1 = PairCalibrator.calc_marker_location(self.frame_buffer_1)
        result2, status2 = PairCalibrator.calc_marker_location(self.frame_buffer_2)
        # TODO do something with the status field...

        if result1 is None or result2 is None:
            # Only continue if reflector is found in both new frames
            return

        # Save the obtained reflector locations
        cluster1, index1 = result1
        cluster1points = self.frame_buffer_1[-1].get_cluster_points(index1)
        self.reflector_locations_1.append(ReflectorLocation(cluster1, cluster1points))

        cluster2, index2 = result2
        cluster2points = self.frame_buffer_2[-1].get_cluster_points(index2)
        self.reflector_locations_2.append(ReflectorLocation(cluster2, cluster2points))

        # Recalculate and publish transformation with new data
        P = np.ndarray([rl.cluster_mean[:3] for rl in self.reflector_locations_1])
        Q = np.ndarray([rl.cluster_mean[:3] for rl in self.reflector_locations_2])
        # TODO discuss how to calculate the single weight for each point pair?
        weights = np.array([
            min(rl1.weight, rl2.weight)
            for rl1, rl2 in zip(self.reflector_locations_1, self.reflector_locations_2)
        ])

        # TODO Here the weights matrix needs to be updated if weights based on prior transformations are available
        self.publish_transformation(calc_transformation_scipy(P, Q, weights))

    def publish_transformation(self, trafo: Transformation):
        self.transformation = trafo
        # Adapted from http://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Broadcaster-Py.html
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        # Trafo is chosen such that it transforms P (frame1) to Q (frame2)
        t.header.frame_id = self.topic2
        t.child_frame_id = self.topic1
        t.transform.translation.x = trafo.t[0]
        t.transform.translation.y = trafo.t[1]
        t.transform.translation.z = trafo.t[2]
        t.transform.rotation.x = trafo.R_quat[0]
        t.transform.rotation.y = trafo.R_quat[1]
        t.transform.rotation.z = trafo.R_quat[2]
        t.transform.rotation.w = trafo.R_quat[3]
        self.trafo_publisher.publish(t)


def main(args=None):
    rclpy.init(args=args)

    # TODO get parameter path and sensor pairs from some config/parameters
    parameters.init()
    DEBUG_PAIRS = [("/rslidar_points_l", "/rslidar_points_r")]
    calibrator = OnlineCalibrator(DEBUG_PAIRS)
    try:
        rclpy.spin(calibrator)
    except KeyboardInterrupt:
        print("Got KeyboardInterrupt, stopping.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
