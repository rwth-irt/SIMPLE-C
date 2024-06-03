from datetime import datetime

import numpy as np
from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from sensor_msgs_py import point_cloud2

from shared import parameters
from shared.frame import Frame
from shared.pair_calibrator import PairCalibrator
from shared.transformation import Transformation


class OnlineCalibrator(Node):
    def __init__(self):
        """
        Create an object to manage online calibration for multiple Lidar sensors/sensor pairs.
        Will subscribe to all relevant sensors and pass frames down to respective PairCalibrator instances.
        These will then publish the current transformations as soon as they are available.

        The list of sensor pairs as well as the other parameters for reflector detection
        are read from ROS parameters.
        """

        # init ROS
        super().__init__("online_calibration")
        parameters.ros_declare_parameters(self)
        parameters.init_from_rosnode(self)

        # Get detector pairs from ROS parameter "sensor_pairs"
        self.declare_parameter(
            name="sensor_pairs",
            descriptor=ParameterDescriptor(
                name="sensor_pairs",
                type=ParameterType.PARAMETER_STRING
            )
        )
        sensor_pairs_raw = str(self.get_parameter("sensor_pairs").get_parameter_value().string_value)
        sensor_pairs = [
            list(map(str.strip, sp.split(",")))
            for sp in sensor_pairs_raw.split(";")
        ]
        print(f"Parsed the following sensor pairs: {sensor_pairs}")  # debug/info print

        self.trafo_publisher = self.create_publisher(TransformStamped, "transformations", 10)

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
                lambda pc2, topic=topic: self.on_message(topic, pc2),
                10
            )

        # create pair calibrators
        for a, b in sensor_pairs:
            pc = PairCalibrator(a, b, self.new_transformation)
            self.pair_calibrators[a].append(pc)
            self.pair_calibrators[b].append(pc)

        print("Waiting for sensor data...")

    def on_message(self, topic: str, pc2: point_cloud2):
        data = np.array(point_cloud2.read_points_numpy(pc2, skip_nans=True))
        frame = Frame(data, datetime.now(), topic)
        # TODO the frame should get the original timestamp from the sensor not from system
        # pass the new frame to all interested PairCalibrators, which will perform
        # buffering and calculate a transformation if possible
        for pc in self.pair_calibrators[topic]:
            pc.new_frame(frame, topic)

    def new_transformation(self, trafo: Transformation, P_topic: str, Q_topic: str):
        """
        This is a callback method which will be called by PairCalibrators as soon as a new transformation is available.
        Publishes the transformation as stamped transformation (including reference frames) in ROS.

        :param trafo: the transformation
        :param P_topic: the name of the "P" frame (see `transformation.py` for explanation)
        :param Q_topic: the name of the "Q" frame (see `transformation.py` for explanation)
        :return:
        """
        assert isinstance(self, OnlineCalibrator)
        # TODO if this fails, the self-binding probably didn't work when passing the instance method as a callback

        print(f"New transformation for '{P_topic}' --> '{Q_topic}':\nR=")
        print(trafo.R)
        print("t =")
        print(trafo.t)
        print("sensitivity matrix for rotation =")
        print(trafo.R_sensitivity)

        # Adapted from http://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Broadcaster-Py.html
        t = TransformStamped()
        t.header.stamp = self.node.get_clock().now().to_msg()
        # Trafo is chosen such that it transforms P (frame1) to Q (frame2)
        t.header.frame_id = Q_topic
        t.child_frame_id = P_topic
        t.transform.translation.x = trafo.t[0]
        t.transform.translation.y = trafo.t[1]
        t.transform.translation.z = trafo.t[2]
        t.transform.rotation.x = trafo.R_quat[0]
        t.transform.rotation.y = trafo.R_quat[1]
        t.transform.rotation.z = trafo.R_quat[2]
        t.transform.rotation.w = trafo.R_quat[3]
        self.trafo_publisher.publish(t)
