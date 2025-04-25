import logging
import os
import time

from geometry_msgs.msg import TransformStamped
from rcl_interfaces.msg import ParameterDescriptor, ParameterType
from rclpy.node import Node
from sensor_msgs_py import point_cloud2
from sensor_msgs_py.numpy_compat import structured_to_unstructured
from std_srvs.srv import Trigger  # Importieren des Service-Typs

from . import resolve_trafo_chain
from .core import parameters, websocket_server
from .core.frame import Frame
from .core.pair_calibrator import PairCalibrator
from .core.transformation import Transformation


def get_numpy_from_pc2(pc2: point_cloud2.PointCloud2, field_names: list[str]):
    """
    Docs for sensor_msgs_py.point_cloud2 can be found here:
    https://docs.ros.org/en/ros2_packages/rolling/api/sensor_msgs_py/sensor_msgs_py.point_cloud2.html
    (page is empty for humble..........). However, the documentation is not sufficient to fully understand
    all behaviour. Therefore, the source code of sensor_msgs_py.point_cloud2 can be found here:
    https://github.com/ros2/common_interfaces/blob/rolling/sensor_msgs_py/sensor_msgs_py/point_cloud2.py

    In the source code of sensor_msgs_py.point_cloud2.read_points_numpy we can see that the assertion for
    equal types of all fields is performed for all fields of the objects, not only for the ones actually
    selected using the field_names parameter. The assertion therefore fails even if all selected fields do
    have the same type.

    This function resembles the functionality of read_points_numpy, only without the type assertion.
    If the behaviour of read_points_numpy is eventually corrected, this function can be replaced by a simple
    call to read_points_numpy.
    """
    data_structured = point_cloud2.read_points(
        pc2,
        field_names=field_names,
        skip_nans=True
    )
    data = structured_to_unstructured(data_structured)
    return data


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
        
        # Add instance variables for timing
        self._current_on_message_start_time = None
        self._transformation_calculated_in_this_call = False

        # Get detector pairs from ROS parameter "sensor_pairs"
        self.declare_parameter(
            name="sensor_pairs",
            descriptor=ParameterDescriptor(
                name="sensor_pairs",
                type=ParameterType.PARAMETER_STRING
            )
        )
        sensor_pairs_raw = str(self.get_parameter("sensor_pairs").get_parameter_value().string_value)
        self.sensor_pairs = [
            list(map(str.strip, sp.split(",")))
            for sp in sensor_pairs_raw.split(";")
        ]

        # Get the log directory path from ROS argument (if provided)
        log_dir = parameters.get_param("log_path")

        try:
            # Check if the default log directory exists, if not, create it
            if not os.path.exists(log_dir):
                os.makedirs(log_dir)
        except Exception:
            raise Exception("log_dir is not correct", log_dir)

        log_file_path = os.path.join(log_dir, 'transformations.log')

        # init logger
        self.logger = logging.getLogger(__name__)

        self.logger.info(f"Log-File saved at: {log_file_path}")

        # save log info to log-file
        file_handler = logging.FileHandler(log_file_path)
        file_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

        # ouput log info to console
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(console_handler)

        self.logger.info(f"Parsed the following sensor pairs: {self.sensor_pairs}")  # debug/info print

        # Get ROS parameter "main_sensor"
        self.declare_parameter(
            name="main_sensor",
            descriptor=ParameterDescriptor(
                name="main_sensor",
                type=ParameterType.PARAMETER_STRING
            )
        )
        self.main_sensor_topic = str(self.get_parameter("main_sensor").get_parameter_value().string_value.strip())
        self.logger.info(f"Main sensor is {self.main_sensor_topic}")

        # Resolve transformation chains from given pairs and main sensor
        self.trafo_chains = resolve_trafo_chain.get_shortest_pair_paths(self.sensor_pairs, self.main_sensor_topic)
        self.logger.info(f"Trafo chains: {self.trafo_chains}")

        # Ros Publisher
        self.trafo_publisher = self.create_publisher(TransformStamped, "transformations", 10)

        # Ros Service Server (Trigger service to send the latest transformation)
        self.service = self.create_service(
            Trigger, 'send_latest_transformation',
            self.handle_send_latest_transformation
        )

        # Initialize latest transformation variable
        self.latest_transformation = None

        # initialize convergence parameters
        self.minimum_iterations_until_convergence = parameters.get_param("minimum_iterations_until_convergence")
        self.transformation_count = 0

        # Get all topics we have to subscribe to
        self.topics = set()  # collect the topics we have to subscribe to
        for a, b in self.sensor_pairs:
            self.topics.add(a)
            self.topics.add(b)

        # Create ROS subscriptions to relevant topics
        for topic in self.topics:
            self.create_subscription(  # (This function is inherited from rclpy.Node)
                point_cloud2.PointCloud2,
                topic,
                lambda pc2, t=topic: self.on_message(t, pc2),
                10
            )

        self.logger.info("ROS initialization finished")

        self._init_reset()
        self.reset = lambda: self._init_reset()  # expose a function to reset this calibrator's data

    def _init_reset(self):
        # (re)initialize all calibration data related variables
        self.logger.info("Initializing new calibration")
        
        # Add instance variables for timing
        self._current_on_message_start_time = None
        self._transformation_calculated_in_this_call = False

        self.transformations: dict[str, Transformation | None] = {
            t: None for t in self.topics
        }

        # a dict which holds a list of interested paircalibrators per topic
        self.pair_calibrators: dict[str, list[PairCalibrator]] = {
            topic: [] for topic in self.topics
        }

        # create pair calibrators
        for a, b in self.sensor_pairs:
            pc = PairCalibrator(a, b, self.new_transformation)
            self.pair_calibrators[a].append(pc)
            self.pair_calibrators[b].append(pc)

        self.logger.info("Waiting for sensor data...")

    def on_message(self, topic: str, pc2: point_cloud2.PointCloud2):
        """
        This method should be called whenever a new PointCloud2 message is available for any sensor of interest.
        This is mainly called by the ROS receiver callback (see constructor of OnlineCalibrator).

        :param topic: The topic a new PointCloud has been received for
        :param pc2: The PointCloud2 message object
        :return: Nothing, passes the data to the respective PairCalibrators, which in turn call their callbacks
            when a new Transformation has been found.
        """
        # <<< Start Timer >>>
        self._current_on_message_start_time = time.perf_counter()
        self._transformation_calculated_in_this_call = False
        
        data = get_numpy_from_pc2(pc2, ["x", "y", "z", "intensity"])

        # As timestamp, we use the one from the ROS message, as it is easy to obtain.
        # The single points of the point cloud (if the SDK is set to the correct point format)
        # also contain a timestamp (for each point!). However, the timestamp of the **last** point
        # is only microseconds off from the one of the ROS message. So using the ones from the
        # actual sensor measurements brings no benefit here.
        timestamp_ros = pc2.header.stamp
        t_sec = timestamp_ros.sec + timestamp_ros.nanosec * 1.e-9

        frame = Frame(data, t_sec, topic)
        websocket_server.broadcast_frame(frame)
        # pass the new frame to all interested PairCalibrators, which will perform
        # buffering and calculate a transformation if possible
        for pc in self.pair_calibrators[topic]:
            pc.new_frame(frame)
            
        # <<< Stop Timer >>>
        if self._transformation_calculated_in_this_call:
            end_time = time.perf_counter()
            if self._current_on_message_start_time is not None:
                duration_ms = (end_time - self._current_on_message_start_time) * 1000
                self.logger.info(
                    f"End-to-end transformation calculation time "
                    f"(from {topic} reception to result): {duration_ms:.2f} ms"
                )
            # reset start time after use for this call chain
            self._current_on_message_start_time = None

    def new_transformation(self, trafo: Transformation, P_topic: str, Q_topic: str):
        """
        This is a callback method which will be called by PairCalibrators as soon as a new transformation is available.
        Publishes the transformation as stamped transformation (including reference frames) in ROS.

        :param trafo: the transformation
        :param P_topic: the name of the "P" frame (see `transformation.py` for explanation)
        :param Q_topic: the name of the "Q" frame (see `transformation.py` for explanation)
        :return:
        """
        self._transformation_calculated_in_this_call = True

        self.logger.info(f"New transformation for '{P_topic}' --> '{Q_topic}':\nR=\n")
        self.logger.info(trafo.R)
        self.logger.info("t =\n")
        self.logger.info(trafo.t)
        self.logger.info("sensitivity matrix for rotation =\n")
        self.logger.info(trafo.R_sensitivity)

        # Publish in ROS
        # Adapted from http://docs.ros.org/en/humble/Tutorials/Intermediate/Tf2/Writing-A-Tf2-Broadcaster-Py.html
        t = TransformStamped()
        t.header.stamp = self.get_clock().now().to_msg()
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

        self.latest_transformation = t

        self._update_transformations()
        self._broadcast_websocket()

        self.transformation_count += 1

    def _get_specific_transformation(self, from_topic: str, to_topic: str) -> Transformation | None:
        """
        For two topics (from, to) returns the Transformation object of the specific pairCalibrator (or None).
        If required, the inverse transformation is returned, depending of the "direction" of the PairCalibrator.

        :param from_topic: the "from" topic
        :param to_topic: the "to" topic
        :return: The Transformation object of the respective PairCalibrator (inverted, if necessary)
            or None if not present.
        """
        for pc in self.pair_calibrators[from_topic]:
            if from_topic == pc.topic1 and to_topic == pc.topic2:
                return pc.transformation  # may be None, which is ok
            if to_topic == pc.topic1 and from_topic == pc.topic2:
                # return inverse
                if pc.transformation is None:
                    return None
                return pc.transformation.inverse
        raise Exception("No paircalibrator for required transformation, this should not happen")

    def _update_transformations(self):
        """
        Based on the current transformations of all PairCalibrators, try to calculate
        transformations to directly transform sensor [topic]s data to self.main_sensor_topic.

        Only calculates where possible (i.e. all partial transformations are present and there
        is a chain of transformations which actually leads to self.main_sensor_topic!).
        Other entries are left at None.

        :return: Nothing, updates self.transformations
        """
        for topic in self.transformations:
            if topic == self.main_sensor_topic:
                self.transformations[topic] = Transformation.unity()
                continue

            # Obtain all transformations of the transformation chain
            trafos = [
                self._get_specific_transformation(from_topic, to_topic)
                for from_topic, to_topic in self.trafo_chains[topic]
            ]

            # Check if any entry in 'trafos' is None
            none_in_trafos = False
            for t in trafos:
                if t is None:
                    none_in_trafos = True
                    break
            if none_in_trafos:
                # Chain to main_sensor yet unkown, so no transformation to calculate
                continue

            # All transformations are available, proceed and switch to homogeneous transformation 
            trafos = [t.matrix for t in trafos]
            combined = trafos.pop(0)
            while trafos:
                combined = trafos.pop(0) @ combined
            self.transformations[topic] = Transformation.from_matrix(combined)  # switch back to Transformation object

    def _broadcast_websocket(self):
        """
        Broadcast the latest transformation data and sensor metadata over the websocket.
        
        This method iterates over all topics in the transformations dictionary. For each topic,
        it checks if there are any transformations available. If transformations are available,
        it retrieves the corresponding PairCalibrator and broadcasts the sensor metadata, including
        reflector locations and the transformation data, to all connected websocket clients.
        """
        for topic in self.transformations:
            if not self.transformations[topic]:
                continue
            pc = self.pair_calibrators[topic][0]
            websocket_server.broadcast_sensor_metadata(
                topic=topic,
                reflector_locations=pc.reflector_locations_1 if pc.topic1 == topic else pc.reflector_locations_2,
                transformation=self.transformations[topic]
            )

    def handle_send_latest_transformation(self, request, response):
        """
        Handle the request to send the latest transformation.

        This method processes a request to send the latest transformation data. If no transformation
        has been published yet, it sets the response to indicate failure. If a transformation is available,
        it attempts to send the transformation data and sets the response accordingly. In case of an error
        during the process, it catches the exception and sets the response to indicate failure.

        :param request: The request object containing the details of the request.
        :param response: The response object to be populated with the result of the request.
        :return: The populated response object.
        """
        if self.latest_transformation is None:
            response.success = False
            response.message = 'No transformation has been published yet.'
        else:
            try:
                msg = self.latest_transformation
                response.success = True
                response.message = f'Successfully sent latest transformation \n{msg}'
                return response
            except Exception as e:
                response.success = False
                response.message = f'Error sending latest transformation \n{str(e)}'
                return response

    def check_convergence(self) -> bool:
        """
        Check if the convergence criteria are met.
        :return: True if convergence is achieved, False otherwise.
        """
        # Check if the minimum number of iterations has been reached
        if not self.transformation_count >= self.minimum_iterations_until_convergence:
            return False

        # Check if one pair calibrator has not yet converged
        for pcs in self.pair_calibrators.values():
            for pc in pcs:
                if not pc.check_convergence():
                    return False
        
        return True
