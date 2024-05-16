import argparse
import pathlib
import sys
from typing import Deque, List

import numpy as np

from collections import deque
from datetime import datetime, timedelta
import parameters
from frame import Frame
from reflector_location import ReflectorLocation
from locate_reflector.find_cluster_centers import get_cluster_centers_multiple_frames
from locate_reflector.track_marker import track_marker_multiple_frames, find_marker_single_frame
from rosbag_import.rosbag_to_numpy import bag_to_numpy, write_to_numpy_file
from rosbag_import.rosbag_utils import print_rosbag_info
from transformation import calc_transformation_scipy, Transformation
from visualization.correlation_plot import plot_match_distances
from visualization.tkinter_ui import create_gui
from visualization.tracking_visualization import prepare_tracking_visualization, visualize_tracking_animation
from visualization.trafo_visualization import visualize_trafo

EXPIRY_DURATION = None


class DebugPairCalibrator:

    def __init__(self, topic1: str, topic2: str):
        self.frame_buffer_1: Deque[Frame] = deque(maxlen=int(parameters.get_param("window size")))
        self.frame_buffer_2: Deque[Frame] = deque(maxlen=int(parameters.get_param("window size")))
        self.topic1 = topic1
        self.topic2 = topic2
        self.last1: Frame | None = None
        self.last2: Frame | None = None
        self.reflector_locations_1: List[ReflectorLocation] = []
        self.reflector_locations_2: List[ReflectorLocation] = []
        self.transformation: Transformation | None = None

    def new_frame_debug(self, f: Frame, topic: str):
        if topic == self.topic1:
            self.frame_buffer_1.append(f)
        else:
            self.frame_buffer_2.append(f)

    @staticmethod
    def calc_marker_location(buffer: Deque[Frame]):
        centers = [f.cluster_centers for f in buffer]
        return find_marker_single_frame(
            centers,
            max_distance=parameters.get_param("maximum neighbor distance"),
            min_velocity=parameters.get_param("minimum velocity"),
            max_vector_angle_rad=2 * np.pi * parameters.get_param("max. vector angle [deg]") / 360,
        )

    def update_transformation(self):
        # first call calculate_marker_location of latest frames
        result1, status1 = DebugPairCalibrator.calc_marker_location(self.frame_buffer_1)
        result2, status2 = DebugPairCalibrator.calc_marker_location(self.frame_buffer_2)
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
        self.transformation = calc_transformation_scipy(P, Q, weights)

def main():
    parser = argparse.ArgumentParser(description="Multi-Lidar alignment calibration")
    parser.add_argument("--rosbag", help="Mandatory, location of the rosbag file to process", required=True)
    parser.add_argument("--param-file", help="Location of parameter JSON file, otherwise default will be used")
    parser.add_argument("--show-topics", action="store_true",
                        help="Print information about the topics found in the given rosbag file and exit.")
    parser.add_argument("--visualize-tracking",
                        help="Specify a topic name for which marker tracking is visualized.")
    parser.add_argument("--no-write-cache", action="store_true",
                        help="Disable automatically writing a cache file after import from rosbag")
    parser.add_argument("--no-read-cache", action="store_true",
                        help="Disable trying to automatically read cache file if found")
    parser.add_argument("--transformation",
                        help="Pass a comma-separated pair of sensors/topic names to calculate a transformation for. "
                             "Example: --transformation 'topic1,topic2' "
                             "The resulting transformation transforms topic1 to the coordinates of topic2.")
    parser.add_argument("--visualize-trafo", action="store_true",
                        help="Show open3d visualization of points with applied transformation."
                             "(Only applies if --transformation is used.)")
    parser.add_argument("--visualize-alignment", action="store_true",
                        help="Show the transformed reflector points in open3d and a plot of the distance between"
                             "corresponding points to verify correct alignment."
                             "(Only applies if --transformation is used.)")

    args = parser.parse_args()

    if args.show_topics:
        print_rosbag_info(args.rosbag)
        sys.exit()

    # load parameters (first because loading data might take long)
    if args.param_file:
        if not pathlib.Path(args.param_file).is_file():
            print("Could not find given parameter file! Omit option to use defaults. Aborting.")
            sys.exit(1)
        paramfile = args.param_file
    else:
        paramfile = None  # use default
    parameters.init(paramfile)

    # set global variable: Maximum age for a frame before it expires (T/2)
    global EXPIRY_DURATION
    EXPIRY_DURATION = timedelta(seconds=1 / float(parameters.get_param("sample rate [Hz]")) / 2) 

    # load data from file
    print(f"Processing rosbag file {pathlib.Path(args.rosbag).name}")
    data = None
    cache_filename = args.rosbag + "_cache.npz"
    if not args.no_read_cache:
        # try to load from cache
        if pathlib.Path(cache_filename).is_file():
            print("Found cache file, reading... ", end="")
            data = np.load(cache_filename)
            print("done")
        else:
            print("No cache file found")
    if data is None:
        # did not read from cache or no cache found
        if not pathlib.Path(args.rosbag).is_file():
            print("Given rosbag file does not exist, aborting.")
            sys.exit(1)
        data = bag_to_numpy(args.rosbag)
        if not args.no_write_cache:
            write_to_numpy_file(cache_filename, data)

    if args.visualize_tracking:
        # visualize reflector tracking for given topic name
        visualize_tracking(data[args.visualize_tracking], parameters.params)

    if args.transformation:
        # calculate transformation
        trafo_topics = args.transformation.split(",")
        if len(trafo_topics) != 2:
            print("Must specify exactly two topics for transformation calculation. Aborting.")
            sys.exit(1)

        # create pair calibrator
        pc = DebugPairCalibrator(trafo_topics[0], trafo_topics[1])

        marker_locations = {}

        cnt = 1
        # take n frames of both sensors to construct buffer than go one step further
        for idx, (pcd_1, pcd_2) in enumerate(zip(data[trafo_topics[0]], data[trafo_topics[1]])):
            if idx <= int(parameters.get_param("window size"))*cnt:
                frame_sensor_1 = Frame(pcd_1, datetime.now())
                pc.new_frame_debug(frame_sensor_1, trafo_topics[0])

                frame_sensor_2 = Frame(pcd_2, datetime.now())
                pc.new_frame_debug(frame_sensor_2, trafo_topics[1])

            pc.update_transformation()        

            print("Transformation result:\nR=")
            print(trafo.R)
            print("t =")
            print(trafo.t)
            print("sensitivity matrix for rotation =")
            print(trafo.R_sensitivity)

            cnt += 1

    if args.visualize_alignment:
        # show verification plot with distances between matched points
        plot_match_distances(
            apply_transformation(filtered[trafo_topics[0]], trafo),
            filtered[trafo_topics[1]]
        )
        visualize_trafo([
            apply_transformation(filtered[trafo_topics[0]], trafo),
            filtered[trafo_topics[1]]
        ], draw_point_match_markers=True)

    if args.visualize_trafo:
        # transform point cloud from first frame for visualization
        pts0 = data[trafo_topics[0]][0, ..., :3]
        pts1 = data[trafo_topics[1]][0, ..., :3]
        pts0tr = apply_transformation(pts0, trafo)
        print("Opening open3d visualization of result...")
        visualize_trafo([pts0tr, pts1])


    def visualize_tracking(frames, params_initial):
        """
        For a frames array of **a single sensor**, a tkinter param selection will be shown,
        from which the open3d tracking visualization can be started.

        Will call all clustering/tracking functions independently of the --transformation CLI argument.

        :param frames: sensor data to use
        :param params_initial: initial parameters to be loaded to UI
        """

    def visualize_tracking_with_params(_frames, params):
        """
        gets **frames for single sensor** and params, calls track_marker and opens open3d visualization.
        Is called as callback from tkinter UI once the "calculate" button is pressed.
        :param _frames: frames for single sensor/topic
        :param params: parameter dict
        """
        centers, visualization = get_cluster_centers_multiple_frames(
            _frames,
            rel_intensity_threshold=params["relative intensity threshold"],
            DBSCAN_epsilon=params["DBSCAN epsilon"],
            DBSCAN_min_samples=int(params["DBSCAN min samples"]),
            create_visualization=True
        )
        marker_pos, selection_indices = track_marker_multiple_frames(
            centers,
            max_distance=params["maximum neighbor distance"],
            min_velocity=params["minimum velocity"],
            window_size=int(params["window size"]),
            max_vector_angle_rad=2 * np.pi * params["max. vector angle [deg]"] / 360,
        )

        prepare_tracking_visualization(selection_indices, visualization)
        print("showing open3d visualization, this will block the settings UI")
        print("press escape to close 3d view, then enter new values")
        visualize_tracking_animation(visualization, marker_pos)
        print("returning to settings UI")

    create_gui(
        params_initial,
        callback=lambda params_from_gui: visualize_tracking_with_params(frames, params_from_gui),
    )


if __name__ == "__main__":
    main()
