import argparse
import json
import pathlib
import sys

import numpy as np

from locate_reflector.find_cluster_centers import get_cluster_centers_multiple_frames
from locate_reflector.track_marker import track_marker_multiple_frames
from rosbag_import.rosbag_to_numpy import bag_to_numpy, write_to_numpy_file
from rosbag_import.rosbag_utils import print_rosbag_info
from transformation.calculate_transformation import filter_locations, calc_transformation_scipy
from visualization.tkinter_ui import create_gui
from visualization.tracking_visualization import prepare_tracking_visualization, visualize_tracking_animation
from visualization.trafo_visualization import visualize_trafo


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
                        help="Show open3d visualization of points with applied transformation")

    args = parser.parse_args()

    if args.show_topics:
        print_rosbag_info(args.rosbag)
        sys.exit()

    # load parameters (first because loading data might take long)
    paramfile = pathlib.Path(__file__).parent.parent.absolute() / "default_params.json"
    if args.param_file:
        if not pathlib.Path(args.param_file).is_file():
            print("Could not find given parameter file! Omit option to use defaults. Aborting.")
            sys.exit(1)
        paramfile = args.param_file
    with open(paramfile, "r") as f:
        params = json.load(f)

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
        visualize_tracking(data[args.visualize_tracking], params)

    if args.transformation:
        # calculate transformation
        trafo_topics = args.transformation.split(",")
        if len(trafo_topics) != 2:
            print("Must specify exactly two topics for transformation calculation. Aborting.")
            sys.exit(1)

        marker_locations = {}
        for topic in trafo_topics:
            if topic not in data:
                print(f"Topic '{topic}' not found in data!")
                print_rosbag_info(args.rosbag)
                print("Aborting due to invalid topic names.")
                sys.exit(1)
            print(f"Topic {topic}:")
            print("  calculating cluster centers")
            centers = get_cluster_centers_multiple_frames(
                data[topic],
                rel_intensity_threshold=params["relative intensity threshold"],
                DBSCAN_epsilon=params["DBSCAN epsilon"],
                DBSCAN_min_samples=int(params["DBSCAN min samples"]),
                create_visualization=False,
            )
            print("  tracking marker")
            selected_locations, _ = track_marker_multiple_frames(
                centers,
                max_distance=params["maximum neighbor distance"],
                min_velocity=params["minimum velocity"],
                window_size=int(params["window size"]),
                max_vector_angle_rad=2 * np.pi * params["max. vector angle [deg]"] / 360,
            )
            marker_locations[topic] = selected_locations
        print("Searching for marker occurrences in both frames")
        filtered = filter_locations(marker_locations, trafo_topics)
        print(f"Calculating transformation using {len(filtered[trafo_topics[0]])} points")
        R, t, _, sensitivity = calc_transformation_scipy(filtered[trafo_topics[0]], filtered[trafo_topics[1]])
        print("Transformation result:\nR=")
        print(R)
        print("t =")
        print(t)
        print("sensitivity matrix for rotation =")
        print(sensitivity)
        if args.visualize_trafo:
            pts0 = data[trafo_topics[0]][0, ..., :3]  # 1st frame, only xyz
            pts1 = data[trafo_topics[1]][0, ..., :3]
            # transform points
            pts0tr = (R @ pts0.T).T + t
            # show
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
