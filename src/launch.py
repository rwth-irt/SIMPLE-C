import argparse
import json
import pathlib

import numpy as np

from imports import bag_to_numpy, write_to_numpy_file
from src.locate_reflector.find_cluster_centers import get_cluster_centers_per_frame
from src.locate_reflector.track_marker import track_marker
from src.visualization import prepare_visualization, visualize_animation
from tkinter_ui import create_gui


def main():
    parser = argparse.ArgumentParser(description="Multi-Lidar alignment calibration")
    parser.add_argument("--rosbag", help="Mandatory, location of the rosbag file to process", required=True)
    parser.add_argument("--param-file", help="Location of parameter JSON file, otherwise default will be used")
    parser.add_argument("--visualize", action="store_true", help="Flag to enable visualization")
    parser.add_argument("--no-write-cache", action="store_true",
                        help="Disable automatically writing a cache file after import from rosbag")
    parser.add_argument("--no-read-cache", action="store_true",
                        help="Disable trying to automatically read cache file if found")
    # parser.add_argument("-t", "--topic", action="append",
    #                     help="Topic to use (can be specified multiple times for multiple sensors).\n"
    #                          "If not given, all topics containing PointCloud2 data will be used.\n")

    args = parser.parse_args()

    # load parameters (first because loading data might take long)
    paramfile = pathlib.Path(__file__).parent.parent.absolute() / "default_params.json"
    if args.param_file:
        if not pathlib.Path(args.param_file).is_file():
            raise Exception("Could not find given parameter file! Omit option to use defaults.")
        paramfile = args.param_file
    with open(paramfile, "r") as f:
        params = json.load(f)

    # load data from file
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
            raise Exception("Given rosbag file does not exist")
        data = bag_to_numpy(args.rosbag)
        if not args.no_write_cache:
            write_to_numpy_file(cache_filename, data)

    if args.visualize:
        visualize(data, params)
    else:
        marker_locations = {}
        for topic, frames in data.items():
            centers = get_cluster_centers_per_frame(
                frames,
                rel_intensity_threshold=params["relative intensity threshold"],
                DBSCAN_epsilon=params["DBSCAN epsilon"],
                DBSCAN_min_samples=int(params["DBSCAN min samples"]),
                create_visualization=False,
            )
            _, marker = track_marker(centers, params)
            marker_locations[topic] = marker


def visualize(data, params_initial):
    """
    For each topic in data, a tkinter param selection will be shown, from which the open3d visualization can be started.
    :param data: data dict, frames per sensor
    :param params_initial: initial parameters to be loaded to UI
    """
    def visualize_with_params(frames, params):
        """
        gets frames for single sensor and params, calls track_marker and opens open3d visualization.
        Is called as callback from tkinter UI once the "calculate" button is pressed.
        :param frames: frames for single sensor/topic
        :param params: parameter dict
        """
        print(f"\nCURRENT TOPIC: {topic}")
        print("close parameter chooser to go to next topic")

        centers, visualization = get_cluster_centers_per_frame(
            frames,
            rel_intensity_threshold=params["relative intensity threshold"],
            DBSCAN_epsilon=params["DBSCAN epsilon"],
            DBSCAN_min_samples=int(params["DBSCAN min samples"]),
            create_visualization=True,
        )
        selection_indices, marker_pos = track_marker(centers, params)

        prepare_visualization(selection_indices, visualization)
        print("showing open3d visualization, this will block the settings UI")
        print("press escape to close 3d view, then enter new values")
        visualize_animation(visualization, marker_pos)
        print("returning to settings UI")

    for topic in data:
        create_gui(
            params_initial,
            callback=lambda params_from_gui: visualize_with_params(data[topic], params_from_gui),
        )
        # due to this problem https://stackoverflow.com/q/75927299 this will fail to open the next tkinter param
        # chooser window if one open3d window had been open in the past. This seems to be a problem in tkinter.
        # so please just go to the right topic for the visualization, let it crash and open the program again.


if __name__ == "__main__":
    main()
