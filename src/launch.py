import argparse
import json
import pathlib

import numpy as np

from imports import bag_to_numpy, write_to_numpy_file
from launch_trafo import get_centers
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
        for topic in data:
            print(f"\nCURRENT TOPIC: {topic}")
            print("close parameter chooser to go to next topic")
            create_gui(
                params,
                callback=lambda p: get_centers(data[topic], p, True),
            )

    # after visualization of object tracking, continue with alignment
    centers = {
        topic: get_centers(data[topic], params, False) for topic in data
    }


if __name__ == "__main__":
    main()
