import argparse
import json
import pathlib

from imports import bag_to_numpy
from launch_trafo import get_centers
from tkinter_ui import create_gui


def main():
    parser = argparse.ArgumentParser(description="Multi-Lidar alignment calibration")
    parser.add_argument("--rosbag", help="Mandatory, location of the rosbag file to process", required=True)
    parser.add_argument("--param_file", help="Location of parameter JSON file, otherwise default will be used",
                        default=pathlib.Path(__file__).parent.absolute() / "default_params.json")
    parser.add_argument("--visualize", action="store_true", help="Flag to enable visualization")
    parser.add_argument("-t", "--topic", action="append",
                        help="Topic in rosbag file to use (can be specified multiple times for multiple sensors)")
    # TODO use all topics with point clouds if not specified

    args = parser.parse_args()

    with open(args.param_file, "r") as f:
        params = json.load(f)

    frames = bag_to_numpy(args.rosbag, args.topic[0])  # TODO currently using only one topic

    if args.visualize:
        create_gui(
            params,
            callback=lambda p: get_centers(frames, p, True),
        )
    else:
        centers = get_centers(frames, params, False)
        # TODO do sth with the centers


if __name__ == "__main__":
    main()
