import argparse
import logging
import pathlib
import sys
from collections import deque

import numpy as np

from .rosbag_import import print_rosbag_info, get_frames_from_rosbag
from .visualization.correlation_plot import plot_match_distances
from .visualization.tracking_visualization import FrameVisInfo, TrackingVisualization
from .visualization.trafo_visualization import visualize_trafo
from ..core import parameters
from ..core.pair_calibrator import PairCalibrator
from ..core.transformation import apply_transformation

logger = logging.getLogger(__name__)

# save log info to log-file
file_handler = logging.FileHandler('transformations.log')
file_handler.setLevel(logging.INFO)
formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# ouput log info to console
console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)


def main():
    parser = argparse.ArgumentParser(description="Multi-Lidar alignment calibration")
    parser.add_argument("--rosbag", help="Mandatory, location of the rosbag file to process", required=True)
    parser.add_argument("--param-file", help="Location of parameter YAML file, otherwise default will be used")
    parser.add_argument("--show-topics", action="store_true",
                        help="Print information about the topics found in the given rosbag file and exit.")
    parser.add_argument("--visualize-tracking",
                        help="Specify a topic name for which marker tracking is visualized.")
    # Caching is not available anymore after transition to Frame objects. If required, reimplement in rosbag_import.py.
    # parser.add_argument("--no-write-cache", action="store_true",
    #                     help="Disable automatically writing a cache file after import from rosbag")
    # parser.add_argument("--no-read-cache", action="store_true",
    #                     help="Disable trying to automatically read cache file if found")

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

    if (args.visualize_tracking and args.transformation) or (not args.visualize_tracking and not args.transformation):
        logger.warning("Please choose either '--visualize-tracking' or '--transformation'.")
        sys.exit(1)

    # load parameters (do this first because loading data might take long)
    if args.param_file:
        if not pathlib.Path(args.param_file).is_file():
            logger.warning("Could not find given parameter file! Omit option to use defaults. Aborting.")
            sys.exit(1)
        paramfile = args.param_file
    else:
        paramfile = (
                pathlib.Path(__file__).parent.parent.parent / "default_parameters.yaml"
        ).absolute()  # use default params
    parameters.init_from_yaml(paramfile)

    # load data from rosbag
    topics = None
    if args.visualize_tracking:
        topics = [args.visualize_tracking]
    if args.transformation:
        topics = list(map(str.strip, args.transformation.split(",")))
        if len(topics) != 2:
            logger.warning("Must specify exactly two topics for transformation calculation. Aborting.")
            sys.exit(1)
    assert topics is not None

    if not pathlib.Path(args.rosbag).is_file():
        logger.warning("Given rosbag file does not exist, aborting.")
        sys.exit(1)
    logger.info(f"Processing rosbag file {pathlib.Path(args.rosbag).name}")

    frames = get_frames_from_rosbag(args.rosbag, topics)

    if args.visualize_tracking:
        visualize_tracking(frames)
    if args.transformation:
        transformation(frames, topics, args)


def transformation(frames, topics, args):
    pc = PairCalibrator(topics[0], topics[1], None)
    logger.info("Searching for reflector location in frames...")
    for f in frames:
        pc.new_frame(f)
    logger.info(f"Found {len(pc.reflector_locations_1)} point pairs.")
    if not pc.transformation:
        return
    logger.info("Transformation result:\nR=")
    logger.info(pc.transformation.R)
    logger.info("t =")
    logger.info(pc.transformation.t)
    logger.info("sensitivity matrix for rotation =")
    logger.info(pc.transformation.R_sensitivity)

    # Visualizations based on transformed reflector locations
    points1 = np.array([p.centroid for p in pc.reflector_locations_1])
    points2 = np.array([p.centroid for p in pc.reflector_locations_2])
    points1_transformed = apply_transformation(points1, pc.transformation)
    if args.visualize_alignment:
        plot_match_distances(points1_transformed, points2)
    if args.visualize_trafo:
        visualize_trafo([points1_transformed, points2], draw_point_match_markers=True)


def visualize_tracking(frames):
    buffer = deque(maxlen=int(parameters.get_param("window size")))
    visualization_infos: list[FrameVisInfo] = []
    for f in frames:
        buffer.append(f)
        reflector_location, status = PairCalibrator.calc_marker_location(buffer)
        if reflector_location:
            visualization_infos.append(FrameVisInfo(f, reflector_location))
        else:
            visualization_infos.append(FrameVisInfo(f, None))

    TrackingVisualization(visualization_infos)
