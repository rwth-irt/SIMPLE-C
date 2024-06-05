# pip install --extra-index-url https://rospypi.github.io/simple/ rosbag sensor_msgs geometry_msgs
# maybe also installing roslz4 (from same pip repo) fixes the insanely bad performance?
# installing via `conda install -c conda-forge ros-rosbag` might work as well
import sys
from datetime import datetime, timedelta

import numpy as np
import rosbag
from sensor_msgs import point_cloud2

from ..core.frame import Frame


def get_frames_from_rosbag(rosbag_filename: str, topics: list[str]) -> list[Frame]:
    """
    Reads Lidar sensor data from a given rosbag file and returns a list of Frame objects.

    :param rosbag_filename: the filename of the rosbag file
    :param topics: Alist of strings containing the topics to consider.
    :return: list of Frame objects
    """

    # Read relevant topics from rosbag file
    total_message_count = 0
    with rosbag.Bag(rosbag_filename) as bag:
        pc2_topics = []
        for topic, topictuple in bag.get_type_and_topic_info().topics.items():
            if topictuple.msg_type.endswith("PointCloud2"):
                pc2_topics.append(topic)
                if topic in topics:
                    total_message_count += bag.get_message_count(topic_filters=topic)
        for t in topics:
            if t not in pc2_topics:
                print(f"Topic '{t}' does not exist in the given rosbag or does not have PointCloud2 data!")
                sys.exit(1)

        frames = []

        # we'll take the difference of timestamps afterwards, value does not matter
        base_time = datetime(year=1970, month=1, day=1)

        print("starting import")
        for i, (topic, message, timestamp_ros) in enumerate(bag.read_messages(topics=topics)):
            # print progress
            sys.stdout.write(
                "\r" + f"reading msg {str(i).rjust(3, ' ')} / {total_message_count}".rjust(30, " ")
            )
            sys.stdout.flush()

            # Get data
            msg_data = np.array(point_cloud2.read_points_list(message, skip_nans=False))
            timestamp = base_time + timedelta(seconds=timestamp_ros.secs, microseconds=timestamp_ros.nsecs / 1000)
            frames.append(Frame(msg_data, timestamp, topic))

            # TODO:
            #  docstring says "For more efficient access use read_points directly.". Is this the performance problem?
            #  or see sensor_msgs.PointCloud2's deserialize_numpy method, which may do what we want.
            #  It seems like the rospy version used from https://rospypi.github.io/simple/ is very outdated.
            #  point_cloud2.read_points *should*
            #  (https://docs.ros.org/en/ros2_packages/rolling/api/sensor_msgs_py/sensor_msgs_py.point_cloud2.html#sensor_msgs_py.point_cloud2.read_points)
            #  return a numpy array, so it has probably been rewritten.

    print("\nimport done")
    return frames


# TODO re-implement caching with good performance? Or rely on using faster/newer rosbag library?


def print_rosbag_info(rosbag_filename):
    bag = rosbag.Bag(rosbag_filename)
    print("Topics found in this rosbag file:")
    for topic, topictuple in bag.get_type_and_topic_info().topics.items():
        print(f"    Name {str(topic).ljust(36)} Message type {str(topictuple.msg_type).ljust(36)}, "
              f"{bag.get_message_count(topic_filters=topic)} messages")
