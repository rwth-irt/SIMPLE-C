# pip install --extra-index-url https://rospypi.github.io/simple/ rosbag sensor_msgs geometry_msgs

import sys

import numpy as np
import rosbag
from sensor_msgs import point_cloud2


def bag_to_numpy(rosbag_filename, topic_filter=None):
    """
    Reads Lidar sensor data from a given rosbag file and converts it into numpy arrays for further processing.
    Searches for all topics containing PointCloud2 objects and reads their data into numpy arrays of the
    structure given below. The data is then written to the given filename.

    1st dimension is frame index (-> time)
    2nd dimension is point index
    each point is stored as [x, y, z, intensity]

    :param rosbag_filename: the filename of the rosbag file
    :param topic_filter: None or a list of strings containing the topics to consider. If None, all topics are used which
        contain PointCloud2 data.
    :return: Dictionary { topic_name : data_numpy_array }
    """
    """
    """
    data = {}
    msg_counts = {}
    current_index = {}
    bag = rosbag.Bag(rosbag_filename)
    topics = []
    for topic, topictuple in bag.get_type_and_topic_info().topics.items():
        print(f"Found topic {topic} with dtype {topictuple.msg_type}")
        if (not topictuple.msg_type.endswith("PointCloud2")) or (topic_filter and topic not in topic_filter):
            print(f"Ignoring topic {topic}.")
            continue
        msg_count = bag.get_message_count(topic_filters=topic)
        topics.append(topic)
        msg_counts[topic] = msg_count
        print(f"Will read topic '{topic}' with PointCloud2 data and {msg_count} messages.")

    print("starting import")
    i = 0
    for topic, msg, _ in bag.read_messages(topics=topics):
        # print progress
        i += 1
        sys.stdout.write("\r" + f"reading msg {i} / {sum(msg_counts.values())}".rjust(30, " "))
        sys.stdout.flush()

        # receiving the messages for all topics/sensors mixed, sort them into the corresponding numpy array
        msg_data = np.array(point_cloud2.read_points_list(msg, skip_nans=False))
        if topic not in data:
            # create np array to hold the output data, assuming that all frames have the same amount of points
            data[topic] = np.zeros((msg_counts[topic], *msg_data.shape), dtype=np.float64)
            current_index[topic] = 0
        data[topic][current_index[topic]] = msg_data
        current_index[topic] += 1

    bag.close()
    print()
    print("import done")
    return data


def write_to_numpy_file(filename, data):
    """
    Write data obtained by bag_to_numpy to compressed numpy cache file
    :param filename: filename of numpy file to write to
    :param data: Dictionary { topic_name : data_numpy_array }
    :return: Nothing
    """
    print("writing cache file... (this may take several minutes)")
    np.savez_compressed(filename, **data)
    print("writing done")


def read_from_numpy_file(filename):
    """
    Read data from (compressed) numpy cache file
    :param filename: filename of numpy file to read from
    :return: Dictionary { topic_name : data_numpy_array }
    """
    print("reading numpy file...")
    data = np.load(filename)
    print("reading done")
    return data
