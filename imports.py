# pip install --extra-index-url https://rospypi.github.io/simple/ rosbag sensor_msgs geometry_msgs

import rosbag
from sensor_msgs import point_cloud2
import numpy as np
from pathlib import Path
import sys


def bag_to_numpy(filename, topic, only_frames=None, cache=True):
    """
    Returns a 3d-numpy array.
    1st dimension is frame over time
    2nd dimension is point index
    each point is stored as [x, y, z, intensity]

    You can pass a list of frame numbers to read (speeding up import without cache significantly).
    If None, all frames are imported.

    Uses numpy files to cache slow imports, but only if all frames should be imported.
    """
    if only_frames:
        cache = False
        # otherwise we would save a limited selection to cache, leading to confusing imports later
    cache_filename = filename + "_cache.npz"
    if cache and Path(cache_filename).is_file():
        print("using cache")
        return np.load(cache_filename)["data"]

    print("starting import")
    out = []
    bag = rosbag.Bag(filename)
    msg_count = bag.get_message_count(topic_filters=topic)
    i = 0
    for topic, msg, t in bag.read_messages(topics=[topic]):
        i += 1
        sys.stdout.write("\r" + f"reading msg {i} / {msg_count}".rjust(30, " "))
        sys.stdout.flush()
        if only_frames and i - 1 not in only_frames:
            continue
        out.append(np.array(point_cloud2.read_points_list(msg, skip_nans=False)))
    bag.close()
    out = np.array(out)
    print()
    print("import done")

    if cache:
        print("writing cache file")
        np.savez_compressed(cache_filename, data=out)
    return out
