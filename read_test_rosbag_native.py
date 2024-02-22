# pip install --extra-index-url https://rospypi.github.io/simple/ rosbag sensor_msgs geometry_msgs
# pip install open3d-cpu

import rosbag
from sensor_msgs import point_cloud2
import numpy as np
from pathlib import Path

import open3d as o3d


def bag_to_numpy(filename, only_frames=None, cache=True):
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
    cache_filename = filename + "_cache.npy"
    if cache and Path(cache_filename).is_file():
        print("using cache")
        return np.load(cache_filename)

    print("starting import")
    out = []
    bag = rosbag.Bag(filename)
    i = 0
    for topic, msg, t in bag.read_messages(topics=["rslidar_points_ref"]):
        i += 1
        if only_frames and i - 1 not in only_frames:
            continue
        out.append(np.array(point_cloud2.read_points_list(msg, skip_nans=False)))
    bag.close()
    out = np.array(out)
    print("import done")

    if cache:
        print("writing cache file")
        np.save(cache_filename, out)
    return out


def np_to_pointcloud(frame_in):
    frame = frame_in[~(np.isnan(frame_in).any(axis=1))]  # remove nans from frame_in
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame[:, :3])
    colors = np.zeros((len(frame), 3))
    colors[:, 0] = frame[:, 3] / 255
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_frame(frame_in):
    o3d.visualization.draw_geometries([np_to_pointcloud(frame_in)])


i = -1
last = None


def callback(vis):
    global i, last
    vs = vis.get_view_status()
    if last:
        vis.remove_geometry(last)
    i = (i + 1) % len(frames)
    last = np_to_pointcloud(frames[i])
    vis.add_geometry(last, reset_bounding_box=True)
    vis.set_view_status(vs)
    return True


def visualize_animation(frames):
    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window()
    vis.add_geometry(np_to_pointcloud(frames[0]))
    vis.poll_events()
    vis.update_renderer()
    vis.register_key_callback(ord("K"), callback)  # apparently, not all keys do work!
    vis.run()
    vis.destroy_window()


filename = (
    "/home/max/UNI/Job_IRT/LIDAR/temporal_reflector/temporal_reflector_disturbance.bag"
)
# filename = "labor_lidar.bag"
frames = bag_to_numpy(filename)
# visualize_frame(frames[0])
visualize_animation(frames)
