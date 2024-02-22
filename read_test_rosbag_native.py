import rosbag
from sensor_msgs import point_cloud2
import numpy as np

from pathlib import Path


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


def visualize_frame(frame_in):
    import open3d as o3d
    
    # remove nans from frame_in
    frame = frame_in[~(np.isnan(frame_in).any(axis=1))]

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame[:, :3])
    colors = np.zeros((len(frame), 3))
    colors[:, 0] = frame[:, 3] / 255
    print(colors)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


frames = bag_to_numpy("labor_lidar.bag")
visualize_frame(frames[0])
