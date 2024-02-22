import rosbag
from sensor_msgs import point_cloud2
import numpy as np


def bag_to_numpy(filename, only_frames=None):
    print("starting import")
    out = []
    bag = rosbag.Bag(filename)
    i = 0
    for topic, msg, t in bag.read_messages(topics=["rslidar_points_ref"]):
        i += 1
        if only_frames and i - 1 not in only_frames:
            continue
        out.append(np.array(point_cloud2.read_points_list(msg, skip_nans=True)))
    bag.close()
    print("import done")
    return out


def visualize_frame(frame):
    import open3d as o3d

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame[:, :3])
    colors = np.zeros((len(frame), 3))
    colors[:, 0] = frame[:, 3] / 255
    print(colors)
    pcd.colors = o3d.utility.Vector3dVector(colors)
    o3d.visualization.draw_geometries([pcd])


frames = bag_to_numpy("labor_lidar.bag", only_frames=[40])
visualize_frame(frames[0])
