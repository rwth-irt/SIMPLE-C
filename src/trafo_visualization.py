import numpy as np
import open3d as o3d

colors = [
    np.array([1.0, 0.0, 0.0]),  # RED
    np.array([0.0, 1.0, 0.0]),  # GREEN
    np.array([0.0, 0.0, 1.0]),  # BLUE
]


def np_to_pointcloud(frame_in):
    frame = frame_in[~(np.isnan(frame_in).any(axis=1))]  # remove nans from frame_in
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame[:, :3])
    return pcd


def visualize_trafo(points):
    """
    open open3d visualization for list of pointcloud data (numpy array, per point x y z).
    Does not apply transformations.
    Draws different point clouds in different colors.

    :param points:
    :return:
    """
    if len(points) > len(colors):
        # we are out of colors, add some in the `colors` list
        raise Exception("Not enough colors specified in code!")
    geoms = []
    for p, col in zip(points, colors[:len(points)]):
        geom = np_to_pointcloud(p)
        geom.paint_uniform_color(col)
        geoms.append(geom)
    o3d.visualization.draw_geometries(geoms)
