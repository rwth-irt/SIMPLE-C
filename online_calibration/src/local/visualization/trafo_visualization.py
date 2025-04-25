import numpy as np
import open3d as o3d

colors = [
    np.array([240, 185, 64]) / 255,  # RED
    np.array([51, 105, 159]) / 255,  # GREEN
    np.array([4, 114, 77]) / 255,  # BLUE
]


def np_to_pointcloud(frame_in):
    frame = frame_in[~(np.isnan(frame_in).any(axis=1))]  # remove nans from frame_in
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame[:, :3])
    return pcd


def visualize_trafo(points, draw_point_match_markers=False):
    """
    Draws multiple point clouds in the same coordinate system.
    Does not apply transformations.
    Draws each cloud in a different color.

    Optionally draws each 5th point in a gradient from black to white to make it easier
    finding points with the same index. This is relevant to verify alignment of matched reflector locations.

    :param points: list of numpy arrays containing the points of each point cloud
    :param draw_point_match_markers: Toggles to color some points depending on their index, see above.
    """
    if len(points) > len(colors):
        # we are out of colors, add some in the `colors` list
        raise Exception("Not enough colors specified in code!")
    geoms = []
    for p, col in zip(points, colors[:len(points)]):
        geom = np_to_pointcloud(p)
        if draw_point_match_markers:
            color_pointcloud_by_index(geom, col)
        else:
            geom.paint_uniform_color(col)
        geoms.append(geom)
    o3d.visualization.draw_geometries(geoms)


def color_pointcloud_by_index(cloud, color, step=5):
    # TODO highlight which points have been used and which were filtered out
    l = len(cloud.points)
    colors = np.tile(color, (l, 1))
    for i in range(0, l, step):
        colors[i] = np.array([1, 1, 1]) * (i / l)
    cloud.colors = o3d.utility.Vector3dVector(colors)
