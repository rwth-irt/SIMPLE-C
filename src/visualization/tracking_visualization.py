# pip install open3d-cpu numpy
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

# for snapshot creation
snapshot_dir = None


colors = {
    -2: np.array([0.5, 0.5, 0.5]),  # irrelevant points             GRAY
    -1: np.array([1.0, 0.0, 0.0]),  # bright points                 RED
    1: np.array([0.0, 1.0, 0.0]),  # points in cluster              GREEN
    2: np.array([0.0, 0.0, 1.0]),  # points in selected cluster     BLUE
}
marker_color = [1, 0.706, 0]
trace_color = [0.5, 0.706 / 2, 0]  # like marker, but darker


def np_to_pointcloud_with_colors(frame_in):
    frame = frame_in[~(np.isnan(frame_in).any(axis=1))]  # remove nans from frame_in
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame[:, :3])
    point_colors = np.zeros((len(frame), 3))  # irrelevant points black
    for ci in colors:
        point_colors[frame[:, 3] == ci] = colors[ci]
    pcd.colors = o3d.utility.Vector3dVector(point_colors)
    return pcd


def visualize_frame(frame_in):
    # show a single frame
    o3d.visualization.draw_geometries([np_to_pointcloud_with_colors(frame_in)])


def reset():
    global _i, _last, _frames, _markers, _last_marker, _trace
    _i = -1  # index of current frame to show
    _last = None  # last frame's geometry, if exists
    _frames = None  # list of all frame geometry objects
    _markers = None  # list of marker centroids (one per frame, or None)
    _last_marker = None
    _trace = []


def _callback(vis):
    global _i, _last, _last_marker, _trace
    first_time = not _last
    vs = vis.get_view_status()  # cache current view to restore after changing objects
    _i = (_i + 1) % len(_frames)
    if _i == 0:
        # clear old trace, restarting
        for m in _trace:
            vis.remove_geometry(m)
        _trace = []
    new = np_to_pointcloud_with_colors(_frames[_i])
    vis.add_geometry(new, reset_bounding_box=True)
    if _last:
        vis.remove_geometry(_last)
    _last = new

    # add marker for this frame if needed
    marker_radius = 0.14
    if _last_marker:  # remove last marker in any case if present
        vis.remove_geometry(_last_marker[0])
        vis.remove_geometry(_last_marker[1])
    if _markers and _markers[_i] is not None:
        # marker: sphere and cylinder
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
        marker2 = o3d.geometry.TriangleMesh.create_cylinder(
            radius=marker_radius / 4, height=marker_radius * 160
        )

        marker.translate(_markers[_i][:3])
        marker2.translate(_markers[_i][:3])

        marker.paint_uniform_color(marker_color)
        marker2.paint_uniform_color(marker_color)

        vis.add_geometry(marker)
        vis.add_geometry(marker2)

        _last_marker = (marker, marker2)

        # trace
        trace_marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius / 2)
        trace_marker.translate(_markers[_i][:3])
        trace_marker.paint_uniform_color(trace_color)
        vis.add_geometry(trace_marker)
        _trace.append(trace_marker)

    if not first_time:
        vis.set_view_status(vs)
    vis.update_renderer()
    return True


def visualize_tracking_animation(frames, markers=None):
    reset()
    global _frames, _markers
    assert (not markers) or len(frames) == len(markers)

    _frames = frames
    _markers = markers

    vis = o3d.visualization.VisualizerWithKeyCallback()
    vis.create_window()

    vis.poll_events()
    vis.update_renderer()
    vis.register_key_callback(ord("K"), _callback)
    vis.register_key_callback(ord("J"), on_capture_key)
    _callback(vis)
    print("PRESS K FOR THE NEXT FRAME! (Press J to save snapshot and proceed to next frame for creating videos.)")
    vis.run()
    vis.destroy_window()
    reset()  # free RAM


def on_capture_key(vis: o3d.visualization.VisualizerWithKeyCallback):
    global snapshot_dir
    _callback(vis)
    if not snapshot_dir:
        snapshot_dir = tempfile.mkdtemp(prefix="tracking_snapshots_")
        print(f"WRITING SNAPSHOTS TO DIRECTORY {snapshot_dir}")
    vis.capture_screen_image(str(Path(snapshot_dir) / f"frame_{str(_i).zfill(4)}.png"), do_render=True)


def prepare_tracking_visualization(selection_indices, visualization):
    """
    Expects visualization to contain cluster indices in intensity channel of points.
    The intensity channel is used as indicator for rendering color in the o3d UI, see tracking_visualization.py.
    For given index of the selected cluster per frame (`selection_indices`), set the
    respectively selected cluster's value to 2 for highlighting and all others to 1.

    **Alters visualization**, which can then be passed to the UI.

    :param selection_indices: list with index of selected cluster per frame
    :param visualization: lidar data numpy array, but with visualization info in intensity, as obtained \
    from `get_cluster_centers_per_frame(create_visualization=True)`
    :return: nothing, writes into visualization array
    """
    for frame_i in range(len(selection_indices)):
        # currently, visualization contains indices of clusters
        # for o3d visualization, we want to convert this to codes meaning "any cluster" or "chosen cluster"
        chosen_cluster_selection = visualization[frame_i, :, 3] == selection_indices[frame_i]
        any_cluster_selection = visualization[frame_i, :, 3] >= 0  # any cluster
        visualization[frame_i, any_cluster_selection, 3] = 1
        visualization[frame_i, chosen_cluster_selection, 3] = 2