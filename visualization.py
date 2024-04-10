# pip install open3d-cpu numpy
import open3d as o3d
import numpy as np

colors = {
    -2: np.array([0.5, 0.5, 0.5]),  # irrelevant points             GRAY
    -1: np.array([1.0, 0.0, 0.0]),  # bright points                 RED
    1: np.array([0.0, 1.0, 0.0]),  # points in cluster              GREEN
    2: np.array([0.0, 0.0, 1.0]),  # points in selected cluster     BLUE
}
marker_color = [1, 0.706, 0]


def np_to_pointcloud(frame_in):
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
    o3d.visualization.draw_geometries([np_to_pointcloud(frame_in)])


def reset():
    global _i, _last, _frames, _markers, _last_marker
    _i = -1  # index of current frame to show
    _last = None  # last frame's geometry, if exists
    _frames = None  # list of all frame geometry objects
    _markers = None  # list of marker centroids (one per frame, or None)
    _last_marker = None


def _callback(vis):
    global _i, _last, _last_marker
    first_time = not _last
    vs = vis.get_view_status()  # cache current view to restore after changing objects
    _i = (_i + 1) % len(_frames)
    new = np_to_pointcloud(_frames[_i])
    vis.add_geometry(new, reset_bounding_box=True)
    if _last:
        vis.remove_geometry(_last)
    _last = new

    # add marker for this frame if needed
    marker_radius = 0.1
    if _last_marker:  # remove last marker in any case if present
        vis.remove_geometry(_last_marker)
    if _markers and _markers[_i] is not None:
        marker = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
        marker.translate(_markers[_i][:3])
        marker.paint_uniform_color(marker_color)
        vis.add_geometry(marker)
        _last_marker = marker

    if not first_time:
        vis.set_view_status(vs)
    vis.update_renderer()
    return True


def visualize_animation(frames, markers=None):
    reset()
    global _frames, _markers
    assert (not markers) or len(frames) == len(markers)

    _frames = frames
    _markers = markers

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window()

    vis.poll_events()
    vis.update_renderer()
    vis.register_key_callback(
        ord("K"), _callback
    )  # apparently, not all keys are available
    _callback(vis)
    print("PRESS K FOR THE NEXT FRAME!")
    vis.run()
    vis.destroy_window()
