# pip install open3d-cpu numpy
import open3d as o3d
import numpy as np


def np_to_pointcloud(frame_in):
    frame = frame_in[~(np.isnan(frame_in).any(axis=1))]  # remove nans from frame_in
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame[:, :3])
    colors = np.zeros((len(frame), 3))  # irrelevant points black
    colors[frame[:, 3] == 1, 0] = 1  # N brightest points have a 1 -> red
    colors[frame[:, 3] == 2, 1] = 1  # points in biggest cluster have a 2 -> yellow
    # colors[:, 0] = frame[:, 3] / np.max(frame[:, 3]) # red brightness for intensity
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_frame(frame_in):
    # show a single frame
    o3d.visualization.draw_geometries([np_to_pointcloud(frame_in)])


_i = -1  # index of current frame to show
_last = None  # last frame's geometry, if exists
_frames = None  # list of all frame geometry objects
_markers = None  # list of marker centroids (one per frame, or None)


def _callback(vis):
    global _i, _last
    vs = vis.get_view_status()
    if _last:
        vis.remove_geometry(_last)
    _i = (_i + 1) % len(_frames)
    _last = np_to_pointcloud(_frames[_i])
    vis.add_geometry(_last, reset_bounding_box=True)
    marker_radius = 0.025
    if _markers and _markers[_i] is not None:
        marker = o3d.geometry.TriangleMesh.create_box(
            width=2 * marker_radius, height=2 * marker_radius, depth=2 * marker_radius
        )
        marker.translate(_markers[_i][:3] - marker_radius)
        vis.add_geometry(marker)
        # TODO remove old markers
    vis.set_view_status(vs)
    return True


def visualize_animation(frames, markers=None):
    global _frames, _markers
    assert (not markers) or len(frames) == len(markers)

    _frames = frames
    _markers = markers

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window()
    vis.add_geometry(np_to_pointcloud(frames[0]))
    # TODO bug: this frame is not removed. I do not want it...
    # but if I try to do so, open3d crashes. Annoying...

    vis.poll_events()
    vis.update_renderer()
    vis.register_key_callback(ord("K"), _callback)  # apparently, not all keys do work!
    vis.run()
    vis.destroy_window()
