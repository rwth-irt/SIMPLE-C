# pip install open3d-cpu numpy
import open3d as o3d
import numpy as np


def np_to_pointcloud(frame_in):
    frame = frame_in[~(np.isnan(frame_in).any(axis=1))]  # remove nans from frame_in
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(frame[:, :3])
    colors = np.zeros((len(frame), 3)) # irrelevant points black
    colors[frame[:, 3] == 1, 0] = 1 # N brightest points have a 1 -> red
    colors[frame[:, 3] == 2, 1] = 1 # points in biggest cluster have a 2 -> yellow
    # colors[:, 0] = frame[:, 3] / np.max(frame[:, 3]) # red brightness for intensity
    pcd.colors = o3d.utility.Vector3dVector(colors)
    return pcd


def visualize_frame(frame_in):
    # show a single frame
    o3d.visualization.draw_geometries([np_to_pointcloud(frame_in)])


_i = -1
_last = None
_frames = None


def _callback(vis):
    global _i, _last
    vs = vis.get_view_status()
    if _last:
        vis.remove_geometry(_last)
    _i = (_i + 1) % len(_frames)
    _last = np_to_pointcloud(_frames[_i])
    vis.add_geometry(_last, reset_bounding_box=True)
    vis.set_view_status(vs)
    return True


def visualize_animation(frames):
    global _frames
    _frames = frames

    o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)
    vis = o3d.visualization.VisualizerWithKeyCallback()

    vis.create_window()
    vis.add_geometry(np_to_pointcloud(frames[0]))
    vis.poll_events()
    vis.update_renderer()
    vis.register_key_callback(ord("K"), _callback)  # apparently, not all keys do work!
    vis.run()
    vis.destroy_window()
