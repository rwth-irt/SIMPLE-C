# pip install open3d-cpu numpy
import tempfile
from pathlib import Path

import numpy as np
import open3d as o3d
from open3d.cpu.pybind.geometry import TriangleMesh

from ...core.frame import Frame
from ...core.reflector_location import ReflectorLocation

o3d.utility.set_verbosity_level(o3d.utility.VerbosityLevel.Info)

# for snapshot creation
snapshot_dir = None

colors = {
    "any": np.array([0.5, 0.5, 0.5]),  # GRAY
    "bright": np.array([1.0, 0.0, 0.0]),  # RED
    "cluster": np.array([0.0, 1.0, 0.0]),  # GREEN
    "reflector": np.array([0.0, 0.0, 1.0]),  # BLUE
    "marker": np.array([1., 0.706, 0.]),  # yellow-orange
    "trace": np.array([0.5, 0.706 / 2, 0.]),  # Brown
    "normal": np.array([1., 1., 0.])  # yellow
}
marker_radius = 0.14


class FrameVisInfo:
    def __init__(
            self,
            frame: Frame,
            reflector_location: ReflectorLocation | None,
    ):
        self.frame = frame
        self.reflector_location = reflector_location

        # create o3d pointcloud object with colors
        c = self.frame.clustering
        point_colors = np.full((len(self.frame.data), 3), colors["any"])
        point_colors[c == -1] = colors["bright"]
        point_colors[c >= 0] = colors["cluster"]
        if self.reflector_location:
            point_colors[c == self.reflector_location.cluster_index_in_frame] = colors["reflector"]

        self.pcd = o3d.geometry.PointCloud()
        self.pcd.points = o3d.utility.Vector3dVector(self.frame.data[:, :3])
        self.pcd.colors = o3d.utility.Vector3dVector(point_colors)

        # create markers: ball and long cylinder
        if self.reflector_location:
            self.marker1: TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius)
            self.marker1.translate(self.reflector_location.centroid)
            self.marker1.paint_uniform_color(colors["marker"])

            self.marker2: TriangleMesh = o3d.geometry.TriangleMesh.create_cylinder(
                radius=marker_radius / 4,
                height=marker_radius * 160
            )
            self.marker2.translate(self.reflector_location.centroid)
            self.marker2.paint_uniform_color(colors["marker"])

            self.trace_marker: TriangleMesh = o3d.geometry.TriangleMesh.create_sphere(radius=marker_radius * .5)
            self.trace_marker.translate(self.reflector_location.centroid)
            self.trace_marker.paint_uniform_color(colors["trace"])

            self.normal_marker: TriangleMesh = o3d.geometry.TriangleMesh.create_cylinder(
                radius=marker_radius * .2,
                height=marker_radius * 10
            )  # shows in positive z direction after construction
            # Now rotate the normal_marker such that it resembles the reflector normal.
            v1 = self.reflector_location.normal_vector
            v2 = np.array([0., 0., 1.])
            rot_vector = np.cross(v1, v2) * np.arccos(v1 @ v2)
            # both v1 and v2 are unit vectors, length of rot_vector is angle in radians
            self.normal_marker.rotate(
                o3d.geometry.get_rotation_matrix_from_axis_angle(rot_vector),
                center=np.array([0., 0., 0.])  # around origin
            )
            self.normal_marker.translate(self.reflector_location.centroid)
            self.normal_marker.paint_uniform_color(colors["normal"])
        else:
            self.marker1 = None
            self.marker2 = None
            self.trace_marker = None
            self.normal_marker = None


class TrackingVisualization:
    def __init__(self, vis_infos: list[FrameVisInfo]):
        self.vis_infos = vis_infos
        self.i = -1
        self.trace = []
        self.last: FrameVisInfo | None = None

        self.vis = o3d.visualization.VisualizerWithKeyCallback()
        self.vis.create_window()

        self.vis.poll_events()
        self.vis.update_renderer()
        self.vis.register_key_callback(ord("K"), lambda _: self.on_next_key())
        self.vis.register_key_callback(ord("J"), lambda _: self.on_capture_key)
        self.on_next_key()

        print("showing open3d visualization, this will block the settings UI")
        print("press escape to close 3d view, then enter new values")
        print("PRESS K FOR THE NEXT FRAME! (Press J to save snapshot and proceed to next frame for creating videos.)")

        self.vis.run()
        self.vis.destroy_window()

    def on_next_key(self):
        """
        Called by open3d on keypress. Switches to the next frame. Removes last frame's points and optionally markers
        and adds new ones. Restores the 3d view to the state before swapping point clouds because open3d would usually
        try to reset the view to the new data.
        """
        vs = self.vis.get_view_status()  # cache current view to restore after changing objects
        self.i = (self.i + 1) % len(self.vis_infos)
        print(f"Showing frame {str(self.i + 1).rjust(3)} / {len(self.vis_infos)}")
        if self.i == 0:
            # clear old trace, restarting
            for m in self.trace:
                self.vis.remove_geometry(m)
            self.trace = []

        first_time = self.last is None
        new = self.vis_infos[self.i]
        self.vis.add_geometry(new.pcd, reset_bounding_box=first_time)
        if new.marker1:
            self.vis.add_geometry(new.marker1, reset_bounding_box=False)
            self.vis.add_geometry(new.marker2, reset_bounding_box=False)
            self.vis.add_geometry(new.normal_marker, reset_bounding_box=False)
            # trace
            self.vis.add_geometry(new.trace_marker, reset_bounding_box=False)
            self.trace.append(new.trace_marker)

        if self.last:
            self.vis.remove_geometry(self.last.pcd)
            self.vis.remove_geometry(self.last.marker1)
            self.vis.remove_geometry(self.last.marker2)
            self.vis.remove_geometry(self.last.normal_marker)
        self.last = new

        if not first_time:
            self.vis.set_view_status(vs)
        self.vis.update_renderer()

    def on_capture_key(self):
        global snapshot_dir
        self.on_next_key()
        if not snapshot_dir:
            snapshot_dir = tempfile.mkdtemp(prefix="tracking_snapshots_")
            print(f"WRITING SNAPSHOTS TO DIRECTORY {snapshot_dir}")
        self.vis.capture_screen_image(str(Path(snapshot_dir) / f"frame_{str(self.i).zfill(4)}.png"), do_render=True)
