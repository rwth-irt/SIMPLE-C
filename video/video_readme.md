# Creating a video

1. set the view
    - config can be copied to clipboard in the 3d ui by pressing Ctrl+C and pasted by Ctrl+V
    - `view_config.txt` contains a view which fits the file config below
2. alternately press P (for saving a snapshot) and K (for next frame). Yes, this is annoying.
3. stitch the pngs together, e.g. using `ffmpeg`, see `create_video.sh`.

## TODO: Automation
Using [this function](https://www.open3d.org/docs/release/python_api/open3d.visualization.Visualizer.html#open3d.visualization.Visualizer.capture_screen_image), one could automate saving the png files and alternating frames.



--------------------
File config for `view_config.txt`:
```python
filename2 = (
    "/home/max/UNI/Job_IRT/LIDAR/Calibration_Target_Deneb/calibration_target_1.bag",
    "/rslidar_points_vr_v",  # topic
)
```