# Running with ROS2 and Docker
This file explains how to run the online calibration inside a Docker container **without** access
to live sensor data for debugging purposes. Pre-recorded data can be replayed using rosbag files.

1.  Run the ROS2 Docker container with access to source code and pre-recorded rosbag data:
    ```sh
    sudo docker run --volume <THIS_REPO>:/calib_src --volume <DATA_DIRECTORY>:/DATA:ro -it <VOLUME_TAG> bash
    ```
    Note that we need python dependencies such as `sklearn` in there. For this you can use the
    Dockerfile to build a working image: `sudo docker build -t <VOLUME_TAG> .`

2.  *In the container*, navigate to `cd /calib_src/ros_workspace`.

3.  *Optional:* Run `rosdep` using `rosdep check --from-path lidar_align/ --rosdistro iron`.

4.  Run `colcon build` to build the ROS2 package.
    (This installs *all* packages found in the current workspace.)
    ~~The `--symlink-install` option creates links to the python code, so that updates to the code on the host system
    appear in ROS immediately without having to rebuild. However, the build fails on my machine using this option.~~

5.  Source the new ROS2 Package: `source install/setup.bash`

6.  Run the calibration node: `ros2 run lidar_align main`

7.  To play back data, you need a Rosbag file in a format compatible with ROS2.
    (See below on how to convert ROS1-Rosbags to the correct format.)

    - Open a new terminal in the docker container: `sudo docker exec -it <CONTAINER_NAME> bash`.  
      (Run this *outside* of docker! Get `<CONTAINER_NAME>` from `sudo docker ps`.)
    - In this new terminal, ROS2 is not sourced yet, so call `source /opt/ros/iron/setup.bash`.
    - Then play the rosbag data using `ros2 bag play /DATA/<NAME_OF_ROSBAG>`.

8.  If you update your code on the host:
    - ~~If you used `--symlink-install` (see above), simply rerun `ros2 run lidar_align main`.~~
    - Without `--symlink-install`, you have to rebuild: `colcon build && source install/setup.bash && ros2 run lidar_align main`


## Converting Rosbags from ROS1 to ROS2
- Install the conversion tool via pip: `pip install rosbags`
- Convert Rosbags using `rosbags-convert --src <ROS1_rosbag> --dst <OUTPUT_NAME> --dst-typestore ros2_iron`
- The converted file is (in a new directory) `<OUTPUT_NAME>/<OUTPUT_NAME>.db3`.