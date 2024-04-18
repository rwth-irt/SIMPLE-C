# based on https://hub.docker.com/_/ros/

FROM ros:noetic-ros-base

# apply Robosense Lidar SDK installation, see https://github.com/RoboSense-LiDAR/rslidar_sdk
# we use the "compile directly" variant:
# https://github.com/RoboSense-LiDAR/rslidar_sdk?tab=readme-ov-file#41-compile-directly
# and use sed to set the compile method line.

WORKDIR /robosense
RUN apt-get update && \
    apt-get install -y git libpcap-dev libyaml-cpp-dev && \
    rm -rf /var/lib/apt/lists/* && \
    git clone https://github.com/RoboSense-LiDAR/rslidar_sdk.git && \
    cd rslidar_sdk && \
    git submodule init && \
    git submodule update && \
    sed -i 's/set(COMPILE_METHOD CATKIN)/set(COMPILE_METHOD ORIGINAL)' CMakeLists.txt && \


# launch ros package
# note that ROS is automatically sourced inside the base image!
CMD ["ros2", "launch", "demo_nodes_cpp", "talker_listener.launch.py"]
