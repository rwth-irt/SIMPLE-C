# TEMPORARY DOCKERFILE
#
# Installs all dependencies to run the ROS2 online calibration code with Ros-Bags.
# TODO: This should later be merged with the Dockerfile containing the RSLidar SDK.
FROM ros:iron-ros-base

RUN apt-get update && \
    apt-get install -y python3-sklearn && \
    rm -rf /var/lib/apt/lists/*
