# TraBaDyFiCa

> Trajectory Based Dynamic Filtering for Calibration

## Running online calibration in Docker

To run in a docker container with all dependencies installed, the ROS package built and everything from ROS properly 
sourced (this is all ensured when using the docker image built from the definitions in the
[docker repository](https://git-ce.rwth-aachen.de/g-nav-mob-irt/projects/galileonautic2plus/calibration/robosense_docker),
run the following command:

`ros2 run online_calibration main --ros-args --params-file /PATH/TO/parameters.yaml -p sensor_pairs:="<PAIRS>"`

The `/PATH/TO/parameters.yaml` must be the path **inside** the docker container!
Make sure that the parameter file is somehow accessible inside the docker container for ROS to read it.
(As the source code of this ROS package contains it, it has probably already been copied to docker.)

Replace `<PAIRS>` with a sensor pair definition using the following syntax: `topicA,topicB;topicB,topicC`.
Pairs are separated using semicolons, in a single pair, the two sensors are comma-separated.

The ros node will then listen to PointCloud2 messages on the provided topics and push calibrations to the
topic `transformations`. Status information is currently (**TODO**) provided on stdout (command line of the ros node). 
