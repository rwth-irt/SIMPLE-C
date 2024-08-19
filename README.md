# SIMPLEX
##  Uncertainty-aware Multi-LiDAR Extrinsic Calibration using a Simple Dynamic Target in Moving Feature-Sparse Environments

This calibration tool can be used for calibrating multiple LiDAR sensors using a simple reflective targaet that is moved through the environment while being tracked. The tool was created to allow for pairwise calibration of LiDAR sensors on large vessels. The vessel is docked on the water introducing a unintended moving sensor platform and a moving environment (water surface) meaning that we cannot use any feature- or environment-based calibration algorithm. Also, we cannot use static targets as the target would constantly move on the floating watersurface. Further, large vessels are often expensive to move such that we created a tool that does not require any movement to the platform to calibrate sensors. We also omit the use of external auxilary sensors such as IMU or GNSS, the need to manufacture complex shaped targets or algorithms that rely on geometric shape detection and calculation. Our calibration tool can be used indoor, outdoor and on waterside making the tool available for many users, applications and environments. Additionally, the tool is online-capable such that live feedback can be used to optimize the calibration procedure or parameters. For point cloud registration we rely on a simple weighted Kabsch Algorithm.

We demonstrate a robust and simple online calibration tool for LiDAR sensors in maritime applications.


## Running the calibration

**Online calibration in ROS/Docker**:

To run in a docker container with all dependencies installed, the ROS package built and everything from ROS properly  sourced (this is all ensured when using the docker image built from the definitions in the [docker repository](https://git-ce.rwth-aachen.de/g-nav-mob-irt/projects/galileonautic2plus/calibration/robosense_docker)), run the following command:

`ros2 run online_calibration main --ros-args --params-file /PATH/TO/parameters.yaml -p sensor_pairs:="<PAIRS>"`

The `/PATH/TO/parameters.yaml` must be the path **inside** the docker container! Make sure that the parameter file is somehow accessible inside the docker container for ROS to read it. (As the source code of this ROS package contains it, it has probably already been copied to docker.)

Replace `<PAIRS>` with a sensor pair definition using the following syntax: `topicA,topicB;topicB,topicC`. Pairs are separated using semicolons, in a single pair, the two sensors are comma-separated.

The ros node will then listen to PointCloud2 messages on the provided topics and push calibrations to the topic `transformations`. Status information is currently (**TODO**) provided on stdout (command line of the ros node).  All transformations use the message type `TransformStamped`, which includes information about the involved coordinate frames (here: sensor topic names). Therefore, transformations for multiple sensor can be pushed to the same ROS topic.

**Offline calibration**

- Using the script `online_calibration/main_cli.py`, the calibration can be used without ROS (pass `--help` for usage information) by importing ROS-bags as numpy arrays.
- Reading rosbag data is used using separate libraries.
- Currently only reading ROS1 bags is supported
- Visualizations of reflector tracking and point pair alignment are available, which are not available inside of ROS.
- As rosbag import is very slow, it might be faster to run the ROS node and play back the rosbag using ROS itself as long as no visualization is required.


**Useful**
- Set the data and repository directory in the docker to mount your data location and the calibration docker repository before building the container: `export DATA_DIR=<path to your data directory>`, `export REPO_DIR=<path to your calibration repository> `
- Open another terminal in the docker: `sudo docker exec -it lidar1 bash`
- Visualization is hosted on Port 8000: http://localhost:8000
- Parameters of the calibration tool can be adjusted in the `launch.py`
- The calibration tool can be started in the docker container using rebuild.sh

## Code structure
- All code lives inside the `src` directory in the ROS package `online_calibration`. 
- The code is split up into three parts:
  1. ROS-specific code. This is all code outside of the `core` and `local` directories.
  2. A part for local execution (and debugging/visualization using open3d) independent of ROS. This code is in `local` and is started from the script file `online_calibration/main_cli.py`.
  3. The shared logic code which is used by both of these launching methods is contained inside the `core` directory.
- To get familiar with the code, starting in `main_ros.py` should be a good entry point for understanding the main logic. The ROS-independent code uses other data imports and calls the logic differently.

## Functional overview

**Data import**:

Single ROS PointCloud2 messages are parsed (either coming from a Rosbag file, see `local/rosbag_import.py` or from a ROS message, see `online_calibrator.py`) to numpy arrays. These contain the unordered single scanned points in the format (x, y, z, intensity, ring, timestamp). The fields ring and timestamp are currently unused.

The frame data is then fed into `Frame` objects (`core/frame.py`). These do some caching of expensive per-frame analysis results, which is useful because one frame might be used to align multiple sensor pairs.

**Calibrating (multiple) pairs of sensors**  
In a single run of the program, multiple sensor pairs should be calibrated, which usually share sensors (e.g. A-B and B-C). In the following, only calibration of a single sensor pair is described, which is handled by a single `PairCalibrator` object. In the ROS node, an `OnlineCalibrator` object distributes the correct frame data to multiple of these to allow for simultaneous calibration of multiple pairs.

**PairCalibrator: Timing of frame data**  
A PairCalibrator has a callback function `new_frame`, which is passed `Frame` objects for both sensors (in real-time) as soon as they are recorded. As the Kabsch algorithm requires pairs of points, i.e. corresponding frames of both sensors, these frames are cached until data has been received from each sensor. Frame acquisition can not be triggered at the Lidar sensors used for this project. Therefore, frames are simply dropped if they are older than 60% of the expected time delay between two frames.

**Reflector detection**  
A series of processing steps is used to identify the detector in the data *of a single sensor*.

1. The reflector is expected to result in very intense points in the lidar frame. Hence points are filtered using a threshold value for the intensity channel. The threshold value is calculated adaptively so that a fixed ratio of points passes (configurable by the parameter `"relative intensity threshold"`).

2. These remaining points are then expected to be from distinct bright objects. The DBSCAN algorithm is used for clustering, configurable by the parameters `"DBSCAN epsilon"` (in meters, as is all coordinate data from the sensors) and `"DBSCAN min samples"`.

These steps are performed in the file `core/locate_reflector/find_cluster_centers.py`. Results are cached in the `Frame` class because until now, they are independent of sensor pairs.

To differentiate between the reflector and static reflective objects (or almost static, such as trees), sequences of successive frames are now used for further analysis. The length of each sequence is adjusted by `"window size"`. Results are only calculated for the *last* frame in a sequence though.

3. For each cluster found in the last frame of a sequence, the trace of successive nearest neighbors (in the previous frame respectively) is determined. These traces are then filtered using multiple rules. It is assumed that if one trace passes all those filters, it belongs to the reflector.
   - We expect the reflector to be visible in the whole sequence. Sequences which are too short are dropped.
   - The reflector is supposed to move to distinguish from static objects. If the *average* distance between two adjacent points is too small (`"minimum velocity"`, in meters \[per frame]), the trace is dropped.
   - The reflector is also not supposed to jump/move very quickly, so with `"maximum neighbor distance"` there is also an upper threshold for the distance between two adjacent frames (here, no average is used).
   - The reflector is expected to not change its movement direction too drastically. Therefore, the angle between adjacent movement vectors must not exceed a certain limit (`"max. vector angle [deg]"`). This filter aims to distinguish it from randomly moving objects, such as water reflections or trees.
   - The number of points in the cluster resembling the reflector is expected to not change drastically. If the difference in point number changes by a certain percentage between two frames (`"max_point_number_change_ratio"`), traces are also dropped.

4. If a single trace passes all those filters, a `ReflectorLocation` object is created, which stores information about the cluster that is identified to originate from the reflector. It is intentionally avoided to store the whole frame data to reduce memory usage.

5. If the reflector has been found in corresponding frames of both sensors, these locations are considered a *point pair*. These point pairs are stored and if more than three have been found, they can be passed to the Kabsch algorithm to calculate the transformation. As more data is acquired, the transformation is successively updated. This allows to run the calibration online in real-time. Using the sensitivity (uncertainty) of the current transformation, the operator can decide whether enough data has been collected for the required precision.

6. As soon as an initial transformation exists, further filtering is applied to remove outliers that might have passed all filters described in step 3. To accomplish this, all point pairs of one sensor are transformed, which should ideally result in perfectly aligned point clouds. The mean distance between two adjacent points (after transformation) is calculated and point pairs with a much greater distance (`mean_distance * "outlier_mean_factor"`) are excluded before the transformation is calculated.

7. The Kabsch algorithm accepts weights for each point pair. This weight is composed of multiple weights (whose influence can be adjusted individually):
   - The number of points in the cluster, divided by the maximum number of all clusters identified as the reflector.
   - The cosine similarity of the normal vector of the reflector surface and the vector from sensor (in the origin) to the cluster centroid. The normal vector is calculated using an SVD, assuming the points to be distributed approximately planar. This follows the assumption that the position of the reflector's center can be calculated more accurately if the reflector surface points directly towards the sensor.

   Note that the algorithm only accepts a *single* weight per point *pair*, not per point. Therefore, the minimum of the points in a pair is used.
