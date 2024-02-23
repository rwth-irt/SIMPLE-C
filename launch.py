from imports import bag_to_numpy
from visualization import visualize_animation
import numpy as np
from sklearn.cluster import DBSCAN
from pprint import pprint


def get_cluster_centers_per_frame(
    frames,  # input points: frames[frame_index, point_index] = [x,y,z,intensity]
    N_highest_intensity=30,  # the N points with highest intens. used for DBSCAN clustering
    DBSCAN_epsilon=0.15,  # in meters, as is LIDAR output (TODO: is that correct?)
    DBSCAN_min_samples=3,
):
    # per frame, get the indices of the N points with highest intensity
    brightest_indices = np.argsort(-frames[:, :, 3], axis=1)[:, :N_highest_intensity]

    # mark points for visualization, rewriting the intensity channel in `frames`
    frames[:, :, 3] = 0  # 0 -> any point
    for i in range(len(frames)):
        frames[i, brightest_indices[i], 3] = 1  # 1 -> N with highest intensity

    # get selected points in extra array
    brightest_points = np.zeros((len(frames), N_highest_intensity, 4))
    for i, (sel, frame) in enumerate(zip(brightest_indices, frames)):
        brightest_points[i] = frame[sel]
        # TODO don't know how to do this with one of those billion np functions, select, take, take_along_axis

    cluster_centers_per_frame = []
    for i in range(len(frames)):
        # perform DBSCAN per frame
        clusterlabels = DBSCAN(
            eps=DBSCAN_epsilon, min_samples=DBSCAN_min_samples
        ).fit_predict(
            brightest_points[i, :, :3]  # :3, without intensity
        )

        # get centroids of clusters and mark them for visualization
        centers = []
        for clusterlabel in np.unique(clusterlabels):
            if clusterlabel == -1:
                continue  # ignore noise
            centers.append(
                np.mean(
                    frames[
                        i, brightest_indices[i, clusterlabels == clusterlabel], :3
                    ],  # :3, without intensity
                    axis=0,
                )
            )
            # mark points as 2 -> "in cluster" for visualization
            frames[i, brightest_indices[i, clusterlabels == clusterlabel], 3] = 2
        cluster_centers_per_frame.append(np.array(centers))

    return cluster_centers_per_frame


def get_all_traces(start: np.ndarray, radius: float, candidates: list[np.ndarray]):
    """
    - Gets a starting point `start`
    - `candidates` is a list containing a numpy array with candidate
      points for the N next frames, i.e. a list of N x 3 numpy arrays.
      It **must not include** the frame where `start` is taken from!
    - Will return a list of all possible traces starting with `start`.
    """
    # no more points left?
    if len(candidates) == 0:
        return [[start]]

    # get all possible successors: close candidate points in next frame
    distances = np.linalg.norm(candidates[0] - start, axis=1)
    successors = candidates[0][distances <= radius]

    # get all traces for all successors
    traces = []
    for s in successors:
        s_traces = get_all_traces(s, radius, candidates[1:])
        traces.extend(s_traces)
    # add start element at the start of all traces
    for t in traces:
        t.insert(0, start)
    return traces


def filter_traces_by_length(traces, min_length):
    return [t for t in traces if len(t) >= min_length]


def find_primary_traces(
    centers,  # a list containing a 2d numpy array of found cluster centers per frame
    max_inter_frame_distance=0.3,  # the maximum distance a cluster may move between two frames
    min_trace_length=4,  # minimum trace length to be considered
):
    """
    Finds traces which are unique per frame (so there are no "competing" traces in any frame of the trace).

    Returns a list of tuples with (start_frame_index, [point_0, point_1, ...])
    """
    results = []
    current_frame_i = 0
    while current_frame_i < len(centers):
        print(current_frame_i)
        print(results)
        current_trace = []
        # get longest trace starting in current frame
        for center in centers[current_frame_i]:
            traces = get_all_traces(
                center, max_inter_frame_distance, centers[current_frame_i + 1:]
            )
            traces = filter_traces_by_length(traces, min_trace_length)
            if not traces:
                continue
            longest = max(traces, key=len)
            current_trace = max(current_trace, longest, key=len)

        if not current_trace:
            current_frame_i += 1
            continue

        # check if there are competing traces for this trace
        # i.e. in each frame this trace has a point in, check traces of all other points
        competition = None
        competition_start = None  # just declaring
        for trace_frame_i in range(len(current_trace)):
            for other_start in centers[current_frame_i + trace_frame_i]:
                if other_start in current_trace:
                    continue
                competition_candidates = get_all_traces(
                    other_start,
                    max_inter_frame_distance,
                    centers[current_frame_i + trace_frame_i + 1 :],
                )
                competition_candidates = filter_traces_by_length(competition_candidates)
                if competition_candidates:
                    competition = max(competition_candidates, key=len)
                    competition_start = trace_frame_i
                    break
            if competition:
                break
        if not competition:
            results.append((current_frame_i, current_trace))
            current_frame_i += (
                len(current_trace) + 1
            )  # continue searching after this trace
        else:
            # cut longest trace at this point where we have serious competition
            # and continue searching after either competition or current trace are over
            cropped = current_trace[:competition_start]
            results.append((current_frame_i, cropped))
            end_of_current = current_frame_i + len(current_trace)
            end_of_competitor = current_frame_i + competition_start + len(competition)
            continue_at = min(end_of_current, end_of_competitor)
            current_frame_i = continue_at  # one of them has ended here


# MAIN
filename = (
    "/home/max/UNI/Job_IRT/LIDAR/temporal_reflector/temporal_reflector_disturbance.bag"
)
# filename = "labor_lidar.bag"
frames = bag_to_numpy(filename)
centers = get_cluster_centers_per_frame(frames)
print(list(map(len, centers)))
visualize_animation(frames)
exit()
traces = find_primary_traces(centers)
pprint(traces)

