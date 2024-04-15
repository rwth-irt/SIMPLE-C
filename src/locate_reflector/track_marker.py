import numpy as np


def get_nearest_neighbor_index(origin, candidates, radius):
    if len(candidates) == 0:
        return -1
    dist = np.linalg.norm(candidates[:, :3] - origin[:3], axis=1)
    best_index = np.argmin(dist)
    if dist[best_index] <= radius:
        return best_index
    return -1


def calc_velocity(origin, future, neighbor_radius, max_vector_angle_rad):
    # calculates velocity for one center `origin`
    # given future centroids
    # if no neighbor within distance `neighbor_radius` is found or the
    # angle between two movement steps is greater than `max_vector_angle_rad`, -1 is returned instead
    assert len(future) > 0
    deltas = []
    current = origin
    last_movement = None
    for next_candidates in future:
        nni = get_nearest_neighbor_index(current, next_candidates, neighbor_radius)
        if nni == -1:
            return -1  # can not calculate
        nn = next_candidates[nni]  # next neighbor
        movement = (nn - current)[:3]
        deltas.append(np.linalg.norm(movement))
        if last_movement is not None:
            # calculate angle between vectors
            angle_rad = np.arccos(
                np.dot(movement, last_movement)
                / (np.linalg.norm(movement) * np.linalg.norm(last_movement))
            )
            if angle_rad > max_vector_angle_rad:
                return -1
        last_movement = movement
        current = nn
    return np.mean(deltas)


def filter_clusters_2(clusters, max_distance, min_velocity, velocity_lookahead, max_vector_angle_rad):
    """
    TODO

    :param clusters: List with (possibly empty, meaning no clusters) numpy array
        containing cluster center means
    :param max_distance:
    :param min_velocity:
    :param velocity_lookahead:
    :param max_vector_angle_rad:
    :return:
    """
    # per frame, choose the best cluster
    # return list of their indices or None
    selection_indices = []
    for frame_i in range(len(clusters) - velocity_lookahead):
        if len(clusters[frame_i]) == 0:
            selection_indices.append(None)
            continue  # no clusters in this frame

        # calculate velocity for each cluster
        velocities = np.array(
            [
                calc_velocity(
                    c,
                    clusters[frame_i + 1: frame_i + 1 + velocity_lookahead],
                    max_distance,
                    max_vector_angle_rad
                )
                for c in clusters[frame_i]
            ]
        )

        # exclude stationary clusters
        velocities[velocities < min_velocity] = -1

        # now choose *largest* cluster with *highest velocity*
        # (excluded clusters have velocity = -1)

        c_i = clusters[frame_i]
        # sort by size
        sortargs_size = np.argsort(c_i[:, 4])
        c_i = c_i[sortargs_size]
        # sort by velocity
        sortargs_vel = np.argsort(velocities, kind="stable")
        best_index = sortargs_vel[-1]

        # has best one velocity != -1? Otherwise, all have been excluded
        if velocities[sortargs_vel[-1]] == -1:
            selection_indices.append(None)
        else:
            selection_indices.append(best_index)

    # add None for last frames, no velocity calculation possible
    selection_indices += [None] * velocity_lookahead
    return selection_indices


def track_marker(centers, params):
    """
    Tracks the marker for **a single sensor**.
    Applies `filter_clusters_2` to given cluster centers and return the chosen centers as array.

    :param centers: list of cluster centers as obtained from `get_cluster_centers_per_frame`
    :param params: dict with parameters, e.g. obtained by reading a parameter JSON file.
    :param visualize: bool whether to show the open3d visualization of the analysis results
    :return: a list of the found cluster centers as numpy arrays (only containing x y z values!)
    """
    selection_indices = filter_clusters_2(
        centers,
        max_distance=params["maximum neighbor distance"],
        min_velocity=params["minimum velocity"],
        velocity_lookahead=int(params["velocity lookahead"]),
        max_vector_angle_rad=2 * np.pi * params["max. vector angle [deg]"] / 360,
    )
    # TODO: Idea for uncertainty of marker position
    #  Maybe calculate an uncertainty score by checking if the number of points in the cluster is close to
    #  the number in adjacent frames. If by accident the tracker cluster is merging with some close disturbance cluster,
    #  this will allow to at least reduce the uncertainty. Otherwise, maybe just exclude such clusters at all.

    # extract chosen centers by indices
    chosen_centers = []
    for frame_i in range(len(selection_indices)):
        if selection_indices[frame_i] is None:
            chosen_centers.append(None)
        else:
            chosen_centers.append(centers[frame_i][selection_indices[frame_i], :3])  # only x,y,z

    return selection_indices, chosen_centers
