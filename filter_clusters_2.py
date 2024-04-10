import numpy as np


def get_nearest_neighbor_index(origin, candidates, radius):
    if len(candidates) == 0:
        return -1
    dist = np.linalg.norm(candidates[:, :3] - origin[:3], axis=1)
    best_index = np.argmin(dist)
    if dist[best_index] <= radius:
        return best_index
    return -1


def calc_velocity(origin, future, neighbor_radius):
    # calculates velocity for one center `origin`
    # given future centroids
    assert len(future) > 0
    deltas = []
    current = origin
    for next_candidates in future:
        nni = get_nearest_neighbor_index(current, next_candidates, neighbor_radius)
        if nni == -1:
            return -1  # can not calculate
        nn = next_candidates[nni]  # next neighbor
        deltas.append(np.linalg.norm(current - nn))
        current = nn
    return np.mean(deltas)


def filter_clusters_2(clusters, max_distance, min_velocity, velocity_lookahead=3):
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
                    clusters[frame_i + 1 : frame_i + 1 + velocity_lookahead],
                    max_distance,
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
