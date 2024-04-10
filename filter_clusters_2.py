import numpy as np


def get_nearest_neighbor_index(origin, candidates, radius):
    if len(candidates) == 0:
        return -1
    dist = np.linalg.norm(candidates[:, :3] - origin[:3], axis=1)
    best_index = np.argmin(dist)
    if dist[best_index] <= radius:
        return best_index
    return -1


def filter_clusters_2(clusters, max_distance, min_velocity):
    # per frame, find most relevant cluster
    selection = []
    for frame_i in range(len(clusters) - 1):
        # TODO this is currently hardcoded for velocity lookahead 1
        if len(clusters[frame_i]) == 0:
            selection.append(None)
            continue  # no clusters in this frame

        # calculate velocity for each cluster
        velocities = []
        for center_i in range(len(clusters[frame_i])):
            origin = clusters[frame_i][center_i]
            nni = get_nearest_neighbor_index(
                origin, clusters[frame_i + 1], max_distance
            )
            if nni == -1:  # no neighbor found
                velocities.append(-1)  # exclude clusters without neighbors
            else:
                v = np.linalg.norm((clusters[frame_i + 1][nni] - origin)[:3])
                velocities.append(v)
        velocities = np.array(velocities)

        # exclude stationary clustersnp.array(velocities)
        velocities[velocities < min_velocity] = -1

        # now choose *largest* cluster with *highest velocity*
        # (excluded clusters have velocity = -1)

        c_i = clusters[frame_i]
        # sort by size
        sortargs_size = np.argsort(c_i[:, 4])
        c_i = c_i[sortargs_size]
        # sort by velocity
        sortargs_vel = np.argsort(velocities, kind="stable")
        c_i = c_i[sortargs_vel]

        # pick best one
        best = c_i[-1]
        if velocities[sortargs_vel[-1]] == -1:
            selection.append(None)  # no candidate found: all were excluded
        else:
            selection.append(best)
    selection.append(None)  # add None for last frame, no velocity calculation possible
    return selection
