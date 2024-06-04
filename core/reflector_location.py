class ReflectorLocation:
    def __init__(self, cluster_mean, cluster_points):
        # Note that this object does intentionally not have any reference to the frame object it is based on,
        # so that it (including the large point cloud array) can be garbage collected.

        self.cluster_mean = cluster_mean
        self.cluster_points = cluster_points
        self.weight = 1

        # TODO extend weight calculation, using normal vector and weight
