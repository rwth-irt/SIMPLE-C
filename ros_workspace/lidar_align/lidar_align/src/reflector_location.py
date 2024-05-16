class ReflectorLocation:
    def __init__(self, cluster_mean, cluster_points):
        self.cluster_mean = cluster_mean
        self.cluster_points = cluster_points
        self.weight = 1

        # TODO extend weight calculation, using normal vector and weight
