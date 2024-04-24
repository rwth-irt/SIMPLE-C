### Idea for measuring the uncertainty of marker position
- check if the number of points in the cluster is close to
the number in adjacent frames. If by accident the tracker cluster is merging with some close disturbance cluster,
this will allow to at least reduce the uncertainty. Otherwise, maybe just exclude such clusters at all.
- We know that the reflector is planar. Calculate how planar the found cluster is, and how well its normal points in the direction of the sensor. This should correlate with the number of points in the cluster.