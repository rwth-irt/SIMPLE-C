import matplotlib.pyplot as plt
import numpy as np


def plot_match_distances(points_a, points_b):
    distances = np.linalg.norm(points_a - points_b, axis=1)
    plt.plot(distances)
    plt.ylabel("Distance between correlated points after transformation")
    plt.xlabel("Point index")
    plt.show()
