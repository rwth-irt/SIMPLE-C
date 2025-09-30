import numpy as np
from scipy.linalg import block_diag

def centroid_covariance(cluster_points: np.ndarray) -> np.ndarray:
    """
    Estimate the 3×3 covariance of the centroid of a cluster.

    Σ_centroid = (1/n) * Σ_cluster
    where Σ_cluster is the empirical covariance of the raw points.
    """
    n = cluster_points.shape[0]

    if n < 2:
        # fallback – at least sensor noise
        return (0.05**2) * np.eye(3)   # 5 cm²

    # unbiased empirical covariance of raw points (3×3)
    Sigma_cluster = np.cov(cluster_points.T, bias=False)

    # Covariance of the centroid
    return (1/n) * Sigma_cluster


def compute_whitened_jacobian_metrics(J: np.ndarray, P: np.ndarray, Sigma_e: np.ndarray):
    """
    Computes the whitened and scaled Jacobian and its smallest singular value.
    We do this to account for the different scales of rotation and translation as well as measurement noise.

    :param J: The Jacobian matrix (3N, 6).
    :param P: The point cloud from the first sensor (N, 3).
    :param Sigma_e: The block-diagonal measurement covariance matrix (3N, 3N).
    :return: A tuple containing:
             - sigma_min_w (float): The smallest singular value of the whitened and scaled Jacobian.
             - J_w (np.ndarray): The whitened and scaled Jacobian matrix.
    """
    # Calculate length scale L
    p_bar = np.mean(P, axis=0)
    L = np.sqrt(np.mean(np.sum((P - p_bar)**2, axis=1)))

    # Scaling matrix S
    S = np.diag([L, L, L, 1, 1, 1])

    # Whitening matrix Sigma_e^(-1/2)
    try:
        # Since Sigma_e is block-diagonal, we can work with blocks if needed for performance.
        # For now, direct computation is fine.
        Sigma_e_inv = np.linalg.inv(Sigma_e)
        # Use Cholesky decomposition for numerical stability to get the square root
        Sigma_e_inv_sqrt = np.linalg.cholesky(Sigma_e_inv).T
    except np.linalg.LinAlgError:
        # If Sigma_e is singular, we cannot whiten.
        return 0.0, None

    # Whitened and scaled Jacobian
    J_w = Sigma_e_inv_sqrt @ J @ np.linalg.inv(S)

    # Smallest singular value of J_w
    try:
        _, s_w, _ = np.linalg.svd(J_w)
        sigma_min_w = s_w[-1] if s_w.size > 0 else 0.0
    except np.linalg.LinAlgError:
        sigma_min_w = 0.0

    return sigma_min_w, J_w

def build_measurement_covariance(list_of_clusters: list[np.ndarray]) -> np.ndarray:
    """
    Build block-diagonal Σ_e from all clusters (N clusters ⇒ 3N×3N matrix).
    """
    blocks = [centroid_covariance(C) for C in list_of_clusters]
    return block_diag(*blocks)

def parameter_covariance(J: np.ndarray,
                         Sigma_e: np.ndarray) -> np.ndarray:
    """
    Return Σ_(φ,t) = (Jᵀ Σ_e^{-1} J)^{-1}.
    """
    JT_Sinv_J = J.T @ np.linalg.inv(Sigma_e) @ J
        
    return np.linalg.pinv(JT_Sinv_J)
    # return np.linalg.inv(JT_Sinv @ J)

def skew_matrix(p: np.ndarray) -> np.ndarray:
    """Return the skew-symmetric matrix of a 3D vector p."""
    return np.array([[  0,   -p[2],  p[1]],
                     [ p[2],    0,  -p[0]],
                     [-p[1],  p[0],    0]])

def compute_jacobian(points: np.ndarray) -> np.ndarray:
    """
    Build the (3N × 6) Jacobian for N transformed 3D points.
    Each 3×6 block is: [- skew(p_i),  -I3].
    """
    N = points.shape[0]
    J = np.zeros((3 * N, 6), dtype=float)

    for i, p in enumerate(points):
        J[3*i : 3*(i+1), :3] = -skew_matrix(p)
        J[3*i : 3*(i+1), 3:] = np.eye(3)

    return J

def compute_observability_metrics(points: np.ndarray) -> tuple[float, float, float, np.ndarray, np.ndarray]:
    
    """
    From transformed points, compute:
      - largest singular value
      - smallest singular value
      - condition number = largest/smallest
    Returns (largest_sv, smallest_sv, condition_number).
    """
    J = compute_jacobian(points)

    # SVD to get singular values
    _, S, Vt = np.linalg.svd(J, full_matrices=False)
    
    largest = float(S[0])
    smallest = float(S[-1]) if S[-1] > 0 else 0.0
    cond    = largest / smallest if smallest > 0 else float('inf')

    # Extract nullspace direction (6D) for smallest singular value
    nullspace_vector = Vt.T[:, -1]

    return largest, smallest, cond, nullspace_vector, J
