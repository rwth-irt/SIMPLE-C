import numpy as np
from typing import TYPE_CHECKING
from scipy.linalg import block_diag, cho_factor, cho_solve
from . import parameters

if TYPE_CHECKING:
    from .transformation import Transformation

def centroid_covariance(cluster_points: np.ndarray) -> np.ndarray:
    """
    Estimate the 3×3 covariance of the centroid of a cluster.

    Σ_centroid = (1/n) * Σ_cluster
    where Σ_cluster is the empirical covariance of the raw points.
    """
    n = cluster_points.shape[0]

    if n < 2:
        # Fallback for clusters with less than 2 points.
        # Propagate uncertainty based on the point's position, assuming it's the centroid.
        return propagate_spherical_to_cartesian_covariance(cluster_points[0])

    # unbiased empirical covariance of raw points (3×3)
    Sigma_cluster = np.cov(cluster_points.T, bias=False)

    # Covariance of the centroid
    return (1/n) * Sigma_cluster


def compute_whitened_jacobian_metrics(J: np.ndarray, P: np.ndarray, Sigma_e: np.ndarray):
    """
    Computes the whitened and scaled Jacobian, its smallest singular value, and the trace of the whitened covariance.
    """
    # Calculate length scale L
    p_bar = np.mean(P, axis=0)
    L = np.sqrt(np.mean(np.sum((P - p_bar)**2, axis=1)))
    # Guard against degenerate geometry
    if not np.isfinite(L) or L <= 0.0:
        L = 1e-6

    # Scaling matrix S
    S = np.diag([L, L, L, 1, 1, 1])

    # Whitening: use Cholesky of Sigma_e and triangular solve, avoid inv(Sigma_e)
    try:
        c, lower = cho_factor(Sigma_e, overwrite_a=False, check_finite=True)
        # We need W^(1/2) = Sigma_e^(-1/2) applied to J, i.e., L^{-T} J where Sigma_e = L L^T
        # Solve L^T X = J  => X = L^{-T} J
        J_w_left = cho_solve((c, lower), J, overwrite_b=False)  # solves Sigma_e X = J
        # But cho_solve gives X = Sigma_e^{-1} J; we want Sigma_e^{-1/2} J.
        # So do it in two steps using the Cholesky factor:
        # Solve L Y = J  and then set J_w = Y  (since ||Y||^2 = J^T Sigma_e^{-1} J in normal matrix)
        # Efficient way: get the Cholesky factor and explicitly apply L^{-1}:
        # Recompute to apply L^{-1} directly
        import numpy.linalg as npl
        Lc = c if lower else c.T
        # Solve L Z = J  => Z = L^{-1} J
        Z = npl.solve(Lc, J)
        # Now Sigma^{-1/2} J can be taken as L^{-T} J or equivalently Z with correct quadratic form.
        # For J_w^T J_w we can use Z^T Z, which equals J^T Sigma^{-1} J.
        J_whitened = Z
    except np.linalg.LinAlgError:
        return 0.0, float('inf'), None

    # Apply scaling on the right
    try:
        S_inv = np.linalg.inv(S)
    except np.linalg.LinAlgError:
        S_inv = np.diag([1.0, 1.0, 1.0, 1.0, 1.0, 1.0])

    J_w = J_whitened @ S_inv

    # Smallest singular value of J_w
    try:
        _, s_w, _ = np.linalg.svd(J_w, full_matrices=False)
        sigma_min_w = s_w[-1] if s_w.size > 0 else 0.0
    except np.linalg.LinAlgError:
        sigma_min_w = 0.0

    # Whitened parameter covariance and its trace (no double whiten)
    try:
        # Equivalent to inv(J^T Sigma^{-1} J) with scaling
        Sigma_theta_w = np.linalg.pinv(J_w.T @ J_w)
        trace_of_whitened_covariance = float(np.trace(Sigma_theta_w))
    except np.linalg.LinAlgError:
        trace_of_whitened_covariance = float('inf')

    return sigma_min_w, trace_of_whitened_covariance, J_w

def huber_rho_norm(m: float, delta: float) -> float:
    """Huber loss on the Mahalanobis norm m = ||r_w|| (not squared)."""
    if m <= delta:
        return 0.5 * m * m
    else:
        return delta * (m - 0.5 * delta)

def compute_residual_metrics(
    P_hat, Q, Sigma_e: np.ndarray,
    d: int = 6, jitter_eps: float = 1e-9,
    use_huber: bool = True, huber_delta: float = 1.5
) -> tuple[float, float, float, float]:
    """
    Computes whitened residual metrics.
    Returns: (S_t, L_t, L_t_adj, L_t_rob)
      S_t      = sum of squared whitened residuals
      L_t      = mean squared whitened residual (per pair)
      L_t_adj  = DoF-bias-corrected mean (per pair), using N_eff - d
      L_t_rob  = robust Huber mean (per pair) with DoF correction (np.nan if use_huber=False)
    """
    assert P_hat.shape == Q.shape and P_hat.shape[1] == 3
    N = Q.shape[0]
    if N == 0:
        return -1, -1, -1, -1

    # Transform P into Q's frame
    R = Q - P_hat                              # residuals (N,3)

    S_t = 0.0
    sum_rho = 0.0
    N_eff = 0

    for i in range(N):
        # Extract 3x3 covariance block and regularize (avoid skipping!)
        Sigma = Sigma_e[3*i:3*(i+1), 3*i:3*(i+1)]
        # scale-aware jitter
        lam = jitter_eps * (np.trace(Sigma) / 3.0 + 1.0)
        Sigma_reg = Sigma + lam * np.eye(3)

        try:
            # Cholesky: Sigma_reg = L L^T
            L = np.linalg.cholesky(Sigma_reg)
        except np.linalg.LinAlgError:
            # Fallback: small extra jitter
            L = np.linalg.cholesky(Sigma_reg + 10.0 * lam * np.eye(3))

        # Whiten: y = L^{-1} r (=> ||y||^2 = r^T Sigma^{-1} r)
        y = np.linalg.solve(L, R[i])       # solve L y = r
        m2 = float(y @ y)                  # squared whitened norm
        S_t += m2
        N_eff += 1

        if use_huber:
            m = float(np.sqrt(m2))
            sum_rho += huber_rho_norm(m, huber_delta)

    if N_eff <= d:
        return -1, -1, -1, -1

    L_t = S_t / N_eff
    L_t_adj = S_t / (N_eff - d)            # DoF-corrected mean
    L_t_rob = (sum_rho / (N_eff - d)) if use_huber else -1

    return S_t, L_t, L_t_adj, L_t_rob

def build_measurement_covariance(list_of_clusters: list[np.ndarray]) -> np.ndarray:
    """
    Build block-diagonal Σ_e from all clusters (N clusters ⇒ 3N×3N matrix).
    """
    if not list_of_clusters:
        return np.zeros((0, 0))
    blocks = [centroid_covariance(C) for C in list_of_clusters]
    return block_diag(*blocks)

def parameter_covariance(J: np.ndarray, Sigma_e: np.ndarray) -> np.ndarray:
    """
    Return Σ_(φ,t) = (Jᵀ Σ_e^{-1} J)^{-1}, computed stably without forming Σ_e^{-1}.
    """
    try:
        c, lower = cho_factor(Sigma_e, overwrite_a=False, check_finite=True)
        # Compute X = Σ_e^{-1} J by solving Σ_e X = J
        X = cho_solve((c, lower), J, overwrite_b=False)
        JT_Sinv_J = J.T @ X
        return np.linalg.pinv(JT_Sinv_J)
    except np.linalg.LinAlgError:
        # Fall back to pseudo-inverse if Sigma_e is not SPD
        try:
            X = np.linalg.pinv(Sigma_e) @ J
            JT_Sinv_J = J.T @ X
            return np.linalg.pinv(JT_Sinv_J)
        except Exception:
            return np.eye(6) * 1e6 

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

def renyi_quadratic_entropy(points: np.ndarray, sigma: float, log_base: str = "e") -> float:
    """
    Renyi quadratic entropy H2 for a point cloud under a Gaussian Parzen window of bandwidth sigma.

    Parzen density: p(x) = (1/N) * sum_i N(x; x_i, σ^2 I)
    Information potential:
        V2 = ∫ p(x)^2 dx
           = (1/N^2) * sum_{i,j} N(x_i; x_j, 2σ^2 I)
           = (1/N^2) * (4πσ^2)^(-d/2) * sum_{i,j} exp(-||x_i - x_j||^2 / (4σ^2))
    H2 = -log(V2)  (natural log by default; set log_base='2' for log2)

    Returns H2 as a finite float (avoids NaN/Inf for logging).
    """
    # Basic validation
    if points is None or sigma is None or sigma <= 0:
        return 0.0
    X = np.asarray(points, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return 0.0

    N, d = X.shape
    # For N=1, closed-form V2 = (4πσ^2)^(-d/2); return corresponding H2
    if N == 1:
        logV2 = -0.5 * d * np.log(4.0 * np.pi * (sigma**2))
        H2 = -logV2 / np.log(2.0) if log_base == "2" else -logV2
        return float(H2)

    # Pairwise squared distances (includes diagonal zeros)
    from scipy.spatial.distance import cdist
    from scipy.special import logsumexp

    D2 = cdist(X, X, metric="sqeuclidean")

    # log V2 = - (d/2) log(4πσ^2) - 2 log N + log( sum_{i,j} exp(-D2 / (4σ^2)) )
    logC = -0.5 * d * np.log(4.0 * np.pi * (sigma**2))
    log_sum = logsumexp(-D2 / (4.0 * (sigma**2)))
    logV2 = logC - 2.0 * np.log(N) + log_sum

    H2 = -logV2 / np.log(2.0) if log_base == "2" else -logV2
    # Ensure finite value for logging (clip extremely large)
    if not np.isfinite(H2):
        return 0.0
    return float(H2)


def renyi_quadratic_entropy_u_statistic(
    points: np.ndarray,
    sigma: float,
    log_base: str = "e",
    cap_sigma_mult: float | None = None,
    return_count: bool = False,
) -> float | tuple[float, int]:
    """
    U-statistic variant of H2 that removes diagonal self-pairs and normalizes by N(N-1),
    then adds the fixed Gaussian scale term so values are comparable across N and σ.

    	ilde H2 = -log( (1/(N(N-1))) * sum_{i!=j} exp(-||xi-xj||^2 / (4σ^2)) ) + (d/2) log(4πσ^2)

        Optional radius cap:
            - If cap_sigma_mult is provided (>0), include only pairs with ||xi-xj|| <= (cap_sigma_mult * sigma).
                Normalization remains N(N-1) (i!=j), so excluded pairs effectively contribute 0.

        Returns a finite float; 0.0 for invalid inputs or degenerate cases.
        If return_count=True, returns a tuple (H2_value, K_included) where K_included is the number of included pairs.
    """
    if points is None or sigma is None or sigma <= 0:
        return 0.0
    X = np.asarray(points, dtype=float)
    if X.ndim != 2 or X.shape[0] == 0:
        return 0.0

    N, d = X.shape
    if N < 2:
        # With <2 points, cross-term is undefined; return baseline scale term (no structure)
        scale_term = 0.5 * d * np.log(4.0 * np.pi * (sigma**2))
        return float(scale_term / np.log(2.0)) if log_base == "2" else float(scale_term)

    from scipy.spatial.distance import cdist
    from scipy.special import logsumexp

    D2 = cdist(X, X, metric="sqeuclidean")
    # Exclude diagonal pairs
    np.fill_diagonal(D2, np.inf)

    # Build inclusion mask (exclude diagonal), optionally cap by radius
    if cap_sigma_mult is not None and cap_sigma_mult > 0:
        r2 = (cap_sigma_mult * sigma) ** 2
        mask = (D2 <= r2)
        np.fill_diagonal(mask, False)
    else:
        mask = np.ones_like(D2, dtype=bool)
        np.fill_diagonal(mask, False)

    K_included = int(np.count_nonzero(mask))
    D2_eff = np.where(mask, D2, np.inf)

    # Compute log of the mean kernel value over included pairs with denominator K_included
    log_sum = logsumexp(-D2_eff / (4.0 * (sigma**2)))
    log_mean = log_sum - (np.log(K_included) if K_included > 0 else 0.0)

    # Add the scale correction term +(d/2) log(4πσ^2)
    scale_term = 0.5 * d * np.log(4.0 * np.pi * (sigma**2))
    H2_tilde = -(log_mean) + scale_term

    if not np.isfinite(H2_tilde):
        H2_value = 0.0
    else:
        H2_value = float(H2_tilde / np.log(2.0)) if log_base == "2" else float(H2_tilde)

    if return_count:
        return H2_value, K_included
    return H2_value


def renyi_quadratic_entropy_cross(
    points_A: np.ndarray,
    points_B: np.ndarray,
    sigma: float,
    log_base: str = "e",
    cap_sigma_mult: float | None = None,
    return_count: bool = False,
    mnn: bool = False,
    mnn_k: int = 1,
) -> float | tuple[float, int]:
    """
    Cross-only variant: measure only alignment between two sets A and B.
        Uses normalization by N_A * N_B and adds the fixed Gaussian scale term.
        Optional radius cap:
            - If cap_sigma_mult is provided (>0), include only cross pairs with ||xi-xj|| <= (cap_sigma_mult * sigma).
                Normalization remains N_A*N_B so excluded pairs effectively contribute 0.

    	ilde H2_cross = -log( (1/(N_A N_B)) * sum_{i in A, j in B} exp(-||xi-xj||^2 / (4σ^2)) ) + (d/2) log(4πσ^2)
    """
    if points_A is None or points_B is None or sigma is None or sigma <= 0:
        return 0.0
    A = np.asarray(points_A, dtype=float)
    B = np.asarray(points_B, dtype=float)
    if A.ndim != 2 or B.ndim != 2 or A.shape[0] == 0 or B.shape[0] == 0 or A.shape[1] != B.shape[1]:
        return 0.0

    NA, d = A.shape
    NB = B.shape[0]

    from scipy.spatial.distance import cdist
    from scipy.special import logsumexp

    if mnn:
        # Mutual nearest-neighbor pairs only
        from scipy.spatial import cKDTree
        k = max(1, int(mnn_k))
        treeB = cKDTree(B)
        d_AB, nnAB = treeB.query(A, k=k)
        treeA = cKDTree(A)
        d_BA, nnBA = treeA.query(B, k=k)

        # Ensure shapes are 2D when k=1
        if k == 1:
            d_AB = d_AB.reshape(-1, 1)
            nnAB = nnAB.reshape(-1, 1)
            d_BA = d_BA.reshape(-1, 1)
            nnBA = nnBA.reshape(-1, 1)

        # Build mutual pairs: j in nnAB[i,:] and i in nnBA[j,:]
        pairs_i_list = []
        pairs_j_list = []
        d2_list = []
        NA = A.shape[0]
        for i in range(NA):
            js = nnAB[i]
            ds = d_AB[i]
            for t, j in enumerate(js):
                # Reciprocal check: is i among j's k-NNs?
                if i in nnBA[j]:
                    pairs_i_list.append(i)
                    pairs_j_list.append(j)
                    d2_list.append(ds[t] ** 2)

        if not d2_list:
            if return_count:
                return 0.0, 0
            return 0.0

        d2 = np.asarray(d2_list, dtype=float)
        # Optional radius-cap
        if cap_sigma_mult is not None and cap_sigma_mult > 0:
            r2 = (cap_sigma_mult * sigma) ** 2
            cap_mask = (d2 <= r2)
            d2 = d2[cap_mask]
        K_included = int(d2.size)
        if K_included == 0:
            if return_count:
                return 0.0, 0
            return 0.0
        from scipy.special import logsumexp
        log_sum = logsumexp(-d2 / (4.0 * (sigma**2)))
        log_mean = log_sum - np.log(K_included)
    else:
        # Full cross-pair kernel with optional radius-cap and normalization by included pairs
        D2 = cdist(A, B, metric="sqeuclidean")
        if cap_sigma_mult is not None and cap_sigma_mult > 0:
            r2 = (cap_sigma_mult * sigma) ** 2
            mask = D2 <= r2
        else:
            mask = np.ones_like(D2, dtype=bool)

        K_included = int(np.count_nonzero(mask))
        D2_eff = np.where(mask, D2, np.inf)

        from scipy.special import logsumexp
        log_sum = logsumexp(-D2_eff / (4.0 * (sigma**2)))
        log_mean = log_sum - (np.log(K_included) if K_included > 0 else 0.0)

    scale_term = 0.5 * d * np.log(4.0 * np.pi * (sigma**2))
    H2_tilde = -(log_mean) + scale_term

    if not np.isfinite(H2_tilde):
        H2_value = 0.0
    else:
        H2_value = float(H2_tilde / np.log(2.0)) if log_base == "2" else float(H2_tilde)

    if return_count:
        return H2_value, K_included
    return H2_value

def propagate_spherical_to_cartesian_covariance(point: np.ndarray) -> np.ndarray:
    """
    Propagates sensor uncertainty from spherical coordinates (range, azimuth, elevation)
    to Cartesian coordinates (x, y, z) for a given point in cartesian coordinates.
    """
    sigma_range = float(parameters.get_param("sensor_sigma_range_m"))
    sigma_azimuth_deg = float(parameters.get_param("sensor_sigma_azimuth_deg"))
    sigma_elevation_deg = float(parameters.get_param("sensor_sigma_elevation_deg"))

    sigma_azimuth_rad = np.deg2rad(sigma_azimuth_deg)
    sigma_elevation_rad = np.deg2rad(sigma_elevation_deg)

    x, y, z = point
    
    # Prevent division by zero for points at the origin
    if np.allclose(point, 0):
        return np.diag([sigma_range**2, sigma_range**2, sigma_range**2])

    # Convert Cartesian to Spherical coordinates
    r_xy = np.sqrt(x**2 + y**2)
    r = np.sqrt(x**2 + y**2 + z**2)
    
    # Handle case where r_xy is zero to avoid division by zero in elevation calculation
    if r_xy == 0:
        # Point is on the Z-axis, azimuth is undefined but can be set to 0.
        # Elevation is +/- 90 degrees.
        az = 0.0
        el = np.pi / 2 if z > 0 else -np.pi / 2
    else:
        az = np.arctan2(y, x)
        el = np.arctan2(z, r_xy)

    # Jacobian of the transformation from spherical (r, az, el) to Cartesian (x, y, z)
    # J = [[∂x/∂r, ∂x/∂az, ∂x/∂el],
    #      [∂y/∂r, ∂y/∂az, ∂y/∂el],
    #      [∂z/∂r, ∂z/∂az, ∂z/∂el]]
    # calculates how a small change in spherical coords leads to change in cartesian coords
    J = np.array([
        [np.cos(el) * np.cos(az), -r * np.cos(el) * np.sin(az), -r * np.sin(el) * np.cos(az)],
        [np.cos(el) * np.sin(az),  r * np.cos(el) * np.cos(az), -r * np.sin(el) * np.sin(az)],
        [np.sin(el),               0,                           r * np.cos(el)]
    ])

    # Covariance matrix in spherical coordinates (assuming independent errors)
    # Diagonal: [var_range, var_azimuth, var_elevation]
    Sigma_spherical = np.diag([sigma_range**2, sigma_azimuth_rad**2, sigma_elevation_rad**2])

    # Error propagation --> propagate covariance: Σ_cartesian = J * Σ_spherical * J^T
    Sigma_cartesian = J @ Sigma_spherical @ J.T

    return Sigma_cartesian
