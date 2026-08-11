'''
Python implementation of SUGAR Geometry-Based Data Generation.

Adapted from https://github.com/KrishnaswamyLab/SUGAR

'''
from __future__ import annotations

from typing import List, Union, Callable, Optional, Tuple, TYPE_CHECKING

import numpy as np
import scipy
from scipy.spatial.distance import squareform
from sklearn.neighbors import NearestNeighbors
import matplotlib.pyplot as plt
from sklearn.datasets import make_swiss_roll

if TYPE_CHECKING:
    import torch


def alpha_kernel(D: np.ndarray, 
    sigma: Union[float, str, Callable], k: int = 5, a: float = 2.0, fac: float = 1.0, sparsify: float = 1e-3):
    """
    Alpha-kernel (Gaussian when a=2).
    
    Args:
        D: [N,N] pairwise distances
        sigma: bandwidth mode ('std'|'median'|'minmax'|'knn'|scalar|callable)
        k: Number of nearest neighbors for bandwidth mode 'knn'.
        a: exponent (default=2 -> Gaussian)
        fac: scale factor
        sparsify: threshold below which entries are zeroed
    
    Returns:
        K: [N,N] kernel matrix
        sigma_used: scalar or [N] per-row bandwidth
    """
    sigma_used = None
    # Compute bandwidth
    if isinstance(sigma, (float, int)):
        sigma_used = float(sigma)
    elif callable(sigma):
        sigma_used = sigma(D)
    elif sigma == 'std':
        sigma_used = float(np.std(D))
    elif sigma == 'median':
        sigma_used = float(np.median(D))
    elif sigma == 'minmax':
        sigma_used = float((D.max() - D.min()) / 2.0)
    elif sigma == 'knn':
        D_sorted = np.sort(D, axis=1)
        kk = min(max(int(k), 1), D.shape[1]-1)
        sigma_used = D_sorted[:, kk]                  # per-row bandwidth
    else:
        raise ValueError(f"Unknown sigma mode: {sigma}")
    
    sigma_used = np.asarray(sigma_used) * fac

    if np.isscalar(sigma_used):
        K = np.exp(- (D / (sigma_used + 1e-12))**a)
    else:
        # row-wise division: D_ij / sigma_i
        K = np.exp(- (D / (sigma_used[:, None] + 1e-12))**a)

    K[np.isnan(K)] = 0.0
    K[K < sparsify] = 0.0

    # Symmetrize if square
    # import pdb; pdb.set_trace()
    if K.shape[0] == K.shape[1]:
        K = 0.5 * (K + K.T)

    return K, sigma_used    

def degree(X, sigma: Union[str, float, Callable] = 'std', k: int = 5, a: float = 2.0, fac: float = 1.0):
    '''
    Compute node degrees using either fixed or adaptive bandwidth.

    Parameters:
        X: Input data tensor or numpy array of shape (N, D).
        sigma: bandwidth mode ('std'|'median'|'minmax'|'knn'|scalar|callable)
        k: Number of nearest neighbors for bandwidth mode 'knn'.
        a: Exponent for alpha-decay kernel. 2.0 is Gaussian.
        fac: multiplicative factor on sigma.

    Returns:
        d_hat: [N] degree for each point
        s_hat: [N] sparsity = 1 / d_hat
        sigma_used: scalar or [N] per-row bandwidth
    '''

    D = squareform(scipy.spatial.distance.pdist(X, metric='euclidean')) # pairwise distances (N, N)
    assert D.shape[0] == D.shape[1], "Distance matrix must be square"

    # import pdb; pdb.set_trace()
    # Compute alpha-decay kernel
    K, sigma_used = alpha_kernel(D, sigma, k, a, fac)

    N = X.shape[0]
    p = K.sum(axis=1)                         # [N]
    d_hat = p * (N / (p.sum() + 1e-12))       # normalized degree
    s_hat = 1.0 / (d_hat + 1e-12)

    return d_hat, s_hat, sigma_used


def local_covariance(data: np.ndarray, k: int = 5, ridge: float = 0.0) -> List[np.ndarray]:
    """
    Compute k-NN local covariance around each point (optionally ridge-regularized).

    Args:
        data: (N, D) array
        k:    neighborhood size (includes the point itself, like MATLAB knnsearch)
        ridge: add ridge*I to each covariance (e.g., 1e-3) to stabilize in high-D

    Returns:
        covs: list of N covariance matrices, each (D, D)
    """
    data = np.asarray(data, dtype=float)
    N, D = data.shape
    if k < 2:
        raise ValueError("k must be >= 2 for a valid covariance.")

    nbrs = NearestNeighbors(n_neighbors=k, algorithm="auto").fit(data)
    # indices of k nearest neighbors for each point (includes self)
    idx = nbrs.kneighbors(return_distance=False)  # shape (N, k)

    covs: List[np.ndarray] = []
    for i in range(N):
        neigh = data[idx[i]]           # (k, D)
        mu = neigh.mean(axis=0, keepdims=True) # (1, D)
        Xc = neigh - mu # (k, D)
        Si = (Xc.T @ Xc) / (k - 1) # (D, D)
        if ridge > 0.0:
            Si = Si + ridge * np.eye(D)
        covs.append(Si)
    return covs

def numpts_localcov(
    degree: np.ndarray,
    noise_cov: List[np.ndarray],                # list of per-point (D x D) covariances
    kernel_sigma: Union[float, np.ndarray] = 1.0,  # scalar or [N] per-point bandwidth
    M: int = 0,                             # total new points; 0 => no scaling
    equalize: bool = False,                 # density equalization flag
    suppress: bool = True                   # if False, raise; if True, warn/print
) -> np.ndarray:
    """
    Compute number of new points per anchor for the case 
    where noise_cov is a list of local covariance matrices (one per data point).

    Args:
      degree: [N] degree estimates
      noise_cov: list of N cov mats (each D x D, SPD-ish)
      kernel_sigma: scalar or [N] per-point bandwidth(s)
      M: desired total # of generated points (0 => leave unnormalized)
      equalize: if True, allocate more to sparse regions using local covariances
      suppress: if False, raise errors; True => print warnings

    Returns:
      npts: [N] integer counts per point
    """

    N = degree.shape[0]
    assert len(noise_cov) == N, "noise_cov length must equal N"

    # handle kernel_sigma (scalar or per-point)
    if isinstance(kernel_sigma, (float, int)):
        sig_vec = np.full(N, float(kernel_sigma), dtype=float) # [N]
    else:
        sig_vec = np.asarray(kernel_sigma, dtype=float).reshape(-1)
        assert sig_vec.shape[0] == N, "kernel_sigma length must equal N"

    Const = float(np.max(degree))
    NumberEstimate = np.zeros(N, dtype=float) 

    if equalize:
        # Equalization with local covariances:
        # n_i ∝ (Const - d_i) * det(I + Σ_i / (2 σ_i^2))^(1/2)
        for i in range(N):
            Sigma_i = np.asarray(noise_cov[i], dtype=float)
            sig2 = sig_vec[i]**2
            # A = I + Σ / (2σ^2)
            D = Sigma_i.shape[0]
            A = np.eye(D, dtype=float) + Sigma_i / (2.0 * sig2 + 1e-12)
            sign, logdet = np.linalg.slogdet(A)
            if sign <= 0:
                print(f"slogdet non-positive at i={i}; adding a tiny ridge to Σ.")
                A = A + 1e-8 * np.eye(D)
                sign, logdet = np.linalg.slogdet(A)
            NumberEstimate[i] = (Const - degree[i]) * np.exp(0.5 * logdet)

        if M:
            total = NumberEstimate.sum()
            if total <= 0:
                print("Equalized sum <= 0; falling back to ones.")
                return np.ones(N, dtype=int)
            if (M / total) < 1e-1:
                print(f"M is {100.0 * M / total:.2f}% of equalized total; consider increasing M.")
            npts = np.floor(NumberEstimate * (M / (total + 1e-17))).astype(int)
        else:
            npts = np.floor(NumberEstimate).astype(int)

    else:
        # No density equalization: proportional to (Const - degree)
        if not M:
            print("Generating without density equalization; no M supplied, using M = N.")
            M = N
        base = (Const - degree)
        total = base.sum()
        if total <= 0:
            print("Base sum <= 0; falling back to ones.")
            return np.ones(N, dtype=int)
        npts = np.floor(base * (M / (total + 1e-17))).astype(int)

    # sanity checks
    S = int(npts.sum())
    if S == 0:
        print("Estimated total points is 0; increasing all to 1.")
        npts = np.ones(N, dtype=int)
    elif S > 10**4:
        print("Estimated total points > 1e4; consider smaller M or larger noise.")

    return npts


def sample_points(
    data: np.ndarray,                 # shape: [N, D]
    npts: np.ndarray,                 # shape: [N], integers (will be floored)
    noise_cov: Union[float, np.ndarray, List[np.ndarray]],  # scalar var or list of [D,D] covs
    labels: Optional[np.ndarray] = None               # shape: [N] or None
) -> Tuple[np.ndarray, Optional[np.ndarray]]:
    """
    Generate sum(npts) Gaussian samples around each data[i] using either a single
    scalar variance (isotropic) or per-point covariance matrices.

    Returns:
        random_points: [M, D], M=sum(npts)
        labels_out:   [M] or None (labels replicated alongside samples)
    """
    npts = np.asarray(npts, dtype=int).reshape(-1)
    N, D = data.shape
    assert npts.shape[0] == N, "npts must have length N"
    M = int(npts.sum())

    random_points = np.zeros((M, D), dtype=float)
    labels_out = None if labels is None else np.zeros((M,), dtype=labels.dtype)

    # scalar variance (isotropic)
    if isinstance(noise_cov, (float, int)) or (np.isscalar(noise_cov)):
        var = float(noise_cov)
        cov = np.eye(D) * var

        # Build all centers in one go
        reps = np.repeat(np.arange(N), np.maximum(npts, 0))
        centers = data[reps]  # [M, D]

        # Draw once for all with same cov
        random_points[:] = np.random.multivariate_normal(mean=np.zeros(D), cov=cov, size=M) + centers

        if labels is not None:
            labels = np.asarray(labels)
            labels_out[:] = labels[reps]

        return random_points, labels_out

    # per-point local covariance matrices
    assert isinstance(noise_cov, list) and len(noise_cov) == N, \
        "For local-cov mode, noise_cov must be a list of length N with (D,D) arrays."

    cur_idx= 0
    for i in range(N):
        m = int(npts[i])
        if m <= 0:
            continue

        mu = data[i] # (D,)
        Sigma = np.asarray(noise_cov[i], dtype=float) # (D, D)

        # Draw m samples around data[i] with covariance Sigma
        samples = np.random.multivariate_normal(mean=mu, cov=Sigma, size=m)  # (m, D)
        random_points[cur_idx:cur_idx + m] = samples

        if labels is not None:
            labels_out[cur_idx:cur_idx + m] = labels[i]

        cur_idx += m

    return random_points, labels_out

def magic_diffuse(Y: np.ndarray, kernel: np.ndarray, t: int, rescale: bool):
    '''
    Perform MAGIC diffusion using kernel affinity matrix.
    
    Args:
        Y:   [N, D] data matrix (rows = points, cols = features)
        kernel: [N, N] affinity/kernel matrix, i.e., K_mgc
        t:      integer # of diffusion steps (>=1)
        rescale: if True, match per-feature 95th percentile to original

    Returns:
        data_imputed:        [N, D] imputed data after diffusion
        diffusion_operator:  [N, N] row-stochastic Markov matrix
    '''
    if t == 0:
        return Y, np.eye(Y.shape[0])
    
    # Normalize kernel
    row_sums = kernel.sum(axis=1, keepdims=True) # [N, 1]
    row_sums[row_sums == 0] = 1.0
    diffusion_operator = kernel / row_sums # [N, N]
    
    data_imputed = Y.copy()
    # Precompute original 95th percentiles for optional rescale
    if rescale:
        p95_orig = np.percentile(Y, 95, axis=0)
        # Avoid zero to prevent divide-by-zero later
        p95_orig = np.where(p95_orig == 0, 1e-12, p95_orig)

    # Diffuse t steps
    for _ in range(int(max(t, 1))):
        data_imputed = diffusion_operator @ data_imputed
        if rescale:
            p95_imp = np.percentile(data_imputed, 95, axis=0)
            p95_imp = np.where(p95_imp == 0, 1e-12, p95_imp)
            scale = p95_orig / p95_imp
            data_imputed = data_imputed * scale  # broadcast per feature

    return data_imputed, diffusion_operator


def mgc_magic(X: np.ndarray, Y: np.ndarray, s_hat: np.ndarray, 
sigma: Optional[Union[float, str, Callable]] = 'std', a: float = 2.0, k: int = 5, fac: float = 1.0, t: int = 1, magic_rescale: bool = True):
    '''
    Compute Measure Gaussian Correction kernel (MGC kernel) and diffusion operator.
    new_Y = MAGIC( K_mgc, Y ), with K_mgc = (diag(s_hat) * K_{Y->X}) @ K_{X->Y}

    Inputs:
      X: (N,D) anchor data
      Y: (M,D) points to be aligned
      sigma: MGC kernel bandwidth
      s_hat: (M,) sparsity/measure weights for rows of Y

    Returns:
      new_Y: (M,D) diffused/aligned Y
      K_mgc: (M,M) MGC kernel on Y
      P:     (M,M) row-stochastic diffusion operator used in MAGIC
    '''    
    # Compute distances between X and Y
    new_to_old_D = scipy.spatial.distance.cdist(Y, X, metric='euclidean')
    old_to_new_D = scipy.spatial.distance.cdist(X, Y, metric='euclidean')

    assert len(new_to_old_D.shape) == 2
    assert len(old_to_new_D.shape) == 2
    
    # Cross kernels.
    new_to_old_K, sigma_used_new_to_old = alpha_kernel(new_to_old_D, sigma, k, a, fac) #[M, N]
    old_to_new_K, sigma_used_old_to_new = alpha_kernel(old_to_new_D, sigma, k, a, fac) #[N, M]

    # Sparsity weighting on rows of K_{Y->X}
    new_to_old_K = new_to_old_K * s_hat # [M, N] x [N, 1] = [M, N]

    # MGC kernel over new points.
    K_mgc = new_to_old_K @ old_to_new_K  # [M,M]
    K_mgc = 0.5 * (K_mgc + K_mgc.T)     # symmetrize

    # MAGIC diffusion
    if t == 0:
        print(f'[SUGAR] t is 0, returning Y ...')
        return Y, K_mgc, K_mgc
    else:
        print(f'[SUGAR] Diffusing {t} steps ...')
        new_Y, P = magic_diffuse(Y, K_mgc, t=int(t), rescale=float(magic_rescale)) # [M, D], [M, M]
        return new_Y, K_mgc, P
    

class SUGAR:
    def __init__(self, 
                degree_sigma='std',
                degree_k=5,
                degree_a=2.0,
                degree_fac=1.0,
                M=0, # number of points to generate, 0 => leave unnormalized
                equalize=False, #density equalization flag
                noise_cov='knn', # sigma mode for the noise covariance
                noise_k=5,  # number of nearest neighbors for the noise covariance
                mgc_sigma='knn', # sigma mode for the MGC kernel
                mgc_k=5,
                mgc_a=2.0,
                mgc_fac=1.0,
                mgc_t=1,
                magic_t=1,
                magic_rescale=True # whether rescale corrected points to match the 95th percentile of the original points
                ):

        self.degree_sigma = degree_sigma
        self.degree_k = degree_k
        self.degree_a = degree_a
        self.degree_fac = degree_fac

        self.M = M
        self.equalize = equalize

        self.noise_cov = noise_cov
        self.noise_k = noise_k

        self.mgc_sigma = mgc_sigma
        self.mgc_k = mgc_k
        self.mgc_a = mgc_a
        self.mgc_fac = mgc_fac
        self.mgc_t = mgc_t

        self.magic_t = magic_t
        self.magic_rescale = magic_rescale

    def generate(self, X: np.ndarray, labels: Optional[np.ndarray] = None):
        # First compute degree and sparsity.
        print(f'[SUGAR] Computing degree and sparsity ...')

        d_hat, s_hat, sigma_used = degree(X, self.degree_sigma, self.degree_k, self.degree_a, self.degree_fac)
        print(f'[SUGAR] d_hat: {d_hat.shape}, s_hat: {s_hat.shape}, sigma_used: {sigma_used}')

        # Estimate local covariance
        print(f'[SUGAR] Estimating local covariance ...')
        covs = local_covariance(X, self.noise_k)

        # Estimate number of points to sample along each point.
        print(f'[SUGAR] Estimating number of points to sample along each point ...')
        npts = numpts_localcov(d_hat, covs, sigma_used, self.M, self.equalize)

        # Sample random points along each point from Gaussian distribution with covariance matrix.
        print(f'[SUGAR] Sampling random points along each point from Gaussian distribution with covariance matrix ...')
        random_points, labels_out = sample_points(X, npts, covs, labels) # (M, D), (M,)

        # Compute Measure Gaussian Correction kernel (MGC kernel), diffusion operator, and corrected points.
        print(f'[SUGAR] Computing Measure Gaussian Correction kernel (MGC kernel) and diffusion operator ...')
        new_Y, K_mgc, P = mgc_magic(X, random_points, s_hat, self.mgc_sigma, 
        self.mgc_a, self.mgc_k, self.mgc_fac, self.mgc_t, self.magic_rescale)

        # Return the corrected points.
        return new_Y, K_mgc, P, random_points


def augment_latents_with_sugar(
    latents: torch.Tensor,
    sugar_w: float,
    *,
    seed: int,
    max_fit_points: int = 10000,
) -> torch.Tensor:
    """Append SUGAR-generated latents to ``latents``.

    Fit SUGAR on up to ``max_fit_points`` train latents, then keep
    ``int(sugar_w * n_sugar)`` of the generated points (with replacement if
    ``sugar_w > 1``).
    """
    if float(sugar_w) <= 0.0:
        return latents

    import torch as _torch

    x = latents.detach().cpu()
    n = int(x.shape[0])
    rng = np.random.default_rng(int(seed))

    sugar_op = SUGAR(degree_sigma="knn", degree_k=5, mgc_t=1, magic_rescale=False)
    if n > int(max_fit_points):
        print(f"[SUGAR] n={n} > {max_fit_points}, subsampling {max_fit_points} points ...")
        idx = rng.choice(n, size=int(max_fit_points), replace=False)
        candidates = x[idx]
    else:
        print(f"[SUGAR] n={n} <= {max_fit_points}, using all points ...")
        candidates = x

    sugar_np, _, _, _random_points = sugar_op.generate(candidates.numpy())
    sugar = _torch.from_numpy(np.asarray(sugar_np, dtype=np.float32))
    print(
        f"[SUGAR] generated {tuple(sugar.shape)}; "
        f"mean={float(sugar.mean()):.4g} std={float(sugar.std()):.4g}"
    )

    num_sugar = int(float(sugar_w) * sugar.shape[0])
    if num_sugar <= 0:
        print(f"[SUGAR] sugar_w={sugar_w} -> num_sugar=0; skipping augmentation")
        return latents

    replace = num_sugar > sugar.shape[0]
    keep = rng.choice(sugar.shape[0], size=num_sugar, replace=replace)
    sugar = sugar[keep]
    print(f"[SUGAR] keeping {sugar.shape[0]} points (sugar_w={sugar_w})")

    out = _torch.cat([x.float(), sugar.float()], dim=0)
    print(f"[SUGAR] manifold size {n} -> {out.shape[0]}")
    return out


if __name__ == '__main__':
    X = np.random.randn(100, 10)

    sugar_op = SUGAR()
    new_Y, K_mgc, P = sugar_op.generate(X)
     
    print('new_Y: ', new_Y.shape) # (M, D)
    print('K_mgc: ', K_mgc.shape) # (M, M)
    print('P: ', P.shape) # (M, M)

    # Sample a 3D swiss roll
    print('Sampling a 3D swiss roll ...')
    X = make_swiss_roll(n_samples=1000, noise=0.05)[0]
    print('X: ', X.shape) # (N, D)

    # drop out random 20% of the points
    dropout = 0.5
    X = X[np.random.rand(X.shape[0]) > dropout]

    sugar_op = SUGAR(degree_sigma='std')
    new_Y, K_mgc, P, random_points = sugar_op.generate(X)
     
    print('new_Y: ', new_Y.shape) # (M, D)
    print('K_mgc: ', K_mgc.shape) # (M, M)
    print('P: ', P.shape) # (M, M)

    # Visualize the old and new points in 3D
    nrows = 3
    fig = plt.figure(figsize=(nrows*6, 8))
    ax = fig.add_subplot(1, nrows, 1, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', s=2)
    ax.set_title('Original Points')

    ax = fig.add_subplot(1, nrows, 2, projection='3d')
    ax.scatter(new_Y[:, 0], new_Y[:, 1], new_Y[:, 2], c='red', s=2)
    ax.set_title('Generated Points')

    ax = fig.add_subplot(1, nrows, 3, projection='3d')
    ax.scatter(X[:, 0], X[:, 1], X[:, 2], c='blue', s=2)
    ax.scatter(new_Y[:, 0], new_Y[:, 1], new_Y[:, 2], c='red', s=2)
    ax.scatter(random_points[:, 0], random_points[:, 1], random_points[:, 2], c='green', s=2)
    ax.set_title('Generated Points vs. Original Points vs. Random Points')
    
    plt.show()