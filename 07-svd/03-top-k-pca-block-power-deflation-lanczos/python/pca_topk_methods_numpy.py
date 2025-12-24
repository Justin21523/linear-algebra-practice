"""  # EN: Start module docstring.
Top-k PCA without full SVD (NumPy): block power (orthogonal iteration), deflation, and Lanczos (Ritz top-k).  # EN: Summarize what this script demonstrates.

We focus on the covariance eigen-problem for centered data Xc:  # EN: Explain the core PCA eigen-problem.
  C = (Xc^T Xc)/(n_samples-1), and we want the top-k eigenvectors of C.  # EN: Define covariance and the goal.

For large feature dimension, you avoid forming C explicitly.  # EN: Explain why we avoid forming covariance matrix.
Instead, you implement a matvec:  Cv = Xc^T (Xc v)/(n_samples-1).  # EN: Provide matvec identity.

Algorithms included:  # EN: List included methods.
  1) Block power / orthogonal iteration (subspace iteration)  # EN: Describe method 1.
  2) Deflation (repeat power iteration with deflated matvec)  # EN: Describe method 2.
  3) Lanczos (build Krylov basis once, extract top-k Ritz pairs)  # EN: Describe method 3.

We also compute a full SVD reference on a moderate-size dataset to verify:  # EN: Explain verification plan.
  - Subspace similarity to the true top-k principal directions  # EN: Mention subspace metric.
  - Eigen-residual norms ||C v_i - λ_i v_i||_2  # EN: Mention residual metric.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured results.
from typing import Callable  # EN: Use Callable to type matvec and other function interfaces.

import numpy as np  # EN: Import NumPy for numerical computation.


EPS = 1e-12  # EN: Small epsilon to avoid division-by-zero and numerical blow-ups.
SEED = 0  # EN: RNG seed for deterministic demos.
PRINT_PRECISION = 6  # EN: Console float precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias for a matrix-vector product function.


@dataclass(frozen=True)  # EN: Immutable record for a set of eigenpairs (top-k).
class TopKEigenpairs:  # EN: Store eigenvalues, eigenvectors, and diagnostic residuals.
    eigenvalues: np.ndarray  # EN: Array of eigenvalues (k,), sorted descending.
    eigenvectors: np.ndarray  # EN: Matrix of eigenvectors (n×k), orthonormal columns.
    residual_norms: np.ndarray  # EN: Residual norms ||A v_i - λ_i v_i|| for each eigenpair.
    n_iters: int  # EN: Iteration count (or Krylov dimension) used by the method.


def print_separator(title: str) -> None:  # EN: Print a section separator for readability.
    print()  # EN: Add a blank line before sections.
    print("=" * 78)  # EN: Print a horizontal divider.
    print(title)  # EN: Print the title.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm) of a vector.
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy's stable norm computation.


def normalize(v: np.ndarray) -> np.ndarray:  # EN: Normalize a vector to unit norm.
    n = l2_norm(v)  # EN: Compute norm.
    if n < EPS:  # EN: Guard against near-zero vectors.
        raise ValueError("Cannot normalize near-zero vector")  # EN: Fail fast for invalid input.
    return v / n  # EN: Return normalized vector.


def orthonormalize_columns(X: np.ndarray) -> np.ndarray:  # EN: Orthonormalize columns of X using QR.
    Q, _ = np.linalg.qr(X, mode="reduced")  # EN: Thin QR yields Q with orthonormal columns.
    return Q  # EN: Return Q.


def eigen_residuals(matvec: Matvec, V: np.ndarray, lam: np.ndarray) -> np.ndarray:  # EN: Compute residual norms for multiple eigenpairs.
    k = V.shape[1]  # EN: Number of eigenvectors.
    res = np.zeros((k,), dtype=float)  # EN: Allocate residual norm array.
    for i in range(k):  # EN: Loop over eigenpairs.
        r = matvec(V[:, i]) - lam[i] * V[:, i]  # EN: Residual r_i = A v_i - λ_i v_i.
        res[i] = l2_norm(r)  # EN: Store residual norm.
    return res  # EN: Return residual norms.


def rayleigh_ritz_from_subspace(matvec: Matvec, Q: np.ndarray) -> TopKEigenpairs:  # EN: Perform Rayleigh–Ritz on subspace span(Q).
    k = Q.shape[1]  # EN: Subspace dimension.
    AQ = np.column_stack([matvec(Q[:, i]) for i in range(k)])  # EN: Compute A Q without forming A.
    T = Q.T @ AQ  # EN: Build small k×k Rayleigh quotient matrix.
    evals, evecs = np.linalg.eigh(T)  # EN: Diagonalize symmetric T.
    order = np.argsort(evals)[::-1]  # EN: Sort eigenvalues descending.
    evals = evals[order]  # EN: Reorder eigenvalues.
    evecs = evecs[:, order]  # EN: Reorder eigenvectors accordingly.
    V = Q @ evecs  # EN: Lift Ritz vectors back to original space.
    V = orthonormalize_columns(V)  # EN: Re-orthonormalize to reduce numerical drift.
    res = eigen_residuals(matvec, V, evals)  # EN: Compute residuals in original space.
    return TopKEigenpairs(eigenvalues=evals, eigenvectors=V, residual_norms=res, n_iters=0)  # EN: Return result (iters not tracked here).


def block_power_iteration(  # EN: Compute top-k eigenvectors using orthogonal iteration (block power).
    matvec: Matvec,  # EN: Matrix-vector product function for symmetric matrix A.
    n: int,  # EN: Dimension of vectors (A is n×n).
    k: int,  # EN: Number of leading eigenvectors to compute.
    n_iters: int = 50,  # EN: Number of subspace iterations.
    rng: np.random.Generator | None = None,  # EN: RNG for deterministic initialization.
) -> TopKEigenpairs:  # EN: Return approximate top-k eigenpairs.
    if k <= 0 or k > n:  # EN: Validate k range.
        raise ValueError("k must be in [1, n]")  # EN: Reject invalid k.
    if rng is None:  # EN: Create a default RNG if needed.
        rng = np.random.default_rng(SEED)  # EN: Use fixed seed for reproducibility.

    Q = rng.standard_normal((n, k))  # EN: Initialize random subspace basis.
    Q = orthonormalize_columns(Q)  # EN: Orthonormalize to get a valid starting subspace.

    for _ in range(n_iters):  # EN: Perform subspace iterations.
        Z = np.column_stack([matvec(Q[:, i]) for i in range(k)])  # EN: Apply A to each basis vector.
        Q = orthonormalize_columns(Z)  # EN: Re-orthonormalize to keep basis stable.

    # EN: After subspace iteration, use Rayleigh–Ritz to get eigenvalues and improved eigenvectors.
    rr = rayleigh_ritz_from_subspace(matvec, Q)  # EN: Compute Ritz pairs on the final subspace.
    return TopKEigenpairs(  # EN: Return with iteration count filled.
        eigenvalues=rr.eigenvalues,  # EN: Store eigenvalues.
        eigenvectors=rr.eigenvectors,  # EN: Store eigenvectors.
        residual_norms=rr.residual_norms,  # EN: Store residual norms.
        n_iters=n_iters,  # EN: Store subspace iteration count.
    )  # EN: End return.


def power_iteration_single(  # EN: Compute the dominant eigenpair of symmetric A via power iteration.
    matvec: Matvec,  # EN: Matrix-vector product function.
    n: int,  # EN: Dimension.
    max_iters: int = 2000,  # EN: Iteration cap.
    tol: float = 1e-12,  # EN: Direction-change tolerance.
    rng: np.random.Generator | None = None,  # EN: RNG for initialization.
) -> tuple[float, np.ndarray, int]:  # EN: Return (lambda, v, iters).
    if rng is None:  # EN: Provide default RNG.
        rng = np.random.default_rng(SEED)  # EN: Use deterministic seed.
    v = normalize(rng.standard_normal(n))  # EN: Random initial unit vector.
    for it in range(1, max_iters + 1):  # EN: Power iteration loop.
        w = matvec(v)  # EN: Multiply by A.
        v_next = normalize(w)  # EN: Normalize.
        align = abs(float(v @ v_next))  # EN: Cosine similarity (sign-invariant).
        v = v_next  # EN: Update iterate.
        if 1.0 - align <= tol:  # EN: Check convergence by direction stability.
            break  # EN: Stop when converged.
    lam = float(v @ matvec(v))  # EN: Rayleigh quotient eigenvalue estimate.
    return lam, v, it  # EN: Return eigenpair estimate and iteration count.


def deflation_power_iteration(  # EN: Compute top-k eigenpairs by repeated power iteration with deflation.
    matvec: Matvec,  # EN: Matrix-vector product for symmetric A.
    n: int,  # EN: Dimension.
    k: int,  # EN: Number of eigenpairs to compute.
    max_iters_each: int = 2000,  # EN: Power iteration cap for each eigenpair.
    tol: float = 1e-12,  # EN: Convergence tolerance for each power iteration.
    rng: np.random.Generator | None = None,  # EN: RNG for initialization.
) -> TopKEigenpairs:  # EN: Return approximate top-k eigenpairs.
    if k <= 0 or k > n:  # EN: Validate k.
        raise ValueError("k must be in [1, n]")  # EN: Reject invalid k.
    if rng is None:  # EN: Provide RNG if missing.
        rng = np.random.default_rng(SEED)  # EN: Use deterministic seed.

    eigenvalues: list[float] = []  # EN: Store eigenvalues in order.
    eigenvectors: list[np.ndarray] = []  # EN: Store eigenvectors in order.
    total_iters = 0  # EN: Accumulate iteration count across all eigenpairs.

    def matvec_deflated(v: np.ndarray) -> np.ndarray:  # EN: Define deflated matvec A'v = Av - sum λ_i v_i(v_i^T v).
        y = matvec(v)  # EN: Start with y = A v.
        for lam_i, vi in zip(eigenvalues, eigenvectors):  # EN: Subtract contributions of already-found eigenpairs.
            y = y - lam_i * vi * float(vi @ v)  # EN: Apply rank-1 deflation term λ v v^T v.
        return y  # EN: Return deflated matvec result.

    for _ in range(k):  # EN: Compute k eigenpairs sequentially.
        lam, v, iters = power_iteration_single(  # EN: Find dominant eigenpair of the deflated operator.
            matvec=matvec_deflated,  # EN: Use deflated matvec.
            n=n,  # EN: Dimension.
            max_iters=max_iters_each,  # EN: Iteration cap.
            tol=tol,  # EN: Convergence tolerance.
            rng=rng,  # EN: RNG for initialization.
        )  # EN: End power iteration call.

        # EN: Re-orthogonalize v against previously found vectors to reduce drift (numerical safety).
        if eigenvectors:  # EN: Only orthogonalize if we have previous vectors.
            V_prev = np.column_stack(eigenvectors)  # EN: Build matrix of previous eigenvectors.
            v = v - V_prev @ (V_prev.T @ v)  # EN: Remove projection onto previous subspace.
            v = normalize(v)  # EN: Re-normalize after orthogonalization.
            lam = float(v @ matvec(v))  # EN: Recompute Rayleigh quotient with original operator.

        eigenvalues.append(lam)  # EN: Store eigenvalue.
        eigenvectors.append(v)  # EN: Store eigenvector.
        total_iters += iters  # EN: Accumulate iterations.

    lam_arr = np.array(eigenvalues, dtype=float)  # EN: Convert eigenvalues list to array.
    V = np.column_stack(eigenvectors)  # EN: Stack eigenvectors into n×k matrix.
    order = np.argsort(lam_arr)[::-1]  # EN: Sort by eigenvalue descending (deflation should already be in order, but be safe).
    lam_arr = lam_arr[order]  # EN: Reorder eigenvalues.
    V = V[:, order]  # EN: Reorder eigenvectors.
    V = orthonormalize_columns(V)  # EN: Ensure orthonormal columns.
    res = eigen_residuals(matvec, V, lam_arr)  # EN: Compute residuals for original operator.
    return TopKEigenpairs(eigenvalues=lam_arr, eigenvectors=V, residual_norms=res, n_iters=total_iters)  # EN: Return results.


def lanczos_topk_ritz(  # EN: Approximate top-k eigenpairs using a single Lanczos run and Rayleigh–Ritz on T.
    matvec: Matvec,  # EN: Matrix-vector product function for symmetric A.
    n: int,  # EN: Dimension.
    k: int,  # EN: Number of eigenpairs requested.
    n_steps: int = 40,  # EN: Lanczos steps (Krylov subspace dimension), must be >= k.
    reorthogonalize: bool = True,  # EN: Full re-orthogonalization for numerical stability (demo-friendly).
    rng: np.random.Generator | None = None,  # EN: RNG for initialization.
) -> TopKEigenpairs:  # EN: Return approximate top-k eigenpairs.
    if k <= 0 or k > n:  # EN: Validate k.
        raise ValueError("k must be in [1, n]")  # EN: Reject invalid k.
    if n_steps < k:  # EN: Ensure Krylov subspace is large enough to extract k pairs.
        raise ValueError("n_steps must be >= k")  # EN: Reject invalid n_steps.
    if rng is None:  # EN: Create default RNG.
        rng = np.random.default_rng(SEED)  # EN: Use deterministic seed.

    q_prev = np.zeros((n,), dtype=float)  # EN: q_0 = 0.
    q = normalize(rng.standard_normal(n))  # EN: q_1 random unit vector.

    Q: list[np.ndarray] = []  # EN: Store Lanczos basis vectors.
    alphas: list[float] = []  # EN: Diagonal elements of T.
    betas: list[float] = []  # EN: Off-diagonal elements of T.

    beta_prev = 0.0  # EN: β_0 = 0.

    for j in range(n_steps):  # EN: Lanczos recurrence loop.
        Q.append(q)  # EN: Store current q_j.
        w = matvec(q) - beta_prev * q_prev  # EN: w = A q_j - β_{j-1} q_{j-1}.
        alpha = float(q @ w)  # EN: α_j = q_j^T w.
        w = w - alpha * q  # EN: Remove component along q_j.

        if reorthogonalize:  # EN: Full re-orthogonalization against previous basis vectors.
            for qi in Q[:-1]:  # EN: Loop over previous q's.
                w = w - float(qi @ w) * qi  # EN: Remove component along qi.

        beta = l2_norm(w)  # EN: β_j = ||w||.
        alphas.append(alpha)  # EN: Store α_j.
        if j < n_steps - 1:  # EN: Store β_j if there will be a next step.
            betas.append(beta)  # EN: Store β_j.

        if beta < EPS:  # EN: Breakdown indicates we reached an invariant subspace.
            break  # EN: Stop early.

        q_prev = q  # EN: Shift q_{j-1}.
        q = w / beta  # EN: Normalize to get q_{j+1}.
        beta_prev = beta  # EN: Update β_{j} for next iteration.

    m = len(alphas)  # EN: Effective Krylov dimension.
    T = np.zeros((m, m), dtype=float)  # EN: Build tridiagonal matrix T.
    for i in range(m):  # EN: Fill diagonal.
        T[i, i] = alphas[i]  # EN: Set α_i.
    for i in range(m - 1):  # EN: Fill off-diagonal.
        T[i, i + 1] = betas[i]  # EN: Set β_i.
        T[i + 1, i] = betas[i]  # EN: Mirror β_i.

    evals, evecs = np.linalg.eigh(T)  # EN: Diagonalize T to get Ritz pairs.
    order = np.argsort(evals)[::-1]  # EN: Sort eigenvalues descending.
    evals = evals[order]  # EN: Reorder.
    evecs = evecs[:, order]  # EN: Reorder Ritz vectors.

    k_eff = min(k, m)  # EN: If breakdown reduced m, clamp k.
    evals_k = evals[:k_eff]  # EN: Take top-k Ritz eigenvalues.
    Y_k = evecs[:, :k_eff]  # EN: Take corresponding Ritz vectors in Krylov coordinates.

    Q_mat = np.column_stack(Q[:m])  # EN: Form Q matrix of size n×m.
    V = Q_mat @ Y_k  # EN: Form Ritz vectors in original space.
    V = orthonormalize_columns(V)  # EN: Orthonormalize to reduce numerical drift.
    res = eigen_residuals(matvec, V, evals_k)  # EN: Compute residual norms.

    return TopKEigenpairs(eigenvalues=evals_k, eigenvectors=V, residual_norms=res, n_iters=m)  # EN: Return top-k result.


def subspace_overlap(V_ref: np.ndarray, V_est: np.ndarray) -> np.ndarray:  # EN: Compute singular values of V_ref^T V_est as subspace similarity.
    M = V_ref.T @ V_est  # EN: Compute k×k overlap matrix.
    s = np.linalg.svd(M, compute_uv=False)  # EN: Singular values are cosines of principal angles.
    return s  # EN: Return overlap singular values (close to 1 is good).


def build_synthetic_data(  # EN: Build a synthetic dataset with known low-rank structure + noise.
    rng: np.random.Generator,  # EN: RNG for reproducibility.
    n_samples: int,  # EN: Number of samples.
    n_features: int,  # EN: Number of features.
    latent_rank: int,  # EN: Intrinsic rank.
    noise_std: float,  # EN: Noise level.
) -> np.ndarray:  # EN: Return data matrix X.
    Z = rng.standard_normal((n_samples, latent_rank))  # EN: Sample latent factors.
    W = rng.standard_normal((n_features, latent_rank))  # EN: Feature loadings.
    X_low = Z @ W.T  # EN: Low-rank signal.
    X = X_low + noise_std * rng.standard_normal((n_samples, n_features))  # EN: Add Gaussian noise.
    return X.astype(float)  # EN: Return float matrix.


def main() -> None:  # EN: Run a demo comparing top-k PCA methods.
    rng = np.random.default_rng(SEED)  # EN: Deterministic RNG.

    n_samples = 400  # EN: Sample count.
    n_features = 120  # EN: Feature dimension.
    latent_rank = 5  # EN: True latent rank (dominant directions).
    noise_std = 0.5  # EN: Noise strength.
    k = 5  # EN: Number of principal components to recover.

    X = build_synthetic_data(rng=rng, n_samples=n_samples, n_features=n_features, latent_rank=latent_rank, noise_std=noise_std)  # EN: Create dataset.
    mean = X.mean(axis=0)  # EN: Compute mean.
    Xc = X - mean  # EN: Center data.
    scale = 1.0 / max(n_samples - 1, 1)  # EN: Covariance scaling.

    def cov_matvec(v: np.ndarray) -> np.ndarray:  # EN: Covariance matvec Cv without forming C.
        return scale * (Xc.T @ (Xc @ v))  # EN: Apply C using two multiplications.

    print_separator("Dataset summary")  # EN: Print summary.
    print(f"X shape: {n_samples}×{n_features}, latent_rank={latent_rank}, noise_std={noise_std}, k={k}")  # EN: Print parameters.

    # EN: Reference top-k via full SVD (feasible at this moderate size).
    _, s_ref, Vt_ref = np.linalg.svd(Xc, full_matrices=False)  # EN: Compute economy SVD.
    V_ref = Vt_ref[:k, :].T  # EN: Take first k right singular vectors as principal directions.
    V_ref = orthonormalize_columns(V_ref)  # EN: Ensure orthonormal (numerical safety).
    evals_ref = (s_ref[:k] ** 2) * scale  # EN: Convert to covariance eigenvalues.
    res_ref = eigen_residuals(cov_matvec, V_ref, evals_ref)  # EN: Compute reference residuals.

    print_separator("Reference (full SVD) top-k")  # EN: Print reference stats.
    print("eigenvalues_ref =", evals_ref)  # EN: Print reference eigenvalues.
    print("residual norms ref =", res_ref)  # EN: Print reference residual norms.

    print_separator("Block Power / Orthogonal Iteration")  # EN: Run block power method.
    block = block_power_iteration(matvec=cov_matvec, n=n_features, k=k, n_iters=40, rng=rng)  # EN: Compute top-k via block power.
    overlap_block = subspace_overlap(V_ref, block.eigenvectors)  # EN: Subspace similarity via principal-angle cosines.
    print("eigenvalues =", block.eigenvalues)  # EN: Print eigenvalue estimates.
    print("residual norms =", block.residual_norms)  # EN: Print residual norms.
    print("subspace overlap singular values =", overlap_block)  # EN: Print overlap (near 1 is good).
    print("iters =", block.n_iters)  # EN: Print iteration count.

    print_separator("Deflation + Power Iteration")  # EN: Run deflation method.
    defl = deflation_power_iteration(matvec=cov_matvec, n=n_features, k=k, max_iters_each=2000, tol=1e-12, rng=rng)  # EN: Compute top-k via deflation.
    overlap_defl = subspace_overlap(V_ref, defl.eigenvectors)  # EN: Compute subspace overlap.
    print("eigenvalues =", defl.eigenvalues)  # EN: Print eigenvalues.
    print("residual norms =", defl.residual_norms)  # EN: Print residual norms.
    print("subspace overlap singular values =", overlap_defl)  # EN: Print overlap.
    print("total power iters =", defl.n_iters)  # EN: Print accumulated iteration count.

    print_separator("Lanczos (top-k Ritz pairs)")  # EN: Run Lanczos method.
    lanczos = lanczos_topk_ritz(matvec=cov_matvec, n=n_features, k=k, n_steps=40, reorthogonalize=True, rng=rng)  # EN: Compute top-k via Lanczos.
    overlap_lanczos = subspace_overlap(V_ref, lanczos.eigenvectors)  # EN: Compute subspace overlap.
    print("eigenvalues =", lanczos.eigenvalues)  # EN: Print eigenvalues.
    print("residual norms =", lanczos.residual_norms)  # EN: Print residual norms.
    print("subspace overlap singular values =", overlap_lanczos)  # EN: Print overlap.
    print("krylov dim used =", lanczos.n_iters)  # EN: Print Krylov dimension used.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main when run as a script.

