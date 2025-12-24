"""  # EN: Start module docstring.
Large-scale PCA idea (NumPy): Power Iteration and Lanczos to find only the first principal component.  # EN: Summarize the goal.

Key point: PCA's first component is the dominant eigenvector of the covariance matrix:  # EN: Explain the math link.
  C = (Xc^T Xc)/(n_samples-1), and v1 = argmax_{||v||=1} v^T C v.  # EN: State the Rayleigh quotient formulation.

For large feature dimension, you often avoid forming C explicitly.  # EN: Explain why we avoid forming covariance.
Instead you implement a matvec:  Cv = Xc^T (Xc v)/(n_samples-1).  # EN: Provide the matvec identity.

This file compares two iterative eigen-solvers:  # EN: Introduce the algorithms.
  - Power iteration (simple, but can be slow when eigenvalues are close)  # EN: Summarize power iteration.
  - Lanczos (Krylov method for symmetric matrices, often faster)  # EN: Summarize Lanczos.

We also compute a small-size "reference" solution via full SVD for verification.  # EN: Explain the validation approach.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured return values.
from typing import Callable  # EN: Use Callable for typing matvec functions.

import numpy as np  # EN: Import NumPy for array operations and linear algebra.


EPS = 1e-12  # EN: Small epsilon for numerical safety.
SEED = 0  # EN: RNG seed for reproducible outputs.
PRINT_PRECISION = 6  # EN: Floating-point printing precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing for readability.


@dataclass(frozen=True)  # EN: Immutable record for an approximate eigenpair.
class Eigenpair:  # EN: Store eigenvalue, eigenvector, and diagnostics.
    eigenvalue: float  # EN: Estimated eigenvalue λ.
    eigenvector: np.ndarray  # EN: Estimated unit eigenvector v (||v||=1).
    n_iters: int  # EN: Number of iterations performed.
    resid_norm: float  # EN: Residual norm ||A v - λ v||_2 as an accuracy indicator.


def print_separator(title: str) -> None:  # EN: Print a section separator.
    print()  # EN: Add a blank line.
    print("=" * 78)  # EN: Print horizontal divider.
    print(title)  # EN: Print title line.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm) of a vector.
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy for stable norm computation.


def normalize(v: np.ndarray) -> np.ndarray:  # EN: Normalize a vector to unit norm (with safety checks).
    n = l2_norm(v)  # EN: Compute vector norm.
    if n < EPS:  # EN: Guard against zero/near-zero vectors.
        raise ValueError("Cannot normalize near-zero vector")  # EN: Fail fast for invalid input.
    return v / n  # EN: Return unit vector.


def rayleigh_quotient(matvec: Callable[[np.ndarray], np.ndarray], v: np.ndarray) -> float:  # EN: Compute Rayleigh quotient v^T A v for unit v.
    Av = matvec(v)  # EN: Compute A v.
    return float(v @ Av)  # EN: Return scalar v^T (A v).


def eigen_residual_norm(matvec: Callable[[np.ndarray], np.ndarray], v: np.ndarray, lam: float) -> float:  # EN: Compute ||A v - λ v||_2.
    return l2_norm(matvec(v) - lam * v)  # EN: Return residual norm for eigenpair accuracy.


def power_iteration(  # EN: Compute dominant eigenpair using power iteration.
    matvec: Callable[[np.ndarray], np.ndarray],  # EN: Function computing A v.
    n: int,  # EN: Dimension of the eigenvector.
    max_iters: int = 2000,  # EN: Iteration cap.
    tol: float = 1e-10,  # EN: Stop when direction change is small.
    rng: np.random.Generator | None = None,  # EN: Optional RNG for deterministic initialization.
) -> Eigenpair:  # EN: Return estimated dominant eigenpair.
    if rng is None:  # EN: Create a default RNG when none is provided.
        rng = np.random.default_rng(SEED)  # EN: Use fixed seed for reproducible runs.

    v = rng.standard_normal(n)  # EN: Initialize with a random vector (generic non-orthogonal to top eigenvector).
    v = normalize(v)  # EN: Normalize initial vector.

    lam = rayleigh_quotient(matvec, v)  # EN: Initial eigenvalue estimate via Rayleigh quotient.
    resid = eigen_residual_norm(matvec, v, lam)  # EN: Initial residual norm.

    for it in range(1, max_iters + 1):  # EN: Iterate power method updates.
        w = matvec(v)  # EN: Multiply by A to amplify dominant eigen-direction.
        v_next = normalize(w)  # EN: Normalize to avoid overflow and keep unit vector.
        lam_next = rayleigh_quotient(matvec, v_next)  # EN: Update eigenvalue estimate.

        # EN: Use angle-based convergence: 1 - |v^T v_next| close to 0 indicates convergence (sign-invariant).
        align = abs(float(v @ v_next))  # EN: Compute cosine similarity between successive iterates.
        if 1.0 - align <= tol:  # EN: Stop when direction changes very little.
            v = v_next  # EN: Accept converged vector.
            lam = lam_next  # EN: Accept converged eigenvalue estimate.
            resid = eigen_residual_norm(matvec, v, lam)  # EN: Compute final residual norm.
            return Eigenpair(eigenvalue=lam, eigenvector=v, n_iters=it, resid_norm=resid)  # EN: Return result.

        v = v_next  # EN: Update iterate vector.
        lam = lam_next  # EN: Update eigenvalue estimate.

    resid = eigen_residual_norm(matvec, v, lam)  # EN: Compute residual norm after max_iters.
    return Eigenpair(eigenvalue=lam, eigenvector=v, n_iters=max_iters, resid_norm=resid)  # EN: Return best effort.


def lanczos_top_eigenpair(  # EN: Approximate dominant eigenpair of symmetric A using Lanczos.
    matvec: Callable[[np.ndarray], np.ndarray],  # EN: Function computing A v.
    n: int,  # EN: Dimension.
    n_steps: int = 25,  # EN: Lanczos steps (Krylov subspace dimension).
    reorthogonalize: bool = True,  # EN: Whether to do full re-orthogonalization (more stable for demos).
    rng: np.random.Generator | None = None,  # EN: RNG for initial vector.
) -> Eigenpair:  # EN: Return dominant Ritz eigenpair approximation.
    if n_steps <= 0:  # EN: Validate steps.
        raise ValueError("n_steps must be positive")  # EN: Reject invalid steps.
    if rng is None:  # EN: Create default RNG if needed.
        rng = np.random.default_rng(SEED)  # EN: Use fixed seed for reproducibility.

    q_prev = np.zeros((n,), dtype=float)  # EN: q_0 = 0 vector for recurrence.
    q = normalize(rng.standard_normal(n))  # EN: q_1 random unit vector.

    Q: list[np.ndarray] = []  # EN: Store Lanczos basis vectors for constructing Ritz vector.
    alphas: list[float] = []  # EN: Store diagonal entries of tridiagonal matrix T.
    betas: list[float] = []  # EN: Store off-diagonal entries of T.

    beta_prev = 0.0  # EN: Initialize previous beta for the first iteration.

    for j in range(n_steps):  # EN: Build Krylov basis and tridiagonal coefficients.
        Q.append(q)  # EN: Store current basis vector q_j.
        w = matvec(q) - beta_prev * q_prev  # EN: Compute w = A q_j - β_{j-1} q_{j-1}.
        alpha = float(q @ w)  # EN: α_j = q_j^T w.
        w = w - alpha * q  # EN: Orthogonalize against q_j.

        if reorthogonalize:  # EN: Optionally re-orthogonalize against all previous q's to reduce drift.
            for qi in Q[:-1]:  # EN: Loop over previous basis vectors.
                w = w - float(qi @ w) * qi  # EN: Remove component along qi.

        beta = l2_norm(w)  # EN: β_j = ||w||.
        alphas.append(alpha)  # EN: Store α_j.
        if j < n_steps - 1:  # EN: Only store β_j if we will have a next vector.
            betas.append(beta)  # EN: Store β_j as off-diagonal entry.

        if beta < EPS:  # EN: Break down indicates Krylov subspace reached an invariant subspace.
            break  # EN: Stop early.

        q_prev = q  # EN: Shift q_{j-1} <- q_j.
        q = w / beta  # EN: Normalize to get next q_{j+1}.
        beta_prev = beta  # EN: Update β_{j} for next iteration.

    k = len(alphas)  # EN: Effective Krylov dimension (may be < n_steps due to breakdown).
    T = np.zeros((k, k), dtype=float)  # EN: Initialize tridiagonal matrix T_k.
    for i in range(k):  # EN: Fill diagonal entries.
        T[i, i] = alphas[i]  # EN: Set α_i on diagonal.
    for i in range(k - 1):  # EN: Fill off-diagonal entries.
        T[i, i + 1] = betas[i]  # EN: Set β_i on super-diagonal.
        T[i + 1, i] = betas[i]  # EN: Set β_i on sub-diagonal.

    evals, evecs = np.linalg.eigh(T)  # EN: Diagonalize small symmetric tridiagonal T_k.
    idx = int(np.argmax(evals))  # EN: Index of largest Ritz value.
    lam = float(evals[idx])  # EN: Largest Ritz eigenvalue approximation.
    y = evecs[:, idx]  # EN: Corresponding eigenvector in Krylov basis coordinates.

    Q_mat = np.column_stack(Q[:k])  # EN: Form Q_k matrix from stored basis vectors.
    v = Q_mat @ y  # EN: Ritz vector in original space.
    v = normalize(v)  # EN: Normalize Ritz vector.
    resid = eigen_residual_norm(matvec, v, lam)  # EN: Compute eigen-residual norm.

    return Eigenpair(eigenvalue=lam, eigenvector=v, n_iters=k, resid_norm=resid)  # EN: Return approximate eigenpair.


def build_synthetic_data(  # EN: Create a synthetic dataset with a few dominant directions (low-rank + noise).
    rng: np.random.Generator,  # EN: RNG for reproducibility.
    n_samples: int,  # EN: Number of samples (rows).
    n_features: int,  # EN: Number of features (columns).
    latent_rank: int,  # EN: Intrinsic rank controlling dominant components.
    noise_std: float,  # EN: Noise strength.
) -> np.ndarray:  # EN: Return data matrix X (n_samples×n_features).
    Z = rng.standard_normal((n_samples, latent_rank))  # EN: Latent factors for samples.
    W = rng.standard_normal((n_features, latent_rank))  # EN: Feature loadings for latent factors.
    X_low_rank = Z @ W.T  # EN: Low-rank signal matrix.
    X = X_low_rank + noise_std * rng.standard_normal((n_samples, n_features))  # EN: Add isotropic noise.
    return X.astype(float)  # EN: Return float array.


def main() -> None:  # EN: Run a demo comparing power iteration and Lanczos for PCA's first component.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    n_samples = 300  # EN: Sample count for the synthetic dataset.
    n_features = 80  # EN: Feature dimension (moderate for demo; concept scales to large).
    latent_rank = 3  # EN: True latent rank controlling dominant principal directions.
    noise_std = 0.5  # EN: Noise strength.

    X = build_synthetic_data(  # EN: Generate synthetic data matrix.
        rng=rng,  # EN: Provide RNG.
        n_samples=n_samples,  # EN: Provide sample count.
        n_features=n_features,  # EN: Provide feature count.
        latent_rank=latent_rank,  # EN: Provide latent rank.
        noise_std=noise_std,  # EN: Provide noise strength.
    )  # EN: End data generation.

    mean = X.mean(axis=0)  # EN: Compute feature-wise mean for centering.
    Xc = X - mean  # EN: Center the data (required for PCA).

    scale = 1.0 / max(n_samples - 1, 1)  # EN: Covariance scaling factor (does not change eigenvectors).

    def cov_matvec(v: np.ndarray) -> np.ndarray:  # EN: Compute C v without forming C explicitly.
        return scale * (Xc.T @ (Xc @ v))  # EN: Apply covariance matvec using two multiplications.

    print_separator("Dataset summary")  # EN: Print dataset diagnostics.
    print(f"X shape: {n_samples}×{n_features}, latent_rank={latent_rank}, noise_std={noise_std}")  # EN: Print dataset parameters.

    # EN: Reference solution via full SVD of centered data (only feasible for moderate sizes).
    U_ref, s_ref, Vt_ref = np.linalg.svd(Xc, full_matrices=False)  # EN: Compute economy SVD: Xc = U Σ V^T.
    v_ref = Vt_ref[0, :].copy()  # EN: First right singular vector is the first principal direction.
    v_ref = normalize(v_ref)  # EN: Normalize for safety.
    lam_ref = float((s_ref[0] ** 2) * scale)  # EN: Convert singular value to covariance eigenvalue σ^2/(n-1).
    resid_ref = eigen_residual_norm(cov_matvec, v_ref, lam_ref)  # EN: Compute reference eigen residual.

    print_separator("Reference (full SVD) first principal component")  # EN: Print reference eigenpair.
    print(f"lambda_ref = {lam_ref:.6e}")  # EN: Print reference eigenvalue.
    print(f"residual ||Cv-λv|| = {resid_ref:.3e}")  # EN: Print residual norm for reference.

    print_separator("Power Iteration")  # EN: Run power iteration and print results.
    power = power_iteration(matvec=cov_matvec, n=n_features, max_iters=2000, tol=1e-12, rng=rng)  # EN: Compute dominant eigenpair.
    cos_power = abs(float(power.eigenvector @ v_ref))  # EN: Cosine similarity to reference (sign-invariant).
    print(f"iters = {power.n_iters}")  # EN: Print iteration count.
    print(f"lambda = {power.eigenvalue:.6e}")  # EN: Print eigenvalue estimate.
    print(f"cosine similarity to ref = {cos_power:.6f}")  # EN: Print similarity to reference vector.
    print(f"residual ||Cv-λv|| = {power.resid_norm:.3e}")  # EN: Print eigen residual.

    print_separator("Lanczos (Krylov) method")  # EN: Run Lanczos and print results.
    lanczos = lanczos_top_eigenpair(matvec=cov_matvec, n=n_features, n_steps=25, reorthogonalize=True, rng=rng)  # EN: Compute top Ritz pair.
    cos_lanczos = abs(float(lanczos.eigenvector @ v_ref))  # EN: Cosine similarity to reference.
    print(f"steps used = {lanczos.n_iters}")  # EN: Print Lanczos steps actually used.
    print(f"lambda = {lanczos.eigenvalue:.6e}")  # EN: Print Ritz eigenvalue estimate.
    print(f"cosine similarity to ref = {cos_lanczos:.6f}")  # EN: Print similarity to reference.
    print(f"residual ||Cv-λv|| = {lanczos.resid_norm:.3e}")  # EN: Print eigen residual.

    print_separator("PCA projection demo (1D)")  # EN: Show PCA projection and reconstruction with the approximate component.
    z_ref = Xc @ v_ref  # EN: Project centered data onto the reference first component.
    X_hat_ref = np.outer(z_ref, v_ref) + mean  # EN: Reconstruct rank-1 approximation using reference component.
    err_ref = float(np.linalg.norm(X - X_hat_ref, ord="fro"))  # EN: Compute reconstruction error (Frobenius norm).

    z_lanczos = Xc @ lanczos.eigenvector  # EN: Project using Lanczos component.
    X_hat_lanczos = np.outer(z_lanczos, lanczos.eigenvector) + mean  # EN: Reconstruct using Lanczos component.
    err_lanczos = float(np.linalg.norm(X - X_hat_lanczos, ord="fro"))  # EN: Compute reconstruction error.

    print(f"rank-1 recon error (ref)     ||X-Xhat||_F = {err_ref:.6e}")  # EN: Print reference reconstruction error.
    print(f"rank-1 recon error (lanczos) ||X-Xhat||_F = {err_lanczos:.6e}")  # EN: Print Lanczos reconstruction error.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main function when run as a script.

