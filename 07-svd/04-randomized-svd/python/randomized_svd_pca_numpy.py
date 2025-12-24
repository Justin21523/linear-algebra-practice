"""  # EN: Start module docstring.
Randomized SVD (NumPy): approximate top-k singular triplets with random projections.  # EN: Summarize the purpose of this script.

Core idea (randomized range finder + small SVD):  # EN: Outline the algorithm at a high level.
  1) Draw a random matrix Ω (n×(k+p))  # EN: Describe the sketching step.
  2) Form Y = A Ω (m×(k+p))  # EN: Explain how we probe the column space of A.
  3) Orthonormalize Y -> Q via QR  # EN: Explain building an orthonormal basis for the sampled range.
  4) Optionally do q power iterations to improve accuracy for slow spectral decay  # EN: Mention power iteration refinement.
  5) Compress: B = Q^T A (small) and compute SVD(B)  # EN: Explain reduction to a small matrix.
  6) Lift: U ≈ Q Ũ, then keep top-k components  # EN: Explain how we recover approximate singular vectors.

We verify quality by comparing to full SVD on a moderate-size matrix:  # EN: Explain the verification plan.
  - Relative singular value error for the top-k  # EN: Metric 1.
  - Relative reconstruction error ||A - U_k Σ_k V_k^T||_F / ||A||_F  # EN: Metric 2.
  - Subspace similarity via principal angles (SVD of U_ref^T U_hat)  # EN: Metric 3.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured return values.
from typing import Literal  # EN: Use Literal to constrain options in type hints.

import numpy as np  # EN: Import NumPy for numerical linear algebra.


EPS = 1e-12  # EN: Small epsilon used to avoid division-by-zero.
SEED = 0  # EN: RNG seed for reproducible demos.
PRINT_PRECISION = 6  # EN: Console float precision for readable prints.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing globally for this script.


@dataclass(frozen=True)  # EN: Make the result immutable for safety and clarity.
class RandomizedSVDResult:  # EN: Container for an approximate rank-k SVD.
    U: np.ndarray  # EN: Approximate left singular vectors (m×k), orthonormal columns.
    S: np.ndarray  # EN: Approximate top-k singular values (k,), non-negative and descending.
    Vt: np.ndarray  # EN: Approximate right singular vectors transpose (k×n), rows orthonormal.
    k: int  # EN: Target rank requested by the user.
    p: int  # EN: Oversampling parameter used (extra sketch dimensions).
    q: int  # EN: Power iteration count used.
    sketch_dim: int  # EN: Total sketch dimension = k+p.


def print_separator(title: str) -> None:  # EN: Print a section separator for console readability.
    print()  # EN: Insert a blank line before each section.
    print("=" * 78)  # EN: Print a horizontal divider.
    print(title)  # EN: Print section title.
    print("=" * 78)  # EN: Print a closing divider.


def fro_norm(A: np.ndarray) -> float:  # EN: Compute Frobenius norm of a matrix.
    return float(np.linalg.norm(A, ord="fro"))  # EN: Delegate to NumPy's stable implementation.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm of a vector.
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def orthonormalize_columns(Y: np.ndarray) -> np.ndarray:  # EN: Orthonormalize columns of Y using thin QR.
    Q, _ = np.linalg.qr(Y, mode="reduced")  # EN: Reduced QR yields Q with orthonormal columns.
    return Q  # EN: Return the orthonormal basis.


def randomized_svd(  # EN: Compute an approximate rank-k SVD using randomized range finding.
    A: np.ndarray,  # EN: Input matrix A (m×n).
    k: int,  # EN: Target rank (number of singular triplets to approximate).
    p: int = 10,  # EN: Oversampling dimension (usually 5–20).
    q: int = 1,  # EN: Power iteration count (0 for none; 1–2 often helps for slow decay).
    rng: np.random.Generator | None = None,  # EN: RNG for the random test matrix Ω.
    orth_mode: Literal["qr"] = "qr",  # EN: Orthonormalization method (QR only for this educational demo).
) -> RandomizedSVDResult:  # EN: Return RandomizedSVDResult with U,S,Vt.
    if rng is None:  # EN: Provide a deterministic default RNG.
        rng = np.random.default_rng(SEED)  # EN: Use a fixed seed so results are repeatable.
    if A.ndim != 2:  # EN: Validate input is a matrix.
        raise ValueError("A must be 2D")  # EN: Reject invalid A.
    if k <= 0:  # EN: Validate k.
        raise ValueError("k must be positive")  # EN: Reject invalid k.
    m, n = A.shape  # EN: Extract matrix dimensions.
    if k > min(m, n):  # EN: Validate k does not exceed rank upper bound.
        raise ValueError("k must be <= min(m, n)")  # EN: Reject invalid k.
    if p < 0:  # EN: Validate oversampling.
        raise ValueError("p must be non-negative")  # EN: Reject invalid p.
    if q < 0:  # EN: Validate power iterations.
        raise ValueError("q must be non-negative")  # EN: Reject invalid q.
    sketch_dim = min(n, k + p)  # EN: Choose sketch dimension (cannot exceed n).

    # EN: Step 1: draw a random test matrix Ω in R^{n×(k+p)}.  # EN: Explain the purpose of Ω.
    Omega = rng.standard_normal((n, sketch_dim)).astype(float)  # EN: Use Gaussian test matrix (common default).

    # EN: Step 2: form Y = A Ω to sample the range (column space) of A.  # EN: Describe range finding.
    Y = A @ Omega  # EN: Compute the sketch Y (m×(k+p)).

    # EN: Step 3: orthonormalize Y -> Q so columns of Q span the sampled range.  # EN: Describe orthonormal basis construction.
    if orth_mode != "qr":  # EN: Only QR is implemented in this repo demo.
        raise ValueError("Only orth_mode='qr' is supported")  # EN: Reject unsupported modes.
    Q = orthonormalize_columns(Y)  # EN: Orthonormal basis for the sketch range.

    # EN: Step 4 (optional): power iteration improves accuracy when singular values decay slowly.  # EN: Explain why power iteration helps.
    # EN: We repeatedly apply A A^T to the subspace: Q <- orth(A (A^T Q)).  # EN: State the update formula.
    for _ in range(q):  # EN: Loop q times (q=0 does nothing).
        Z = A.T @ Q  # EN: Compute Z = A^T Q (n×(k+p)) to move into the row space.
        Y = A @ Z  # EN: Compute Y = A Z = A A^T Q (m×(k+p)) to amplify dominant directions.
        Q = orthonormalize_columns(Y)  # EN: Re-orthonormalize to control numerical drift.

    # EN: Step 5: compress A to a small matrix B = Q^T A.  # EN: Explain dimension reduction.
    B = Q.T @ A  # EN: B has shape (k+p)×n and captures most of A's action in span(Q).

    # EN: Step 6: compute SVD of the small matrix B and lift back.  # EN: Explain the final SVD step.
    Ub, S, Vt = np.linalg.svd(B, full_matrices=False)  # EN: Compute thin SVD of B.
    U = Q @ Ub  # EN: Lift left singular vectors: U ≈ Q Ub.

    # EN: Keep only the top-k components (rank-k approximation).  # EN: Explain truncation.
    U_k = U[:, :k]  # EN: Top-k left singular vectors.
    S_k = S[:k]  # EN: Top-k singular values.
    Vt_k = Vt[:k, :]  # EN: Top-k right singular vectors (transposed).

    return RandomizedSVDResult(  # EN: Package results into a dataclass.
        U=U_k,  # EN: Store U_k.
        S=S_k,  # EN: Store S_k.
        Vt=Vt_k,  # EN: Store Vt_k.
        k=k,  # EN: Store k.
        p=p,  # EN: Store p.
        q=q,  # EN: Store q.
        sketch_dim=sketch_dim,  # EN: Store total sketch dimension.
    )  # EN: End return.


def principal_angle_cosines(U_ref: np.ndarray, U_hat: np.ndarray) -> np.ndarray:  # EN: Compute cosines of principal angles between two subspaces.
    M = U_ref.T @ U_hat  # EN: Build k×k cross-Gram matrix for the two orthonormal bases.
    s = np.linalg.svd(M, compute_uv=False)  # EN: Singular values are cosines of principal angles.
    s = np.clip(s, 0.0, 1.0)  # EN: Clip to [0,1] to avoid tiny numerical negatives/overflows.
    return s  # EN: Return cosines (sorted descending by SVD).


def rel_error(a: np.ndarray, b: np.ndarray) -> float:  # EN: Compute relative error ||a-b||/||b|| with EPS safety.
    denom = max(l2_norm(b.ravel()), EPS)  # EN: Compute denominator with epsilon floor.
    return l2_norm((a - b).ravel()) / denom  # EN: Return relative error.


def build_synthetic_matrix(  # EN: Build a synthetic matrix with controllable singular spectrum.
    m: int,  # EN: Number of rows.
    n: int,  # EN: Number of columns.
    r_true: int,  # EN: True underlying rank for the signal part.
    spectrum: Literal["fast", "slow"] = "slow",  # EN: Choose fast vs slow singular value decay.
    noise_std: float = 1e-3,  # EN: Additive Gaussian noise level.
    rng: np.random.Generator | None = None,  # EN: RNG for reproducibility.
) -> np.ndarray:  # EN: Return A ≈ U diag(s) V^T + noise.
    if rng is None:  # EN: Provide default RNG.
        rng = np.random.default_rng(SEED)  # EN: Use deterministic seed.
    if r_true <= 0 or r_true > min(m, n):  # EN: Validate rank.
        raise ValueError("r_true must be in [1, min(m,n)]")  # EN: Reject invalid rank.

    # EN: Create random orthonormal factors U and V via QR.  # EN: Explain factor construction.
    U0, _ = np.linalg.qr(rng.standard_normal((m, r_true)))  # EN: Orthonormal U0 (m×r).
    V0, _ = np.linalg.qr(rng.standard_normal((n, r_true)))  # EN: Orthonormal V0 (n×r).

    # EN: Choose a singular spectrum with either fast or slow decay.  # EN: Explain the two regimes.
    if spectrum == "fast":  # EN: Fast decay -> randomized methods work very well even without power iteration.
        s = 10.0 ** (-np.linspace(0.0, 4.0, r_true))  # EN: Exponentially decaying singular values.
    elif spectrum == "slow":  # EN: Slow decay -> power iterations (q>0) noticeably help.
        s = 1.0 / (1.0 + np.arange(r_true))  # EN: Harmonic-like decay: 1,1/2,1/3,...
    else:  # EN: Guard against unsupported spectrum selection.
        raise ValueError("spectrum must be 'fast' or 'slow'")  # EN: Reject invalid choice.

    A_signal = (U0 * s) @ V0.T  # EN: Build low-rank signal U diag(s) V^T using broadcasting.
    A_noise = noise_std * rng.standard_normal((m, n))  # EN: Add small Gaussian noise (full-rank perturbation).
    return (A_signal + A_noise).astype(float)  # EN: Return final synthetic matrix.


def evaluate_case(A: np.ndarray, k: int, p: int, q: int, rng: np.random.Generator) -> None:  # EN: Run one randomized SVD configuration and print metrics.
    m, n = A.shape  # EN: Extract matrix dimensions for printing and checks.
    print(f"Config: k={k}, p={p}, q={q}  (A shape {m}×{n})")  # EN: Print configuration line.

    # EN: Reference: full SVD on a moderate-size matrix.  # EN: Explain baseline.
    U_ref, S_ref, Vt_ref = np.linalg.svd(A, full_matrices=False)  # EN: Compute exact thin SVD with NumPy.

    # EN: Approximation: randomized SVD.  # EN: Explain target algorithm.
    approx = randomized_svd(A=A, k=k, p=p, q=q, rng=rng)  # EN: Compute approximate top-k SVD.

    # EN: Build rank-k reconstructions for both reference and approximation.  # EN: Explain reconstruction step.
    A_ref_k = (U_ref[:, :k] * S_ref[:k]) @ Vt_ref[:k, :]  # EN: Rank-k reconstruction from the exact SVD.
    A_hat_k = (approx.U * approx.S) @ approx.Vt  # EN: Rank-k reconstruction from randomized SVD.

    # EN: Compute quality metrics.  # EN: Explain what we measure.
    sv_rel = rel_error(approx.S, S_ref[:k])  # EN: Relative error of the top-k singular values.
    rec_rel = fro_norm(A - A_hat_k) / max(fro_norm(A), EPS)  # EN: Relative reconstruction error in Frobenius norm.
    rec_gap = (fro_norm(A - A_hat_k) - fro_norm(A - A_ref_k)) / max(fro_norm(A - A_ref_k), EPS)  # EN: Relative gap to optimal rank-k error.

    # EN: Compare left singular subspaces via principal angles.  # EN: Explain subspace comparison.
    cosines = principal_angle_cosines(U_ref[:, :k], approx.U)  # EN: Compute cosines of principal angles.
    min_cos = float(np.min(cosines))  # EN: Worst-case alignment among the k principal angles.
    mean_cos = float(np.mean(cosines))  # EN: Average alignment.

    print(f"  singular values rel_err = {sv_rel:.3e}")  # EN: Print singular value relative error.
    print(f"  reconstruction rel_err  = {rec_rel:.3e}")  # EN: Print reconstruction relative error.
    print(f"  gap to optimal rank-k   = {rec_gap:.3e}")  # EN: Print reconstruction gap vs best rank-k.
    print(f"  principal angle cosines = {cosines}")  # EN: Print principal-angle cosines (closer to 1 is better).
    print(f"    min cosine={min_cos:.6f}, mean cosine={mean_cos:.6f}")  # EN: Summarize alignment.


def main() -> None:  # EN: Run a small demo comparing configurations on synthetic matrices.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    m, n = 600, 200  # EN: Choose a moderate size where full SVD is still feasible for reference.
    r_true = 20  # EN: Choose a true rank for the underlying signal.
    k = 10  # EN: Target rank to approximate (top-k).

    for spectrum in ["fast", "slow"]:  # EN: Compare fast vs slow singular value decay regimes.
        print_separator(f"Synthetic matrix with {spectrum} spectral decay")  # EN: Print regime header.
        A = build_synthetic_matrix(m=m, n=n, r_true=r_true, spectrum=spectrum, noise_std=1e-3, rng=rng)  # EN: Create synthetic matrix.

        # EN: Compare a few (p,q) settings to show the effect of oversampling and power iterations.  # EN: Explain experiment design.
        evaluate_case(A=A, k=k, p=5, q=0, rng=rng)  # EN: Minimal oversampling, no power iteration.
        evaluate_case(A=A, k=k, p=10, q=0, rng=rng)  # EN: More oversampling, no power iteration.
        evaluate_case(A=A, k=k, p=10, q=1, rng=rng)  # EN: Add one power iteration.
        evaluate_case(A=A, k=k, p=10, q=2, rng=rng)  # EN: Add two power iterations (often enough).

    print_separator("Done")  # EN: Print final marker.


if __name__ == "__main__":  # EN: Standard Python entrypoint guard.
    main()  # EN: Execute the demo.

