"""  # EN: Start module docstring.
Preconditioning for LSQR (and LSMR-style normal-equation methods): column scaling (right preconditioning).  # EN: Summarize what this script demonstrates.

Goal: speed up convergence on ill-conditioned least-squares problems without forming A^T A.  # EN: Explain the objective.

Right preconditioning with a diagonal D (column scaling):  # EN: Introduce the main trick.
  - Let x = D^{-1} y, so we solve: min_y || (A D^{-1}) y - b ||_2.  # EN: Show the transformed problem.
  - If D_j approximates the scale of column j (e.g., ||A[:,j]||_2), then A D^{-1} has better-conditioned columns.  # EN: Explain why it helps.
  - After solving for y, recover x by x = D^{-1} y (elementwise division for diagonal D).  # EN: Show recovery step.

We compare:  # EN: Describe what we compare in this demo.
  1) LSQR on the original system (matvec-only).  # EN: Baseline method.
  2) LSQR on the right-preconditioned system (column-scaled).  # EN: Preconditioned LSQR.
  3) A normal-equation Krylov baseline (CG on A^T A x = A^T b) as a proxy for "LSMR-style" methods.  # EN: Clarify what we mean by LSMR-style.

Note: true LSMR is a MINRES-based method on the normal equations with nicer monotonicity properties,  # EN: Provide an important nuance.
but it uses the same idea for right preconditioning (replace A by A D^{-1}).  # EN: State the transferability of the preconditioning trick.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured results.
from typing import Callable  # EN: Use Callable for matvec typing.

import numpy as np  # EN: Import NumPy for numerical linear algebra.


EPS = 1e-12  # EN: Small epsilon to avoid division-by-zero.
SEED = 0  # EN: RNG seed for deterministic demos.
PRINT_PRECISION = 6  # EN: Console float precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias for a matrix-vector product function.


@dataclass(frozen=True)  # EN: Make results immutable for safer usage.
class LSQRResult:  # EN: Store LSQR solution and convergence diagnostics.
    x_hat: np.ndarray  # EN: Estimated solution (n,).
    n_iters: int  # EN: Number of iterations performed.
    residual_norm: float  # EN: Final ||Ax-b||_2.
    normal_residual_norm: float  # EN: Final ||A^T(Ax-b)||_2.
    history_residual_norm: np.ndarray  # EN: History of ||Ax-b|| over iterations.
    history_normal_residual_norm: np.ndarray  # EN: History of ||A^T(Ax-b)|| over iterations.


@dataclass(frozen=True)  # EN: Make results immutable for safer usage.
class KrylovResult:  # EN: Store CG/PCG outputs on the normal equations.
    method: str  # EN: Method label (e.g., "CGNR" or "PCGNR-Jacobi").
    x_hat: np.ndarray  # EN: Estimated solution (n,).
    n_iters: int  # EN: Iterations performed.
    resid_norm_history: np.ndarray  # EN: History of ||g - (A^T A)x||_2 (i.e., ||A^T r||_2).


def print_separator(title: str) -> None:  # EN: Print a section separator for console readability.
    print()  # EN: Add a blank line before sections.
    print("=" * 78)  # EN: Print a horizontal divider.
    print(title)  # EN: Print the section title.
    print("=" * 78)  # EN: Print a closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def lsqr(  # EN: Solve min ||Ax-b|| using LSQR given only matvecs for A and A^T.
    matvec_A: Matvec,  # EN: Function computing A @ v (m,).
    matvec_AT: Matvec,  # EN: Function computing A^T @ u (n,).
    b: np.ndarray,  # EN: Right-hand side vector (m,).
    n: int,  # EN: Number of unknowns (columns of A).
    max_iters: int = 200,  # EN: Iteration cap.
    tol_normal: float = 1e-10,  # EN: Stop when ||A^T(Ax-b)||_2 <= tol_normal.
) -> LSQRResult:  # EN: Return LSQRResult with solution and history.
    if b.ndim != 1:  # EN: Validate b shape.
        raise ValueError("b must be a 1D vector")  # EN: Reject invalid input.
    if n <= 0:  # EN: Validate dimension.
        raise ValueError("n must be positive")  # EN: Reject invalid dimension.
    if max_iters <= 0:  # EN: Validate iteration cap.
        raise ValueError("max_iters must be positive")  # EN: Reject invalid max_iters.

    x = np.zeros((n,), dtype=float)  # EN: Initialize x0=0 (common default).

    r0 = matvec_A(x) - b  # EN: Initial residual r0 = A x0 - b = -b.
    nr0 = matvec_AT(r0)  # EN: Initial normal residual A^T r0.
    history_r: list[float] = [l2_norm(r0)]  # EN: Track ||r|| starting at iter 0.
    history_nr: list[float] = [l2_norm(nr0)]  # EN: Track ||A^T r|| starting at iter 0.

    # EN: Initialize Golub–Kahan bidiagonalization with u1 = b/||b||.  # EN: Explain initialization.
    u = b.copy()  # EN: Copy b to start u.
    beta = l2_norm(u)  # EN: beta = ||b||.
    if beta < EPS:  # EN: Handle b=0 -> x=0 is optimal.
        return LSQRResult(  # EN: Return early with the zero solution.
            x_hat=x,  # EN: x=0.
            n_iters=0,  # EN: No iterations.
            residual_norm=history_r[-1],  # EN: Residual norm.
            normal_residual_norm=history_nr[-1],  # EN: Normal residual norm.
            history_residual_norm=np.array(history_r, dtype=float),  # EN: Convert history to array.
            history_normal_residual_norm=np.array(history_nr, dtype=float),  # EN: Convert history to array.
        )  # EN: End early return.
    u = u / beta  # EN: Normalize u.

    v = matvec_AT(u)  # EN: Compute v = A^T u.
    alpha = l2_norm(v)  # EN: alpha = ||A^T u||.
    if alpha < EPS:  # EN: Handle A^T u = 0 (rare, but possible).
        return LSQRResult(  # EN: Return with x=0 (cannot proceed).
            x_hat=x,  # EN: Keep x=0.
            n_iters=0,  # EN: No iterations.
            residual_norm=history_r[-1],  # EN: Residual norm.
            normal_residual_norm=history_nr[-1],  # EN: Normal residual norm.
            history_residual_norm=np.array(history_r, dtype=float),  # EN: History.
            history_normal_residual_norm=np.array(history_nr, dtype=float),  # EN: History.
        )  # EN: End return.
    v = v / alpha  # EN: Normalize v.

    w = v.copy()  # EN: Initialize w (the "search direction" basis vector).
    phi_bar = beta  # EN: Initialize φ̄ for residual recurrences.
    rho_bar = alpha  # EN: Initialize ρ̄ for bidiagonal recurrences.

    # EN: Main LSQR iteration loop.  # EN: Explain loop purpose.
    for it in range(1, max_iters + 1):  # EN: Iterate up to the cap.
        u = matvec_A(v) - alpha * u  # EN: u_{k+1} = A v_k - α_k u_k.
        beta = l2_norm(u)  # EN: β_{k+1} = ||u_{k+1}||.
        if beta >= EPS:  # EN: Normalize u when possible.
            u = u / beta  # EN: Normalize u.

        v = matvec_AT(u) - beta * v  # EN: v_{k+1} = A^T u_{k+1} - β_{k+1} v_k.
        alpha = l2_norm(v)  # EN: α_{k+1} = ||v_{k+1}||.
        if alpha >= EPS:  # EN: Normalize v when possible.
            v = v / alpha  # EN: Normalize v.

        rho = float(np.hypot(rho_bar, beta))  # EN: ρ = sqrt(ρ̄^2 + β^2).
        c = rho_bar / max(rho, EPS)  # EN: c = ρ̄ / ρ.
        s = beta / max(rho, EPS)  # EN: s = β / ρ.
        theta = s * alpha  # EN: θ = s α.
        rho_bar = -c * alpha  # EN: Update ρ̄ = -c α.
        phi = c * phi_bar  # EN: φ = c φ̄.
        phi_bar = s * phi_bar  # EN: Update φ̄ = s φ̄.

        x = x + (phi / max(rho, EPS)) * w  # EN: Update solution estimate.
        w = v - (theta / max(rho, EPS)) * w  # EN: Update direction vector.

        r = matvec_A(x) - b  # EN: Compute residual r = Ax-b for diagnostics.
        nr = matvec_AT(r)  # EN: Compute normal residual A^T r for stopping.
        history_r.append(l2_norm(r))  # EN: Record ||r||.
        history_nr.append(l2_norm(nr))  # EN: Record ||A^T r||.

        if history_nr[-1] <= tol_normal:  # EN: Stop when normal residual is small.
            break  # EN: Exit iteration loop.
        if beta < EPS and alpha < EPS:  # EN: Detect breakdown (cannot continue bidiagonalization).
            break  # EN: Exit on breakdown.

    return LSQRResult(  # EN: Package final outputs.
        x_hat=x,  # EN: Final x.
        n_iters=len(history_r) - 1,  # EN: Iterations performed (history includes iter 0).
        residual_norm=history_r[-1],  # EN: Final ||Ax-b||.
        normal_residual_norm=history_nr[-1],  # EN: Final ||A^T(Ax-b)||.
        history_residual_norm=np.array(history_r, dtype=float),  # EN: Residual history.
        history_normal_residual_norm=np.array(history_nr, dtype=float),  # EN: Normal residual history.
    )  # EN: End return.


def cg_spd(  # EN: Conjugate Gradient for SPD systems Hx=g (used here on normal equations).
    matvec_H: Matvec,  # EN: Callable computing H @ v.
    g: np.ndarray,  # EN: Right-hand side vector.
    x0: np.ndarray,  # EN: Initial guess.
    max_iters: int,  # EN: Iteration cap.
    tol: float,  # EN: Residual tolerance on ||r||_2.
    inv_diag: np.ndarray | None = None,  # EN: Optional Jacobi preconditioner (M^{-1} as a diagonal vector).
) -> KrylovResult:  # EN: Return KrylovResult.
    x = x0.copy()  # EN: Copy initial guess.
    r = g - matvec_H(x)  # EN: Initial residual r0.
    if inv_diag is None:  # EN: Choose unpreconditioned vs preconditioned branch.
        z = r.copy()  # EN: For CG, z=r.
        method = "CGNR"  # EN: Label method as CG on normal equations.
    else:  # EN: Preconditioned branch.
        z = inv_diag * r  # EN: Apply Jacobi preconditioner z=M^{-1} r.
        method = "PCGNR-Jacobi"  # EN: Label method accordingly.
    p = z.copy()  # EN: Initial search direction.
    rz_old = float(r @ z)  # EN: Compute r^T z (or r^T r for CG).
    resid_hist: list[float] = [l2_norm(r)]  # EN: Track ||r|| over iterations.

    if resid_hist[0] <= tol:  # EN: Early exit if already converged.
        return KrylovResult(method=method, x_hat=x, n_iters=0, resid_norm_history=np.array(resid_hist, dtype=float))  # EN: Return.

    for it in range(1, max_iters + 1):  # EN: CG iteration loop.
        Hp = matvec_H(p)  # EN: Compute H p.
        denom = float(p @ Hp)  # EN: Compute p^T H p (positive for SPD).
        if abs(denom) < EPS:  # EN: Guard against breakdown.
            break  # EN: Stop if denominator is too small.
        alpha = rz_old / denom  # EN: Step size α.
        x = x + alpha * p  # EN: Update solution.
        r = r - alpha * Hp  # EN: Update residual.
        resid = l2_norm(r)  # EN: Compute ||r|| for stopping.
        resid_hist.append(resid)  # EN: Record ||r||.
        if resid <= tol:  # EN: Stop when converged.
            return KrylovResult(method=method, x_hat=x, n_iters=it, resid_norm_history=np.array(resid_hist, dtype=float))  # EN: Return.
        if inv_diag is None:  # EN: CG update uses z=r.
            z = r  # EN: Set z=r.
        else:  # EN: PCG update uses z=M^{-1} r.
            z = inv_diag * r  # EN: Apply preconditioner.
        rz_new = float(r @ z)  # EN: Compute new r^T z.
        beta = rz_new / max(rz_old, EPS)  # EN: β coefficient for direction update.
        p = z + beta * p  # EN: Update search direction.
        rz_old = rz_new  # EN: Carry dot-product for next iteration.

    return KrylovResult(method=method, x_hat=x, n_iters=len(resid_hist) - 1, resid_norm_history=np.array(resid_hist, dtype=float))  # EN: Return best effort.


def make_conditioned_matrix(  # EN: Build A with controlled singular values to create an ill-conditioned least-squares problem.
    rng: np.random.Generator,  # EN: RNG for reproducibility.
    m: int,  # EN: Number of rows.
    n: int,  # EN: Number of columns.
    singular_values: np.ndarray,  # EN: Desired singular values (length n).
) -> np.ndarray:  # EN: Return A with approximately the given singular spectrum.
    G1 = rng.standard_normal((m, n))  # EN: Random Gaussian matrix for left factor.
    Q1, _ = np.linalg.qr(G1, mode="reduced")  # EN: Orthonormalize columns to get m×n matrix.
    G2 = rng.standard_normal((n, n))  # EN: Random Gaussian matrix for right factor.
    Q2, _ = np.linalg.qr(G2)  # EN: Orthonormalize to get n×n orthogonal matrix.
    S = np.diag(singular_values.astype(float))  # EN: Diagonal matrix of singular values.
    return Q1 @ S @ Q2.T  # EN: Construct A = Q1 S Q2^T.


def column_scaling_D(A: np.ndarray) -> np.ndarray:  # EN: Build a simple diagonal right preconditioner D from column norms of A.
    col_norms = np.linalg.norm(A, axis=0)  # EN: Compute ||A[:,j]||_2 for each column.
    D = np.maximum(col_norms, EPS)  # EN: Avoid zero divisions for zero columns.
    return D.astype(float)  # EN: Return D as a float vector.


def solve_lsqr_dense(A: np.ndarray, b: np.ndarray, tol_normal: float, max_iters: int) -> LSQRResult:  # EN: Convenience wrapper for dense LSQR.
    def matvec_A(v: np.ndarray) -> np.ndarray:  # EN: Dense matvec for A.
        return A @ v  # EN: Compute A v.

    def matvec_AT(u: np.ndarray) -> np.ndarray:  # EN: Dense matvec for A^T.
        return A.T @ u  # EN: Compute A^T u.

    return lsqr(matvec_A=matvec_A, matvec_AT=matvec_AT, b=b, n=A.shape[1], max_iters=max_iters, tol_normal=tol_normal)  # EN: Solve with LSQR.


def solve_lsqr_right_preconditioned(  # EN: Solve least squares with LSQR on A D^{-1} and map back x = D^{-1} y.
    A: np.ndarray,  # EN: Design matrix (m×n).
    b: np.ndarray,  # EN: RHS vector (m,).
    D: np.ndarray,  # EN: Diagonal scaling vector (n,) with positive entries.
    tol_normal: float,  # EN: Stopping tolerance for LSQR on the preconditioned operator.
    max_iters: int,  # EN: Iteration cap.
) -> LSQRResult:  # EN: Return LSQRResult for x (mapped back).
    D_safe = np.maximum(D, EPS)  # EN: Guard against zeros in D.

    def matvec_B(y: np.ndarray) -> np.ndarray:  # EN: Compute (A D^{-1}) y.
        return A @ (y / D_safe)  # EN: Right preconditioning uses elementwise division by D.

    def matvec_BT(u: np.ndarray) -> np.ndarray:  # EN: Compute (A D^{-1})^T u = D^{-1} A^T u.
        return (A.T @ u) / D_safe  # EN: Apply transpose then scale by D^{-1}.

    res_y = lsqr(matvec_A=matvec_B, matvec_AT=matvec_BT, b=b, n=A.shape[1], max_iters=max_iters, tol_normal=tol_normal)  # EN: Solve for y.
    x_hat = res_y.x_hat / D_safe  # EN: Map back x = D^{-1} y.

    r = A @ x_hat - b  # EN: Compute residual in the original coordinates.
    nr = A.T @ r  # EN: Compute normal residual in the original coordinates.

    return LSQRResult(  # EN: Return an LSQRResult but expressed in x-space diagnostics.
        x_hat=x_hat,  # EN: Store mapped-back solution.
        n_iters=res_y.n_iters,  # EN: Iteration count is the same.
        residual_norm=l2_norm(r),  # EN: Original residual norm.
        normal_residual_norm=l2_norm(nr),  # EN: Original normal residual norm.
        history_residual_norm=res_y.history_residual_norm,  # EN: Keep preconditioned residual history (still informative for convergence).
        history_normal_residual_norm=res_y.history_normal_residual_norm,  # EN: Keep preconditioned normal residual history.
    )  # EN: End return.


def solve_cgnr_dense(A: np.ndarray, b: np.ndarray, tol: float, max_iters: int, use_jacobi: bool) -> KrylovResult:  # EN: Solve normal equations with (P)CG as a baseline.
    g = A.T @ b  # EN: Form g = A^T b (cheap compared to A^T A).

    def matvec_H(v: np.ndarray) -> np.ndarray:  # EN: Compute H v with H = A^T A, without forming H.
        return A.T @ (A @ v)  # EN: Return A^T(A v).

    x0 = np.zeros((A.shape[1],), dtype=float)  # EN: Start from x0=0.
    if not use_jacobi:  # EN: Plain CGNR case.
        return cg_spd(matvec_H=matvec_H, g=g, x0=x0, max_iters=max_iters, tol=tol, inv_diag=None)  # EN: Run CG.

    diag = np.sum(A * A, axis=0)  # EN: diag(A^T A) equals column-wise sum of squares.
    inv_diag = 1.0 / np.maximum(diag, EPS)  # EN: Jacobi preconditioner M^{-1}.
    return cg_spd(matvec_H=matvec_H, g=g, x0=x0, max_iters=max_iters, tol=tol, inv_diag=inv_diag)  # EN: Run PCG.


def main() -> None:  # EN: Run a demo comparing LSQR with/without right preconditioning on an ill-conditioned problem.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    m = 400  # EN: Number of rows (samples).
    n = 80  # EN: Number of columns (features).
    noise_std = 1e-3  # EN: Observation noise level.

    # EN: Create a strongly ill-conditioned matrix by prescribing a wide singular value range.  # EN: Explain matrix design.
    singular_values = np.logspace(0.0, -10.0, num=n, base=10.0)  # EN: Singular values from 1 to 1e-10.
    A = make_conditioned_matrix(rng=rng, m=m, n=n, singular_values=singular_values)  # EN: Build A.

    x_true = rng.standard_normal(n)  # EN: Choose a random ground-truth coefficient vector.
    b = A @ x_true + noise_std * rng.standard_normal(m)  # EN: Build noisy targets.

    # EN: Diagnostics (small n only): condition numbers for intuition.  # EN: Explain why we can compute cond here.
    cond_A = float(np.linalg.cond(A))  # EN: Compute cond(A).
    cond_ATA = float(np.linalg.cond(A.T @ A))  # EN: Compute cond(A^T A) to show squaring effect.

    print_separator("Problem Summary")  # EN: Print summary header.
    print(f"A shape: {m}×{n}, noise_std={noise_std:.1e}")  # EN: Print problem sizes and noise.
    print(f"cond(A)   ≈ {cond_A:.3e}")  # EN: Print condition number of A.
    print(f"cond(A^T A) ≈ {cond_ATA:.3e} (roughly cond(A)^2)")  # EN: Print condition number of normal matrix.

    max_iters = 400  # EN: Iteration cap for the Krylov methods.
    tol_normal = 1e-8  # EN: Stop tolerance for ||A^T r|| (LSQR) and ||g - Hx|| (CGNR).

    print_separator("LSQR: unpreconditioned vs column-scaled (right preconditioned)")  # EN: Announce LSQR comparison.
    lsqr_plain = solve_lsqr_dense(A=A, b=b, tol_normal=tol_normal, max_iters=max_iters)  # EN: Run LSQR on original A.

    D = column_scaling_D(A)  # EN: Build column scaling preconditioner D_j = ||A[:,j]||_2.
    lsqr_scaled = solve_lsqr_right_preconditioned(A=A, b=b, D=D, tol_normal=tol_normal, max_iters=max_iters)  # EN: Run LSQR on A D^{-1}.

    print("method              | iters | ||Ax-b||_2 | ||A^T r||_2")  # EN: Print table header.
    print("-" * 66)  # EN: Print table divider.
    print(f"LSQR                | {lsqr_plain.n_iters:5d} | {lsqr_plain.residual_norm:.3e} | {lsqr_plain.normal_residual_norm:.3e}")  # EN: Print LSQR row.
    print(f"LSQR + col scaling  | {lsqr_scaled.n_iters:5d} | {lsqr_scaled.residual_norm:.3e} | {lsqr_scaled.normal_residual_norm:.3e}")  # EN: Print preconditioned LSQR row.

    print_separator("Normal-equation baseline: CGNR vs PCGNR (Jacobi)")  # EN: Announce baseline comparison.
    cgnr = solve_cgnr_dense(A=A, b=b, tol=tol_normal, max_iters=max_iters, use_jacobi=False)  # EN: Solve normal equations with CG.
    pcgnr = solve_cgnr_dense(A=A, b=b, tol=tol_normal, max_iters=max_iters, use_jacobi=True)  # EN: Solve normal equations with PCG (Jacobi).

    # EN: Compute residuals for CG-based solutions in original LS sense.  # EN: Explain why we compute both norms.
    r_cg = A @ cgnr.x_hat - b  # EN: Residual for CGNR solution.
    r_pcg = A @ pcgnr.x_hat - b  # EN: Residual for PCGNR solution.
    nr_cg = A.T @ r_cg  # EN: Normal residual for CGNR.
    nr_pcg = A.T @ r_pcg  # EN: Normal residual for PCGNR.

    print("method              | iters | ||Ax-b||_2 | ||A^T r||_2")  # EN: Print header.
    print("-" * 66)  # EN: Print divider.
    print(f"{cgnr.method:19} | {cgnr.n_iters:5d} | {l2_norm(r_cg):.3e} | {l2_norm(nr_cg):.3e}")  # EN: Print CGNR row.
    print(f"{pcgnr.method:19} | {pcgnr.n_iters:5d} | {l2_norm(r_pcg):.3e} | {l2_norm(nr_pcg):.3e}")  # EN: Print PCGNR row.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main.

