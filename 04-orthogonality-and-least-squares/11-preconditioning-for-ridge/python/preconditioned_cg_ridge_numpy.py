"""  # EN: Start module docstring.
Preconditioning for Ridge regression: compare CG vs PCG (Jacobi preconditioner) on an ill-conditioned system.  # EN: Describe what this script does.

We solve the Ridge normal equation system:  # EN: Explain the mathematical problem.
  H x = g, where H = A^T A + λ I and g = A^T b.  # EN: Define H and g.

CG is a strong baseline for SPD systems, but its convergence can still degrade when H is ill-conditioned.  # EN: Explain why CG may be slow.
PCG improves convergence by applying a preconditioner M ≈ H that is cheap to invert.  # EN: Explain what preconditioning does.

Here we use the simplest preconditioner (Jacobi / diagonal):  # EN: Describe the chosen preconditioner.
  M = diag(H), i.e., M_jj = ||A[:,j]||_2^2 + λ.  # EN: Provide exact formula for diagonal entries.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references for type annotations.

from dataclasses import dataclass  # EN: Use dataclass for a structured result record.

import numpy as np  # EN: Import NumPy for matrix computations.


EPS = 1e-12  # EN: Small epsilon to avoid division by zero.
SEED = 0  # EN: RNG seed for deterministic runs.
PRINT_PRECISION = 6  # EN: Printing precision for floating-point values.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing for readability.


@dataclass(frozen=True)  # EN: Immutable record for CG/PCG solver outputs.
class KrylovResult:  # EN: Store final solution, iterations, and residual history.
    method: str  # EN: Method label ("CG" or "PCG-Jacobi").
    x_hat: np.ndarray  # EN: Final solution vector.
    n_iters: int  # EN: Iterations performed.
    resid_norm_history: np.ndarray  # EN: History of ||r||_2 per iteration, including iteration 0.


def print_separator(title: str) -> None:  # EN: Print a simple separator to structure console output.
    print()  # EN: Add a blank line before the section.
    print("=" * 78)  # EN: Print horizontal divider.
    print(title)  # EN: Print section title.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Use NumPy's norm implementation.


def make_conditioned_matrix(  # EN: Build A with controlled singular values to create an ill-conditioned system.
    rng: np.random.Generator,  # EN: RNG for reproducibility.
    m: int,  # EN: Row count.
    n: int,  # EN: Column count.
    singular_values: np.ndarray,  # EN: Desired singular values (length n).
) -> np.ndarray:  # EN: Return A with approximately the given singular spectrum.
    G1 = rng.standard_normal((m, n))  # EN: Random Gaussian matrix for left orthonormal factor.
    Q1, _ = np.linalg.qr(G1, mode="reduced")  # EN: Orthonormalize columns to get m×n matrix.
    G2 = rng.standard_normal((n, n))  # EN: Random Gaussian matrix for right orthonormal factor.
    Q2, _ = np.linalg.qr(G2)  # EN: Orthonormalize to get n×n orthogonal matrix.
    S = np.diag(singular_values.astype(float))  # EN: Diagonal matrix of singular values.
    return Q1 @ S @ Q2.T  # EN: Construct A = Q1 S Q2^T.


def ridge_system_matvec(A: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:  # EN: Compute Hx with H=A^T A + λI without forming H.
    return A.T @ (A @ x) + lam * x  # EN: Return A^T(Ax) + λx.


def ridge_rhs(A: np.ndarray, b: np.ndarray) -> np.ndarray:  # EN: Compute g = A^T b for the normal equation.
    return A.T @ b  # EN: Return right-hand side vector.


def cg_spd(  # EN: Standard Conjugate Gradient for SPD systems Hx=g.
    matvec,  # EN: Callable computing H@v.
    g: np.ndarray,  # EN: Right-hand side.
    x0: np.ndarray,  # EN: Initial guess.
    max_iters: int,  # EN: Iteration cap.
    tol: float,  # EN: Stop when ||r|| <= tol.
) -> KrylovResult:  # EN: Return KrylovResult for CG.
    x = x0.copy()  # EN: Copy initial guess.
    r = g - matvec(x)  # EN: Initial residual r0 = g - Hx0.
    p = r.copy()  # EN: Initial search direction p0 = r0.
    rs_old = float(r @ r)  # EN: Compute r^T r for step size.
    resid_hist: list[float] = [float(np.sqrt(rs_old))]  # EN: Store ||r|| at iteration 0.

    if resid_hist[0] <= tol:  # EN: Allow early exit if already converged.
        return KrylovResult(method="CG", x_hat=x, n_iters=0, resid_norm_history=np.array(resid_hist, dtype=float))  # EN: Return.

    for it in range(1, max_iters + 1):  # EN: CG main loop.
        Hp = matvec(p)  # EN: Compute H p.
        denom = float(p @ Hp)  # EN: Compute p^T H p (positive for SPD).
        if abs(denom) < EPS:  # EN: Guard against numerical breakdown.
            break  # EN: Stop if denominator is too small.
        alpha = rs_old / denom  # EN: α = (r^T r)/(p^T H p).
        x = x + alpha * p  # EN: Update x.
        r = r - alpha * Hp  # EN: Update residual r.
        rs_new = float(r @ r)  # EN: Compute new r^T r.
        resid = float(np.sqrt(rs_new))  # EN: Compute ||r||.
        resid_hist.append(resid)  # EN: Record residual norm.
        if resid <= tol:  # EN: Stop when residual is below tolerance.
            return KrylovResult(method="CG", x_hat=x, n_iters=it, resid_norm_history=np.array(resid_hist, dtype=float))  # EN: Return.
        beta = rs_new / max(rs_old, EPS)  # EN: β = (r_{k+1}^T r_{k+1})/(r_k^T r_k).
        p = r + beta * p  # EN: Update search direction.
        rs_old = rs_new  # EN: Carry rs for next iteration.

    return KrylovResult(method="CG", x_hat=x, n_iters=len(resid_hist) - 1, resid_norm_history=np.array(resid_hist, dtype=float))  # EN: Return best effort.


def pcg_jacobi(  # EN: Preconditioned Conjugate Gradient using Jacobi (diagonal) preconditioner.
    matvec,  # EN: Callable computing H@v.
    g: np.ndarray,  # EN: Right-hand side.
    x0: np.ndarray,  # EN: Initial guess.
    inv_diag: np.ndarray,  # EN: Vector representing M^{-1} (inverse diagonal entries).
    max_iters: int,  # EN: Iteration cap.
    tol: float,  # EN: Stop when ||r|| <= tol.
) -> KrylovResult:  # EN: Return KrylovResult for PCG.
    x = x0.copy()  # EN: Copy initial guess.
    r = g - matvec(x)  # EN: Compute initial residual r0 = g - Hx0.
    z = inv_diag * r  # EN: Apply Jacobi preconditioner z0 = M^{-1} r0.
    p = z.copy()  # EN: Initial direction p0 = z0.
    rz_old = float(r @ z)  # EN: Use r^T z (preconditioned residual inner product).
    resid_hist: list[float] = [l2_norm(r)]  # EN: Track ||r|| at iteration 0 (unpreconditioned norm).

    if resid_hist[0] <= tol:  # EN: Early exit if already converged.
        return KrylovResult(method="PCG-Jacobi", x_hat=x, n_iters=0, resid_norm_history=np.array(resid_hist, dtype=float))  # EN: Return.

    for it in range(1, max_iters + 1):  # EN: PCG main loop.
        Hp = matvec(p)  # EN: Compute H p.
        denom = float(p @ Hp)  # EN: Compute p^T H p.
        if abs(denom) < EPS:  # EN: Guard against breakdown.
            break  # EN: Stop if denominator is too small.
        alpha = rz_old / denom  # EN: α = (r^T z)/(p^T H p).
        x = x + alpha * p  # EN: Update x.
        r = r - alpha * Hp  # EN: Update residual r.
        resid = l2_norm(r)  # EN: Compute ||r|| for stopping and diagnostics.
        resid_hist.append(resid)  # EN: Record residual norm.
        if resid <= tol:  # EN: Stop when residual is below tolerance.
            return KrylovResult(method="PCG-Jacobi", x_hat=x, n_iters=it, resid_norm_history=np.array(resid_hist, dtype=float))  # EN: Return.
        z = inv_diag * r  # EN: Apply preconditioner z = M^{-1} r.
        rz_new = float(r @ z)  # EN: Compute new r^T z.
        beta = rz_new / max(rz_old, EPS)  # EN: β = (r_{k+1}^T z_{k+1})/(r_k^T z_k).
        p = z + beta * p  # EN: Update search direction p.
        rz_old = rz_new  # EN: Carry r^T z for next iteration.

    return KrylovResult(method="PCG-Jacobi", x_hat=x, n_iters=len(resid_hist) - 1, resid_norm_history=np.array(resid_hist, dtype=float))  # EN: Return best effort.


def main() -> None:  # EN: Build an ill-conditioned ridge problem and compare CG vs PCG.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    m = 250  # EN: Number of rows (samples).
    n = 60  # EN: Number of columns (features).
    lam = 1e-2  # EN: Ridge parameter (λ>0 makes H SPD).
    noise_std = 1e-3  # EN: Small observation noise.

    s_bad = np.logspace(0.0, -8.0, num=n, base=10.0)  # EN: Strongly decaying singular values -> very ill-conditioned A.
    A = make_conditioned_matrix(rng=rng, m=m, n=n, singular_values=s_bad)  # EN: Build design matrix A.
    x_true = rng.standard_normal(n)  # EN: Choose a random ground-truth coefficient vector.
    b = A @ x_true + noise_std * rng.standard_normal(m)  # EN: Build noisy targets.

    H_cond = float(np.linalg.cond(A.T @ A + lam * np.eye(n)))  # EN: Compute condition number of H for diagnostics (small n only).
    cond_A = float(np.linalg.cond(A))  # EN: Compute condition number of A.

    print_separator("Problem Summary")  # EN: Announce problem summary.
    print(f"A shape: {m}×{n}, λ={lam:.3e}")  # EN: Print sizes and λ.
    print(f"cond(A) = {cond_A:.3e}")  # EN: Print cond(A).
    print(f"cond(H) = cond(A^T A + λI) ≈ {H_cond:.3e}")  # EN: Print cond(H) for understanding convergence.

    g = ridge_rhs(A, b)  # EN: Compute g = A^T b.

    def matvec(v: np.ndarray) -> np.ndarray:  # EN: Closure computing H@v.
        return ridge_system_matvec(A, v, lam)  # EN: Use implicit matvec.

    diag_H = np.sum(A * A, axis=0) + lam  # EN: Compute diagonal entries of H: diag(A^T A) + λ.
    inv_diag = 1.0 / np.maximum(diag_H, EPS)  # EN: Build inverse diagonal for Jacobi preconditioner.

    max_iters = n * 2  # EN: Allow up to 2n iterations for a fair comparison under floating-point arithmetic.
    tol = 1e-8  # EN: Residual tolerance for stopping.
    x0 = np.zeros((n,), dtype=float)  # EN: Start from zero vector.

    cg_res = cg_spd(matvec=matvec, g=g, x0=x0, max_iters=max_iters, tol=tol)  # EN: Run plain CG.
    pcg_res = pcg_jacobi(matvec=matvec, g=g, x0=x0, inv_diag=inv_diag, max_iters=max_iters, tol=tol)  # EN: Run PCG with Jacobi preconditioner.

    def last(hist: np.ndarray) -> float:  # EN: Helper to safely get last element.
        return float(hist[-1]) if hist.size else float("nan")  # EN: Return last element or NaN.

    print_separator("Convergence Summary")  # EN: Print convergence summary.
    print("method      | iters | final ||r||_2")  # EN: Print table header.
    print("-" * 36)  # EN: Print table separator.
    print(f"{cg_res.method:11} | {cg_res.n_iters:5d} | {last(cg_res.resid_norm_history):.3e}")  # EN: Print CG result row.
    print(f"{pcg_res.method:11} | {pcg_res.n_iters:5d} | {last(pcg_res.resid_norm_history):.3e}")  # EN: Print PCG result row.

    print_separator("Selected Residual Checkpoints")  # EN: Show a few checkpoints for both methods.
    checkpoints = [0, 1, 2, 5, 10, 20, 40, 80]  # EN: Pick checkpoint iteration numbers.
    for c in checkpoints:  # EN: Print checkpoint info.
        cg_val = cg_res.resid_norm_history[c] if c < cg_res.resid_norm_history.size else None  # EN: Get CG value if in range.
        pcg_val = pcg_res.resid_norm_history[c] if c < pcg_res.resid_norm_history.size else None  # EN: Get PCG value if in range.
        cg_str = "n/a" if cg_val is None else f"{float(cg_val):.3e}"  # EN: Format CG value.
        pcg_str = "n/a" if pcg_val is None else f"{float(pcg_val):.3e}"  # EN: Format PCG value.
        print(f"iter {c:3d}: CG ||r||={cg_str:>11} | PCG ||r||={pcg_str:>11}")  # EN: Print both residuals at checkpoint.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Run main when executed as a script.

