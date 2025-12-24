"""  # EN: Start module docstring.
LSQR (manual): iterative least-squares solver without forming A^T A.  # EN: Describe the purpose of this script.

We solve the classic least-squares problem:  # EN: Introduce the optimization target.
  x_hat = argmin_x ||A x - b||_2.  # EN: State the objective.

Why LSQR (vs normal equation):  # EN: Explain motivation.
  - Normal equation uses (A^T A)x = A^T b, which roughly squares the condition number.  # EN: Mention cond(A^T A) ≈ cond(A)^2.
  - LSQR uses only matrix-vector products with A and A^T (no explicit A^T A).  # EN: Explain matvec-only property.
  - This is important for large-scale ML, where A may be huge or sparse.  # EN: Connect to real-world use cases.

Algorithm note: LSQR is based on Golub–Kahan bidiagonalization and is closely related to CG on the normal equations,  # EN: Provide algorithmic context.
but it is typically more numerically stable because it avoids forming A^T A explicitly.  # EN: Emphasize stability advantage.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass to store solver outputs.
from typing import Callable  # EN: Use Callable for typing matrix-vector product functions.

import numpy as np  # EN: Import NumPy for vector/matrix operations.


EPS = 1e-12  # EN: Small epsilon to avoid division by zero.
SEED = 0  # EN: RNG seed for deterministic demos.
PRINT_PRECISION = 6  # EN: Console float precision for readability.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy array printing.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias: function that multiplies a matrix by a vector.


@dataclass(frozen=True)  # EN: Immutable container for LSQR results.
class LSQRResult:  # EN: Store solution and basic convergence diagnostics.
    x_hat: np.ndarray  # EN: Estimated least-squares solution vector (n,).
    n_iters: int  # EN: Number of LSQR iterations performed.
    residual_norm: float  # EN: Final residual norm ||A x_hat - b||_2.
    normal_residual_norm: float  # EN: Final normal residual ||A^T(Ax_hat-b)||_2.
    history_residual_norm: np.ndarray  # EN: Residual-norm history over iterations (optional diagnostic).
    history_normal_residual_norm: np.ndarray  # EN: Normal-residual history (optional diagnostic).


def print_separator(title: str) -> None:  # EN: Print a section separator for console readability.
    print()  # EN: Add whitespace before the section.
    print("=" * 78)  # EN: Print a horizontal divider.
    print(title)  # EN: Print section title.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (vector 2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy's stable norm implementation.


def lsqr(  # EN: Solve min ||A x - b|| using the LSQR algorithm (matvec-only).
    matvec_A: Matvec,  # EN: Function computing A @ v for v in R^n.
    matvec_AT: Matvec,  # EN: Function computing A^T @ u for u in R^m.
    b: np.ndarray,  # EN: Right-hand side vector in R^m.
    n: int,  # EN: Number of unknowns (columns of A).
    max_iters: int = 200,  # EN: Maximum LSQR iterations.
    tol: float = 1e-10,  # EN: Stopping tolerance on ||A^T r||.
) -> LSQRResult:  # EN: Return LSQRResult with solution and diagnostics.
    if b.ndim != 1:  # EN: Validate b shape.
        raise ValueError("b must be a 1D vector")  # EN: Reject invalid input shape.
    if n <= 0:  # EN: Validate dimension n.
        raise ValueError("n must be positive")  # EN: Reject invalid dimension.
    if max_iters <= 0:  # EN: Validate iteration cap.
        raise ValueError("max_iters must be positive")  # EN: Reject invalid max_iters.

    m = b.size  # EN: Number of equations / rows of A.
    x = np.zeros((n,), dtype=float)  # EN: Start from zero solution (common default).

    # EN: Initialize Golub–Kahan bidiagonalization with u_1 = b / ||b||.
    u = b.copy()  # EN: Copy b into u.
    beta = l2_norm(u)  # EN: beta = ||b||.
    if beta < EPS:  # EN: Handle the trivial case b=0 (solution x=0).
        r0 = b - matvec_A(x)  # EN: Residual is b (since x=0).
        nr0 = matvec_AT(r0)  # EN: Normal residual A^T r.
        return LSQRResult(  # EN: Return early with zero solution.
            x_hat=x,  # EN: x=0 is optimal when b=0.
            n_iters=0,  # EN: No iterations needed.
            residual_norm=l2_norm(r0),  # EN: Residual norm.
            normal_residual_norm=l2_norm(nr0),  # EN: Normal residual norm.
            history_residual_norm=np.array([l2_norm(r0)], dtype=float),  # EN: History with one entry.
            history_normal_residual_norm=np.array([l2_norm(nr0)], dtype=float),  # EN: History with one entry.
        )  # EN: End early return.
    u = u / beta  # EN: Normalize u to unit norm.

    # EN: v_1 = A^T u_1 / ||A^T u_1||.
    v = matvec_AT(u)  # EN: Compute A^T u.
    alpha = l2_norm(v)  # EN: alpha = ||A^T u||.
    if alpha < EPS:  # EN: Handle the case where A^T u is zero (A is zero or b orthogonal to range).
        r0 = b - matvec_A(x)  # EN: Residual with x=0.
        nr0 = matvec_AT(r0)  # EN: Normal residual with x=0.
        return LSQRResult(  # EN: Return early.
            x_hat=x,  # EN: x=0 is a valid minimizer in this degenerate case.
            n_iters=0,  # EN: No iterations performed.
            residual_norm=l2_norm(r0),  # EN: Residual norm.
            normal_residual_norm=l2_norm(nr0),  # EN: Normal residual norm.
            history_residual_norm=np.array([l2_norm(r0)], dtype=float),  # EN: One-entry history.
            history_normal_residual_norm=np.array([l2_norm(nr0)], dtype=float),  # EN: One-entry history.
        )  # EN: End early return.
    v = v / alpha  # EN: Normalize v to unit norm.

    w = v.copy()  # EN: Initialize auxiliary vector w (search direction accumulator).

    # EN: Initialize the orthogonal transformation scalars.
    phi_bar = beta  # EN: phi_bar starts at beta and tracks residual-related quantity.
    rho_bar = alpha  # EN: rho_bar starts at alpha and tracks bidiagonal rotations.

    history_r: list[float] = []  # EN: Track ||r||_2 over iterations (computed explicitly for clarity).
    history_nr: list[float] = []  # EN: Track ||A^T r||_2 over iterations (normal residual).

    # EN: Record iteration 0 diagnostics using x=0.
    r = matvec_A(x) - b  # EN: r = Ax - b (note sign; norm is invariant).
    nr = matvec_AT(r)  # EN: Compute A^T r for normal residual.
    history_r.append(l2_norm(r))  # EN: Store ||r|| at iter 0.
    history_nr.append(l2_norm(nr))  # EN: Store ||A^T r|| at iter 0.

    for it in range(1, max_iters + 1):  # EN: Main LSQR iteration loop.
        # EN: Continue bidiagonalization: u_{k+1} = A v_k - alpha_k u_k.
        u = matvec_A(v) - alpha * u  # EN: Update u using current v and previous u.
        beta = l2_norm(u)  # EN: beta_{k+1} = ||u||.
        if beta > EPS:  # EN: Normalize u when beta is nonzero.
            u = u / beta  # EN: Normalize u to unit norm.
        else:  # EN: If beta is zero, we hit an invariant subspace (exact solution in Krylov space).
            beta = 0.0  # EN: Keep beta as 0 for safe downstream math.

        # EN: v_{k+1} = A^T u_{k+1} - beta_{k+1} v_k.
        v = matvec_AT(u) - beta * v  # EN: Update v.
        alpha = l2_norm(v)  # EN: alpha_{k+1} = ||v||.
        if alpha > EPS:  # EN: Normalize v when alpha is nonzero.
            v = v / alpha  # EN: Normalize v to unit norm.
        else:  # EN: alpha=0 indicates we cannot expand further (degenerate case).
            alpha = 0.0  # EN: Keep alpha at 0.

        # EN: Apply the next orthogonal transformation to the bidiagonal system.
        rho = float(np.sqrt(rho_bar * rho_bar + beta * beta))  # EN: rho = sqrt(rho_bar^2 + beta^2).
        if rho < EPS:  # EN: Guard against breakdown.
            break  # EN: Stop iterations if rho is too small.
        c = rho_bar / rho  # EN: c = rho_bar / rho.
        s = beta / rho  # EN: s = beta / rho.
        theta = s * alpha  # EN: theta = s * alpha.
        rho_bar = -c * alpha  # EN: rho_bar <- -c * alpha (Paige–Saunders convention).
        phi = c * phi_bar  # EN: phi = c * phi_bar.
        phi_bar = s * phi_bar  # EN: phi_bar <- s * phi_bar.

        # EN: Update x and w (the solution and direction accumulator).
        x = x + (phi / rho) * w  # EN: x_{k} = x_{k-1} + (phi/rho) w_{k-1}.
        w = v - (theta / rho) * w  # EN: w_{k} = v_{k} - (theta/rho) w_{k-1}.

        # EN: Compute explicit residual diagnostics (adds extra matvecs but keeps the demo clear).
        r = matvec_A(x) - b  # EN: Residual r = Ax - b.
        nr = matvec_AT(r)  # EN: Normal residual A^T r.
        r_norm = l2_norm(r)  # EN: ||r||_2.
        nr_norm = l2_norm(nr)  # EN: ||A^T r||_2.
        history_r.append(r_norm)  # EN: Store residual norm.
        history_nr.append(nr_norm)  # EN: Store normal residual norm.

        if nr_norm <= tol:  # EN: Stop when normal residual is small (first-order optimality).
            return LSQRResult(  # EN: Return converged result.
                x_hat=x,  # EN: Store solution.
                n_iters=it,  # EN: Store iterations used.
                residual_norm=r_norm,  # EN: Store final residual norm.
                normal_residual_norm=nr_norm,  # EN: Store final normal residual norm.
                history_residual_norm=np.array(history_r, dtype=float),  # EN: Store residual history.
                history_normal_residual_norm=np.array(history_nr, dtype=float),  # EN: Store normal residual history.
            )  # EN: End return.

        if beta == 0.0 and alpha == 0.0:  # EN: If bidiagonalization can no longer proceed, stop.
            break  # EN: Exit loop on breakdown.

    # EN: Return best-effort result after reaching max_iters or encountering breakdown.
    r = matvec_A(x) - b  # EN: Compute final residual.
    nr = matvec_AT(r)  # EN: Compute final normal residual.
    return LSQRResult(  # EN: Construct final result.
        x_hat=x,  # EN: Store solution.
        n_iters=len(history_r) - 1,  # EN: Iterations performed (history includes iter 0).
        residual_norm=l2_norm(r),  # EN: Residual norm.
        normal_residual_norm=l2_norm(nr),  # EN: Normal residual norm.
        history_residual_norm=np.array(history_r, dtype=float),  # EN: Residual history.
        history_normal_residual_norm=np.array(history_nr, dtype=float),  # EN: Normal residual history.
    )  # EN: End result construction.


def build_design_matrix_multicollinear(  # EN: Build a multicollinear design matrix to stress least-squares solvers.
    rng: np.random.Generator,  # EN: RNG for reproducibility.
    m: int,  # EN: Number of rows (samples).
    col_eps: float,  # EN: Collinearity strength (smaller -> more collinear).
) -> np.ndarray:  # EN: Return A with columns [1, x1, x2≈x1].
    x1 = rng.standard_normal(m)  # EN: First feature column.
    x2 = x1 + col_eps * rng.standard_normal(m)  # EN: Second feature nearly equals x1.
    A = np.column_stack([np.ones(m), x1, x2]).astype(float)  # EN: Build design matrix with intercept.
    return A  # EN: Return A.


def run_case(name: str, A: np.ndarray, b: np.ndarray, x_true: np.ndarray | None) -> None:  # EN: Run LSQR and compare to NumPy lstsq.
    m, n = A.shape  # EN: Extract shapes.
    print_separator(f"Case: {name}")  # EN: Print case header.
    cond_A = float(np.linalg.cond(A))  # EN: Condition number of A (diagnostic).
    print(f"A shape: {m}×{n}, cond(A)={cond_A:.3e}")  # EN: Print shape and cond.

    def matvec_A(v: np.ndarray) -> np.ndarray:  # EN: Dense matvec for A.
        return A @ v  # EN: Compute A v.

    def matvec_AT(u: np.ndarray) -> np.ndarray:  # EN: Dense matvec for A^T.
        return A.T @ u  # EN: Compute A^T u.

    lsqr_res = lsqr(matvec_A=matvec_A, matvec_AT=matvec_AT, b=b, n=n, max_iters=200, tol=1e-10)  # EN: Solve with LSQR.
    x_lstsq, *_ = np.linalg.lstsq(A, b, rcond=None)  # EN: Reference solution using NumPy's stable least-squares solver.

    r_lsqr = A @ lsqr_res.x_hat - b  # EN: Residual for LSQR solution.
    r_ref = A @ x_lstsq - b  # EN: Residual for reference solution.
    nr_lsqr = A.T @ r_lsqr  # EN: Normal residual for LSQR.
    nr_ref = A.T @ r_ref  # EN: Normal residual for reference.

    def rel_err(x: np.ndarray, x0: np.ndarray) -> float:  # EN: Relative error helper.
        return l2_norm(x - x0) / max(l2_norm(x0), EPS)  # EN: Return normalized error.

    print("LSQR iterations:", lsqr_res.n_iters)  # EN: Print iteration count.
    print(f"||Ax-b||_2   (LSQR) = {l2_norm(r_lsqr):.3e}")  # EN: Print LSQR residual norm.
    print(f"||A^T r||_2  (LSQR) = {l2_norm(nr_lsqr):.3e}")  # EN: Print LSQR normal residual norm.
    print(f"||Ax-b||_2   (lstsq)= {l2_norm(r_ref):.3e}")  # EN: Print reference residual norm.
    print(f"||A^T r||_2  (lstsq)= {l2_norm(nr_ref):.3e}")  # EN: Print reference normal residual norm.
    print(f"rel_err(x_lsqr, x_lstsq) = {rel_err(lsqr_res.x_hat, x_lstsq):.3e}")  # EN: Compare LSQR to reference solution.

    if x_true is not None:  # EN: Optionally compare to the true coefficients when available.
        print(f"rel_err(x_lstsq, x_true) = {rel_err(x_lstsq, x_true):.3e}")  # EN: Print reference relative error to x_true.
        print(f"rel_err(x_lsqr,  x_true) = {rel_err(lsqr_res.x_hat, x_true):.3e}")  # EN: Print LSQR relative error to x_true.

    print("\nSolutions (for inspection):")  # EN: Announce solution prints.
    print("x_lstsq =", x_lstsq)  # EN: Print reference solution.
    print("x_lsqr  =", lsqr_res.x_hat)  # EN: Print LSQR solution.


def main() -> None:  # EN: Run LSQR demo on well-conditioned and ill-conditioned least-squares problems.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    m = 200  # EN: Number of samples (rows).
    x_true = np.array([1.0, 2.0, -1.0], dtype=float)  # EN: Choose a fixed ground-truth parameter vector.
    noise_std = 1e-3  # EN: Noise level for observations.

    A_good = build_design_matrix_multicollinear(rng=rng, m=m, col_eps=1e0)  # EN: Less collinearity -> better conditioning.
    b_good = A_good @ x_true + noise_std * rng.standard_normal(m)  # EN: Build noisy targets for good case.

    A_bad = build_design_matrix_multicollinear(rng=rng, m=m, col_eps=1e-8)  # EN: Strong collinearity -> ill-conditioning.
    b_bad = A_bad @ x_true + noise_std * rng.standard_normal(m)  # EN: Build noisy targets for bad case.

    run_case(name="Well-conditioned-ish (weak collinearity)", A=A_good, b=b_good, x_true=x_true)  # EN: Run good case.
    run_case(name="Ill-conditioned (strong collinearity)", A=A_bad, b=b_bad, x_true=x_true)  # EN: Run bad case.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main.

