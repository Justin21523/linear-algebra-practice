"""  # EN: Start module docstring.
Iterative solvers for Ridge regression (NumPy): Gradient Descent (GD) vs Conjugate Gradient (CG).  # EN: Describe what the script demonstrates.

Why this matters in ML / numerical linear algebra:  # EN: Provide motivation.
  - For large problems you rarely form (A^T A) explicitly or invert matrices.  # EN: Explain practical constraints.
  - You solve using *iterative methods* that use only matrix-vector products.  # EN: Highlight the iterative approach.
  - Convergence speed depends heavily on conditioning (multicollinearity / ill-conditioning).  # EN: Connect to conditioning.

We solve Ridge regression in two equivalent forms:  # EN: Explain the two formulations.
  (1) Optimization:  min f(x) = 1/2||Ax-b||^2 + 1/2*λ||x||^2  # EN: State objective.
  (2) Linear system: (A^T A + λI)x = A^T b  (SPD when λ>0)  # EN: State normal equation for Ridge.

We compare:  # EN: Introduce comparison items.
  - GD on the objective using a step size based on the Lipschitz constant L = ||A||_2^2 + λ.  # EN: Describe GD step size rule.
  - CG on the SPD system using only matvec(x) = A^T(Ax) + λx.  # EN: Describe CG approach.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured results.

import numpy as np  # EN: Import NumPy for numerical computation.


EPS = 1e-12  # EN: Small epsilon for safe divisions.
RCOND = 1e-12  # EN: Relative cutoff for SVD-based pseudo-inverse computations.
SEED = 0  # EN: RNG seed for reproducible demos.
PRINT_PRECISION = 6  # EN: Print precision for console outputs.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure array printing.


@dataclass(frozen=True)  # EN: Immutable container for optimization run history.
class IterativeResult:  # EN: Store final solution and convergence diagnostics.
    method: str  # EN: Method name (GD / CG).
    x_hat: np.ndarray  # EN: Final coefficient vector.
    n_iters: int  # EN: Number of iterations executed.
    final_obj: float  # EN: Final objective value f(x_hat).
    final_grad_norm: float  # EN: Final gradient norm ||∇f|| for optimization diagnostics.
    final_sys_resid_norm: float  # EN: Final system residual ||(A^T A + λI)x - A^T b||.
    history_obj: np.ndarray  # EN: Objective history over iterations (length n_iters+1).
    history_metric: np.ndarray  # EN: Per-iteration primary metric history (e.g., grad norm or r norm).


def print_separator(title: str) -> None:  # EN: Print a readable separator line.
    print()  # EN: Add whitespace before a section.
    print("=" * 78)  # EN: Print a horizontal divider.
    print(title)  # EN: Print section title.
    print("=" * 78)  # EN: Print a closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (vector 2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Use NumPy's norm implementation.


def ridge_objective(A: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> float:  # EN: Compute ridge objective f(x).
    r = A @ x - b  # EN: Compute residual vector Ax-b.
    return 0.5 * float(r @ r) + 0.5 * lam * float(x @ x)  # EN: Return 1/2||r||^2 + 1/2 λ||x||^2.


def ridge_gradient(A: np.ndarray, b: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:  # EN: Compute ∇f(x) for ridge objective.
    return A.T @ (A @ x - b) + lam * x  # EN: Gradient is A^T(Ax-b) + λx.


def ridge_system_matvec(A: np.ndarray, x: np.ndarray, lam: float) -> np.ndarray:  # EN: Compute (A^T A + λI)x without forming A^T A.
    return A.T @ (A @ x) + lam * x  # EN: Return A^T(Ax) + λx.


def ridge_rhs(A: np.ndarray, b: np.ndarray) -> np.ndarray:  # EN: Compute right-hand side g = A^T b for ridge normal equation.
    return A.T @ b  # EN: Return A^T b.


def solve_ridge_closed_form_svd(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:  # EN: Solve ridge exactly via SVD filter factors.
    if lam < 0.0:  # EN: Validate λ is non-negative.
        raise ValueError("lam must be non-negative")  # EN: Reject invalid λ.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD A = U diag(s) V^T.
    if s.size == 0:  # EN: Handle degenerate shapes defensively.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return zeros for empty problem.
    if lam == 0.0:  # EN: For λ=0, return pseudo-inverse least squares solution.
        cutoff = RCOND * float(s.max())  # EN: Compute cutoff for small singular values.
        keep = s > cutoff  # EN: Identify numerically nonzero singular values.
        if not np.any(keep):  # EN: If rank is 0, the min-norm solution is zeros.
            return np.zeros((A.shape[1],), dtype=float)  # EN: Return zeros.
        U_r = U[:, keep]  # EN: Select kept U columns.
        s_r = s[keep]  # EN: Select kept singular values.
        Vt_r = Vt[keep, :]  # EN: Select kept V^T rows.
        return Vt_r.T @ ((U_r.T @ b) / s_r)  # EN: x = V diag(1/s) U^T b.
    factors = s / (s**2 + lam)  # EN: Ridge filter factors σ/(σ^2+λ).
    return Vt.T @ (factors * (U.T @ b))  # EN: x = V diag(factors) U^T b implemented efficiently.


def gradient_descent_ridge(  # EN: Run gradient descent on the ridge objective.
    A: np.ndarray,  # EN: Design matrix.
    b: np.ndarray,  # EN: Target vector.
    lam: float,  # EN: Ridge regularization parameter.
    step_size: float,  # EN: GD step size α.
    max_iters: int,  # EN: Maximum iterations.
    tol_grad: float,  # EN: Stop when ||∇f|| <= tol_grad.
) -> IterativeResult:  # EN: Return IterativeResult with history.
    n = A.shape[1]  # EN: Number of parameters.
    x = np.zeros((n,), dtype=float)  # EN: Initialize x at zeros (common baseline).

    history_obj: list[float] = []  # EN: Collect objective values per iteration.
    history_grad: list[float] = []  # EN: Collect gradient norms per iteration.

    for it in range(max_iters + 1):  # EN: Iterate (including iteration 0 for initial diagnostics).
        obj = ridge_objective(A, b, x, lam)  # EN: Compute current objective.
        g = ridge_gradient(A, b, x, lam)  # EN: Compute current gradient.
        g_norm = l2_norm(g)  # EN: Compute gradient norm for stopping condition.

        history_obj.append(obj)  # EN: Store objective value.
        history_grad.append(g_norm)  # EN: Store gradient norm.

        if g_norm <= tol_grad:  # EN: Stop early when gradient is small enough.
            break  # EN: Exit loop on convergence.

        if it == max_iters:  # EN: Avoid stepping after recording the final iteration.
            break  # EN: Exit loop when reaching max iterations.

        x = x - step_size * g  # EN: GD update: x_{t+1} = x_t - α ∇f(x_t).

    g_final = ridge_gradient(A, b, x, lam)  # EN: Compute final gradient for report.
    sys_resid = ridge_system_matvec(A, x, lam) - ridge_rhs(A, b)  # EN: Compute normal-equation residual.

    return IterativeResult(  # EN: Construct result record.
        method="GD",  # EN: Method label.
        x_hat=x,  # EN: Final solution.
        n_iters=len(history_obj) - 1,  # EN: Number of completed updates (history includes iteration 0).
        final_obj=history_obj[-1],  # EN: Final objective value.
        final_grad_norm=l2_norm(g_final),  # EN: Final gradient norm.
        final_sys_resid_norm=l2_norm(sys_resid),  # EN: Final system residual norm.
        history_obj=np.array(history_obj, dtype=float),  # EN: Convert objective history to ndarray.
        history_metric=np.array(history_grad, dtype=float),  # EN: Convert gradient history to ndarray.
    )  # EN: End result construction.


def conjugate_gradient_spd(  # EN: Run Conjugate Gradient on Hx=g where H is SPD.
    matvec,  # EN: Callable computing H @ v.
    g: np.ndarray,  # EN: Right-hand side vector.
    x0: np.ndarray,  # EN: Initial guess.
    max_iters: int,  # EN: Maximum CG iterations.
    tol_resid: float,  # EN: Stop when ||r|| <= tol_resid where r=g-Hx.
) -> tuple[np.ndarray, int, np.ndarray]:  # EN: Return (x, n_iters, resid_norm_history).
    x = x0.copy()  # EN: Copy initial guess to avoid mutating caller's array.
    r = g - matvec(x)  # EN: Compute initial residual r0 = g - Hx0.
    p = r.copy()  # EN: Initialize search direction p0 = r0.
    rs_old = float(r @ r)  # EN: Compute squared residual norm r^T r.

    resid_hist: list[float] = [float(np.sqrt(rs_old))]  # EN: Track ||r||_2 per iteration (include iter 0).
    if resid_hist[0] <= tol_resid:  # EN: Allow immediate convergence if x0 already solves the system.
        return x, 0, np.array(resid_hist, dtype=float)  # EN: Return early with 0 iterations.

    for it in range(1, max_iters + 1):  # EN: Run up to max_iters CG updates.
        Hp = matvec(p)  # EN: Compute H p_k.
        denom = float(p @ Hp)  # EN: Compute denominator p^T H p (positive for SPD).
        if abs(denom) < EPS:  # EN: Guard against numerical breakdown.
            break  # EN: Exit if denominator is too small.
        alpha = rs_old / denom  # EN: Step size α_k = (r^T r)/(p^T H p).
        x = x + alpha * p  # EN: Update solution estimate x_{k+1} = x_k + α p_k.
        r = r - alpha * Hp  # EN: Update residual r_{k+1} = r_k - α H p_k.
        rs_new = float(r @ r)  # EN: Compute new squared residual norm.
        resid_norm = float(np.sqrt(rs_new))  # EN: Compute residual norm ||r||_2.
        resid_hist.append(resid_norm)  # EN: Record residual norm for diagnostics.
        if resid_norm <= tol_resid:  # EN: Stop when residual is below tolerance.
            return x, it, np.array(resid_hist, dtype=float)  # EN: Return converged solution and history.
        beta = rs_new / max(rs_old, EPS)  # EN: Compute β_k = (r_{k+1}^T r_{k+1})/(r_k^T r_k).
        p = r + beta * p  # EN: Update direction p_{k+1} = r_{k+1} + β p_k.
        rs_old = rs_new  # EN: Update stored residual norm for next iteration.

    return x, len(resid_hist) - 1, np.array(resid_hist, dtype=float)  # EN: Return best effort solution and history.


def cg_ridge(  # EN: Convenience wrapper: run CG to solve Ridge normal equation without forming A^T A.
    A: np.ndarray,  # EN: Design matrix.
    b: np.ndarray,  # EN: Target vector.
    lam: float,  # EN: Ridge λ (>0 recommended for SPD).
    max_iters: int,  # EN: Maximum CG iterations.
    tol_resid: float,  # EN: Residual tolerance for stopping.
) -> IterativeResult:  # EN: Return IterativeResult with objective and residual history.
    n = A.shape[1]  # EN: Number of unknowns.
    g = ridge_rhs(A, b)  # EN: Compute g = A^T b.

    def matvec(v: np.ndarray) -> np.ndarray:  # EN: Closure to compute H v for H = A^T A + λI.
        return ridge_system_matvec(A, v, lam)  # EN: Use matvec form to avoid explicit A^T A.

    x0 = np.zeros((n,), dtype=float)  # EN: Start from zero vector (common choice).
    x, n_iters, resid_hist = conjugate_gradient_spd(  # EN: Run CG on the SPD system.
        matvec=matvec,  # EN: Provide matvec callback.
        g=g,  # EN: Provide right-hand side.
        x0=x0,  # EN: Provide initial guess.
        max_iters=max_iters,  # EN: Provide iteration cap.
        tol_resid=tol_resid,  # EN: Provide stopping tolerance.
    )  # EN: End CG call.

    obj_hist: list[float] = []  # EN: Build objective history aligned with residual history.
    for _ in range(len(resid_hist)):  # EN: For simplicity, compute objective at the final x for each history step.
        obj_hist.append(ridge_objective(A, b, x, lam))  # EN: Use final x for a single summary objective value.

    g_final = ridge_gradient(A, b, x, lam)  # EN: Compute gradient at final x for comparison to GD.
    sys_resid = ridge_system_matvec(A, x, lam) - g  # EN: Compute system residual at final x (should match last CG residual).

    return IterativeResult(  # EN: Construct CG result record.
        method="CG",  # EN: Method label.
        x_hat=x,  # EN: Final solution.
        n_iters=n_iters,  # EN: Iterations performed.
        final_obj=ridge_objective(A, b, x, lam),  # EN: Final objective value.
        final_grad_norm=l2_norm(g_final),  # EN: Final gradient norm.
        final_sys_resid_norm=l2_norm(sys_resid),  # EN: Final system residual norm.
        history_obj=np.array(obj_hist, dtype=float),  # EN: Objective history placeholder (constant, but present for uniformity).
        history_metric=np.array(resid_hist, dtype=float),  # EN: Use residual norm history as the primary metric.
    )  # EN: End result construction.


def make_conditioned_matrix(  # EN: Build a matrix with controlled singular values to control conditioning.
    rng: np.random.Generator,  # EN: RNG for generating random orthonormal factors.
    m: int,  # EN: Number of rows.
    n: int,  # EN: Number of columns.
    singular_values: np.ndarray,  # EN: Desired singular values (length n, descending recommended).
) -> np.ndarray:  # EN: Return A with approximate singular spectrum.
    G1 = rng.standard_normal((m, n))  # EN: Random Gaussian matrix for left factor.
    Q1, _ = np.linalg.qr(G1, mode="reduced")  # EN: Orthonormalize to get m×n matrix with orthonormal columns.
    G2 = rng.standard_normal((n, n))  # EN: Random Gaussian matrix for right factor.
    Q2, _ = np.linalg.qr(G2)  # EN: Orthonormalize to get n×n orthogonal matrix.
    S = np.diag(singular_values.astype(float))  # EN: Build diagonal matrix of singular values.
    return Q1 @ S @ Q2.T  # EN: Construct A = Q1 S Q2^T with specified singular values.


def run_case(  # EN: Run GD/CG on one synthetic ridge problem and print comparisons.
    name: str,  # EN: Case name label.
    A: np.ndarray,  # EN: Design matrix.
    b: np.ndarray,  # EN: Target vector.
    lam: float,  # EN: Ridge λ.
    x_star: np.ndarray,  # EN: Reference solution (closed-form).
) -> None:  # EN: Print results; return nothing.
    print_separator(f"Case: {name}")  # EN: Announce case.
    cond_A = float(np.linalg.cond(A))  # EN: Compute condition number of A.
    s = np.linalg.svd(A, compute_uv=False)  # EN: Compute singular values for step-size and diagnostics.
    sigma_max = float(s.max())  # EN: Largest singular value (spectral norm).
    L = sigma_max**2 + lam  # EN: Lipschitz constant for ridge gradient (||A||_2^2 + λ).
    step_size = 1.0 / L  # EN: Safe GD step size for convergence on a convex quadratic.
    print(f"Shape: A is {A.shape[0]}×{A.shape[1]}, λ={lam:.3e}")  # EN: Print shape and λ.
    print(f"cond(A) = {cond_A:.3e}, sigma_max = {sigma_max:.3e}, L = {L:.3e}, step = 1/L = {step_size:.3e}")  # EN: Print conditioning + step.

    max_gd_iters = 5000  # EN: Cap GD iterations so the demo finishes quickly.
    max_cg_iters = A.shape[1]  # EN: CG theoretically converges in at most n steps (exact arithmetic).
    tol_grad = 1e-8  # EN: Gradient tolerance for GD stopping.
    tol_resid = 1e-8  # EN: Residual tolerance for CG stopping.

    gd = gradient_descent_ridge(  # EN: Run gradient descent solver.
        A=A,  # EN: Provide A.
        b=b,  # EN: Provide b.
        lam=lam,  # EN: Provide λ.
        step_size=step_size,  # EN: Provide step size.
        max_iters=max_gd_iters,  # EN: Provide iteration cap.
        tol_grad=tol_grad,  # EN: Provide tolerance.
    )  # EN: End GD call.

    cg = cg_ridge(  # EN: Run conjugate gradient solver.
        A=A,  # EN: Provide A.
        b=b,  # EN: Provide b.
        lam=lam,  # EN: Provide λ.
        max_iters=max_cg_iters,  # EN: Provide iteration cap.
        tol_resid=tol_resid,  # EN: Provide tolerance.
    )  # EN: End CG call.

    def rel_error(x: np.ndarray) -> float:  # EN: Compute relative error to x_star.
        return l2_norm(x - x_star) / max(l2_norm(x_star), EPS)  # EN: Return ||x-x*||/||x*|| with safe denom.

    print_separator("Summary (vs closed-form SVD ridge solution)")  # EN: Print summary header.
    print(f"x_star norm ||x*|| = {l2_norm(x_star):.3e}")  # EN: Print reference solution norm.
    print(  # EN: Print a compact comparison table header.
        "method | iters | rel_err_to_x* | final_obj | ||grad|| | ||system_resid|| | metric_history_last"  # EN: Column names.
    )  # EN: End header print.
    print("-" * 94)  # EN: Print separator.

    print(  # EN: Print GD row.
        f"{gd.method:6} | {gd.n_iters:5d} | {rel_error(gd.x_hat):13.3e} | "
        f"{gd.final_obj:9.3e} | {gd.final_grad_norm:8.3e} | {gd.final_sys_resid_norm:13.3e} | {gd.history_metric[-1]:.3e}"
    )  # EN: End GD row.

    print(  # EN: Print CG row.
        f"{cg.method:6} | {cg.n_iters:5d} | {rel_error(cg.x_hat):13.3e} | "
        f"{cg.final_obj:9.3e} | {cg.final_grad_norm:8.3e} | {cg.final_sys_resid_norm:13.3e} | {cg.history_metric[-1]:.3e}"
    )  # EN: End CG row.

    print_separator("Selected GD checkpoints (objective / ||grad||)")  # EN: Print GD checkpoints.
    checkpoints = [0, 1, 10, 100, 1000, gd.n_iters]  # EN: Choose checkpoint iterations to display.
    checkpoints = [c for c in checkpoints if 0 <= c < len(gd.history_obj)]  # EN: Clamp checkpoints to valid indices.
    for c in checkpoints:  # EN: Print each checkpoint.
        print(f"iter {c:5d}: f={gd.history_obj[c]:.3e}, ||grad||={gd.history_metric[c]:.3e}")  # EN: Print checkpoint metrics.

    print_separator("Selected CG checkpoints (||system residual||)")  # EN: Print CG checkpoints.
    cg_checkpoints = [0, 1, 2, min(5, cg.n_iters), cg.n_iters]  # EN: Choose a few CG checkpoints.
    cg_checkpoints = sorted({c for c in cg_checkpoints if 0 <= c < len(cg.history_metric)})  # EN: Deduplicate and clamp.
    for c in cg_checkpoints:  # EN: Print checkpoint residual norms.
        print(f"iter {c:3d}: ||r||={cg.history_metric[c]:.3e}")  # EN: Print CG residual norm.


def main() -> None:  # EN: Run well-conditioned and ill-conditioned ridge problems.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    m = 200  # EN: Number of samples / equations (rows).
    n = 40  # EN: Number of parameters / features (columns).
    noise_std = 1e-3  # EN: Small noise for targets.
    lam = 1e-2  # EN: Ridge regularization strength (λ>0 ensures SPD system).

    x_true = rng.standard_normal(n)  # EN: Draw a random ground-truth coefficient vector.

    s_good = np.ones(n, dtype=float)  # EN: Flat singular values -> well-conditioned.
    A_good = make_conditioned_matrix(rng=rng, m=m, n=n, singular_values=s_good)  # EN: Build well-conditioned A.
    b_good = A_good @ x_true + noise_std * rng.standard_normal(m)  # EN: Build noisy targets.
    x_star_good = solve_ridge_closed_form_svd(A_good, b_good, lam=lam)  # EN: Reference ridge solution via SVD.

    s_bad = np.logspace(0.0, -6.0, num=n, base=10.0)  # EN: Exponentially decaying singular values -> ill-conditioned.
    A_bad = make_conditioned_matrix(rng=rng, m=m, n=n, singular_values=s_bad)  # EN: Build ill-conditioned A.
    b_bad = A_bad @ x_true + noise_std * rng.standard_normal(m)  # EN: Build noisy targets.
    x_star_bad = solve_ridge_closed_form_svd(A_bad, b_bad, lam=lam)  # EN: Reference ridge solution via SVD.

    run_case(name="Well-conditioned spectrum", A=A_good, b=b_good, lam=lam, x_star=x_star_good)  # EN: Run solvers on good case.
    run_case(name="Ill-conditioned spectrum", A=A_bad, b=b_bad, lam=lam, x_star=x_star_bad)  # EN: Run solvers on bad case.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard script entrypoint guard.
    main()  # EN: Execute the demo.

