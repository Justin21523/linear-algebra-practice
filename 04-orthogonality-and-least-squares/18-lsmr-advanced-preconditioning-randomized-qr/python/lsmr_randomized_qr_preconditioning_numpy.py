"""  # EN: Start module docstring.
Advanced preconditioning for LSMR/LSQR: randomized sketch + QR (Blendenpik/LSRN-style right preconditioner).  # EN: Summarize the purpose.

We solve least squares:  # EN: State the base problem.
  min_x ||A x - b||_2.  # EN: Objective.

Right preconditioning:  # EN: Introduce right preconditioning.
  Let x = M^{-1} y, then solve min_y ||A M^{-1} y - b||_2, and map back x = M^{-1} y.  # EN: Show the transformation.

Preconditioners compared:  # EN: Describe the three preconditioners we compare.
  1) None (baseline)  # EN: No preconditioning.
  2) Column scaling D = diag(||A[:,j]||_2)  # EN: Simple diagonal scaling.
  3) Randomized QR: build R from QR(SA), where S is an oversampled random sketch matrix  # EN: Sketch-and-QR idea.

Why randomized QR helps:  # EN: Explain the intuition.
  If R ≈ "well-scaled" factor of A, then A R^{-1} tends to have a much better condition number,  # EN: Conditioning statement.
  so Krylov solvers (LSMR/LSQR) converge in far fewer iterations.  # EN: Convergence benefit.

This demo uses dense NumPy for clarity; the same operator-wrapping approach generalizes to sparse/matvec-only settings.  # EN: Note about generalization.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured results.
from time import perf_counter  # EN: Use perf_counter for rough timing.
from typing import Callable  # EN: Use Callable for matvec typing.

import numpy as np  # EN: Import NumPy for numerical linear algebra.


EPS = 1e-12  # EN: Small epsilon for safe divisions.
SEED = 0  # EN: RNG seed for reproducible demos.
PRINT_PRECISION = 6  # EN: Console float precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias for a matrix-vector product function.


@dataclass(frozen=True)  # EN: Immutable record for one solver run.
class SolveReport:  # EN: Store solution and diagnostics in original x-space.
    label: str  # EN: Human-readable label for the method/config.
    n_iters: int  # EN: Iterations performed by the Krylov solver.
    seconds_build: float  # EN: Seconds spent building the preconditioner (0 for none).
    seconds_solve: float  # EN: Seconds spent in the iterative solve.
    rnorm: float  # EN: ||Ax-b||_2 for the mapped-back x.
    arnorm: float  # EN: ||A^T(Ax-b)||_2 for the mapped-back x.
    xnorm: float  # EN: ||x||_2 for the mapped-back x.


def print_separator(title: str) -> None:  # EN: Print a section separator for readable console output.
    print()  # EN: Add a blank line.
    print("=" * 78)  # EN: Print divider.
    print(title)  # EN: Print title.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def build_tridiagonal_T(  # EN: Build T_k = B_k^T B_k from alpha[1..k], beta[2..k+1].
    alphas: np.ndarray,  # EN: Alpha sequence (length >= k).
    betas: np.ndarray,  # EN: Beta sequence including beta_1 (length >= k+1).
    k: int,  # EN: Tridiagonal size.
) -> np.ndarray:  # EN: Return dense k×k tridiagonal matrix.
    diag = (alphas[:k] ** 2) + (betas[1 : k + 1] ** 2)  # EN: diag_i = alpha_i^2 + beta_{i+1}^2.
    off = alphas[1:k] * betas[1:k]  # EN: off_i = alpha_{i+1} * beta_{i+1}.
    T = np.diag(diag.astype(float))  # EN: Initialize diagonal.
    if k > 1:  # EN: Add off-diagonals for k>=2.
        T = T + np.diag(off.astype(float), 1) + np.diag(off.astype(float), -1)  # EN: Add symmetric off-diagonals.
    return T  # EN: Return T.


def lsmr_teaching_minres(  # EN: Teaching LSMR: MINRES on normal equations via small LS solve each iteration.
    matvec_A: Matvec,  # EN: A matvec.
    matvec_AT: Matvec,  # EN: A^T matvec.
    b: np.ndarray,  # EN: RHS vector (m,).
    n: int,  # EN: Unknown dimension.
    max_iters: int,  # EN: Iteration cap.
    tol_rel_arnorm: float,  # EN: Relative tolerance on ||A^T(Ax-b)||.
) -> tuple[np.ndarray, int]:  # EN: Return (y_hat, n_iters) in the solver's coordinates.
    x = np.zeros((n,), dtype=float)  # EN: Initialize x0=0.

    u = b.copy()  # EN: Initialize u from b.
    beta1 = l2_norm(u)  # EN: beta1 = ||b||.
    if beta1 < EPS:  # EN: Handle b=0.
        return x, 0  # EN: Return y=0 with zero iterations.
    u = u / beta1  # EN: Normalize u1.

    v = matvec_AT(u)  # EN: v1 = A^T u1.
    alpha1 = l2_norm(v)  # EN: alpha1 = ||v1||.
    if alpha1 < EPS:  # EN: Handle A^T b = 0.
        return x, 0  # EN: Return y=0 with zero iterations.
    v = v / alpha1  # EN: Normalize v1.

    V_basis = np.zeros((n, max_iters + 1), dtype=float)  # EN: Store v basis vectors for reconstruction.
    V_basis[:, 0] = v  # EN: Store v1.

    alphas: list[float] = [float(alpha1)]  # EN: Store alpha_1.
    betas: list[float] = [float(beta1)]  # EN: Store beta_1 (beta_{k+1} appended each iteration).

    g_norm = float(alpha1 * beta1)  # EN: ||A^T b|| = alpha1*beta1.
    if g_norm < EPS:  # EN: Defensive fallback for degenerate case.
        g_norm = float(l2_norm(matvec_AT(b)))  # EN: Compute directly.

    y_hat = np.zeros((n,), dtype=float)  # EN: Placeholder for the current y estimate.
    n_done = 0  # EN: Track iterations completed.

    for k in range(1, max_iters + 1):  # EN: Expand subspace to dimension k.
        u_next = matvec_A(v) - alphas[-1] * u  # EN: u_{k+1} = A v_k - alpha_k u_k.
        beta_next = l2_norm(u_next)  # EN: beta_{k+1}.
        if beta_next >= EPS:  # EN: Normalize when possible.
            u_next = u_next / beta_next  # EN: Normalize u_{k+1}.

        v_next = matvec_AT(u_next) - beta_next * v  # EN: v_{k+1} = A^T u_{k+1} - beta_{k+1} v_k.
        alpha_next = l2_norm(v_next)  # EN: alpha_{k+1}.
        if alpha_next >= EPS:  # EN: Normalize when possible.
            v_next = v_next / alpha_next  # EN: Normalize v_{k+1}.

        betas.append(float(beta_next))  # EN: Append beta_{k+1}.
        alphas.append(float(alpha_next))  # EN: Append alpha_{k+1}.
        V_basis[:, k] = v_next  # EN: Store v_{k+1}.

        alpha_arr = np.array(alphas, dtype=float)  # EN: Convert alphas to array.
        beta_arr = np.array(betas, dtype=float)  # EN: Convert betas to array.
        T_k = build_tridiagonal_T(alphas=alpha_arr, betas=beta_arr, k=k)  # EN: Build T_k.

        rhs = np.zeros((k,), dtype=float)  # EN: Build right-hand side vector.
        rhs[0] = g_norm  # EN: Place ||g|| on e1.
        y_k, *_ = np.linalg.lstsq(T_k, rhs, rcond=None)  # EN: Compute MINRES iterate in the Krylov basis.
        y_hat = V_basis[:, :k] @ y_k  # EN: Map y back to R^n.

        arnorm = l2_norm(rhs - (T_k @ y_k))  # EN: Normal residual norm in solver coordinates.
        n_done = k  # EN: Update completed iterations.
        if arnorm <= tol_rel_arnorm * max(g_norm, EPS):  # EN: Stop on relative normal residual.
            break  # EN: Exit loop.
        if beta_next < EPS and alpha_next < EPS:  # EN: Breakdown case.
            break  # EN: Exit loop.

        u = u_next  # EN: Advance u.
        v = v_next  # EN: Advance v.

    return y_hat, int(n_done)  # EN: Return final y and iteration count.


def column_scaling_D(A: np.ndarray) -> np.ndarray:  # EN: Build diagonal column scaling D_j = ||A[:,j]||_2.
    col_norms = np.linalg.norm(A, axis=0)  # EN: Compute column 2-norms.
    return np.maximum(col_norms, EPS).astype(float)  # EN: Avoid zeros and return as float.


def randomized_qr_preconditioner_R(  # EN: Build an upper-triangular right preconditioner R via QR(SA).
    A: np.ndarray,  # EN: Design matrix (m×n).
    sketch_factor: float,  # EN: Oversampling factor for sketch rows, e.g., 4.0 means s≈4n.
    rng: np.random.Generator,  # EN: RNG for sketch construction.
) -> np.ndarray:  # EN: Return R (n×n) upper triangular.
    m, n = A.shape  # EN: Extract dimensions.
    s = int(max(n, min(m, int(round(sketch_factor * n)))))  # EN: Choose sketch rows s between n and m.

    # EN: Use a simple Rademacher sketch S with entries ±1/sqrt(s).  # EN: Explain sketch choice.
    S = rng.choice(np.array([-1.0, 1.0]), size=(s, m)) / np.sqrt(max(s, 1))  # EN: Build S (s×m).
    A_sketch = S @ A  # EN: Compute SA (s×n).

    # EN: Thin QR on SA yields SA = Q R with R (n×n).  # EN: Explain QR output.
    _, R = np.linalg.qr(A_sketch, mode="reduced")  # EN: Compute reduced QR to obtain R.

    # EN: Normalize signs so diag(R) is non-negative (helps stable, consistent triangular solves).  # EN: Explain sign fix.
    d = np.sign(np.diag(R))  # EN: Extract diagonal signs.
    d[d == 0.0] = 1.0  # EN: Replace zeros with +1.
    R = np.diag(d) @ R  # EN: Flip rows to make diagonal non-negative.
    return R.astype(float)  # EN: Return R as float.


def solve_with_right_preconditioning(  # EN: Solve min||Ax-b|| using LSMR on A M^{-1}, then map back x=M^{-1} y.
    A: np.ndarray,  # EN: Design matrix (m×n).
    b: np.ndarray,  # EN: RHS vector (m,).
    apply_Minv: Callable[[np.ndarray], np.ndarray],  # EN: Function applying M^{-1} to a vector in R^n.
    apply_Minv_T: Callable[[np.ndarray], np.ndarray],  # EN: Function applying M^{-T} to a vector in R^n.
    max_iters: int,  # EN: LSMR iteration cap.
    tol_rel_arnorm: float,  # EN: Relative normal residual tolerance for LSMR.
) -> tuple[np.ndarray, int]:  # EN: Return (x_hat, iters) in original x-space.
    m, n = A.shape  # EN: Extract dimensions.

    def matvec_B(y: np.ndarray) -> np.ndarray:  # EN: Compute B y = A (M^{-1} y).
        return A @ apply_Minv(y)  # EN: Apply M^{-1} then multiply by A.

    def matvec_BT(u: np.ndarray) -> np.ndarray:  # EN: Compute B^T u = M^{-T} (A^T u).
        return apply_Minv_T(A.T @ u)  # EN: Multiply by A^T then apply M^{-T}.

    y_hat, iters = lsmr_teaching_minres(  # EN: Solve in y-space using operator B.
        matvec_A=matvec_B,  # EN: Provide B matvec.
        matvec_AT=matvec_BT,  # EN: Provide B^T matvec.
        b=b,  # EN: Provide RHS.
        n=n,  # EN: Unknown dimension is n.
        max_iters=max_iters,  # EN: Iteration cap.
        tol_rel_arnorm=tol_rel_arnorm,  # EN: Tolerance.
    )  # EN: End solve call.

    x_hat = apply_Minv(y_hat)  # EN: Map back x = M^{-1} y.
    return x_hat, int(iters)  # EN: Return mapped-back x and iteration count.


def main() -> None:  # EN: Run a demo comparing preconditioners for LSMR on an ill-conditioned problem.
    rng = np.random.default_rng(SEED)  # EN: Deterministic RNG.

    m = 1500  # EN: Number of rows (samples).
    n = 120  # EN: Number of columns (features).
    noise_std = 1e-3  # EN: Noise level for b.

    # EN: Build an ill-conditioned matrix using a prescribed singular spectrum.  # EN: Explain matrix construction.
    s = np.logspace(0.0, -12.0, num=n, base=10.0)  # EN: Very wide spectrum to stress solvers.
    G1 = rng.standard_normal((m, n))  # EN: Random matrix for left factor.
    Q1, _ = np.linalg.qr(G1, mode="reduced")  # EN: Orthonormalize columns to get Q1.
    G2 = rng.standard_normal((n, n))  # EN: Random matrix for right factor.
    Q2, _ = np.linalg.qr(G2)  # EN: Orthonormalize to get Q2.
    A = Q1 @ np.diag(s) @ Q2.T  # EN: Construct A with chosen singular values.

    x_true = rng.standard_normal(n)  # EN: Ground-truth coefficients.
    b = A @ x_true + noise_std * rng.standard_normal(m)  # EN: Build noisy RHS.

    cond_A = float(np.linalg.cond(A))  # EN: Condition number for intuition (dense/small n only).
    print_separator("Problem Summary")  # EN: Print summary header.
    print(f"A shape: {m}×{n}, cond(A)≈{cond_A:.3e}, noise_std={noise_std:.1e}")  # EN: Print diagnostics.

    max_iters = 200  # EN: Iteration cap for LSMR.
    tol_rel_arnorm = 1e-8  # EN: Relative tolerance on ||A^T r||.

    reports: list[SolveReport] = []  # EN: Collect reports for each method.

    # EN: Case 1: No preconditioner (M=I).  # EN: Explain baseline.
    t0 = perf_counter()  # EN: Start build timer (none).
    seconds_build = perf_counter() - t0  # EN: No build time.
    t1 = perf_counter()  # EN: Start solve timer.
    x_hat, iters = solve_with_right_preconditioning(  # EN: Solve with identity preconditioner.
        A=A,  # EN: A.
        b=b,  # EN: b.
        apply_Minv=lambda y: y,  # EN: M^{-1}=I.
        apply_Minv_T=lambda z: z,  # EN: M^{-T}=I.
        max_iters=max_iters,  # EN: Cap.
        tol_rel_arnorm=tol_rel_arnorm,  # EN: Tolerance.
    )  # EN: End solve.
    seconds_solve = perf_counter() - t1  # EN: Stop solve timer.
    r = A @ x_hat - b  # EN: Compute residual.
    ar = A.T @ r  # EN: Compute normal residual.
    reports.append(  # EN: Store report.
        SolveReport(  # EN: Construct report.
            label="LSMR (none)",  # EN: Label.
            n_iters=iters,  # EN: Iterations.
            seconds_build=float(seconds_build),  # EN: Build time.
            seconds_solve=float(seconds_solve),  # EN: Solve time.
            rnorm=l2_norm(r),  # EN: rnorm.
            arnorm=l2_norm(ar),  # EN: arnorm.
            xnorm=l2_norm(x_hat),  # EN: xnorm.
        )  # EN: End report.
    )  # EN: End append.

    # EN: Case 2: Column scaling (M = D).  # EN: Explain diagonal preconditioner.
    t0 = perf_counter()  # EN: Start build timer.
    D = column_scaling_D(A)  # EN: Build D from column norms.
    seconds_build = perf_counter() - t0  # EN: Stop build timer.
    t1 = perf_counter()  # EN: Start solve timer.
    x_hat, iters = solve_with_right_preconditioning(  # EN: Solve with diagonal right preconditioner.
        A=A,  # EN: A.
        b=b,  # EN: b.
        apply_Minv=lambda y, D=D: y / D,  # EN: M^{-1} y = D^{-1} y.
        apply_Minv_T=lambda z, D=D: z / D,  # EN: M^{-T} z = D^{-1} z (D is diagonal).
        max_iters=max_iters,  # EN: Cap.
        tol_rel_arnorm=tol_rel_arnorm,  # EN: Tolerance.
    )  # EN: End solve.
    seconds_solve = perf_counter() - t1  # EN: Stop solve timer.
    r = A @ x_hat - b  # EN: Residual in original coordinates.
    ar = A.T @ r  # EN: Normal residual in original coordinates.
    reports.append(  # EN: Store report.
        SolveReport(  # EN: Construct report.
            label="LSMR + col scaling",  # EN: Label.
            n_iters=iters,  # EN: Iterations.
            seconds_build=float(seconds_build),  # EN: Build time.
            seconds_solve=float(seconds_solve),  # EN: Solve time.
            rnorm=l2_norm(r),  # EN: rnorm.
            arnorm=l2_norm(ar),  # EN: arnorm.
            xnorm=l2_norm(x_hat),  # EN: xnorm.
        )  # EN: End report.
    )  # EN: End append.

    # EN: Case 3: Randomized QR (M = R from QR(SA)).  # EN: Explain advanced preconditioner.
    t0 = perf_counter()  # EN: Start build timer.
    R = randomized_qr_preconditioner_R(A=A, sketch_factor=4.0, rng=rng)  # EN: Build R using a 4n-row sketch.
    seconds_build = perf_counter() - t0  # EN: Stop build timer.

    def apply_Rinv(y: np.ndarray, R: np.ndarray = R) -> np.ndarray:  # EN: Apply R^{-1} via triangular solve.
        return np.linalg.solve(R, y)  # EN: Solve R x = y.

    def apply_Rinv_T(z: np.ndarray, R: np.ndarray = R) -> np.ndarray:  # EN: Apply R^{-T} via triangular solve.
        return np.linalg.solve(R.T, z)  # EN: Solve R^T y = z.

    t1 = perf_counter()  # EN: Start solve timer.
    x_hat, iters = solve_with_right_preconditioning(  # EN: Solve with randomized-QR right preconditioner.
        A=A,  # EN: A.
        b=b,  # EN: b.
        apply_Minv=apply_Rinv,  # EN: Use R^{-1}.
        apply_Minv_T=apply_Rinv_T,  # EN: Use R^{-T}.
        max_iters=max_iters,  # EN: Cap.
        tol_rel_arnorm=tol_rel_arnorm,  # EN: Tolerance.
    )  # EN: End solve.
    seconds_solve = perf_counter() - t1  # EN: Stop solve timer.
    r = A @ x_hat - b  # EN: Residual.
    ar = A.T @ r  # EN: Normal residual.
    reports.append(  # EN: Store report.
        SolveReport(  # EN: Construct report.
            label="LSMR + rand-QR (4n)",  # EN: Label.
            n_iters=iters,  # EN: Iterations.
            seconds_build=float(seconds_build),  # EN: Build time.
            seconds_solve=float(seconds_solve),  # EN: Solve time.
            rnorm=l2_norm(r),  # EN: rnorm.
            arnorm=l2_norm(ar),  # EN: arnorm.
            xnorm=l2_norm(x_hat),  # EN: xnorm.
        )  # EN: End report.
    )  # EN: End append.

    # EN: Print summary table.  # EN: Explain output section.
    print_separator("Summary (original x-space diagnostics)")  # EN: Announce summary.
    header = "method                 | iters | build_s | solve_s | ||Ax-b|| | ||A^T r|| | ||x||"  # EN: Build header.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Divider.
    for rep in reports:  # EN: Print each report row.
        print(  # EN: Print formatted row.
            f"{rep.label:22} | "  # EN: Label column.
            f"{rep.n_iters:5d} | "  # EN: Iterations.
            f"{rep.seconds_build:7.3f} | "  # EN: Build seconds.
            f"{rep.seconds_solve:7.3f} | "  # EN: Solve seconds.
            f"{rep.rnorm:8.2e} | "  # EN: rnorm.
            f"{rep.arnorm:9.2e} | "  # EN: arnorm.
            f"{rep.xnorm:8.2e}"  # EN: xnorm.
        )  # EN: End print.

    print_separator("Notes")  # EN: Print notes section.
    print("- Column scaling is cheap and often helps a bit, especially when feature scales differ.")  # EN: Note 1.
    print("- Randomized QR is more expensive to build, but can drastically cut Krylov iterations.")  # EN: Note 2.
    print("- For large sparse A, replace dense matmuls with matvec operators; the wrapping structure stays the same.")  # EN: Note 3.

    print_separator("Done")  # EN: End marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main.

