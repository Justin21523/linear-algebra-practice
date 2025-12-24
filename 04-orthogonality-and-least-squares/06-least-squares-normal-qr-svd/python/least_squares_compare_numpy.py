"""  # EN: Start module docstring.
Least Squares solver stability: Normal Equation vs QR vs SVD (NumPy).  # EN: Summarize what this script compares.

This script focuses on *numerical linear algebra* behavior that matters in ML:  # EN: Explain motivation for ML readers.
  - Residual size: ||A x_hat - b||_2  # EN: Mention core metric (fit quality).
  - Optimality condition: A^T r ≈ 0 where r = b - A x_hat  # EN: Mention least-squares first-order condition.
  - Stability: how much x_hat changes under a tiny perturbation in b  # EN: Mention sensitivity/stability test.

Key takeaway: forming A^T A (Normal Equation) roughly squares the condition number,  # EN: Highlight the numerical issue.
so it can lose accuracy compared with QR / SVD on ill-conditioned (multicollinear) problems.  # EN: State practical implication.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Use postponed evaluation of annotations for forward references.

from dataclasses import dataclass  # EN: Use dataclass for a small immutable result record.
from typing import Callable  # EN: Use Callable to type solver function inputs/outputs.

import numpy as np  # EN: Import NumPy for matrix computations and decompositions.


EPS = 1e-12  # EN: Small epsilon to avoid division by zero in relative comparisons.
RCOND = 1e-12  # EN: Relative cutoff for small singular values when building a pseudo-inverse.
SEED = 0  # EN: Fixed RNG seed for deterministic demo outputs.
PRINT_PRECISION = 6  # EN: Console printing precision for floating-point numbers.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Make printed arrays compact and readable.


@dataclass(frozen=True)  # EN: Make a simple, immutable container for solver diagnostics.
class SolverReport:  # EN: Store per-method results and metrics for comparison.
    method: str  # EN: Human-readable method name (Normal / QR / SVD).
    succeeded: bool  # EN: Whether the solver produced a result without throwing an error.
    x_hat: np.ndarray | None  # EN: Estimated parameter vector (n,) when succeeded; otherwise None.
    residual_norm: float | None  # EN: ||A x_hat - b||_2 when succeeded; otherwise None.
    at_r_norm: float | None  # EN: ||A^T r||_2 where r=b-Ax_hat; near 0 indicates LS optimality.
    x_norm: float | None  # EN: ||x_hat||_2, useful for comparing minimum-norm solutions in rank-deficient cases.
    rel_x_error: float | None  # EN: ||x_hat-x_true||/||x_true|| when x_true is available; otherwise None.
    stability_gain: float | None  # EN: ||Δx||_2 / ||Δb||_2 for a tiny random perturbation Δb; smaller is better.
    stability_rel_change: float | None  # EN: ||Δx||_2 / ||x_hat||_2 for the same perturbation; smaller is better.
    error: str | None  # EN: Exception string when failed; otherwise None.


def print_separator(title: str) -> None:  # EN: Print a readable section separator.
    print()  # EN: Insert a blank line between sections.
    print("=" * 78)  # EN: Print a horizontal bar line.
    print(title)  # EN: Print the section title.
    print("=" * 78)  # EN: Print a closing horizontal bar line.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm for vectors (or Frobenius for matrices treated as flat).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy for stable norm computation.


def compute_residual(A: np.ndarray, b: np.ndarray, x_hat: np.ndarray) -> np.ndarray:  # EN: Compute LS residual r=b-Ax_hat.
    return b - (A @ x_hat)  # EN: Return residual vector.


def solve_normal_equation(A: np.ndarray, b: np.ndarray) -> np.ndarray:  # EN: Solve least squares via (A^T A) x = A^T b.
    AtA = A.T @ A  # EN: Form normal-equation matrix; this can square the condition number.
    Atb = A.T @ b  # EN: Form the right-hand side for normal equation.
    return np.linalg.solve(AtA, Atb)  # EN: Solve the n×n linear system for x_hat.


def solve_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:  # EN: Solve least squares via QR: A=QR, solve R x = Q^T b.
    Q, R = np.linalg.qr(A, mode="reduced")  # EN: Compute thin QR (m×n Q, n×n R) for m>=n.
    return np.linalg.solve(R, Q.T @ b)  # EN: Solve the upper-triangular system using back-substitution via solve().


def solve_svd(A: np.ndarray, b: np.ndarray, rcond: float = RCOND) -> np.ndarray:  # EN: Solve least squares via SVD pseudo-inverse.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD A = U diag(s) V^T.
    if s.size == 0:  # EN: Handle degenerate empty matrices defensively.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return a zero-length/zero vector solution as a fallback.
    cutoff = rcond * float(s.max())  # EN: Define absolute cutoff for singular values (relative to the largest).
    keep = s > cutoff  # EN: Keep only "significant" singular values to avoid dividing by near-zero values.
    if not np.any(keep):  # EN: If all singular values are below cutoff, treat as rank-0.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return the minimum-norm solution (all zeros).
    U_r = U[:, keep]  # EN: Select kept left singular vectors (m×r_eff).
    s_r = s[keep]  # EN: Select kept singular values (r_eff,).
    Vt_r = Vt[keep, :]  # EN: Select kept right singular vectors transposed (r_eff×n).
    return Vt_r.T @ ((U_r.T @ b) / s_r)  # EN: Compute x = V diag(1/s) U^T b without forming diag matrices.


def build_problem(  # EN: Construct a synthetic linear regression least-squares problem (A, b).
    rng: np.random.Generator,  # EN: Random generator used for reproducibility.
    n_samples: int,  # EN: Number of rows (equations / data points).
    collinearity_eps: float | None,  # EN: Controls how close column 3 is to column 2; None means independent.
    noise_std: float,  # EN: Standard deviation of additive noise in b (observation noise).
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # EN: Return (A, b, x_true) for the constructed problem.
    x1 = rng.standard_normal(n_samples)  # EN: Create a base feature column (mean ~0, variance ~1).
    if collinearity_eps is None:  # EN: Branch for the well-conditioned independent-feature case.
        x2 = rng.standard_normal(n_samples)  # EN: Create an independent second feature.
    else:  # EN: Otherwise build x2 nearly collinear with x1 (multicollinearity).
        x2 = x1 + collinearity_eps * rng.standard_normal(n_samples)  # EN: Make x2 ≈ x1 plus small noise.
    A = np.column_stack([np.ones(n_samples), x1, x2]).astype(float)  # EN: Build design matrix with intercept + 2 features.
    x_true = np.array([1.0, 2.0, -1.0], dtype=float)  # EN: Set a fixed ground-truth parameter vector.
    b_clean = A @ x_true  # EN: Generate noiseless observations.
    b = b_clean + noise_std * rng.standard_normal(n_samples)  # EN: Add measurement noise to create an overdetermined system.
    return A, b, x_true  # EN: Return the problem instance for experiments.


def evaluate_solver(  # EN: Run one solver on (A, b) and compute diagnostics + stability metrics.
    method: str,  # EN: Name used in reports and prints.
    solver: Callable[[np.ndarray, np.ndarray], np.ndarray],  # EN: Function that computes x_hat from (A, b).
    A: np.ndarray,  # EN: Design matrix (m×n).
    b: np.ndarray,  # EN: Observation vector (m,).
    x_true: np.ndarray | None,  # EN: Optional ground-truth x for reporting relative error.
    rng: np.random.Generator,  # EN: RNG used to generate a deterministic perturbation for stability testing.
) -> SolverReport:  # EN: Return a structured SolverReport record.
    try:  # EN: Catch numerical failures (singular matrices) so the demo can continue.
        x_hat = solver(A, b)  # EN: Compute the solver's least-squares estimate.
        r = compute_residual(A, b, x_hat)  # EN: Compute residual vector r=b-Ax_hat.
        residual_norm = l2_norm(r)  # EN: Compute the 2-norm of the residual.
        at_r_norm = l2_norm(A.T @ r)  # EN: Compute ||A^T r||_2 (should be ~0 for an optimal LS solution).
        x_norm = l2_norm(x_hat)  # EN: Compute ||x_hat||_2 to compare coefficient magnitudes.

        if x_true is None:  # EN: If no ground truth is provided, skip x error computation.
            rel_x_error = None  # EN: Indicate that relative x error is not available.
        else:  # EN: Otherwise compute relative parameter error against x_true.
            rel_x_error = l2_norm(x_hat - x_true) / max(l2_norm(x_true), EPS)  # EN: Compute ||x-x_true||/||x_true||.

        delta_scale = 1e-8 * max(l2_norm(b), 1.0)  # EN: Scale perturbation relative to ||b|| to make it tiny but nonzero.
        delta_b = delta_scale * rng.standard_normal(b.shape)  # EN: Create a small random perturbation for b.
        b_perturbed = b + delta_b  # EN: Create the perturbed observation vector.
        x_hat_perturbed = solver(A, b_perturbed)  # EN: Re-solve the LS problem under the perturbed b.
        dx = x_hat_perturbed - x_hat  # EN: Compute solution change Δx.
        db_norm = max(l2_norm(delta_b), EPS)  # EN: Compute ||Δb||_2 with a floor to avoid division by zero.
        dx_norm = l2_norm(dx)  # EN: Compute ||Δx||_2.
        stability_gain = dx_norm / db_norm  # EN: Report amplification factor ||Δx||/||Δb||.
        stability_rel_change = dx_norm / max(l2_norm(x_hat), EPS)  # EN: Report relative change in x_hat for the perturbation.

        return SolverReport(  # EN: Construct a successful report with all metrics filled in.
            method=method,  # EN: Store method name.
            succeeded=True,  # EN: Mark success.
            x_hat=x_hat,  # EN: Store computed solution.
            residual_norm=residual_norm,  # EN: Store residual norm.
            at_r_norm=at_r_norm,  # EN: Store optimality condition norm.
            x_norm=x_norm,  # EN: Store solution norm.
            rel_x_error=rel_x_error,  # EN: Store relative parameter error when available.
            stability_gain=stability_gain,  # EN: Store absolute stability gain.
            stability_rel_change=stability_rel_change,  # EN: Store relative stability change.
            error=None,  # EN: No error string on success.
        )  # EN: End report construction.
    except np.linalg.LinAlgError as exc:  # EN: Catch NumPy/LAPACK errors (singular matrices, etc.).
        return SolverReport(  # EN: Construct a failure report with missing metrics.
            method=method,  # EN: Store method name.
            succeeded=False,  # EN: Mark failure.
            x_hat=None,  # EN: No solution available.
            residual_norm=None,  # EN: Residual not computed.
            at_r_norm=None,  # EN: Optimality check not computed.
            x_norm=None,  # EN: Solution norm not computed.
            rel_x_error=None,  # EN: Relative x error not computed.
            stability_gain=None,  # EN: Stability not computed.
            stability_rel_change=None,  # EN: Stability not computed.
            error=str(exc),  # EN: Keep error message for display.
        )  # EN: End report construction.


def print_problem_summary(name: str, A: np.ndarray, b: np.ndarray) -> None:  # EN: Print conditioning and SVD info for a problem.
    m, n = A.shape  # EN: Extract matrix dimensions for display.
    cond_A = float(np.linalg.cond(A))  # EN: Compute condition number of A using SVD.
    AtA = A.T @ A  # EN: Form A^T A to illustrate condition-number squaring.
    cond_AtA = float(np.linalg.cond(AtA))  # EN: Compute condition number of A^T A (may be inf for rank-deficient).
    rank_A = int(np.linalg.matrix_rank(A))  # EN: Compute numerical rank of A.
    s = np.linalg.svd(A, compute_uv=False)  # EN: Get singular values for intuition about conditioning and rank.

    print_separator(f"Problem: {name}")  # EN: Announce the current problem case.
    print(f"Shape: A is {m}×{n}, b is ({m},)")  # EN: Print matrix/vector shapes.
    print(f"rank(A) = {rank_A}")  # EN: Print rank to distinguish full-rank vs rank-deficient.
    print(f"cond(A)   = {cond_A:.3e}")  # EN: Print condition number of A.
    print(f"cond(A^T A) = {cond_AtA:.3e}  (≈ cond(A)^2)")  # EN: Print squared-conditioning effect for normal equation.
    print(f"singular values s = {s}")  # EN: Print singular values for additional context.
    print(f"||b||_2 = {l2_norm(b):.3e}")  # EN: Print norm of b for scaling context in stability test.


def print_solver_table(reports: list[SolverReport]) -> None:  # EN: Print a compact comparison table of solver metrics.
    header = (  # EN: Build a header string for the printed table.
        "method | success | ||Ax-b||_2 | ||A^T r||_2 | ||x||_2 | rel_x_err | gain ||Δx||/||Δb|| | rel ||Δx||/||x||"  # EN: Column names.
    )  # EN: Finish building header.
    print(header)  # EN: Print the header line.
    print("-" * len(header))  # EN: Print a separator line based on header width.
    for rep in reports:  # EN: Iterate through reports and print one row per solver.
        if not rep.succeeded:  # EN: Handle failures with a short row.
            print(f"{rep.method:6} |   no   | (failed) -> {rep.error}")  # EN: Print failure reason.
            continue  # EN: Skip printing numeric columns for failed solver.
        rel_err_str = "n/a" if rep.rel_x_error is None else f"{rep.rel_x_error:.3e}"  # EN: Format relative error column.
        print(  # EN: Print a formatted row with key metrics.
            f"{rep.method:6} |  yes   | "  # EN: Method name + success flag.
            f"{rep.residual_norm:.3e} | "  # EN: Residual norm column.
            f"{rep.at_r_norm:.3e} | "  # EN: Optimality condition norm column.
            f"{rep.x_norm:.3e} | "  # EN: Solution norm column.
            f"{rel_err_str:>9} | "  # EN: Relative parameter error column.
            f"{rep.stability_gain:.3e} | "  # EN: Stability gain column.
            f"{rep.stability_rel_change:.3e}"  # EN: Relative stability change column.
        )  # EN: End row print.


def main() -> None:  # EN: Run three least-squares cases and compare Normal/QR/SVD solvers.
    rng = np.random.default_rng(SEED)  # EN: Create a deterministic random number generator.

    n_samples = 80  # EN: Pick a tall system (m>n) similar to typical regression problems.
    noise_std = 1e-3  # EN: Add small observation noise so the system is not exactly consistent.

    A_good, b_good, x_true = build_problem(  # EN: Create a well-conditioned baseline problem.
        rng=rng,  # EN: Provide RNG.
        n_samples=n_samples,  # EN: Provide sample count.
        collinearity_eps=None,  # EN: Use independent features for good conditioning.
        noise_std=noise_std,  # EN: Use fixed noise level.
    )  # EN: Finish building well-conditioned problem.

    A_bad, b_bad, x_true_bad = build_problem(  # EN: Create an ill-conditioned multicollinear problem.
        rng=rng,  # EN: Provide RNG.
        n_samples=n_samples,  # EN: Provide sample count.
        collinearity_eps=1e-8,  # EN: Make columns almost identical -> large cond(A).
        noise_std=noise_std,  # EN: Use the same noise level for fair comparison.
    )  # EN: Finish building ill-conditioned problem.

    A_rank_def, b_rank_def, x_true_rank_def = build_problem(  # EN: Create a rank-deficient problem (exact collinearity).
        rng=rng,  # EN: Provide RNG.
        n_samples=n_samples,  # EN: Provide sample count.
        collinearity_eps=0.0,  # EN: Make x2 == x1 exactly -> rank deficiency.
        noise_std=noise_std,  # EN: Keep noise so b is slightly inconsistent.
    )  # EN: Finish building rank-deficient problem.

    problems = [  # EN: Collect problems so we can loop uniformly.
        ("Well-conditioned (independent features)", A_good, b_good, x_true),  # EN: Baseline case with good conditioning.
        ("Ill-conditioned (multicollinearity)", A_bad, b_bad, x_true_bad),  # EN: Multicollinearity case to stress solvers.
        ("Rank-deficient (exact collinearity)", A_rank_def, b_rank_def, None),  # EN: Rank-deficient case (x_true error is not meaningful).
    ]  # EN: End problems list.

    solvers: list[tuple[str, Callable[[np.ndarray, np.ndarray], np.ndarray]]] = [  # EN: Define solver set to compare.
        ("Normal", solve_normal_equation),  # EN: Normal equation method (potentially unstable).
        ("QR", solve_qr),  # EN: QR-based method (typically stable).
        ("SVD", solve_svd),  # EN: SVD pseudo-inverse method (stable and handles rank deficiency).
    ]  # EN: End solvers list.

    for name, A, b, x_true_opt in problems:  # EN: Loop over problem cases and print comparisons.
        print_problem_summary(name=name, A=A, b=b)  # EN: Print conditioning + SVD summary for this case.

        reports: list[SolverReport] = []  # EN: Collect per-solver reports for this problem.
        for method, solver in solvers:  # EN: Evaluate each solver in the solver set.
            rep = evaluate_solver(  # EN: Run solver and compute metrics for this method.
                method=method,  # EN: Provide method name.
                solver=solver,  # EN: Provide solver function.
                A=A,  # EN: Provide A.
                b=b,  # EN: Provide b.
                x_true=x_true_opt,  # EN: Provide x_true (or None) for relative error reporting.
                rng=rng,  # EN: Provide RNG for deterministic perturbation.
            )  # EN: Finish evaluating solver.
            reports.append(rep)  # EN: Add report to the list for printing.

        print_solver_table(reports)  # EN: Print the table of metrics for this problem.

        # EN: Optionally print x_hat vectors for deeper inspection when solvers succeed.
        print("\nSolutions x_hat (for inspection):")  # EN: Announce solution vector prints.
        for rep in reports:  # EN: Iterate through solver reports again.
            if not rep.succeeded:  # EN: Skip printing x_hat for failed solvers.
                continue  # EN: Move to next report.
            print(f"{rep.method}: x_hat = {rep.x_hat}")  # EN: Print x_hat array.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard Python entrypoint guard.
    main()  # EN: Execute main when run as a script.

