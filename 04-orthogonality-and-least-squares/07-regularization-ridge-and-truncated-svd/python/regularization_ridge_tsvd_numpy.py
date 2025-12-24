"""  # EN: Start module docstring.
Regularized least squares on ill-conditioned problems: Ridge (Tikhonov) and Truncated SVD (TSVD).  # EN: Describe scope.

We intentionally build *multicollinear* (ill-conditioned) design matrices to show:  # EN: Explain why we stress the system.
  - Unregularized least squares can have huge coefficients with high sensitivity.  # EN: Describe the failure mode.
  - Ridge shrinks coefficients and improves stability by penalizing ||x||_2.  # EN: Summarize Ridge behavior.
  - TSVD discards tiny-singular-value directions to avoid noise amplification.  # EN: Summarize TSVD behavior.

Diagnostics printed per method:  # EN: List what we will report.
  - Residual: ||A x_hat - b||_2  # EN: Fit quality.
  - LS optimality: ||A^T r||_2 (only for unregularized LS)  # EN: First-order condition for LS.
  - Ridge optimality: ||A^T r - λ x_hat||_2  # EN: First-order condition for Ridge with λ||x||^2.
  - Stability: ||Δx||_2 / ||Δb||_2 under a tiny random perturbation Δb.  # EN: Sensitivity metric.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for a small immutable report container.
from typing import Callable  # EN: Use Callable for solver function typing.

import numpy as np  # EN: Import NumPy for linear algebra and random generation.


EPS = 1e-12  # EN: Small epsilon to avoid division by zero.
RCOND = 1e-12  # EN: Relative cutoff for pseudo-inverse computations.
SEED = 0  # EN: RNG seed for deterministic outputs.
PRINT_PRECISION = 6  # EN: Printing precision for floats in console output.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Make printed arrays readable.


@dataclass(frozen=True)  # EN: Create an immutable record to store per-method metrics.
class MethodReport:  # EN: Store outputs for comparing LS vs Ridge vs TSVD.
    method: str  # EN: Method name shown in tables.
    succeeded: bool  # EN: Whether the method produced a solution without error.
    x_hat: np.ndarray | None  # EN: Estimated coefficients when succeeded.
    residual_norm: float | None  # EN: ||A x_hat - b||_2 when succeeded.
    at_r_norm: float | None  # EN: ||A^T r||_2 for LS diagnostic (r=b-Ax_hat).
    ridge_opt_norm: float | None  # EN: ||A^T r - λ x_hat||_2 for Ridge diagnostic.
    tsvd_opt_norm: float | None  # EN: ||U_k^T r||_2 for TSVD subspace optimality.
    x_norm: float | None  # EN: ||x_hat||_2 to show coefficient shrinkage.
    rel_x_error: float | None  # EN: ||x_hat-x_true||/||x_true|| if x_true is provided.
    stability_gain: float | None  # EN: ||Δx||/||Δb|| under tiny perturbation in b.
    stability_rel_change: float | None  # EN: ||Δx||/||x|| under the same perturbation.
    error: str | None  # EN: Error message for failed methods.


def print_separator(title: str) -> None:  # EN: Print a section separator for readability.
    print()  # EN: Add a blank line before each section.
    print("=" * 78)  # EN: Print a horizontal divider line.
    print(title)  # EN: Print section title.
    print("=" * 78)  # EN: Print another divider line.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (vector 2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy for stable norm computation.


def build_multicollinear_problem(  # EN: Build a regression problem with controllable multicollinearity.
    rng: np.random.Generator,  # EN: RNG for reproducibility.
    n_samples: int,  # EN: Number of rows in A (data points).
    collinearity_eps: float,  # EN: Strength of noise added to create near-collinear columns.
    noise_std: float,  # EN: Observation noise level.
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # EN: Return (A, b, x_true).
    x1 = rng.standard_normal(n_samples)  # EN: First feature column.
    x2 = x1 + collinearity_eps * rng.standard_normal(n_samples)  # EN: Second feature ~ x1 -> multicollinearity.
    A = np.column_stack([np.ones(n_samples), x1, x2]).astype(float)  # EN: Add intercept and form design matrix.
    x_true = np.array([1.0, 2.0, -1.0], dtype=float)  # EN: Choose a fixed ground-truth coefficient vector.
    b_clean = A @ x_true  # EN: Generate noiseless targets.
    b = b_clean + noise_std * rng.standard_normal(n_samples)  # EN: Add observation noise to targets.
    return A, b, x_true  # EN: Return constructed problem instance.


def summarize_matrix(A: np.ndarray) -> tuple[int, int, int, float, np.ndarray]:  # EN: Compute rank/cond/singular values summary.
    m, n = A.shape  # EN: Extract shape for printing.
    rank = int(np.linalg.matrix_rank(A))  # EN: Compute numerical rank for diagnostics.
    cond = float(np.linalg.cond(A))  # EN: Compute condition number using SVD.
    s = np.linalg.svd(A, compute_uv=False)  # EN: Compute singular values only (no U/V).
    return m, n, rank, cond, s  # EN: Return summary fields for printing.


def residual(A: np.ndarray, b: np.ndarray, x_hat: np.ndarray) -> np.ndarray:  # EN: Compute residual r=b-Ax_hat.
    return b - (A @ x_hat)  # EN: Compute residual vector.


def solve_ls_qr(A: np.ndarray, b: np.ndarray) -> np.ndarray:  # EN: Solve unregularized least squares via QR.
    Q, R = np.linalg.qr(A, mode="reduced")  # EN: Compute thin QR decomposition.
    return np.linalg.solve(R, Q.T @ b)  # EN: Solve R x = Q^T b for x.


def solve_ls_svd_pinv(A: np.ndarray, b: np.ndarray, rcond: float = RCOND) -> np.ndarray:  # EN: Solve LS via SVD pseudo-inverse.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD A = U diag(s) V^T.
    if s.size == 0:  # EN: Defensive handling for empty shapes.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return a zero vector solution.
    cutoff = rcond * float(s.max())  # EN: Build absolute cutoff for small singular values.
    keep = s > cutoff  # EN: Keep only well-resolved singular values.
    if not np.any(keep):  # EN: If nothing is kept, treat matrix as rank-0.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return the minimum-norm solution (zeros).
    U_r = U[:, keep]  # EN: Select kept U columns.
    s_r = s[keep]  # EN: Select kept singular values.
    Vt_r = Vt[keep, :]  # EN: Select kept V^T rows.
    return Vt_r.T @ ((U_r.T @ b) / s_r)  # EN: Compute x = V diag(1/s) U^T b without explicit diags.


def solve_ridge_svd_filter(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:  # EN: Solve Ridge using SVD filter factors.
    if lam < 0.0:  # EN: Validate that λ is non-negative.
        raise ValueError("lam must be non-negative")  # EN: Reject invalid regularization strength.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD of A.
    if s.size == 0:  # EN: Handle degenerate empty matrices.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return a zero vector solution.
    if lam == 0.0:  # EN: Ridge with λ=0 reduces to unregularized pseudo-inverse LS.
        return solve_ls_svd_pinv(A, b)  # EN: Delegate to LS SVD solver for the λ=0 case.
    factors = s / (s**2 + lam)  # EN: Ridge filter factors: σ/(σ^2+λ) damp small σ directions.
    return Vt.T @ (factors * (U.T @ b))  # EN: x = V diag(factors) U^T b using elementwise multiplication.


def solve_ridge_normal_equation(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:  # EN: Solve Ridge via (A^T A + λI)x = A^T b.
    if lam < 0.0:  # EN: Validate λ.
        raise ValueError("lam must be non-negative")  # EN: Reject invalid λ.
    n = A.shape[1]  # EN: Number of columns / unknowns.
    AtA = A.T @ A  # EN: Form normal-equation matrix A^T A.
    Atb = A.T @ b  # EN: Form right-hand side A^T b.
    return np.linalg.solve(AtA + lam * np.eye(n), Atb)  # EN: Solve the SPD system for λ>0.


def solve_tsvd(A: np.ndarray, b: np.ndarray, k: int, rcond: float = RCOND) -> tuple[np.ndarray, np.ndarray]:  # EN: Solve using truncated SVD and return (x_hat, U_k).
    if k <= 0:  # EN: Validate k.
        raise ValueError("k must be positive")  # EN: Reject invalid truncation rank.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD A = U diag(s) V^T.
    if s.size == 0:  # EN: Handle degenerate empty matrices.
        return np.zeros((A.shape[1],), dtype=float), np.zeros((A.shape[0], 0), dtype=float)  # EN: Return empty solution + empty U_k.
    cutoff = rcond * float(s.max())  # EN: Define small-singular-value cutoff.
    keep = s > cutoff  # EN: Identify numerically nonzero singular values.
    r_eff = int(np.count_nonzero(keep))  # EN: Effective numerical rank based on cutoff.
    k_eff = min(k, r_eff)  # EN: Clamp k to the effective rank.
    if k_eff == 0:  # EN: If effective rank is zero, return zero solution.
        return np.zeros((A.shape[1],), dtype=float), np.zeros((A.shape[0], 0), dtype=float)  # EN: Return zeros + empty U_k.
    U_k = U[:, :k_eff]  # EN: Keep top-k left singular vectors.
    s_k = s[:k_eff]  # EN: Keep top-k singular values.
    Vt_k = Vt[:k_eff, :]  # EN: Keep top-k rows of V^T.
    x_hat = Vt_k.T @ ((U_k.T @ b) / s_k)  # EN: Compute x_hat using the truncated pseudo-inverse.
    return x_hat, U_k  # EN: Return solution and U_k for the TSVD optimality check.


def evaluate(  # EN: Evaluate a solver and compute metrics, including stability under perturbation.
    method: str,  # EN: Method name.
    solver: Callable[[np.ndarray, np.ndarray], np.ndarray],  # EN: Solver returning x_hat.
    A: np.ndarray,  # EN: Design matrix.
    b: np.ndarray,  # EN: Target vector.
    x_true: np.ndarray | None,  # EN: Optional ground-truth for relative error.
    rng: np.random.Generator,  # EN: RNG used for perturbation generation.
    ridge_lam: float | None = None,  # EN: λ for ridge optimality check, if applicable.
) -> MethodReport:  # EN: Return a filled MethodReport.
    try:  # EN: Catch numerical errors so the demo can keep running.
        x_hat = solver(A, b)  # EN: Compute solution.
        r = residual(A, b, x_hat)  # EN: Compute residual vector.
        residual_norm = l2_norm(A @ x_hat - b)  # EN: Compute ||Ax-b|| directly for clarity.
        at_r_norm = l2_norm(A.T @ r)  # EN: Compute ||A^T r||_2 (LS first-order condition).
        x_norm = l2_norm(x_hat)  # EN: Compute ||x_hat||_2 for shrinkage comparison.

        if x_true is None:  # EN: Skip x error when no unique truth is meaningful (rank-deficient case).
            rel_x_error = None  # EN: Mark as not available.
        else:  # EN: Otherwise compute relative error vs x_true.
            rel_x_error = l2_norm(x_hat - x_true) / max(l2_norm(x_true), EPS)  # EN: Compute normalized parameter error.

        if ridge_lam is None:  # EN: Only Ridge methods should compute ridge optimality.
            ridge_opt_norm = None  # EN: Not applicable.
        else:  # EN: For Ridge: A^T r ≈ λ x.
            ridge_opt_norm = l2_norm((A.T @ r) - ridge_lam * x_hat)  # EN: Compute Ridge first-order residual norm.

        delta_scale = 1e-8 * max(l2_norm(b), 1.0)  # EN: Scale perturbation relative to ||b||.
        delta_b = delta_scale * rng.standard_normal(b.shape)  # EN: Create a tiny perturbation Δb.
        x_hat_pert = solver(A, b + delta_b)  # EN: Re-solve with perturbed b.
        dx = x_hat_pert - x_hat  # EN: Compute solution difference Δx.
        stability_gain = l2_norm(dx) / max(l2_norm(delta_b), EPS)  # EN: Compute amplification ||Δx||/||Δb||.
        stability_rel_change = l2_norm(dx) / max(l2_norm(x_hat), EPS)  # EN: Compute relative change ||Δx||/||x||.

        return MethodReport(  # EN: Construct successful report.
            method=method,  # EN: Store method name.
            succeeded=True,  # EN: Mark success.
            x_hat=x_hat,  # EN: Store solution vector.
            residual_norm=residual_norm,  # EN: Store residual norm.
            at_r_norm=at_r_norm,  # EN: Store LS optimality norm.
            ridge_opt_norm=ridge_opt_norm,  # EN: Store Ridge optimality norm if applicable.
            tsvd_opt_norm=None,  # EN: TSVD optimality is computed separately (needs U_k).
            x_norm=x_norm,  # EN: Store solution norm.
            rel_x_error=rel_x_error,  # EN: Store relative error if available.
            stability_gain=stability_gain,  # EN: Store stability gain.
            stability_rel_change=stability_rel_change,  # EN: Store relative stability.
            error=None,  # EN: No error.
        )  # EN: End report construction.
    except np.linalg.LinAlgError as exc:  # EN: Catch linear algebra errors (singular matrices, etc.).
        return MethodReport(  # EN: Construct failure report.
            method=method,  # EN: Store method.
            succeeded=False,  # EN: Mark failure.
            x_hat=None,  # EN: No x_hat.
            residual_norm=None,  # EN: No residual.
            at_r_norm=None,  # EN: No A^T r.
            ridge_opt_norm=None,  # EN: No Ridge check.
            tsvd_opt_norm=None,  # EN: No TSVD check.
            x_norm=None,  # EN: No x norm.
            rel_x_error=None,  # EN: No relative error.
            stability_gain=None,  # EN: No stability.
            stability_rel_change=None,  # EN: No stability.
            error=str(exc),  # EN: Store error message.
        )  # EN: End failure report.


def format_optional(value: float | None) -> str:  # EN: Format optional floats for table output.
    return "n/a" if value is None else f"{value:.3e}"  # EN: Render None as n/a; otherwise scientific format.


def print_table(reports: list[MethodReport]) -> None:  # EN: Print a comparison table for multiple methods.
    header = (  # EN: Build a header line describing table columns.
        "method | success | ||Ax-b|| | ||A^T r|| | ||A^T r-λx|| | ||U_k^T r|| | ||x|| | rel_x_err | gain ||Δx||/||Δb||"  # EN: Column labels.
    )  # EN: Finish header construction.
    print(header)  # EN: Print the header.
    print("-" * len(header))  # EN: Print underline separator.
    for rep in reports:  # EN: Print one row per method report.
        if not rep.succeeded:  # EN: Handle failed methods.
            print(f"{rep.method:18} |   no   | (failed) -> {rep.error}")  # EN: Print method and error.
            continue  # EN: Skip metric columns on failure.
        print(  # EN: Print one formatted row.
            f"{rep.method:18} |  yes   | "  # EN: Method and success flag.
            f"{rep.residual_norm:.3e} | "  # EN: Residual norm.
            f"{rep.at_r_norm:.3e} | "  # EN: LS optimality norm.
            f"{format_optional(rep.ridge_opt_norm):>11} | "  # EN: Ridge optimality norm.
            f"{format_optional(rep.tsvd_opt_norm):>10} | "  # EN: TSVD subspace optimality norm.
            f"{rep.x_norm:.3e} | "  # EN: Solution norm.
            f"{format_optional(rep.rel_x_error):>9} | "  # EN: Relative x error.
            f"{rep.stability_gain:.3e}"  # EN: Stability gain.
        )  # EN: End row print.


def evaluate_tsvd(  # EN: Evaluate TSVD separately because it needs U_k for its own optimality check.
    method: str,  # EN: Method name.
    A: np.ndarray,  # EN: Design matrix.
    b: np.ndarray,  # EN: Target vector.
    k: int,  # EN: Truncation rank.
    x_true: np.ndarray | None,  # EN: Optional ground truth.
    rng: np.random.Generator,  # EN: RNG for perturbation.
) -> MethodReport:  # EN: Return filled MethodReport with TSVD optimality.
    def solver(A_in: np.ndarray, b_in: np.ndarray) -> np.ndarray:  # EN: Wrap TSVD to match solver signature.
        x_hat_local, _ = solve_tsvd(A_in, b_in, k=k)  # EN: Solve TSVD and ignore U_k.
        return x_hat_local  # EN: Return x_hat only.

    try:  # EN: Compute x_hat and U_k once so we can check U_k^T r.
        x_hat, U_k = solve_tsvd(A, b, k=k)  # EN: Solve TSVD and keep U_k.
        r = residual(A, b, x_hat)  # EN: Compute residual for TSVD solution.
        tsvd_opt_norm = l2_norm(U_k.T @ r)  # EN: TSVD condition: r ⟂ span(U_k) => U_k^T r ≈ 0.
        base = evaluate(  # EN: Reuse evaluate() for shared metrics and stability.
            method=method,  # EN: Provide method name.
            solver=solver,  # EN: Provide wrapped solver.
            A=A,  # EN: Provide A.
            b=b,  # EN: Provide b.
            x_true=x_true,  # EN: Provide optional x_true.
            rng=rng,  # EN: Provide RNG.
            ridge_lam=None,  # EN: Ridge optimality not applicable.
        )  # EN: Finish base evaluation.
        return MethodReport(  # EN: Return base report but with TSVD optimality filled in.
            method=base.method,  # EN: Copy method name.
            succeeded=base.succeeded,  # EN: Copy success.
            x_hat=base.x_hat,  # EN: Copy x_hat.
            residual_norm=base.residual_norm,  # EN: Copy residual norm.
            at_r_norm=base.at_r_norm,  # EN: Copy A^T r norm.
            ridge_opt_norm=base.ridge_opt_norm,  # EN: Copy ridge optimality (None).
            tsvd_opt_norm=tsvd_opt_norm,  # EN: Set TSVD optimality norm.
            x_norm=base.x_norm,  # EN: Copy x norm.
            rel_x_error=base.rel_x_error,  # EN: Copy relative error.
            stability_gain=base.stability_gain,  # EN: Copy stability gain.
            stability_rel_change=base.stability_rel_change,  # EN: Copy relative stability.
            error=base.error,  # EN: Copy error (None on success).
        )  # EN: End report construction.
    except np.linalg.LinAlgError as exc:  # EN: Catch TSVD linear algebra errors.
        return MethodReport(  # EN: Return failure report.
            method=method,  # EN: Method name.
            succeeded=False,  # EN: Mark failure.
            x_hat=None,  # EN: No solution.
            residual_norm=None,  # EN: No residual.
            at_r_norm=None,  # EN: No A^T r.
            ridge_opt_norm=None,  # EN: No ridge check.
            tsvd_opt_norm=None,  # EN: No TSVD check.
            x_norm=None,  # EN: No x norm.
            rel_x_error=None,  # EN: No relative error.
            stability_gain=None,  # EN: No stability.
            stability_rel_change=None,  # EN: No relative stability.
            error=str(exc),  # EN: Store error.
        )  # EN: End failure report.


def run_case(  # EN: Run one problem case and print comparison tables.
    case_name: str,  # EN: Human-readable case name.
    A: np.ndarray,  # EN: Design matrix.
    b: np.ndarray,  # EN: Target vector.
    x_true: np.ndarray | None,  # EN: Optional ground truth.
    rng: np.random.Generator,  # EN: RNG for perturbations.
) -> None:  # EN: Print results; return nothing.
    m, n, rank, cond, s = summarize_matrix(A)  # EN: Compute matrix diagnostics.
    print_separator(f"Case: {case_name}")  # EN: Print case title.
    print(f"A shape: {m}×{n}, rank(A)={rank}, cond(A)={cond:.3e}")  # EN: Print shape/rank/cond summary.
    print(f"singular values: {s}")  # EN: Print singular values for intuition.
    print(f"||b||_2 = {l2_norm(b):.3e}")  # EN: Print b norm for scale context.

    reports: list[MethodReport] = []  # EN: Collect reports for this case.

    reports.append(  # EN: Append QR least squares baseline report.
        evaluate(  # EN: Evaluate QR method.
            method="LS-QR",  # EN: Label method.
            solver=solve_ls_qr,  # EN: Provide QR solver.
            A=A,  # EN: Provide A.
            b=b,  # EN: Provide b.
            x_true=x_true,  # EN: Provide x_true when available.
            rng=rng,  # EN: Provide RNG.
            ridge_lam=None,  # EN: No ridge condition.
        )  # EN: Finish evaluation call.
    )  # EN: End append.

    reports.append(  # EN: Append SVD pseudo-inverse least squares baseline report.
        evaluate(  # EN: Evaluate LS via SVD pseudo-inverse.
            method="LS-SVD(pinv)",  # EN: Label method.
            solver=lambda A_in, b_in: solve_ls_svd_pinv(A_in, b_in),  # EN: Wrap pinv solver to match signature.
            A=A,  # EN: Provide A.
            b=b,  # EN: Provide b.
            x_true=x_true,  # EN: Provide x_true.
            rng=rng,  # EN: Provide RNG.
            ridge_lam=None,  # EN: No ridge condition.
        )  # EN: Finish evaluation call.
    )  # EN: End append.

    ridge_lambdas = [1e-10, 1e-6, 1e-2, 1e0]  # EN: Choose a few λ values to show shrinkage/stability tradeoff.
    for lam in ridge_lambdas:  # EN: Loop over λ values.
        reports.append(  # EN: Add Ridge-SVD report for this λ.
            evaluate(  # EN: Evaluate Ridge via SVD filter factors.
                method=f"Ridge-SVD λ={lam:g}",  # EN: Method label includes λ.
                solver=lambda A_in, b_in, lam=lam: solve_ridge_svd_filter(A_in, b_in, lam=lam),  # EN: Bind λ in lambda.
                A=A,  # EN: Provide A.
                b=b,  # EN: Provide b.
                x_true=x_true,  # EN: Provide x_true.
                rng=rng,  # EN: Provide RNG.
                ridge_lam=lam,  # EN: Use λ for ridge optimality check.
            )  # EN: Finish evaluation call.
        )  # EN: End append.

        reports.append(  # EN: Add Ridge-Normal report for this λ (same math, different computation path).
            evaluate(  # EN: Evaluate Ridge via regularized normal equation.
                method=f"Ridge-Normal λ={lam:g}",  # EN: Method label.
                solver=lambda A_in, b_in, lam=lam: solve_ridge_normal_equation(A_in, b_in, lam=lam),  # EN: Bind λ.
                A=A,  # EN: Provide A.
                b=b,  # EN: Provide b.
                x_true=x_true,  # EN: Provide x_true.
                rng=rng,  # EN: Provide RNG.
                ridge_lam=lam,  # EN: Use λ for ridge optimality check.
            )  # EN: Finish evaluation.
        )  # EN: End append.

    tsvd_ks = [max(1, min(2, n))]  # EN: Use k=2 (drop smallest singular direction) for a 3-column regression.
    for k in tsvd_ks:  # EN: Loop over TSVD truncation ranks.
        reports.append(  # EN: Add TSVD report.
            evaluate_tsvd(  # EN: Evaluate TSVD method.
                method=f"TSVD k={k}",  # EN: Label includes k.
                A=A,  # EN: Provide A.
                b=b,  # EN: Provide b.
                k=k,  # EN: Provide k.
                x_true=x_true,  # EN: Provide x_true (or None).
                rng=rng,  # EN: Provide RNG.
            )  # EN: Finish TSVD evaluation.
        )  # EN: End append.

    print_table(reports)  # EN: Print the comparison table.

    print("\nSelected solutions (x_hat) for inspection:")  # EN: Announce solution prints.
    for rep in reports:  # EN: Iterate through reports to print x_hat when available.
        if not rep.succeeded:  # EN: Skip failed methods.
            continue  # EN: Move on.
        if rep.method in {"LS-QR", "LS-SVD(pinv)", "TSVD k=2"} or rep.method.startswith("Ridge-SVD λ=1e-06"):  # EN: Print a small subset.
            print(f"{rep.method:18}: x_hat = {rep.x_hat}")  # EN: Print solution vector.


def main() -> None:  # EN: Main entrypoint for the demo script.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    n_samples = 80  # EN: Use a tall system (more rows than columns) like typical regression.
    noise_std = 1e-3  # EN: Small noise level to make the system inconsistent (realistic data).

    A_ill, b_ill, x_true = build_multicollinear_problem(  # EN: Build an ill-conditioned multicollinear problem.
        rng=rng,  # EN: Provide RNG.
        n_samples=n_samples,  # EN: Provide sample count.
        collinearity_eps=1e-8,  # EN: Very small epsilon -> large condition number.
        noise_std=noise_std,  # EN: Provide noise level.
    )  # EN: Finish building ill-conditioned problem.

    run_case(  # EN: Run and print results for ill-conditioned case.
        case_name="Ill-conditioned (multicollinearity)",  # EN: Case label.
        A=A_ill,  # EN: Provide A.
        b=b_ill,  # EN: Provide b.
        x_true=x_true,  # EN: Provide x_true (meaningful here).
        rng=rng,  # EN: Provide RNG.
    )  # EN: End ill-conditioned case run.

    A_rank, b_rank, _x_true_rank = build_multicollinear_problem(  # EN: Build an exactly collinear (rank-deficient) problem.
        rng=rng,  # EN: Provide RNG.
        n_samples=n_samples,  # EN: Provide sample count.
        collinearity_eps=0.0,  # EN: x2 == x1 exactly -> rank deficiency.
        noise_std=noise_std,  # EN: Provide noise.
    )  # EN: Finish rank-deficient problem build.

    run_case(  # EN: Run and print results for rank-deficient case.
        case_name="Rank-deficient (exact collinearity)",  # EN: Case label.
        A=A_rank,  # EN: Provide A.
        b=b_rank,  # EN: Provide b.
        x_true=None,  # EN: No unique x_true comparison for rank-deficient systems.
        rng=rng,  # EN: Provide RNG.
    )  # EN: End rank-deficient case run.

    print_separator("Done")  # EN: Print end marker.


if __name__ == "__main__":  # EN: Standard Python entrypoint guard.
    main()  # EN: Execute the demo.

