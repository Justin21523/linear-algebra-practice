"""  # EN: Start module docstring.
Choose the damping parameter for Damped LSQR via k-fold cross-validation (NumPy).  # EN: Summarize what this script does.

Damped LSQR solves the Ridge-style objective without forming A^T A:  # EN: Explain the optimization target.
  min_x  ||A x - b||_2^2 + damp^2 ||x||_2^2,   damp >= 0.  # EN: State the damped least-squares problem.

Key equivalence (augmentation trick):  # EN: Explain the core identity used by damped LSQR.
  min ||A x - b||^2 + damp^2||x||^2  ==  min || [A; damp I] x - [b; 0] ||^2.  # EN: Show augmented formulation.

We select damp using k-fold CV:  # EN: Explain why we do cross-validation.
  - For each damp, fit on the training folds using LSQR on the augmented operator.  # EN: Describe training step.
  - Evaluate validation RMSE on the held-out fold.  # EN: Describe validation step.
  - Choose damp that minimizes mean validation RMSE (and inspect the U-shaped curve).  # EN: Describe selection criterion.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured result records.
from typing import Callable  # EN: Use Callable for solver typing.

import numpy as np  # EN: Import NumPy for linear algebra and RNG.


EPS = 1e-12  # EN: Small epsilon for safe divisions.
SEED = 0  # EN: RNG seed for deterministic demos.
PRINT_PRECISION = 6  # EN: Console float precision for readable output.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias for a matrix-vector product function.


@dataclass(frozen=True)  # EN: Immutable record for a single CV point.
class CVPoint:  # EN: Store mean/std across folds for one damp value.
    key: str  # EN: Display label (e.g., "d=1e-3").
    param: float  # EN: Numeric damp for sorting and selection.
    train_mean: float  # EN: Mean training RMSE across folds.
    train_std: float  # EN: Std training RMSE across folds.
    val_mean: float  # EN: Mean validation RMSE across folds.
    val_std: float  # EN: Std validation RMSE across folds.
    x_norm_mean: float  # EN: Mean ||x_hat||_2 across folds (shrinkage proxy).


def print_separator(title: str) -> None:  # EN: Print a readable separator between sections.
    print()  # EN: Add a blank line before the section.
    print("=" * 78)  # EN: Print a horizontal divider.
    print(title)  # EN: Print section title.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # EN: Compute root-mean-square error.
    err = y_pred - y_true  # EN: Prediction error vector.
    return float(np.sqrt(np.mean(err**2)))  # EN: Return sqrt(mean squared error).


def build_vandermonde(t: np.ndarray, degree: int) -> np.ndarray:  # EN: Build polynomial design matrix with powers 0..degree.
    if t.ndim != 1:  # EN: Validate t shape.
        raise ValueError("t must be 1D")  # EN: Reject invalid input.
    if degree < 0:  # EN: Validate degree.
        raise ValueError("degree must be non-negative")  # EN: Reject invalid degree.
    return np.vander(t, N=degree + 1, increasing=True).astype(float)  # EN: Return Vandermonde matrix (n×(degree+1)).


def k_fold_splits(  # EN: Build deterministic k-fold splits (train_idx, val_idx).
    rng: np.random.Generator,  # EN: RNG used for shuffling.
    n_samples: int,  # EN: Total sample count.
    n_folds: int,  # EN: Number of folds.
) -> list[tuple[np.ndarray, np.ndarray]]:  # EN: Return list of index pairs.
    if n_folds < 2:  # EN: Need at least 2 folds.
        raise ValueError("n_folds must be >= 2")  # EN: Reject invalid fold count.
    if n_samples < n_folds:  # EN: Ensure each fold can be non-empty.
        raise ValueError("n_samples must be >= n_folds")  # EN: Reject impossible split.

    perm = rng.permutation(n_samples)  # EN: Shuffle indices.
    folds = np.array_split(perm, n_folds)  # EN: Split into roughly equal folds.

    splits: list[tuple[np.ndarray, np.ndarray]] = []  # EN: Collect splits.
    for i in range(n_folds):  # EN: Use fold i as validation fold.
        val_idx = folds[i]  # EN: Current fold is validation.
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])  # EN: Remaining folds are training.
        splits.append((train_idx, val_idx))  # EN: Store split.
    return splits  # EN: Return list of splits.


def ascii_bar(value: float, vmin: float, vmax: float, width: int = 30) -> str:  # EN: Render a simple ASCII bar (lower is better).
    if width <= 0:  # EN: Validate width.
        return ""  # EN: Return empty bar for non-positive widths.
    if vmax <= vmin + EPS:  # EN: Handle near-constant series.
        return "#" * width  # EN: Return full bar when values are effectively identical.
    score = (vmax - value) / (vmax - vmin)  # EN: Map lower RMSE -> higher score in [0,1].
    score = float(np.clip(score, 0.0, 1.0))  # EN: Clamp score to [0,1].
    n = int(round(score * width))  # EN: Convert score to bar length.
    return "#" * n  # EN: Return bar string.


def print_cv_table(points: list[CVPoint], best: CVPoint) -> None:  # EN: Print a CV table with an ASCII curve.
    points_sorted = sorted(points, key=lambda p: p.param)  # EN: Sort by damp for readability.
    vals = [p.val_mean for p in points_sorted]  # EN: Collect validation means for scaling.
    vmin = min(vals)  # EN: Minimum val RMSE.
    vmax = max(vals)  # EN: Maximum val RMSE.

    header = (  # EN: Build header string.
        "param        | train_rmse(mean±std) | val_rmse(mean±std) | ||x||_2(mean) | curve (lower val -> longer bar)"  # EN: Column names.
    )  # EN: End header construction.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Print divider line.

    for p in points_sorted:  # EN: Print each parameter row.
        mark = " <== best" if p.key == best.key else ""  # EN: Mark the best damp.
        bar = ascii_bar(value=p.val_mean, vmin=vmin, vmax=vmax, width=32)  # EN: Build ASCII bar for this row.
        print(  # EN: Print formatted row.
            f"{p.key:12} | "  # EN: Parameter label.
            f"{p.train_mean:.3e}±{p.train_std:.1e} | "  # EN: Train RMSE mean±std.
            f"{p.val_mean:.3e}±{p.val_std:.1e} | "  # EN: Val RMSE mean±std.
            f"{p.x_norm_mean:.3e} | "  # EN: Mean coefficient norm.
            f"{bar}{mark}"  # EN: ASCII bar + best marker.
        )  # EN: End print.


@dataclass(frozen=True)  # EN: Immutable record for LSQR outputs.
class LSQRResult:  # EN: Store LSQR solution and basic convergence information.
    x_hat: np.ndarray  # EN: Estimated solution vector (n,).
    n_iters: int  # EN: Iterations performed.
    normal_residual_norm: float  # EN: Final ||A^T(Ax-b)|| (or augmented equivalent) used for stopping.


def lsqr(  # EN: Minimal LSQR solver (matvec-only), stopping on ||A^T(Ax-b)||.
    matvec_A: Matvec,  # EN: Matvec for A.
    matvec_AT: Matvec,  # EN: Matvec for A^T.
    b: np.ndarray,  # EN: RHS vector.
    n: int,  # EN: Unknown dimension.
    max_iters: int,  # EN: Iteration cap.
    tol_normal: float,  # EN: Stop tolerance on normal residual.
) -> LSQRResult:  # EN: Return LSQRResult.
    x = np.zeros((n,), dtype=float)  # EN: Initialize x0=0.

    u = b.copy()  # EN: Initialize u from b.
    beta = l2_norm(u)  # EN: beta = ||b||.
    if beta < EPS:  # EN: Handle b=0.
        r = matvec_A(x) - b  # EN: Residual is -b.
        nr = matvec_AT(r)  # EN: Normal residual.
        return LSQRResult(x_hat=x, n_iters=0, normal_residual_norm=l2_norm(nr))  # EN: Return zero solution.
    u = u / beta  # EN: Normalize u.

    v = matvec_AT(u)  # EN: v = A^T u.
    alpha = l2_norm(v)  # EN: alpha = ||v||.
    if alpha < EPS:  # EN: Handle breakdown.
        r = matvec_A(x) - b  # EN: Residual.
        nr = matvec_AT(r)  # EN: Normal residual.
        return LSQRResult(x_hat=x, n_iters=0, normal_residual_norm=l2_norm(nr))  # EN: Return.
    v = v / alpha  # EN: Normalize v.

    w = v.copy()  # EN: Initialize w.
    phi_bar = beta  # EN: Initialize φ̄.
    rho_bar = alpha  # EN: Initialize ρ̄.

    for it in range(1, max_iters + 1):  # EN: LSQR iteration loop.
        u = matvec_A(v) - alpha * u  # EN: u_{k+1} = A v_k - α_k u_k.
        beta = l2_norm(u)  # EN: β_{k+1} = ||u_{k+1}||.
        if beta >= EPS:  # EN: Normalize if possible.
            u = u / beta  # EN: Normalize u.

        v = matvec_AT(u) - beta * v  # EN: v_{k+1} = A^T u_{k+1} - β_{k+1} v_k.
        alpha = l2_norm(v)  # EN: α_{k+1} = ||v_{k+1}||.
        if alpha >= EPS:  # EN: Normalize if possible.
            v = v / alpha  # EN: Normalize v.

        rho = float(np.hypot(rho_bar, beta))  # EN: ρ = sqrt(ρ̄^2 + β^2).
        c = rho_bar / max(rho, EPS)  # EN: c = ρ̄ / ρ.
        s = beta / max(rho, EPS)  # EN: s = β / ρ.
        theta = s * alpha  # EN: θ = s α.
        rho_bar = -c * alpha  # EN: Update ρ̄ = -c α.
        phi = c * phi_bar  # EN: φ = c φ̄.
        phi_bar = s * phi_bar  # EN: Update φ̄ = s φ̄.

        x = x + (phi / max(rho, EPS)) * w  # EN: Update x.
        w = v - (theta / max(rho, EPS)) * w  # EN: Update w.

        r = matvec_A(x) - b  # EN: Compute residual for stopping.
        nr = matvec_AT(r)  # EN: Compute normal residual.
        if l2_norm(nr) <= tol_normal:  # EN: Stop when normal residual is small.
            return LSQRResult(x_hat=x, n_iters=it, normal_residual_norm=l2_norm(nr))  # EN: Return converged result.
        if beta < EPS and alpha < EPS:  # EN: Detect breakdown.
            break  # EN: Exit loop.

    r = matvec_A(x) - b  # EN: Final residual.
    nr = matvec_AT(r)  # EN: Final normal residual.
    return LSQRResult(x_hat=x, n_iters=max_iters, normal_residual_norm=l2_norm(nr))  # EN: Return best effort.


def solve_damped_lsqr(  # EN: Solve min ||Ax-b||^2 + damp^2||x||^2 using LSQR on the augmented operator.
    A: np.ndarray,  # EN: Design matrix (m×n).
    b: np.ndarray,  # EN: RHS vector (m,).
    damp: float,  # EN: Damping parameter (>=0).
    max_iters: int,  # EN: Iteration cap for LSQR.
    tol_normal: float,  # EN: Stopping tolerance on ||A^T(Ax-b) + damp^2 x||.
) -> LSQRResult:  # EN: Return LSQRResult with x_hat.
    if damp < 0.0:  # EN: Validate damp.
        raise ValueError("damp must be non-negative")  # EN: Reject invalid damp.
    m, n = A.shape  # EN: Extract dimensions.

    def matvec_A_aug(v: np.ndarray) -> np.ndarray:  # EN: Compute [A; damp I] v.
        top = A @ v  # EN: Top block A v.
        bottom = damp * v  # EN: Bottom block damp * v.
        return np.concatenate([top, bottom])  # EN: Return concatenated vector of length m+n.

    def matvec_AT_aug(u_aug: np.ndarray) -> np.ndarray:  # EN: Compute [A; damp I]^T u_aug.
        u_top = u_aug[:m]  # EN: Extract top part.
        u_bottom = u_aug[m:]  # EN: Extract bottom part.
        return (A.T @ u_top) + (damp * u_bottom)  # EN: Return A^T u_top + damp u_bottom.

    b_aug = np.concatenate([b, np.zeros((n,), dtype=float)])  # EN: Augmented RHS [b; 0].
    return lsqr(matvec_A=matvec_A_aug, matvec_AT=matvec_AT_aug, b=b_aug, n=n, max_iters=max_iters, tol_normal=tol_normal)  # EN: Solve augmented system.


def summarize_cv_for_damp(  # EN: Compute k-fold CV summary for a single damp value.
    splits: list[tuple[np.ndarray, np.ndarray]],  # EN: List of (train_idx, val_idx).
    A: np.ndarray,  # EN: Full design matrix.
    b: np.ndarray,  # EN: Full targets.
    damp: float,  # EN: Damping value to evaluate.
    max_iters: int,  # EN: LSQR iteration cap.
    tol_normal: float,  # EN: LSQR stopping tolerance.
) -> CVPoint:  # EN: Return aggregated CV metrics.
    train_scores: list[float] = []  # EN: Collect training RMSE per fold.
    val_scores: list[float] = []  # EN: Collect validation RMSE per fold.
    x_norms: list[float] = []  # EN: Collect coefficient norms per fold.

    for train_idx, val_idx in splits:  # EN: Iterate folds.
        A_tr = A[train_idx, :]  # EN: Training design matrix.
        b_tr = b[train_idx]  # EN: Training targets.
        A_va = A[val_idx, :]  # EN: Validation design matrix.
        b_va = b[val_idx]  # EN: Validation targets.

        res = solve_damped_lsqr(A=A_tr, b=b_tr, damp=damp, max_iters=max_iters, tol_normal=tol_normal)  # EN: Fit on training fold.
        x_hat = res.x_hat  # EN: Extract fitted coefficients.

        train_scores.append(rmse(y_true=b_tr, y_pred=A_tr @ x_hat))  # EN: Compute train RMSE.
        val_scores.append(rmse(y_true=b_va, y_pred=A_va @ x_hat))  # EN: Compute val RMSE.
        x_norms.append(l2_norm(x_hat))  # EN: Record ||x||_2 (shrinkage).

    train_arr = np.array(train_scores, dtype=float)  # EN: Convert to array for stats.
    val_arr = np.array(val_scores, dtype=float)  # EN: Convert to array for stats.
    x_arr = np.array(x_norms, dtype=float)  # EN: Convert to array for stats.

    key = f"d={damp:.0e}" if damp != 0.0 else "d=0"  # EN: Build compact label (scientific notation).
    return CVPoint(  # EN: Construct CVPoint summary.
        key=key,  # EN: Store label.
        param=float(damp),  # EN: Store numeric damp.
        train_mean=float(np.mean(train_arr)),  # EN: Mean train RMSE.
        train_std=float(np.std(train_arr)),  # EN: Std train RMSE.
        val_mean=float(np.mean(val_arr)),  # EN: Mean val RMSE.
        val_std=float(np.std(val_arr)),  # EN: Std val RMSE.
        x_norm_mean=float(np.mean(x_arr)),  # EN: Mean coefficient norm.
    )  # EN: End CVPoint construction.


def main() -> None:  # EN: Run k-fold CV for damp and print an ASCII curve.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    n_samples = 80  # EN: Sample count for synthetic regression dataset.
    degree = 12  # EN: Polynomial degree (higher -> more ill-conditioned Vandermonde).
    noise_std = 0.05  # EN: Observation noise level.
    n_folds = 5  # EN: k for k-fold CV.

    t = np.linspace(-1.0, 1.0, n_samples)  # EN: Input locations.
    A = build_vandermonde(t, degree=degree)  # EN: Build design matrix.
    n_features = A.shape[1]  # EN: Number of coefficients/features.

    x_true = np.zeros((n_features,), dtype=float)  # EN: Create a sparse ground-truth polynomial.
    x_true[0] = 0.5  # EN: Intercept term.
    x_true[1] = 1.0  # EN: Linear term.
    x_true[2] = -2.0  # EN: Quadratic term.
    x_true[3] = 0.7  # EN: Cubic term.

    b_clean = A @ x_true  # EN: Clean targets.
    b = b_clean + noise_std * rng.standard_normal(b_clean.shape)  # EN: Add one noise realization.

    splits = k_fold_splits(rng=rng, n_samples=n_samples, n_folds=n_folds)  # EN: Build CV splits.

    print_separator("Dataset Summary")  # EN: Print dataset summary header.
    cond_A = float(np.linalg.cond(A))  # EN: Condition number diagnostic (small n).
    rank_A = int(np.linalg.matrix_rank(A))  # EN: Numerical rank diagnostic.
    print(f"n_samples={n_samples}, degree={degree}, n_features={n_features}, n_folds={n_folds}")  # EN: Print sizes.
    print(f"rank(A)={rank_A}, cond(A)={cond_A:.3e}, noise_std={noise_std}")  # EN: Print conditioning and noise.

    max_iters = 200  # EN: LSQR iteration cap (enough for this small demo).
    tol_normal = 1e-10  # EN: Stop tolerance on ||A^T(Ax-b)+damp^2 x|| for augmented operator.

    # EN: Candidate damps (include damp=0 baseline).  # EN: Explain candidate choice.
    damps = np.concatenate(([0.0], np.logspace(-8, 1, num=15)))  # EN: Evaluate damp across many magnitudes.

    print_separator("k-fold CV: damp (Ridge λ = damp^2)")  # EN: Announce CV section.
    points: list[CVPoint] = []  # EN: Collect results for each damp.
    for d in damps:  # EN: Loop over candidate damps.
        points.append(  # EN: Append CV summary for this damp.
            summarize_cv_for_damp(splits=splits, A=A, b=b, damp=float(d), max_iters=max_iters, tol_normal=tol_normal)  # EN: Compute CV summary.
        )  # EN: End append.

    best = min(points, key=lambda p: p.val_mean)  # EN: Choose damp with smallest mean val RMSE.
    print_cv_table(points=points, best=best)  # EN: Print table + ASCII curve.
    print(f"\nSelected damp = {best.param:.3e}  (λ = damp^2 ≈ {best.param**2:.3e})")  # EN: Print chosen damp and equivalent λ.

    # EN: Fit on full data with the selected damp and show the optimality condition.  # EN: Explain final fit.
    res_full = solve_damped_lsqr(A=A, b=b, damp=best.param, max_iters=max_iters, tol_normal=tol_normal)  # EN: Fit on full dataset.
    x_hat = res_full.x_hat  # EN: Extract coefficients.
    r = A @ x_hat - b  # EN: Compute data residual.
    grad = (A.T @ r) + (best.param**2) * x_hat  # EN: Compute ridge optimality residual A^T(Ax-b)+damp^2 x.
    print_separator("Final Fit Diagnostics (full data)")  # EN: Print diagnostics header.
    print(f"LSQR iters = {res_full.n_iters}, ||A^T r + damp^2 x||_2 = {l2_norm(grad):.3e}")  # EN: Show stopping-related quantity.
    print(f"||Ax-b||_2 = {l2_norm(r):.3e},  ||x||_2 = {l2_norm(x_hat):.3e}")  # EN: Show residual norm and shrinkage.

    print_separator("Done")  # EN: End marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main.

