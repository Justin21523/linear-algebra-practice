"""  # EN: Start module docstring.
k-fold cross-validation for regularization hyperparameters (NumPy).  # EN: Describe the script purpose.

This script chooses:  # EN: Introduce what we select.
  - Ridge λ (L2 regularization strength)  # EN: Mention Ridge hyperparameter.
  - TSVD k (truncation rank in truncated SVD)  # EN: Mention TSVD hyperparameter.

It prints both a numeric table and a simple ASCII "curve" for validation RMSE,  # EN: Describe outputs.
so you can quickly see the U-shape and the best region without plotting libraries.  # EN: Explain why ASCII curves are used.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for small structured result records.

import numpy as np  # EN: Import NumPy for linear algebra and random generation.


EPS = 1e-12  # EN: Small epsilon to avoid division by zero.
RCOND = 1e-12  # EN: Relative cutoff for pseudo-inverse and rank decisions.
SEED = 0  # EN: RNG seed for deterministic results.
PRINT_PRECISION = 6  # EN: Console float precision for readable prints.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy array printing.


@dataclass(frozen=True)  # EN: Immutable record for a single CV point (one hyperparameter value).
class CVPoint:  # EN: Store mean/std metrics for train/val across folds.
    key: str  # EN: Display label (e.g., "λ=1e-3" or "k=05").
    param: float  # EN: Numeric hyperparameter value used for sorting and selection.
    train_mean: float  # EN: Mean training RMSE across folds.
    train_std: float  # EN: Std training RMSE across folds.
    val_mean: float  # EN: Mean validation RMSE across folds.
    val_std: float  # EN: Std validation RMSE across folds.
    x_norm_mean: float  # EN: Mean ||x_hat||_2 across folds (shrinkage / complexity proxy).


def print_separator(title: str) -> None:  # EN: Print a readable separator between sections.
    print()  # EN: Add a blank line before each section.
    print("=" * 78)  # EN: Print a horizontal divider line.
    print(title)  # EN: Print the section title.
    print("=" * 78)  # EN: Print a closing divider line.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (vector 2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy for stable norm computation.


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # EN: Compute root-mean-square error for predictions.
    err = y_pred - y_true  # EN: Compute prediction error vector.
    return float(np.sqrt(np.mean(err**2)))  # EN: Return RMSE as sqrt(mean squared error).


def build_vandermonde(t: np.ndarray, degree: int) -> np.ndarray:  # EN: Build polynomial feature matrix with powers 0..degree.
    if t.ndim != 1:  # EN: Ensure t is a 1D array.
        raise ValueError("t must be a 1D array")  # EN: Fail fast on invalid input.
    if degree < 0:  # EN: Validate degree.
        raise ValueError("degree must be non-negative")  # EN: Reject invalid degree.
    return np.vander(t, N=degree + 1, increasing=True).astype(float)  # EN: Return Vandermonde matrix (n×(degree+1)).


def k_fold_splits(  # EN: Generate k-fold (train_idx, val_idx) splits with deterministic shuffling.
    rng: np.random.Generator,  # EN: RNG used to permute indices.
    n_samples: int,  # EN: Total number of samples.
    n_folds: int,  # EN: Number of folds (k).
) -> list[tuple[np.ndarray, np.ndarray]]:  # EN: Return list of (train_idx, val_idx) arrays.
    if n_folds < 2:  # EN: Require at least 2 folds.
        raise ValueError("n_folds must be >= 2")  # EN: Reject invalid fold count.
    if n_samples < n_folds:  # EN: Ensure each fold can be non-empty.
        raise ValueError("n_samples must be >= n_folds")  # EN: Reject impossible split.

    perm = rng.permutation(n_samples)  # EN: Permute indices to randomize fold assignment.
    folds = np.array_split(perm, n_folds)  # EN: Split permuted indices into k roughly equal folds.

    splits: list[tuple[np.ndarray, np.ndarray]] = []  # EN: Collect (train, val) index pairs.
    for i in range(n_folds):  # EN: Iterate each fold as the validation fold.
        val_idx = folds[i]  # EN: Current fold indices become the validation set.
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])  # EN: All other folds become training.
        splits.append((train_idx, val_idx))  # EN: Store split pair.
    return splits  # EN: Return the list of k splits.


def solve_ls_svd_pinv(A: np.ndarray, b: np.ndarray, rcond: float = RCOND) -> np.ndarray:  # EN: Solve unregularized LS via SVD pseudo-inverse.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD.
    if s.size == 0:  # EN: Handle degenerate shapes defensively.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return zero vector.
    cutoff = rcond * float(s.max())  # EN: Compute absolute cutoff for small singular values.
    keep = s > cutoff  # EN: Keep only numerically significant singular values.
    if not np.any(keep):  # EN: If effective rank is zero, return min-norm zero solution.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return zeros.
    U_r = U[:, keep]  # EN: Select kept U columns.
    s_r = s[keep]  # EN: Select kept singular values.
    Vt_r = Vt[keep, :]  # EN: Select kept V^T rows.
    return Vt_r.T @ ((U_r.T @ b) / s_r)  # EN: Compute x = V diag(1/s) U^T b without explicit diagonal matrices.


def solve_ridge_svd_filter(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:  # EN: Solve Ridge using SVD filter factors.
    if lam < 0.0:  # EN: Validate λ.
        raise ValueError("lam must be non-negative")  # EN: Reject invalid λ.
    if lam == 0.0:  # EN: Ridge with λ=0 reduces to unregularized LS.
        return solve_ls_svd_pinv(A, b)  # EN: Use SVD pseudo-inverse LS for stability.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD of A.
    if s.size == 0:  # EN: Handle degenerate shapes.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return zero vector.
    factors = s / (s**2 + lam)  # EN: Ridge filter factors σ/(σ^2+λ) that damp small-σ directions.
    return Vt.T @ (factors * (U.T @ b))  # EN: x = V diag(factors) U^T b using elementwise ops.


def solve_tsvd(A: np.ndarray, b: np.ndarray, k: int, rcond: float = RCOND) -> np.ndarray:  # EN: Solve using truncated SVD rank-k pseudo-inverse.
    if k <= 0:  # EN: Validate truncation rank.
        raise ValueError("k must be positive")  # EN: Reject invalid k.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD.
    if s.size == 0:  # EN: Handle degenerate shapes.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return zeros.
    cutoff = rcond * float(s.max())  # EN: Compute cutoff for numerical rank decisions.
    keep = s > cutoff  # EN: Identify non-negligible singular values.
    r_eff = int(np.count_nonzero(keep))  # EN: Effective numerical rank.
    k_eff = min(k, r_eff)  # EN: Clamp k to effective rank to avoid dividing by ~0 singular values.
    if k_eff == 0:  # EN: If effective rank is zero, return the minimum-norm zero solution.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return zeros.
    U_k = U[:, :k_eff]  # EN: Keep top-k left singular vectors.
    s_k = s[:k_eff]  # EN: Keep top-k singular values.
    Vt_k = Vt[:k_eff, :]  # EN: Keep top-k rows of V^T.
    return Vt_k.T @ ((U_k.T @ b) / s_k)  # EN: Compute truncated pseudo-inverse solution.


def summarize_cv(  # EN: Compute mean/std summaries for one hyperparameter across folds.
    splits: list[tuple[np.ndarray, np.ndarray]],  # EN: List of (train_idx, val_idx) splits.
    A: np.ndarray,  # EN: Full design matrix.
    b: np.ndarray,  # EN: Full target vector.
    solver: callable,  # EN: Solver mapping (A_train, b_train) -> x_hat.
    key: str,  # EN: Label for printing.
    param: float,  # EN: Numeric hyperparameter value for sorting/selection.
) -> CVPoint:  # EN: Return a CVPoint summary record.
    train_rmses: list[float] = []  # EN: Collect training RMSE per fold.
    val_rmses: list[float] = []  # EN: Collect validation RMSE per fold.
    x_norms: list[float] = []  # EN: Collect coefficient norms per fold.

    for train_idx, val_idx in splits:  # EN: Iterate folds and evaluate the solver on each split.
        A_train = A[train_idx, :]  # EN: Slice training design matrix.
        b_train = b[train_idx]  # EN: Slice training targets.
        A_val = A[val_idx, :]  # EN: Slice validation design matrix.
        b_val = b[val_idx]  # EN: Slice validation targets.

        x_hat = solver(A_train, b_train)  # EN: Fit model coefficients on the training fold.
        train_pred = A_train @ x_hat  # EN: Predict on training fold.
        val_pred = A_val @ x_hat  # EN: Predict on validation fold.

        train_rmses.append(rmse(b_train, train_pred))  # EN: Store training RMSE.
        val_rmses.append(rmse(b_val, val_pred))  # EN: Store validation RMSE.
        x_norms.append(l2_norm(x_hat))  # EN: Store coefficient norm as complexity proxy.

    train_mean = float(np.mean(train_rmses))  # EN: Compute mean training RMSE across folds.
    train_std = float(np.std(train_rmses, ddof=1)) if len(train_rmses) > 1 else 0.0  # EN: Compute sample std (or 0).
    val_mean = float(np.mean(val_rmses))  # EN: Compute mean validation RMSE across folds.
    val_std = float(np.std(val_rmses, ddof=1)) if len(val_rmses) > 1 else 0.0  # EN: Compute sample std (or 0).
    x_norm_mean = float(np.mean(x_norms))  # EN: Compute mean coefficient norm across folds.

    return CVPoint(  # EN: Construct CVPoint record.
        key=key,  # EN: Store label.
        param=param,  # EN: Store numeric hyperparameter value.
        train_mean=train_mean,  # EN: Store train RMSE mean.
        train_std=train_std,  # EN: Store train RMSE std.
        val_mean=val_mean,  # EN: Store val RMSE mean.
        val_std=val_std,  # EN: Store val RMSE std.
        x_norm_mean=x_norm_mean,  # EN: Store mean coefficient norm.
    )  # EN: End CVPoint construction.


def choose_best(points: list[CVPoint]) -> CVPoint:  # EN: Select the CVPoint with lowest validation mean (tie-break by smaller param).
    if not points:  # EN: Validate non-empty list.
        raise ValueError("points must be non-empty")  # EN: Reject invalid selection input.
    return min(points, key=lambda p: (p.val_mean, p.param))  # EN: Choose by val_mean, then by param for determinism.


def ascii_bar(  # EN: Build a bar where longer means better (lower validation RMSE).
    value: float,  # EN: Current value to visualize.
    vmin: float,  # EN: Minimum value in the series.
    vmax: float,  # EN: Maximum value in the series.
    width: int = 40,  # EN: Bar width in characters.
) -> str:  # EN: Return a string bar for ASCII plotting.
    if width <= 0:  # EN: Validate width.
        return ""  # EN: Return empty bar for non-positive widths.
    if vmax <= vmin + EPS:  # EN: Handle near-constant series to avoid divide-by-zero.
        return "#" * width  # EN: Return full bar when values are effectively identical.
    score = (vmax - value) / (vmax - vmin)  # EN: Map lower RMSE -> higher score in [0,1].
    score = float(np.clip(score, 0.0, 1.0))  # EN: Clamp score to [0,1].
    n = int(round(score * width))  # EN: Convert score to a character count.
    return "#" * n  # EN: Return the bar string.


def print_cv_table(points: list[CVPoint], best: CVPoint) -> None:  # EN: Print a CV results table and an ASCII curve.
    points_sorted = sorted(points, key=lambda p: p.param)  # EN: Sort points by numeric hyperparameter for readability.
    val_values = [p.val_mean for p in points_sorted]  # EN: Collect validation means for ASCII scaling.
    vmin = min(val_values)  # EN: Compute series minimum.
    vmax = max(val_values)  # EN: Compute series maximum.

    header = (  # EN: Build table header line.
        "param        | train_rmse(mean±std) | val_rmse(mean±std) | ||x||_2(mean) | curve (lower val -> longer bar)"  # EN: Column names.
    )  # EN: Finish building header.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Print separator line under header.

    for p in points_sorted:  # EN: Print one row per hyperparameter value.
        mark = " <== best" if p.key == best.key else ""  # EN: Mark the best hyperparameter choice.
        bar = ascii_bar(value=p.val_mean, vmin=vmin, vmax=vmax, width=32)  # EN: Build ASCII bar for visual curve.
        print(  # EN: Print a formatted row with metrics and curve bar.
            f"{p.key:12} | "  # EN: Print param label.
            f"{p.train_mean:.3e}±{p.train_std:.1e} | "  # EN: Print train RMSE mean±std.
            f"{p.val_mean:.3e}±{p.val_std:.1e} | "  # EN: Print val RMSE mean±std.
            f"{p.x_norm_mean:.3e} | "  # EN: Print coefficient norm mean.
            f"{bar}{mark}"  # EN: Print curve bar and optional best marker.
        )  # EN: End print row.


def main() -> None:  # EN: Run k-fold CV for Ridge λ and TSVD k and print ASCII curves.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    n_samples = 80  # EN: Number of samples in the synthetic regression dataset.
    degree = 12  # EN: Polynomial degree; larger degree increases ill-conditioning.
    noise_std = 0.05  # EN: Observation noise level.
    n_folds = 5  # EN: k for k-fold cross-validation.

    t = np.linspace(-1.0, 1.0, n_samples)  # EN: Generate sample locations.
    A = build_vandermonde(t, degree=degree)  # EN: Build polynomial design matrix.
    n_features = A.shape[1]  # EN: Number of coefficients in the polynomial model.

    x_true = np.zeros((n_features,), dtype=float)  # EN: Initialize sparse ground-truth coefficients.
    x_true[0] = 0.5  # EN: Intercept coefficient.
    x_true[1] = 1.0  # EN: Linear coefficient.
    x_true[2] = -2.0  # EN: Quadratic coefficient.
    x_true[3] = 0.7  # EN: Cubic coefficient.

    b_clean = A @ x_true  # EN: Generate noiseless targets.
    b = b_clean + noise_std * rng.standard_normal(b_clean.shape)  # EN: Add one noise realization.

    splits = k_fold_splits(rng=rng, n_samples=n_samples, n_folds=n_folds)  # EN: Create k-fold splits.

    print_separator("Dataset Summary")  # EN: Print dataset diagnostics section.
    cond_A = float(np.linalg.cond(A))  # EN: Compute condition number to show ill-conditioning.
    rank_A = int(np.linalg.matrix_rank(A))  # EN: Compute numerical rank for context.
    s = np.linalg.svd(A, compute_uv=False)  # EN: Compute singular values for additional insight.
    print(f"n_samples={n_samples}, degree={degree}, n_features={n_features}, n_folds={n_folds}")  # EN: Print dataset sizes.
    print(f"rank(A)={rank_A}, cond(A)={cond_A:.3e}")  # EN: Print rank and condition number.
    print(f"singular values(A) = {s}")  # EN: Print singular values.
    print(f"noise_std={noise_std}")  # EN: Print noise level.

    print_separator("k-fold CV: Ridge λ")  # EN: Start Ridge λ selection section.
    ridge_lambdas = np.concatenate(([0.0], np.logspace(-12, 2, num=15)))  # EN: Candidate λ values including λ=0 baseline.
    ridge_points: list[CVPoint] = []  # EN: Collect CV summaries for each λ.
    for lam in ridge_lambdas:  # EN: Evaluate each λ.
        point = summarize_cv(  # EN: Compute CV summary for this λ.
            splits=splits,  # EN: Provide CV splits.
            A=A,  # EN: Provide design matrix.
            b=b,  # EN: Provide targets.
            solver=lambda A_in, b_in, lam=lam: solve_ridge_svd_filter(A_in, b_in, lam),  # EN: Ridge solver bound to λ.
            key=f"λ={lam:.0e}" if lam != 0.0 else "λ=0",  # EN: Compact label formatting (scientific notation).
            param=float(lam),  # EN: Store numeric λ for sorting/selection.
        )  # EN: End summarize_cv call.
        ridge_points.append(point)  # EN: Store result.

    best_ridge = choose_best(ridge_points)  # EN: Choose λ with minimum mean validation RMSE.
    print_cv_table(ridge_points, best=best_ridge)  # EN: Print Ridge table and ASCII curve.
    print(f"\nChosen Ridge λ by CV: {best_ridge.key} (val_mean={best_ridge.val_mean:.3e})")  # EN: Print chosen λ summary.

    print_separator("k-fold CV: TSVD k")  # EN: Start TSVD k selection section.
    tsvd_points: list[CVPoint] = []  # EN: Collect CV summaries for each k.
    for k in range(1, n_features + 1):  # EN: Evaluate k from 1..n_features.
        point = summarize_cv(  # EN: Compute CV summary for this k.
            splits=splits,  # EN: Provide CV splits.
            A=A,  # EN: Provide design matrix.
            b=b,  # EN: Provide targets.
            solver=lambda A_in, b_in, k=k: solve_tsvd(A_in, b_in, k=k),  # EN: TSVD solver bound to k.
            key=f"k={k:02d}",  # EN: Label k with fixed width for table alignment.
            param=float(k),  # EN: Store numeric k for sorting/selection.
        )  # EN: End summarize_cv call.
        tsvd_points.append(point)  # EN: Store CV point.

    best_tsvd = choose_best(tsvd_points)  # EN: Choose k with minimum mean validation RMSE.
    print_cv_table(tsvd_points, best=best_tsvd)  # EN: Print TSVD table and ASCII curve.
    print(f"\nChosen TSVD k by CV: {best_tsvd.key} (val_mean={best_tsvd.val_mean:.3e})")  # EN: Print chosen k summary.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard Python entrypoint guard.
    main()  # EN: Execute main when run as a script.

