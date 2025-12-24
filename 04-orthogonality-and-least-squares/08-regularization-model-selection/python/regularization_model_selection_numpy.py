"""  # EN: Start module docstring.
Regularization model selection (NumPy): choose Ridge λ and TSVD k with a hold-out set,  # EN: Describe the goal.
and observe bias–variance with repeated noise experiments.  # EN: Describe the second goal.

We use a polynomial design matrix (Vandermonde) to intentionally create an ill-conditioned regression problem.  # EN: Explain why we use polynomials.
This mirrors ML situations where features become highly correlated (multicollinearity) or models become too flexible.  # EN: Connect to ML motivation.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for clean, typed result objects.

import numpy as np  # EN: Import NumPy for numerical linear algebra.


EPS = 1e-12  # EN: Small epsilon to avoid division by zero in relative computations.
RCOND = 1e-12  # EN: Relative cutoff for pseudo-inverse / truncated SVD rank decisions.
SEED = 0  # EN: RNG seed for deterministic demos.
PRINT_PRECISION = 6  # EN: Default float printing precision for console output.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing for readability.


@dataclass(frozen=True)  # EN: Use an immutable record to store (train, val) metrics for one hyperparameter.
class FitMetrics:  # EN: Store errors and coefficient norms for selection tables.
    key: str  # EN: Hyperparameter label (e.g., "λ=1e-3" or "k=5").
    train_rmse: float  # EN: Root-mean-square error on training set.
    val_rmse: float  # EN: Root-mean-square error on validation set.
    x_norm: float  # EN: Coefficient norm ||x||_2 (shrinkage indicator).


@dataclass(frozen=True)  # EN: Store bias/variance summary statistics for a method under repeated noise.
class BiasVarianceReport:  # EN: Summarize bias–variance and validation error variability.
    method: str  # EN: Method name (LS / Ridge / TSVD).
    hyperparam: str  # EN: Chosen hyperparameter string (e.g., best λ or best k).
    bias_norm: float  # EN: ||E[x_hat] - x_true||_2 as a bias proxy.
    variance_rms: float  # EN: RMS deviation of x_hat around its mean as a variance proxy.
    val_rmse_mean: float  # EN: Mean validation RMSE over trials.
    val_rmse_std: float  # EN: Std validation RMSE over trials.


def print_separator(title: str) -> None:  # EN: Print a section separator.
    print()  # EN: Add spacing between sections.
    print("=" * 78)  # EN: Print a horizontal rule.
    print(title)  # EN: Print the section title.
    print("=" * 78)  # EN: Print a closing rule.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (vector 2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy's stable norm implementation.


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # EN: Compute root-mean-square error between vectors.
    err = y_pred - y_true  # EN: Compute prediction error vector.
    return float(np.sqrt(np.mean(err**2)))  # EN: Return RMSE as sqrt(mean squared error).


def build_vandermonde(t: np.ndarray, degree: int) -> np.ndarray:  # EN: Build polynomial feature matrix with powers 0..degree.
    if t.ndim != 1:  # EN: Ensure t is a 1D vector of sample locations.
        raise ValueError("t must be a 1D array")  # EN: Fail fast on invalid shapes.
    if degree < 0:  # EN: Validate degree.
        raise ValueError("degree must be non-negative")  # EN: Reject invalid degree values.
    return np.vander(t, N=degree + 1, increasing=True).astype(float)  # EN: Return Vandermonde matrix (n_samples×(degree+1)).


def split_train_val(  # EN: Create a deterministic train/validation split.
    rng: np.random.Generator,  # EN: RNG for shuffling indices.
    n_samples: int,  # EN: Total number of samples.
    val_fraction: float,  # EN: Fraction of samples assigned to validation.
) -> tuple[np.ndarray, np.ndarray]:  # EN: Return (train_idx, val_idx).
    if not (0.0 < val_fraction < 1.0):  # EN: Validate split ratio.
        raise ValueError("val_fraction must be in (0, 1)")  # EN: Reject invalid split ratio.
    perm = rng.permutation(n_samples)  # EN: Randomly permute indices.
    n_val = int(round(val_fraction * n_samples))  # EN: Compute number of validation samples.
    n_val = max(1, min(n_val, n_samples - 1))  # EN: Ensure both splits are non-empty.
    val_idx = perm[:n_val]  # EN: Take first part as validation indices.
    train_idx = perm[n_val:]  # EN: Take remaining part as training indices.
    return train_idx, val_idx  # EN: Return both index arrays.


def solve_ls_svd_pinv(A: np.ndarray, b: np.ndarray, rcond: float = RCOND) -> np.ndarray:  # EN: Solve unregularized least squares using SVD pseudo-inverse.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD A = U diag(s) V^T.
    if s.size == 0:  # EN: Handle degenerate shapes defensively.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return a zero vector solution.
    cutoff = rcond * float(s.max())  # EN: Compute absolute cutoff for tiny singular values.
    keep = s > cutoff  # EN: Keep only numerically significant singular values.
    if not np.any(keep):  # EN: If effective rank is zero, the minimum-norm solution is zeros.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return zeros as the min-norm solution.
    U_r = U[:, keep]  # EN: Select kept left singular vectors.
    s_r = s[keep]  # EN: Select kept singular values.
    Vt_r = Vt[keep, :]  # EN: Select kept right singular vectors transposed.
    return Vt_r.T @ ((U_r.T @ b) / s_r)  # EN: Compute x = V diag(1/s) U^T b without explicit diag.


def solve_ridge_svd_filter(A: np.ndarray, b: np.ndarray, lam: float) -> np.ndarray:  # EN: Solve Ridge via SVD filter factors (penalty λ||x||^2).
    if lam < 0.0:  # EN: Validate λ.
        raise ValueError("lam must be non-negative")  # EN: Reject invalid regularization strengths.
    if lam == 0.0:  # EN: Ridge with λ=0 reduces to unregularized least squares.
        return solve_ls_svd_pinv(A, b)  # EN: Delegate to SVD pseudo-inverse LS solver for numerical safety.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD for filter-factor formula.
    if s.size == 0:  # EN: Defensive handling for empty matrices.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return a zero vector.
    factors = s / (s**2 + lam)  # EN: Ridge filter factors σ/(σ^2+λ), damping small-σ directions.
    return Vt.T @ (factors * (U.T @ b))  # EN: x = V diag(factors) U^T b implemented with elementwise multiplication.


def solve_tsvd(A: np.ndarray, b: np.ndarray, k: int, rcond: float = RCOND) -> np.ndarray:  # EN: Solve via truncated SVD rank-k pseudo-inverse.
    if k <= 0:  # EN: Validate k.
        raise ValueError("k must be positive")  # EN: Reject invalid truncation rank.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Compute economy SVD A = U diag(s) V^T.
    if s.size == 0:  # EN: Handle degenerate shapes defensively.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return a zero vector solution.
    cutoff = rcond * float(s.max())  # EN: Compute absolute cutoff for tiny singular values.
    keep = s > cutoff  # EN: Identify numerically significant singular values.
    r_eff = int(np.count_nonzero(keep))  # EN: Effective numerical rank.
    k_eff = min(k, r_eff)  # EN: Clamp k to the effective rank.
    if k_eff == 0:  # EN: If effective rank is 0, return the min-norm zero solution.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return zeros.
    U_k = U[:, :k_eff]  # EN: Keep top-k left singular vectors.
    s_k = s[:k_eff]  # EN: Keep top-k singular values.
    Vt_k = Vt[:k_eff, :]  # EN: Keep top-k rows of V^T.
    return Vt_k.T @ ((U_k.T @ b) / s_k)  # EN: Compute truncated pseudo-inverse solution.


def fit_and_score(  # EN: Fit one method on train set and compute RMSE on train/val plus ||x||.
    A_train: np.ndarray,  # EN: Training design matrix.
    b_train: np.ndarray,  # EN: Training targets.
    A_val: np.ndarray,  # EN: Validation design matrix.
    b_val: np.ndarray,  # EN: Validation targets.
    solver: callable,  # EN: Solver function returning x_hat.
    solver_key: str,  # EN: Label for the hyperparameter/method used.
) -> FitMetrics:  # EN: Return FitMetrics containing scores and coefficient norm.
    x_hat = solver(A_train, b_train)  # EN: Fit coefficients on training set.
    train_pred = A_train @ x_hat  # EN: Compute predictions on training set.
    val_pred = A_val @ x_hat  # EN: Compute predictions on validation set.
    return FitMetrics(  # EN: Construct metrics record.
        key=solver_key,  # EN: Store hyperparameter label.
        train_rmse=rmse(b_train, train_pred),  # EN: Compute training RMSE.
        val_rmse=rmse(b_val, val_pred),  # EN: Compute validation RMSE.
        x_norm=l2_norm(x_hat),  # EN: Store coefficient norm for shrinkage comparison.
    )  # EN: End FitMetrics construction.


def select_best_by_val(metrics: list[FitMetrics]) -> FitMetrics:  # EN: Choose the metric entry with smallest validation RMSE.
    if not metrics:  # EN: Validate non-empty list.
        raise ValueError("metrics must be non-empty")  # EN: Reject empty selection input.
    return min(metrics, key=lambda m: m.val_rmse)  # EN: Return the entry with minimum validation error.


def bias_variance_on_coefficients(x_hats: np.ndarray, x_true: np.ndarray) -> tuple[float, float]:  # EN: Compute bias and variance proxies from coefficient samples.
    mean_x = x_hats.mean(axis=0)  # EN: Compute E[x_hat] as sample mean across trials.
    bias_norm = l2_norm(mean_x - x_true)  # EN: Bias proxy: ||E[x_hat] - x_true||_2.
    centered = x_hats - mean_x  # EN: Center coefficient samples to measure variability around mean.
    variance_rms = float(np.sqrt(np.mean(np.sum(centered**2, axis=1))))  # EN: RMS deviation across trials as variance proxy.
    return bias_norm, variance_rms  # EN: Return both summary statistics.


def run_bias_variance_experiment(  # EN: Run repeated-noise trials and summarize bias–variance for chosen methods.
    rng: np.random.Generator,  # EN: RNG used for generating noise per trial.
    A: np.ndarray,  # EN: Full design matrix (n_samples×n_features).
    x_true: np.ndarray,  # EN: True coefficient vector.
    train_idx: np.ndarray,  # EN: Training indices.
    val_idx: np.ndarray,  # EN: Validation indices.
    noise_std: float,  # EN: Noise standard deviation for observations.
    n_trials: int,  # EN: Number of noise realizations.
    best_lam: float,  # EN: Selected Ridge λ from hold-out.
    best_k: int,  # EN: Selected TSVD k from hold-out.
) -> list[BiasVarianceReport]:  # EN: Return a list of BiasVarianceReport entries.
    A_train = A[train_idx, :]  # EN: Slice training design matrix.
    A_val = A[val_idx, :]  # EN: Slice validation design matrix.
    b_clean = A @ x_true  # EN: Compute noiseless targets once.

    methods: list[tuple[str, str, callable]] = [  # EN: Define the method set to compare.
        ("LS-SVD(pinv)", "λ=0", lambda A_in, b_in: solve_ls_svd_pinv(A_in, b_in)),  # EN: Unregularized LS baseline.
        ("Ridge-SVD", f"λ={best_lam:.3e}", lambda A_in, b_in: solve_ridge_svd_filter(A_in, b_in, best_lam)),  # EN: Ridge with chosen λ.
        ("TSVD", f"k={best_k}", lambda A_in, b_in: solve_tsvd(A_in, b_in, best_k)),  # EN: TSVD with chosen k.
    ]  # EN: End methods list.

    reports: list[BiasVarianceReport] = []  # EN: Collect bias–variance summaries per method.

    for method_name, hyperparam_str, solver in methods:  # EN: Evaluate each method under repeated noise.
        x_samples: list[np.ndarray] = []  # EN: Collect x_hat samples for bias/variance computation.
        val_rmses: list[float] = []  # EN: Collect validation RMSEs per trial for stability/generalization view.

        for _ in range(n_trials):  # EN: Repeat experiment under different noise realizations.
            noise = noise_std * rng.standard_normal(b_clean.shape)  # EN: Generate new noise vector.
            b_noisy = b_clean + noise  # EN: Create noisy observations.
            b_train = b_noisy[train_idx]  # EN: Slice training targets.
            b_val = b_noisy[val_idx]  # EN: Slice validation targets.

            x_hat = solver(A_train, b_train)  # EN: Fit coefficients on the training split.
            val_pred = A_val @ x_hat  # EN: Predict on validation split.
            val_rmses.append(rmse(b_val, val_pred))  # EN: Store validation RMSE for this trial.
            x_samples.append(x_hat)  # EN: Store coefficient vector sample.

        x_hats = np.stack(x_samples, axis=0)  # EN: Convert list of vectors into an array (n_trials×n_features).
        bias_norm, variance_rms = bias_variance_on_coefficients(x_hats, x_true)  # EN: Compute bias/variance proxies.
        val_rmse_mean = float(np.mean(val_rmses))  # EN: Compute mean validation RMSE.
        val_rmse_std = float(np.std(val_rmses, ddof=1))  # EN: Compute std of validation RMSE (sample std).

        reports.append(  # EN: Append the summary report for this method.
            BiasVarianceReport(  # EN: Construct BiasVarianceReport record.
                method=method_name,  # EN: Store method name.
                hyperparam=hyperparam_str,  # EN: Store chosen hyperparameter.
                bias_norm=bias_norm,  # EN: Store bias proxy.
                variance_rms=variance_rms,  # EN: Store variance proxy.
                val_rmse_mean=val_rmse_mean,  # EN: Store mean validation RMSE.
                val_rmse_std=val_rmse_std,  # EN: Store validation RMSE std.
            )  # EN: End BiasVarianceReport construction.
        )  # EN: End append.

    return reports  # EN: Return list of method summaries.


def main() -> None:  # EN: Entrypoint to run hold-out selection and bias–variance experiment.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    n_samples = 60  # EN: Number of samples for the polynomial regression dataset.
    degree = 12  # EN: Polynomial degree (features = degree+1); higher degree increases ill-conditioning.
    noise_std = 0.05  # EN: Noise level for observations.
    val_fraction = 0.30  # EN: Hold-out validation fraction.

    t = np.linspace(-1.0, 1.0, n_samples)  # EN: Use evenly spaced sample locations.
    A = build_vandermonde(t, degree=degree)  # EN: Build Vandermonde design matrix.
    n_features = A.shape[1]  # EN: Number of polynomial coefficients.

    x_true = np.zeros((n_features,), dtype=float)  # EN: Initialize a sparse ground-truth polynomial.
    x_true[0] = 0.5  # EN: Intercept term.
    x_true[1] = 1.0  # EN: Linear term.
    x_true[2] = -2.0  # EN: Quadratic term.
    x_true[3] = 0.7  # EN: Cubic term.

    b_clean = A @ x_true  # EN: Compute noiseless outputs.
    b = b_clean + noise_std * rng.standard_normal(b_clean.shape)  # EN: Add one noise realization for hold-out selection.

    train_idx, val_idx = split_train_val(rng=rng, n_samples=n_samples, val_fraction=val_fraction)  # EN: Create hold-out split.
    A_train = A[train_idx, :]  # EN: Slice training design matrix.
    b_train = b[train_idx]  # EN: Slice training targets.
    A_val = A[val_idx, :]  # EN: Slice validation design matrix.
    b_val = b[val_idx]  # EN: Slice validation targets.

    print_separator("Dataset Summary (Polynomial / Vandermonde)")  # EN: Print dataset summary header.
    cond_A = float(np.linalg.cond(A_train))  # EN: Compute condition number of training design matrix.
    rank_A = int(np.linalg.matrix_rank(A_train))  # EN: Compute numerical rank of A_train.
    s = np.linalg.svd(A_train, compute_uv=False)  # EN: Compute singular values for context.
    print(f"n_samples={n_samples}, degree={degree}, n_features={n_features}, val_fraction={val_fraction}")  # EN: Print dataset sizes.
    print(f"rank(A_train)={rank_A}, cond(A_train)={cond_A:.3e}")  # EN: Print rank and condition number.
    print(f"singular values (A_train) = {s}")  # EN: Print singular values.
    print(f"x_true (first 6 coeffs) = {x_true[:6]}")  # EN: Print truncated true coefficient vector.
    print(f"noise_std={noise_std}")  # EN: Print noise level.

    print_separator("Hold-out Selection: Ridge λ")  # EN: Announce Ridge selection section.
    ridge_lambdas = np.concatenate(([0.0], np.logspace(-12, 2, num=15)))  # EN: Candidate λ values including λ=0 baseline.
    ridge_metrics: list[FitMetrics] = []  # EN: Collect metrics for each λ candidate.
    for lam in ridge_lambdas:  # EN: Loop over λ candidates.
        metrics = fit_and_score(  # EN: Fit Ridge for this λ and compute train/val RMSE.
            A_train=A_train,  # EN: Provide A_train.
            b_train=b_train,  # EN: Provide b_train.
            A_val=A_val,  # EN: Provide A_val.
            b_val=b_val,  # EN: Provide b_val.
            solver=lambda A_in, b_in, lam=lam: solve_ridge_svd_filter(A_in, b_in, lam),  # EN: Bind λ into solver.
            solver_key=f"λ={lam:.3e}",  # EN: Label this configuration for printing.
        )  # EN: End fit_and_score call.
        ridge_metrics.append(metrics)  # EN: Store metrics for selection.

    best_ridge = select_best_by_val(ridge_metrics)  # EN: Select λ with minimum validation RMSE.
    for m in ridge_metrics:  # EN: Print a compact table of results for Ridge.
        print(f"{m.key:12} | train_rmse={m.train_rmse:.3e} | val_rmse={m.val_rmse:.3e} | ||x||={m.x_norm:.3e}")  # EN: Print one row.
    print(f"\nBest Ridge by val RMSE -> {best_ridge.key} (val_rmse={best_ridge.val_rmse:.3e})")  # EN: Print chosen λ.

    print_separator("Hold-out Selection: TSVD k")  # EN: Announce TSVD selection section.
    tsvd_ks = list(range(1, n_features + 1))  # EN: Candidate k values from 1..n_features.
    tsvd_metrics: list[FitMetrics] = []  # EN: Collect metrics for each k candidate.
    for k in tsvd_ks:  # EN: Loop over k candidates.
        metrics = fit_and_score(  # EN: Fit TSVD for this k and compute train/val RMSE.
            A_train=A_train,  # EN: Provide A_train.
            b_train=b_train,  # EN: Provide b_train.
            A_val=A_val,  # EN: Provide A_val.
            b_val=b_val,  # EN: Provide b_val.
            solver=lambda A_in, b_in, k=k: solve_tsvd(A_in, b_in, k),  # EN: Bind k into solver.
            solver_key=f"k={k:02d}",  # EN: Label this configuration.
        )  # EN: End fit_and_score call.
        tsvd_metrics.append(metrics)  # EN: Store metrics for selection.

    best_tsvd = select_best_by_val(tsvd_metrics)  # EN: Select k with minimum validation RMSE.
    for m in tsvd_metrics:  # EN: Print a compact table of results for TSVD.
        print(f"{m.key:6} | train_rmse={m.train_rmse:.3e} | val_rmse={m.val_rmse:.3e} | ||x||={m.x_norm:.3e}")  # EN: Print one row.
    print(f"\nBest TSVD by val RMSE -> {best_tsvd.key} (val_rmse={best_tsvd.val_rmse:.3e})")  # EN: Print chosen k.

    print_separator("Bias–Variance Experiment (Repeated Noise)")  # EN: Announce bias–variance experiment section.
    n_trials = 200  # EN: Number of repeated noise draws to estimate bias/variance proxies.

    best_lam_value = float(best_ridge.key.split("=")[1])  # EN: Parse λ value from label string "λ=...".
    best_k_value = int(best_tsvd.key.split("=")[1])  # EN: Parse k value from label string "k=NN".

    bv_reports = run_bias_variance_experiment(  # EN: Run repeated-noise experiment for selected methods.
        rng=rng,  # EN: Provide RNG.
        A=A,  # EN: Provide full design matrix (used to build b_clean each trial).
        x_true=x_true,  # EN: Provide ground-truth coefficients.
        train_idx=train_idx,  # EN: Provide train indices.
        val_idx=val_idx,  # EN: Provide val indices.
        noise_std=noise_std,  # EN: Provide noise level.
        n_trials=n_trials,  # EN: Provide number of trials.
        best_lam=best_lam_value,  # EN: Provide chosen λ.
        best_k=best_k_value,  # EN: Provide chosen k.
    )  # EN: End experiment call.

    print("method        | hyperparam     | bias ||E[x]-x*|| | var RMS | val_rmse_mean ± std")  # EN: Print BV table header.
    print("-" * 86)  # EN: Print header underline.
    for rep in bv_reports:  # EN: Print one row per method.
        print(  # EN: Print formatted BV summary row.
            f"{rep.method:12} | {rep.hyperparam:12} | "  # EN: Print method/hyperparam columns.
            f"{rep.bias_norm:.3e} | {rep.variance_rms:.3e} | "  # EN: Print bias and variance proxies.
            f"{rep.val_rmse_mean:.3e} ± {rep.val_rmse_std:.3e}"  # EN: Print mean±std of validation RMSE.
        )  # EN: End print call.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard script entrypoint guard.
    main()  # EN: Run the demo.

