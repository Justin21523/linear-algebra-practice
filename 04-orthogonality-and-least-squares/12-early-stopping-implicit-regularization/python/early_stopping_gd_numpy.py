"""  # EN: Start module docstring.
Early stopping as implicit regularization (NumPy): choose the number of gradient descent steps T using a validation set.  # EN: Summarize the goal.

We solve an unregularized least squares problem:  # EN: Describe the base objective.
  min_x f(x) = 1/2||Ax - b||^2.  # EN: State objective.

On ill-conditioned problems (e.g., polynomial / Vandermonde features), running GD too long can start fitting noise.  # EN: Explain why over-iteration hurts.
Stopping early (small T) often improves generalization, similar to explicit regularizers like Ridge or TSVD.  # EN: Connect to explicit regularization.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for a clean result record.

import numpy as np  # EN: Import NumPy for linear algebra.


EPS = 1e-12  # EN: Small epsilon to avoid division by zero.
SEED = 0  # EN: RNG seed for deterministic output.
PRINT_PRECISION = 6  # EN: Console float precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


@dataclass(frozen=True)  # EN: Immutable record for one early-stopping checkpoint.
class Checkpoint:  # EN: Store metrics at a specific iteration count T.
    T: int  # EN: Iteration count (number of GD steps).
    train_rmse: float  # EN: RMSE on training set.
    val_rmse: float  # EN: RMSE on validation set.
    x_norm: float  # EN: Coefficient norm ||x||_2 (complexity proxy).


def print_separator(title: str) -> None:  # EN: Print a readable separator.
    print()  # EN: Add a blank line before a section.
    print("=" * 78)  # EN: Print a divider line.
    print(title)  # EN: Print section title.
    print("=" * 78)  # EN: Print closing divider.


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # EN: Compute root-mean-square error.
    err = y_pred - y_true  # EN: Compute prediction error.
    return float(np.sqrt(np.mean(err**2)))  # EN: Return RMSE.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def build_vandermonde(t: np.ndarray, degree: int) -> np.ndarray:  # EN: Build polynomial feature matrix with powers 0..degree.
    if t.ndim != 1:  # EN: Validate input shape.
        raise ValueError("t must be a 1D array")  # EN: Reject invalid input.
    if degree < 0:  # EN: Validate degree.
        raise ValueError("degree must be non-negative")  # EN: Reject invalid degree.
    return np.vander(t, N=degree + 1, increasing=True).astype(float)  # EN: Return Vandermonde design matrix.


def split_train_val(  # EN: Deterministically split indices into train/validation sets.
    rng: np.random.Generator,  # EN: RNG used to shuffle indices.
    n_samples: int,  # EN: Total sample count.
    val_fraction: float,  # EN: Fraction for validation set.
) -> tuple[np.ndarray, np.ndarray]:  # EN: Return (train_idx, val_idx).
    if not (0.0 < val_fraction < 1.0):  # EN: Validate fraction.
        raise ValueError("val_fraction must be in (0,1)")  # EN: Reject invalid values.
    perm = rng.permutation(n_samples)  # EN: Shuffle indices.
    n_val = int(round(val_fraction * n_samples))  # EN: Compute val size.
    n_val = max(1, min(n_val, n_samples - 1))  # EN: Ensure both splits are non-empty.
    val_idx = perm[:n_val]  # EN: Take first slice as validation.
    train_idx = perm[n_val:]  # EN: Remaining indices are training.
    return train_idx, val_idx  # EN: Return index arrays.


def least_squares_objective(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> float:  # EN: Compute f(x)=1/2||Ax-b||^2.
    r = A @ x - b  # EN: Compute residual.
    return 0.5 * float(r @ r)  # EN: Return half squared residual norm.


def least_squares_gradient(A: np.ndarray, b: np.ndarray, x: np.ndarray) -> np.ndarray:  # EN: Compute gradient ∇f = A^T(Ax-b).
    return A.T @ (A @ x - b)  # EN: Return gradient vector.


def lipschitz_constant(A: np.ndarray) -> float:  # EN: Compute L = ||A||_2^2 for LS gradient.
    s = np.linalg.svd(A, compute_uv=False)  # EN: Singular values of A.
    sigma_max = float(s.max()) if s.size else 0.0  # EN: Largest singular value (spectral norm).
    return sigma_max**2  # EN: Return L = sigma_max^2.


def ascii_bar(value: float, vmin: float, vmax: float, width: int = 32) -> str:  # EN: Create an ASCII bar where longer is better (lower RMSE).
    if width <= 0:  # EN: Validate width.
        return ""  # EN: Return empty bar.
    if vmax <= vmin + EPS:  # EN: Handle near-constant range.
        return "#" * width  # EN: Return full bar.
    score = (vmax - value) / (vmax - vmin)  # EN: Map lower RMSE -> higher score in [0,1].
    score = float(np.clip(score, 0.0, 1.0))  # EN: Clamp score.
    n = int(round(score * width))  # EN: Convert score to bar length.
    return "#" * n  # EN: Return bar string.


def main() -> None:  # EN: Run early-stopping sweep on a polynomial regression dataset.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    n_samples = 60  # EN: Sample count.
    degree = 12  # EN: Polynomial degree -> ill-conditioned Vandermonde features.
    noise_std = 0.05  # EN: Observation noise level.
    val_fraction = 0.30  # EN: Hold-out validation fraction.

    t = np.linspace(-1.0, 1.0, n_samples)  # EN: Sample locations.
    A = build_vandermonde(t, degree=degree)  # EN: Build design matrix.
    n_features = A.shape[1]  # EN: Feature count.

    x_true = np.zeros((n_features,), dtype=float)  # EN: Define a sparse ground-truth polynomial.
    x_true[0] = 0.5  # EN: Intercept term.
    x_true[1] = 1.0  # EN: Linear term.
    x_true[2] = -2.0  # EN: Quadratic term.
    x_true[3] = 0.7  # EN: Cubic term.

    b_clean = A @ x_true  # EN: Noiseless targets.
    b = b_clean + noise_std * rng.standard_normal(b_clean.shape)  # EN: Add noise to create observed targets.

    train_idx, val_idx = split_train_val(rng=rng, n_samples=n_samples, val_fraction=val_fraction)  # EN: Create hold-out split.
    A_train = A[train_idx, :]  # EN: Training design matrix.
    b_train = b[train_idx]  # EN: Training targets.
    A_val = A[val_idx, :]  # EN: Validation design matrix.
    b_val = b[val_idx]  # EN: Validation targets.

    L = lipschitz_constant(A_train)  # EN: Compute Lipschitz constant for training objective.
    step = 1.0 / max(L, EPS)  # EN: Use safe step size α=1/L.

    print_separator("Dataset Summary")  # EN: Print dataset diagnostics.
    cond = float(np.linalg.cond(A_train))  # EN: Condition number of A_train.
    print(f"n_samples={n_samples}, degree={degree}, n_features={n_features}, val_fraction={val_fraction}")  # EN: Print sizes.
    print(f"cond(A_train)={cond:.3e}, L=||A||_2^2={L:.3e}, step=1/L={step:.3e}")  # EN: Print conditioning and step.
    print(f"noise_std={noise_std}")  # EN: Print noise level.

    # EN: Choose checkpoint iteration counts (T) that cover early, mid, and late phases.
    checkpoints_T = [0, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 4000]  # EN: Candidate early-stopping iteration counts.
    T_max = max(checkpoints_T)  # EN: Maximum iteration to run.

    x = np.zeros((n_features,), dtype=float)  # EN: Initialize GD at zero.
    ckpts: list[Checkpoint] = []  # EN: Collect checkpoint metrics.
    checkpoint_set = set(checkpoints_T)  # EN: Use a set for O(1) membership checks.

    for t_iter in range(T_max + 1):  # EN: Run GD for T_max steps, recording selected checkpoints.
        if t_iter in checkpoint_set:  # EN: Record metrics at this checkpoint iteration.
            train_pred = A_train @ x  # EN: Predict on training set.
            val_pred = A_val @ x  # EN: Predict on validation set.
            ckpts.append(  # EN: Store checkpoint record.
                Checkpoint(  # EN: Construct checkpoint record.
                    T=t_iter,  # EN: Store iteration count.
                    train_rmse=rmse(b_train, train_pred),  # EN: Compute train RMSE.
                    val_rmse=rmse(b_val, val_pred),  # EN: Compute validation RMSE.
                    x_norm=l2_norm(x),  # EN: Compute coefficient norm.
                )  # EN: End checkpoint construction.
            )  # EN: End append.

        if t_iter == T_max:  # EN: Do not update beyond the last checkpoint.
            break  # EN: Exit loop.

        g = least_squares_gradient(A_train, b_train, x)  # EN: Compute gradient on training split.
        x = x - step * g  # EN: GD update step.

    best = min(ckpts, key=lambda c: c.val_rmse)  # EN: Choose checkpoint with minimum validation RMSE.

    val_values = [c.val_rmse for c in ckpts]  # EN: Collect val RMSE values for ASCII scaling.
    vmin = min(val_values)  # EN: Minimum val RMSE.
    vmax = max(val_values)  # EN: Maximum val RMSE.

    print_separator("Early Stopping Sweep (choose T by validation RMSE)")  # EN: Print sweep results.
    print("T    | train_rmse | val_rmse | ||x||_2 | curve (lower val -> longer bar)")  # EN: Print table header.
    print("-" * 74)  # EN: Print header separator.
    for c in ckpts:  # EN: Print each checkpoint row.
        mark = " <== best" if c.T == best.T else ""  # EN: Mark the best checkpoint.
        bar = ascii_bar(value=c.val_rmse, vmin=vmin, vmax=vmax, width=28)  # EN: Build ASCII bar for val RMSE.
        print(f"{c.T:4d} | {c.train_rmse:9.3e} | {c.val_rmse:8.3e} | {c.x_norm:7.3e} | {bar}{mark}")  # EN: Print row.

    print_separator("Best Early-Stopping Choice")  # EN: Print best choice summary.
    print(f"Best T = {best.T} with val_rmse={best.val_rmse:.3e} (train_rmse={best.train_rmse:.3e}, ||x||={best.x_norm:.3e})")  # EN: Print best checkpoint.

    # EN: Print a small interpretation hint using objective values.
    f0 = least_squares_objective(A_train, b_train, np.zeros_like(x))  # EN: Objective at initialization.
    f_end = least_squares_objective(A_train, b_train, x)  # EN: Objective at final iterate (T_max).
    n_train = A_train.shape[0]  # EN: Training sample count for mapping RMSE to objective.
    f_at_best_T = 0.5 * (best.train_rmse**2) * n_train  # EN: Convert RMSE to objective: 0.5 * sum(err^2).
    print(f"Training objective: f(x0)={f0:.3e}, f(x_bestT)≈{f_at_best_T:.3e}, f(x_Tmax)={f_end:.3e}")  # EN: Print objective comparison.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main.
