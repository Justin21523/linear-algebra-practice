"""  # EN: Start module docstring.
Damped LSMR + preconditioning + k-fold CV cost comparison (NumPy).  # EN: Summarize this script in one line.

We solve Ridge (damped least squares):  # EN: State the main optimization problem.
  min_x ||A x - b||_2^2 + damp^2 ||x||_2^2,  damp >= 0.  # EN: Define the objective and the damping parameter.

We compare three right-preconditioning strategies in the SAME problem:  # EN: Explain what is compared.
  1) none (M = I)  # EN: Baseline.
  2) column scaling (M = diag(column norms of [A; damp I]))  # EN: Cheap diagonal preconditioner.
  3) randomized-QR (M = R from QR(S [A; damp I]))  # EN: Stronger (but more expensive) preconditioner.

For each damp we report:  # EN: Describe per-damp outputs.
  - iterations  # EN: Convergence speed.
  - ||Ax-b||_2  # EN: Data residual.
  - ||A^T(Ax-b) + damp^2 x||_2  # EN: Ridge optimality / gradient norm.

Then we run k-fold CV over a damp grid and report the total cost of sweeping the whole curve:  # EN: Explain CV cost reporting.
  - total build time (preconditioners)  # EN: Part 1.
  - total solve time (Krylov iterations)  # EN: Part 2.
  - total iterations across all fits  # EN: Part 3.

We also demonstrate two practical ways to reduce the cost of sweeping a full damp curve (common in ML hyperparameter tuning):  # EN: Add motivation for the "speedups" section.
  - continuation / warm-start across the damp grid (per fold)  # EN: Reuse the previous solution as an initial guess.
  - reusing randomized-QR components (shared sketch or fixed preconditioner)  # EN: Avoid rebuilding the expensive preconditioner for every damp.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured records.
from time import perf_counter  # EN: Use perf_counter for wall-clock timing.
from typing import Callable, Literal  # EN: Use typing helpers for clearer interfaces.

import numpy as np  # EN: Import NumPy for numerical linear algebra.


EPS = 1e-12  # EN: Small epsilon for safe divisions and numeric guards.
SEED = 0  # EN: RNG seed for reproducible experiments.
PRINT_PRECISION = 6  # EN: Console float precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias for a matrix-vector product function.
PrecondKind = Literal["none", "col", "randqr"]  # EN: Supported preconditioner kinds for this unit.
RandQRPolicy = Literal["rebuild", "shared_sketch", "fixed_R"]  # EN: Policies for building/reusing rand-QR preconditioners along a damp sweep.


@dataclass(frozen=True)  # EN: Immutable container for a reusable randomized sketch for [A; damp I].
class RandQRSketch:  # EN: Store pieces so SA_aug can be updated cheaply for different damp values.
    S_top: np.ndarray  # EN: Sketch block multiplying the data rows (shape s×m).
    S_bottom: np.ndarray  # EN: Sketch block multiplying the identity rows (shape s×n).
    SA_top: np.ndarray  # EN: Precomputed product S_top @ A (shape s×n).
    s: int  # EN: Sketch row count.


@dataclass(frozen=True)  # EN: Immutable record for one solver run (single damp, single preconditioner).
class SolveReport:  # EN: Store per-damp solver diagnostics and timing.
    precond: str  # EN: Preconditioner label.
    damp: float  # EN: Damping parameter used.
    n_iters: int  # EN: Iterations performed by the iterative solver.
    stop_reason: str  # EN: Human-readable termination reason.
    build_seconds: float  # EN: Time spent building preconditioner (0 for none).
    solve_seconds: float  # EN: Time spent in the iterative solve (including norm estimation).
    rnorm_data: float  # EN: ||Ax-b||_2 in original coordinates.
    grad_norm: float  # EN: ||A^T(Ax-b) + damp^2 x||_2 in original coordinates.
    xnorm: float  # EN: ||x||_2 in original coordinates.
    x_hat: np.ndarray  # EN: Coefficient vector x_hat (stored so CV can compute predictions).


@dataclass(frozen=True)  # EN: Immutable record for one CV point (one damp value).
class CVPoint:  # EN: Store mean/std metrics across folds for one damp.
    key: str  # EN: Display label (e.g., "d=1e-3").
    damp: float  # EN: Numeric damp.
    train_mean: float  # EN: Mean training RMSE across folds.
    train_std: float  # EN: Std training RMSE across folds.
    val_mean: float  # EN: Mean validation RMSE across folds.
    val_std: float  # EN: Std validation RMSE across folds.
    x_norm_mean: float  # EN: Mean ||x|| across folds (shrinkage proxy).
    iters_mean: float  # EN: Mean iterations across folds (cost proxy).


@dataclass(frozen=True)  # EN: Immutable record for CV sweep totals (whole curve cost).
class CVTotals:  # EN: Store total cost for sweeping all damps for one preconditioner.
    precond: str  # EN: Preconditioner label.
    total_build_seconds: float  # EN: Sum of preconditioner build time across all fits.
    total_solve_seconds: float  # EN: Sum of solver time across all fits.
    total_iters: int  # EN: Sum of iterations across all fits.


def print_separator(title: str) -> None:  # EN: Print a section separator for console readability.
    print()  # EN: Add a blank line before the section.
    print("=" * 78)  # EN: Print divider line.
    print(title)  # EN: Print section title.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # EN: Compute root-mean-square error.
    err = y_pred - y_true  # EN: Prediction error vector.
    return float(np.sqrt(np.mean(err**2)))  # EN: RMSE is sqrt(mean squared error).


def normalize(v: np.ndarray) -> np.ndarray:  # EN: Normalize vector to unit 2-norm.
    n = l2_norm(v)  # EN: Compute norm.
    if n < EPS:  # EN: Guard against near-zero vectors.
        raise ValueError("Cannot normalize near-zero vector")  # EN: Fail fast on invalid input.
    return v / n  # EN: Return normalized vector.


def build_vandermonde(t: np.ndarray, degree: int) -> np.ndarray:  # EN: Build polynomial design matrix (Vandermonde) with powers 0..degree.
    if t.ndim != 1:  # EN: Validate input shape.
        raise ValueError("t must be 1D")  # EN: Reject invalid input.
    if degree < 0:  # EN: Validate degree.
        raise ValueError("degree must be non-negative")  # EN: Reject invalid degree.
    return np.vander(t, N=degree + 1, increasing=True).astype(float)  # EN: Return Vandermonde matrix.


def k_fold_splits(  # EN: Build deterministic k-fold splits (train_idx, val_idx).
    rng: np.random.Generator,  # EN: RNG for shuffling.
    n_samples: int,  # EN: Total sample count.
    n_folds: int,  # EN: Number of folds.
) -> list[tuple[np.ndarray, np.ndarray]]:  # EN: Return list of (train_idx, val_idx).
    if n_folds < 2:  # EN: Require at least 2 folds.
        raise ValueError("n_folds must be >= 2")  # EN: Reject invalid fold count.
    if n_samples < n_folds:  # EN: Ensure each fold can be non-empty.
        raise ValueError("n_samples must be >= n_folds")  # EN: Reject impossible split.

    perm = rng.permutation(n_samples)  # EN: Shuffle indices deterministically.
    folds = np.array_split(perm, n_folds)  # EN: Split into roughly equal folds.

    splits: list[tuple[np.ndarray, np.ndarray]] = []  # EN: Collect split pairs.
    for i in range(n_folds):  # EN: Use fold i as validation fold.
        val_idx = folds[i]  # EN: Validation indices.
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])  # EN: Training indices.
        splits.append((train_idx, val_idx))  # EN: Store split.
    return splits  # EN: Return splits list.


def ascii_bar(value: float, vmin: float, vmax: float, width: int = 30) -> str:  # EN: Render an ASCII bar where lower values -> longer bars.
    if width <= 0:  # EN: Validate width.
        return ""  # EN: Return empty bar.
    if vmax <= vmin + EPS:  # EN: Handle near-constant series.
        return "#" * width  # EN: Return full bar.
    score = (vmax - value) / (vmax - vmin)  # EN: Map lower -> higher score in [0,1].
    score = float(np.clip(score, 0.0, 1.0))  # EN: Clamp to [0,1].
    n = int(round(score * width))  # EN: Convert score to char count.
    return "#" * n  # EN: Return bar string.


def estimate_spectral_norm(  # EN: Estimate ||A||_2 via power iteration on A^T A (matvec-only).
    matvec_A: Matvec,  # EN: Function computing A v.
    matvec_AT: Matvec,  # EN: Function computing A^T u.
    n: int,  # EN: Domain dimension.
    n_steps: int,  # EN: Power iteration steps.
    rng: np.random.Generator,  # EN: RNG for initialization.
) -> float:  # EN: Return estimated ||A||_2.
    if n_steps <= 0:  # EN: Validate steps.
        raise ValueError("n_steps must be positive")  # EN: Reject invalid n_steps.
    v = normalize(rng.standard_normal(n))  # EN: Random initial unit vector.
    for _ in range(n_steps):  # EN: Iterate v <- normalize(A^T A v).
        w = matvec_A(v)  # EN: Compute w = A v.
        v = matvec_AT(w)  # EN: Compute v = A^T w = A^T A v.
        v = normalize(v)  # EN: Normalize to avoid blow-ups.
    sigma_sq = l2_norm(matvec_A(v)) ** 2  # EN: Rayleigh quotient for A^T A is ||A v||^2.
    return float(np.sqrt(max(sigma_sq, 0.0)))  # EN: Return sqrt of estimated top eigenvalue.


def build_tridiagonal_T_from_golub_kahan(  # EN: Build T_k = B_k^T B_k from GK bidiagonalization coefficients.
    alphas: np.ndarray,  # EN: Alpha sequence (length >= k).
    betas: np.ndarray,  # EN: Beta sequence including beta_1 (length >= k+1).
    k: int,  # EN: Tridiagonal size.
) -> np.ndarray:  # EN: Return dense k×k tridiagonal matrix.
    diag = (alphas[:k] ** 2) + (betas[1 : k + 1] ** 2)  # EN: diag_i = alpha_i^2 + beta_{i+1}^2.
    off = alphas[1:k] * betas[1:k]  # EN: off_i = alpha_{i+1} * beta_{i+1}.
    T = np.diag(diag.astype(float))  # EN: Start with diagonal matrix.
    if k > 1:  # EN: Add symmetric off-diagonals for k>=2.
        T = T + np.diag(off.astype(float), 1) + np.diag(off.astype(float), -1)  # EN: Add off-diagonal entries.
    return T  # EN: Return tridiagonal matrix.


def column_scaling_D_aug(A: np.ndarray, damp: float) -> np.ndarray:  # EN: Build diagonal scaling for augmented operator [A; damp I].
    col_sq = np.sum(A * A, axis=0)  # EN: Compute ||A[:,j]||_2^2 for each column.
    D = np.sqrt(np.maximum(col_sq + (damp * damp), EPS))  # EN: Column norm of augmented column is sqrt(||A||^2 + damp^2).
    return D.astype(float)  # EN: Return D as float.


def choose_randqr_sketch_rows(m_aug: int, n: int, sketch_factor: float) -> int:  # EN: Choose sketch row count s for an oversampled rand-QR preconditioner.
    s_target = int(round(sketch_factor * n))  # EN: Oversampled target like 4n.
    s = int(max(n, min(m_aug, s_target)))  # EN: Clamp s into [n, m_aug] so QR is well-posed.
    return int(s)  # EN: Return as Python int.


def build_randqr_sketch(  # EN: Build a reusable sketch S partitioned into [S_top, S_bottom] for [A; damp I].
    A: np.ndarray,  # EN: Design matrix (m×n).
    sketch_factor: float,  # EN: Oversampling factor for sketch rows.
    rng: np.random.Generator,  # EN: RNG for sketch construction.
) -> RandQRSketch:  # EN: Return a RandQRSketch so SA_aug(damp) can be formed cheaply.
    m, n = A.shape  # EN: Extract dimensions.
    m_aug = m + n  # EN: Augmented row count for [A; damp I].
    s = choose_randqr_sketch_rows(m_aug=m_aug, n=n, sketch_factor=sketch_factor)  # EN: Choose sketch rows.

    # EN: Use a Rademacher sketch with entries ±1/sqrt(s).  # EN: Explain sketch choice.
    S = rng.choice(np.array([-1.0, 1.0]), size=(s, m_aug)) / np.sqrt(max(s, 1))  # EN: Build dense sketch S (s×(m+n)).
    S_top = S[:, :m]  # EN: Extract block that multiplies A rows.
    S_bottom = S[:, m:]  # EN: Extract block that multiplies the identity rows.
    SA_top = S_top @ A  # EN: Precompute S_top A once (expensive part).

    return RandQRSketch(S_top=S_top.astype(float), S_bottom=S_bottom.astype(float), SA_top=SA_top.astype(float), s=int(s))  # EN: Return sketch container.


def randqr_R_from_sketch(sketch: RandQRSketch, damp: float) -> np.ndarray:  # EN: Compute R from QR(S [A; damp I]) using a prebuilt sketch.
    # EN: Since A_aug=[A;damp I], we have S A_aug = S_top A + damp S_bottom.  # EN: Show the key algebra.
    A_sketch = sketch.SA_top + float(damp) * sketch.S_bottom  # EN: Form SA_aug(damp) without explicitly building A_aug.

    # EN: Reduced QR gives SA_aug = Q R, with R (n×n) upper-triangular.  # EN: Explain QR output.
    _, R = np.linalg.qr(A_sketch, mode="reduced")  # EN: Compute QR to extract R.

    # EN: Flip signs so diag(R) is non-negative (stabilizes triangular solves).  # EN: Explain sign convention.
    d = np.sign(np.diag(R))  # EN: Diagonal sign vector.
    d[d == 0.0] = 1.0  # EN: Replace zeros with +1.
    R = np.diag(d) @ R  # EN: Apply sign flips.

    # EN: Add a tiny diagonal jitter if R is nearly singular (defensive for extreme cases).  # EN: Explain jitter.
    jitter = 1e-12 * float(np.linalg.norm(R, ord="fro"))  # EN: Scale jitter by Frobenius norm.
    R = R + jitter * np.eye(R.shape[0], dtype=float)  # EN: Stabilize R for solves.
    return R.astype(float)  # EN: Return R as float.


def upper_triangular_preconditioner_from_R(  # EN: Create right-preconditioner apply_Minv/apply_Minv_T closures from an upper-triangular R.
    R: np.ndarray,  # EN: Upper-triangular matrix defining M=R.
    label: str,  # EN: Human-readable label for printing.
) -> tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]]:  # EN: Return (label, Minv, Minv_T).
    def apply_Minv(y: np.ndarray, R: np.ndarray = R) -> np.ndarray:  # EN: Apply R^{-1} via triangular solve.
        return np.linalg.solve(R, y)  # EN: Solve R x = y.

    def apply_Minv_T(z: np.ndarray, R: np.ndarray = R) -> np.ndarray:  # EN: Apply R^{-T} via triangular solve.
        return np.linalg.solve(R.T, z)  # EN: Solve R^T x = z.

    return label, apply_Minv, apply_Minv_T  # EN: Return preconditioner functions.


def randomized_qr_preconditioner_R_aug(  # EN: Build R from QR(S [A; damp I]) for right preconditioning.
    A: np.ndarray,  # EN: Design matrix (m×n).
    damp: float,  # EN: Damping parameter controlling the augmented rows.
    sketch_factor: float,  # EN: Oversampling factor for sketch rows (e.g., 4.0 => s≈4n).
    rng: np.random.Generator,  # EN: RNG for sketch construction.
) -> np.ndarray:  # EN: Return upper-triangular R (n×n).
    sketch = build_randqr_sketch(A=A, sketch_factor=sketch_factor, rng=rng)  # EN: Build reusable sketch blocks.
    R = randqr_R_from_sketch(sketch=sketch, damp=float(damp))  # EN: Compute R for this damp using the sketch.
    return R.astype(float)  # EN: Return R as float.


def build_preconditioner(  # EN: Build right preconditioner M and return apply_Minv/apply_Minv_T plus build time.
    kind: PrecondKind,  # EN: Preconditioner type.
    A: np.ndarray,  # EN: Design matrix (m×n).
    damp: float,  # EN: Damping parameter (affects augmented operator).
    rng: np.random.Generator,  # EN: RNG for randomized QR sketching.
) -> tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], float]:  # EN: Return (label, Minv, Minv_T, build_seconds).
    m, n = A.shape  # EN: Extract dimensions for closures.
    _ = m  # EN: Explicitly acknowledge m is unused in some branches (keeps comments stable).

    if kind == "none":  # EN: No preconditioning.
        label = "none"  # EN: Human label.

        def apply_Minv(y: np.ndarray) -> np.ndarray:  # EN: Identity M^{-1} for none.
            return y  # EN: Return y unchanged.

        def apply_Minv_T(z: np.ndarray) -> np.ndarray:  # EN: Identity M^{-T} for none.
            return z  # EN: Return z unchanged.

        return label, apply_Minv, apply_Minv_T, 0.0  # EN: Return with zero build time.

    if kind == "col":  # EN: Column scaling using augmented column norms.
        t0 = perf_counter()  # EN: Start build timer.
        D = column_scaling_D_aug(A=A, damp=damp)  # EN: Build diagonal scaling D.
        build_seconds = float(perf_counter() - t0)  # EN: Stop build timer.
        label = "col-scaling"  # EN: Human label.

        def apply_Minv(y: np.ndarray, D: np.ndarray = D) -> np.ndarray:  # EN: Apply D^{-1} (right preconditioning).
            return y / D  # EN: Elementwise divide by diagonal.

        def apply_Minv_T(z: np.ndarray, D: np.ndarray = D) -> np.ndarray:  # EN: Apply D^{-T} = D^{-1} (diagonal).
            return z / D  # EN: Elementwise divide.

        return label, apply_Minv, apply_Minv_T, build_seconds  # EN: Return preconditioner and timing.

    if kind == "randqr":  # EN: Randomized QR preconditioner R from QR(S [A; damp I]).
        t0 = perf_counter()  # EN: Start build timer.
        R = randomized_qr_preconditioner_R_aug(A=A, damp=damp, sketch_factor=4.0, rng=rng)  # EN: Build R.
        build_seconds = float(perf_counter() - t0)  # EN: Stop build timer.
        label = "rand-QR(4n)"  # EN: Human label.
        label, apply_Minv, apply_Minv_T = upper_triangular_preconditioner_from_R(R=R, label=label)  # EN: Build apply closures from R.
        return label, apply_Minv, apply_Minv_T, build_seconds  # EN: Return preconditioner and timing.

    raise ValueError("Unknown preconditioner kind")  # EN: Guard against unreachable cases.


def lsmr_damped_minres_teaching(  # EN: Teaching LSMR for ridge: MINRES on normal equations with right preconditioning.
    A: np.ndarray,  # EN: Design matrix (m×n).
    b: np.ndarray,  # EN: RHS vector (m,).
    damp: float,  # EN: Damping parameter.
    apply_Minv: Callable[[np.ndarray], np.ndarray],  # EN: Function applying M^{-1} (maps y -> x).
    apply_Minv_T: Callable[[np.ndarray], np.ndarray],  # EN: Function applying M^{-T}.
    max_iters: int,  # EN: Iteration cap.
    atol: float,  # EN: Absolute tolerance for stopping.
    btol: float,  # EN: Relative tolerance for stopping.
    anorm_est: float,  # EN: Estimated ||A_aug||_2 (for stopping tests).
    b_aug_bottom: np.ndarray | None = None,  # EN: Optional bottom RHS for the augmented system (length n); default is zeros.
) -> tuple[np.ndarray, int, str, float, float, float, float]:  # EN: Return (x, iters, reason, rnorm_data, grad_norm, xnorm, rnorm_aug).
    m, n = A.shape  # EN: Extract dimensions.
    if b_aug_bottom is None:  # EN: Default bottom RHS is zero (standard ridge).
        b_bottom = np.zeros((n,), dtype=float)  # EN: Use zeros for the regularization rows.
    else:  # EN: Accept a non-zero bottom RHS (used for warm-start correction solves).
        b_bottom = np.asarray(b_aug_bottom, dtype=float).reshape(-1)  # EN: Convert to 1D float array.
        if b_bottom.size != n:  # EN: Validate length.
            raise ValueError("b_aug_bottom must have length n")  # EN: Reject mismatched bottom RHS.

    b_aug = np.concatenate([b, b_bottom])  # EN: Build augmented RHS [b; b_bottom].
    bnorm = l2_norm(b_aug)  # EN: Use ||b_aug|| in residual-bound stopping tests.

    # EN: Define augmented operator B = [A; damp I] M^{-1} via matvecs.  # EN: Explain operator design.
    def matvec_B(y: np.ndarray) -> np.ndarray:  # EN: Compute B y = [A x; damp x] where x = M^{-1} y.
        x = apply_Minv(y)  # EN: Map y -> x.
        top = A @ x  # EN: Compute A x.
        bottom = damp * x  # EN: Compute damp x.
        return np.concatenate([top, bottom])  # EN: Return concatenated vector of length (m+n).

    def matvec_BT(u_aug: np.ndarray) -> np.ndarray:  # EN: Compute B^T u_aug = M^{-T}(A^T u_top + damp u_bottom).
        u_top = u_aug[:m]  # EN: Extract top part.
        u_bottom = u_aug[m:]  # EN: Extract bottom part.
        z = (A.T @ u_top) + (damp * u_bottom)  # EN: Compute A_aug^T u_aug in x-space.
        return apply_Minv_T(z)  # EN: Apply M^{-T} to map to y-space.

    # EN: For general augmented RHS [b; b_bottom], normal-equations RHS is A^T b + damp b_bottom.  # EN: Explain scaling.
    gnorm_x = l2_norm((A.T @ b) + (damp * b_bottom))  # EN: Scale for relative-gradient stopping tests (x-space).

    # EN: Initialize Golub–Kahan bidiagonalization with u1 = b_aug / ||b_aug||.  # EN: Explain initialization.
    u = b_aug.copy()  # EN: Start u from b_aug.
    beta1 = l2_norm(u)  # EN: beta1 = ||b_aug||.
    if beta1 < EPS:  # EN: Trivial b=0 case.
        x0 = np.zeros((n,), dtype=float)  # EN: Solution is x=0.
        r0_top = A @ x0 - b  # EN: Top residual is -b.
        r0_bottom = (damp * x0) - b_bottom  # EN: Bottom residual is -b_bottom.
        grad0 = (A.T @ r0_top) + (damp * r0_bottom)  # EN: Gradient A^T r_top + damp r_bottom.
        rnorm_data0 = l2_norm(r0_top)  # EN: ||Ax-b|| (data residual).
        xnorm0 = l2_norm(x0)  # EN: ||x||.
        rnorm_bottom0 = l2_norm(r0_bottom)  # EN: ||damp x - b_bottom||.
        rnorm_aug0 = float(np.sqrt(rnorm_data0 * rnorm_data0 + rnorm_bottom0 * rnorm_bottom0))  # EN: ||[r_top;r_bottom]||.
        return x0, 0, "b is zero (trivial)", rnorm_data0, l2_norm(grad0), xnorm0, rnorm_aug0  # EN: Return.
    u = u / beta1  # EN: Normalize u1.

    v = matvec_BT(u)  # EN: v1 = B^T u1.
    alpha1 = l2_norm(v)  # EN: alpha1 = ||v1||.
    if alpha1 < EPS:  # EN: Degenerate case B^T b_aug = 0.
        x0 = np.zeros((n,), dtype=float)  # EN: Use x=0.
        r0_top = A @ x0 - b  # EN: Top residual.
        r0_bottom = (damp * x0) - b_bottom  # EN: Bottom residual.
        grad0 = (A.T @ r0_top) + (damp * r0_bottom)  # EN: Gradient for augmented objective.
        rnorm_data0 = l2_norm(r0_top)  # EN: ||Ax-b||.
        xnorm0 = l2_norm(x0)  # EN: ||x||.
        rnorm_bottom0 = l2_norm(r0_bottom)  # EN: ||damp x - b_bottom||.
        rnorm_aug0 = float(np.sqrt(rnorm_data0 * rnorm_data0 + rnorm_bottom0 * rnorm_bottom0))  # EN: ||[r_top;r_bottom]||.
        return x0, 0, "B^T b is zero (degenerate)", rnorm_data0, l2_norm(grad0), xnorm0, rnorm_aug0  # EN: Return.
    v = v / alpha1  # EN: Normalize v1.

    # EN: Store v-basis vectors in y-space so we can reconstruct y_k and map to x.  # EN: Explain storage.
    V_basis_y = np.zeros((n, min(max_iters, n) + 1), dtype=float)  # EN: Store v vectors as columns.
    V_basis_y[:, 0] = v  # EN: Store v1.

    alphas: list[float] = [float(alpha1)]  # EN: Store alpha_1.
    betas: list[float] = [float(beta1)]  # EN: Store beta_1 (append beta_{k+1} each iteration).

    # EN: In GK, ||B^T b_aug|| = alpha1 * beta1, and v1 is proportional to that RHS.  # EN: Explain RHS magnitude.
    gnorm_y = float(alpha1 * beta1)  # EN: ||B^T b_aug|| in y-space coordinates.

    x_hat = np.zeros((n,), dtype=float)  # EN: Initialize x estimate (mapped from y).
    stop_reason = "max_iters reached"  # EN: Default stop reason.
    n_done = 0  # EN: Iterations completed.

    # EN: Main loop: expand Krylov subspace and compute MINRES iterate via a small LS solve with T_k.  # EN: Explain loop.
    for k in range(1, min(max_iters, n) + 1):  # EN: Limit to <= n for this small teaching demo.
        u_next = matvec_B(v) - alphas[-1] * u  # EN: u_{k+1} = B v_k - alpha_k u_k.
        beta_next = l2_norm(u_next)  # EN: beta_{k+1}.
        if beta_next >= EPS:  # EN: Normalize when possible.
            u_next = u_next / beta_next  # EN: Normalize u_{k+1}.

        v_next = matvec_BT(u_next) - beta_next * v  # EN: v_{k+1} = B^T u_{k+1} - beta_{k+1} v_k.
        alpha_next = l2_norm(v_next)  # EN: alpha_{k+1}.
        if alpha_next >= EPS:  # EN: Normalize when possible.
            v_next = v_next / alpha_next  # EN: Normalize v_{k+1}.

        betas.append(float(beta_next))  # EN: Append beta_{k+1}.
        alphas.append(float(alpha_next))  # EN: Append alpha_{k+1}.
        V_basis_y[:, k] = v_next  # EN: Store v_{k+1}.

        alpha_arr = np.array(alphas, dtype=float)  # EN: Convert alphas to array.
        beta_arr = np.array(betas, dtype=float)  # EN: Convert betas to array.
        T_k = build_tridiagonal_T_from_golub_kahan(alphas=alpha_arr, betas=beta_arr, k=k)  # EN: Build T_k = B_k^T B_k.

        rhs = np.zeros((k,), dtype=float)  # EN: Build RHS vector in Krylov basis.
        rhs[0] = gnorm_y  # EN: Place ||B^T b_aug|| on e1.
        y_coeffs, *_ = np.linalg.lstsq(T_k, rhs, rcond=None)  # EN: Solve small LS for MINRES iterate.
        y_k = V_basis_y[:, :k] @ y_coeffs  # EN: Reconstruct y_k in R^n.
        x_hat = apply_Minv(y_k)  # EN: Map back x = M^{-1} y.

        # EN: Compute ridge diagnostics in original coordinates.  # EN: Explain compute.
        r_top = A @ x_hat - b  # EN: Top residual (data-fit part).
        r_bottom = (damp * x_hat) - b_bottom  # EN: Bottom residual (regularization rows).
        grad = (A.T @ r_top) + (damp * r_bottom)  # EN: Gradient of ||[A;damp I]x-[b;b_bottom]||^2.
        xnorm = l2_norm(x_hat)  # EN: ||x||.
        rnorm_data = l2_norm(r_top)  # EN: ||Ax-b||.
        rnorm_bottom = l2_norm(r_bottom)  # EN: ||damp x - b_bottom||.
        rnorm_aug = float(np.sqrt(rnorm_data * rnorm_data + rnorm_bottom * rnorm_bottom))  # EN: ||[r_top;r_bottom]||.
        grad_norm = l2_norm(grad)  # EN: ||A^T r + damp^2 x||.

        n_done = k  # EN: Update completed iteration count.

        # EN: Stopping test 1: residual bound (LSQR-style) using ||A_aug|| and ||x||.  # EN: Explain test.
        r_bound = (btol * bnorm) + (atol * anorm_est * xnorm)  # EN: Mixed absolute/relative residual bound.
        if rnorm_aug <= r_bound:  # EN: Stop when residual is small enough.
            stop_reason = "residual bound satisfied"  # EN: Record stop reason.
            break  # EN: Exit loop.

        # EN: Stopping test 2: gradient bound ||A_aug^T r_aug|| <= atol*||A_aug||*||r_aug||.  # EN: Explain test.
        if grad_norm <= atol * anorm_est * max(rnorm_aug, EPS):  # EN: Stop when gradient is small relative to residual.
            stop_reason = "normal residual bound satisfied"  # EN: Record stop reason.
            break  # EN: Exit loop.

        # EN: Stopping test 3: relative gradient ||grad|| <= btol * ||A^T b|| (x-space scale).  # EN: Explain test.
        if gnorm_x >= EPS and grad_norm <= btol * gnorm_x:  # EN: Stop when gradient is small relative to initial scale.
            stop_reason = "relative normal residual satisfied"  # EN: Record stop reason.
            break  # EN: Exit loop.

        if beta_next < EPS and alpha_next < EPS:  # EN: Breakdown: cannot expand Krylov basis further.
            stop_reason = "breakdown (beta and alpha near zero)"  # EN: Record breakdown reason.
            break  # EN: Exit loop.

        u = u_next  # EN: Advance u.
        v = v_next  # EN: Advance v.

    # EN: Final diagnostics in original coordinates.  # EN: Explain final compute.
    r_final_top = A @ x_hat - b  # EN: Final top residual.
    r_final_bottom = (damp * x_hat) - b_bottom  # EN: Final bottom residual.
    grad_final = (A.T @ r_final_top) + (damp * r_final_bottom)  # EN: Final gradient for augmented objective.
    xnorm_final = l2_norm(x_hat)  # EN: Final ||x||.
    rnorm_data_final = l2_norm(r_final_top)  # EN: Final ||Ax-b||.
    rnorm_bottom_final = l2_norm(r_final_bottom)  # EN: Final ||damp x - b_bottom||.
    rnorm_aug_final = float(np.sqrt(rnorm_data_final * rnorm_data_final + rnorm_bottom_final * rnorm_bottom_final))  # EN: Final ||[r_top;r_bottom]||.
    grad_norm_final = l2_norm(grad_final)  # EN: Final ||grad||.

    return x_hat, int(n_done), stop_reason, float(rnorm_data_final), float(grad_norm_final), float(xnorm_final), float(rnorm_aug_final)  # EN: Return final values.


def solve_one(  # EN: Solve one ridge problem for a specific damp and preconditioner, returning a SolveReport.
    A: np.ndarray,  # EN: Design matrix.
    b: np.ndarray,  # EN: RHS vector.
    damp: float,  # EN: Damping parameter.
    precond_kind: PrecondKind,  # EN: Preconditioner kind.
    max_iters: int,  # EN: Iteration cap.
    atol: float,  # EN: Absolute tolerance.
    btol: float,  # EN: Relative tolerance.
    estimate_norm_steps: int,  # EN: Steps for ||A_aug|| estimate.
    rng: np.random.Generator,  # EN: RNG.
    x_init: np.ndarray | None = None,  # EN: Optional warm-start initial guess in x-space; used as continuation across damp.
    precond_override: tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]] | None = None,  # EN: Optional prebuilt (label, Minv, Minv_T).
    build_seconds_override: float | None = None,  # EN: Optional build-time override (used when precond_override is provided).
) -> SolveReport:  # EN: Return SolveReport with diagnostics and timing.
    m, n = A.shape  # EN: Extract dimensions.

    # EN: Build matvecs for the augmented operator A_aug = [A; damp I] (no preconditioning).  # EN: Explain norm estimate operator.
    def matvec_A_aug(v: np.ndarray) -> np.ndarray:  # EN: Compute A_aug v.
        top = A @ v  # EN: Top block A v.
        bottom = damp * v  # EN: Bottom block damp v.
        return np.concatenate([top, bottom])  # EN: Concatenate.

    def matvec_AT_aug(u_aug: np.ndarray) -> np.ndarray:  # EN: Compute A_aug^T u_aug.
        u_top = u_aug[:m]  # EN: Extract top part.
        u_bottom = u_aug[m:]  # EN: Extract bottom part.
        return (A.T @ u_top) + (damp * u_bottom)  # EN: Combine.

    # EN: Estimate ||A_aug||_2 once per call (kept small for CV).  # EN: Explain estimation choice.
    if estimate_norm_steps > 0:  # EN: Use power iteration when requested.
        anorm_A_aug = estimate_spectral_norm(matvec_A=matvec_A_aug, matvec_AT=matvec_AT_aug, n=n, n_steps=estimate_norm_steps, rng=rng)  # EN: Estimate.
        anorm_est = float(anorm_A_aug)  # EN: Store estimate.
    else:  # EN: Fallback to Frobenius norm upper bound if skipping estimation.
        A_aug = np.vstack([A, damp * np.eye(n, dtype=float)])  # EN: Form A_aug for the cheap bound.
        anorm_est = float(np.linalg.norm(A_aug, ord="fro"))  # EN: Use ||A_aug||_F as an upper bound.

    # EN: Choose the preconditioner: either build it here, or reuse a prebuilt one supplied by the caller.  # EN: Explain override.
    if precond_override is None:  # EN: Default behavior (baseline): build preconditioner per call.
        label, apply_Minv, apply_Minv_T, build_seconds = build_preconditioner(kind=precond_kind, A=A, damp=damp, rng=rng)  # EN: Build preconditioner.
    else:  # EN: Reuse a cached preconditioner (e.g., fixed rand-QR across a damp grid).
        label, apply_Minv, apply_Minv_T = precond_override  # EN: Unpack prebuilt closures.
        build_seconds = float(build_seconds_override) if build_seconds_override is not None else 0.0  # EN: Use explicit build time (or 0 when reused).

    t0 = perf_counter()  # EN: Start solve timer.

    # EN: Warm-start via a correction solve: set b_aug=[b-Ax0; -damp x0] so the residual corresponds to x=x0+delta.  # EN: Explain trick.
    if x_init is None:  # EN: No warm-start (x0=0).
        b_top = b  # EN: Top RHS is the original b.
        b_bottom = None  # EN: Bottom RHS defaults to zeros in the solver.
        x0 = np.zeros((n,), dtype=float)  # EN: Define x0 explicitly for uniform handling below.
    else:  # EN: Warm-start enabled.
        x0 = np.asarray(x_init, dtype=float).reshape(-1)  # EN: Convert to 1D float array.
        if x0.size != n:  # EN: Validate length.
            raise ValueError("x_init must have length n")  # EN: Reject invalid warm-start vector.
        b_top = b - (A @ x0)  # EN: Top RHS becomes the data residual at x0.
        b_bottom = -float(damp) * x0  # EN: Bottom RHS becomes -damp x0 so damp residual is damp(x0+delta).

    x_delta, iters, stop_reason, _, _, _, _ = lsmr_damped_minres_teaching(  # EN: Run solver (returns delta when warm-start is used).
        A=A,  # EN: Provide A.
        b=b_top,  # EN: Provide top RHS (possibly shifted for warm-start).
        damp=damp,  # EN: Provide damp.
        apply_Minv=apply_Minv,  # EN: Provide M^{-1}.
        apply_Minv_T=apply_Minv_T,  # EN: Provide M^{-T}.
        max_iters=max_iters,  # EN: Provide cap.
        atol=atol,  # EN: Provide atol.
        btol=btol,  # EN: Provide btol.
        anorm_est=anorm_est,  # EN: Provide ||A_aug|| estimate.
        b_aug_bottom=b_bottom,  # EN: Provide bottom RHS (or None for zeros).
    )  # EN: End solver call.
    solve_seconds = float(perf_counter() - t0)  # EN: Stop solve timer.

    x_hat = (x0 + x_delta).astype(float)  # EN: Combine x0 and delta to get the final coefficients.
    r_final = (A @ x_hat) - b  # EN: Compute final data residual in original coordinates.
    grad_final = (A.T @ r_final) + (float(damp) * float(damp)) * x_hat  # EN: Compute ridge gradient A^T(Ax-b)+damp^2 x.

    rnorm_data = l2_norm(r_final)  # EN: Report ||Ax-b||.
    grad_norm = l2_norm(grad_final)  # EN: Report ||A^T r + damp^2 x||.
    xnorm = l2_norm(x_hat)  # EN: Report ||x|| in original coordinates.

    return SolveReport(  # EN: Package report.
        precond=label,  # EN: Label.
        damp=float(damp),  # EN: Damp value.
        n_iters=int(iters),  # EN: Iterations.
        stop_reason=str(stop_reason),  # EN: Stop reason.
        build_seconds=float(build_seconds),  # EN: Build time.
        solve_seconds=float(solve_seconds),  # EN: Solve time.
        rnorm_data=float(rnorm_data),  # EN: ||Ax-b||.
        grad_norm=float(grad_norm),  # EN: ||A^T r + damp^2 x||.
        xnorm=float(xnorm),  # EN: ||x||.
        x_hat=x_hat.astype(float),  # EN: Store coefficients for CV evaluation.
    )  # EN: End report.


def print_solver_table(reports: list[SolveReport]) -> None:  # EN: Print a compact per-damp solver comparison table.
    if not reports:  # EN: Handle empty input.
        print("(no reports)")  # EN: Print message.
        return  # EN: Exit.

    header = (  # EN: Build table header.
        "damp      | precond        | iters | build_s | solve_s | ||Ax-b||   | ||grad||   | ||x||"  # EN: Column names.
    )  # EN: End header.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Print divider.

    # EN: Sort by damp then by preconditioner label for readability.  # EN: Explain sorting.
    reps = sorted(reports, key=lambda r: (r.damp, r.precond))  # EN: Sort reports.
    for r in reps:  # EN: Print each row.
        damp_key = f"{r.damp:.0e}" if r.damp != 0.0 else "0"  # EN: Compact damp formatting.
        print(  # EN: Print formatted row.
            f"{damp_key:9} | "  # EN: Damp column.
            f"{r.precond:14} | "  # EN: Preconditioner label.
            f"{r.n_iters:5d} | "  # EN: Iterations.
            f"{r.build_seconds:7.3f} | "  # EN: Build time.
            f"{r.solve_seconds:7.3f} | "  # EN: Solve time.
            f"{r.rnorm_data:9.2e} | "  # EN: Data residual norm.
            f"{r.grad_norm:9.2e} | "  # EN: Gradient norm.
            f"{r.xnorm:9.2e}"  # EN: x norm.
        )  # EN: End print.


def print_cv_table(points: list[CVPoint], best: CVPoint) -> None:  # EN: Print a CV table with an ASCII curve and iteration proxy.
    points_sorted = sorted(points, key=lambda p: p.damp)  # EN: Sort by damp for readability.
    vals = [p.val_mean for p in points_sorted]  # EN: Collect validation means for bar scaling.
    vmin = min(vals)  # EN: Minimum validation mean.
    vmax = max(vals)  # EN: Maximum validation mean.

    header = (  # EN: Build the table header.
        "param        | train_rmse(mean±std) | val_rmse(mean±std) | ||x||_2(mean) | iters(mean) | curve"  # EN: Column names.
    )  # EN: End header.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Print divider.

    for p in points_sorted:  # EN: Print each row.
        mark = " <== best" if p.key == best.key else ""  # EN: Mark best damp.
        bar = ascii_bar(value=p.val_mean, vmin=vmin, vmax=vmax, width=26)  # EN: Build ASCII curve bar.
        print(  # EN: Print formatted row.
            f"{p.key:12} | "  # EN: Damp label.
            f"{p.train_mean:.3e}±{p.train_std:.1e} | "  # EN: Train RMSE.
            f"{p.val_mean:.3e}±{p.val_std:.1e} | "  # EN: Val RMSE.
            f"{p.x_norm_mean:.3e} | "  # EN: Mean ||x||.
            f"{p.iters_mean:10.1f} | "  # EN: Mean iterations.
            f"{bar}{mark}"  # EN: Bar + marker.
        )  # EN: End print.


def choose_fixed_randqr_reference_damp(damps_sorted: np.ndarray) -> float:  # EN: Pick a representative damp for building a fixed rand-QR preconditioner.
    positive = damps_sorted[damps_sorted > 0.0]  # EN: Filter to strictly positive damps.
    if positive.size == 0:  # EN: If no positive damp exists, fall back to 0.
        return 0.0  # EN: Return zero reference.
    return float(positive[positive.size // 2])  # EN: Return the median positive damp (in sorted order).


def cv_sweep_curve(  # EN: Sweep a damp grid with k-fold CV, optionally using continuation and rand-QR reuse to reduce total cost.
    splits: list[tuple[np.ndarray, np.ndarray]],  # EN: CV splits (train_idx, val_idx).
    A: np.ndarray,  # EN: Full design matrix.
    b: np.ndarray,  # EN: Full targets.
    damps: np.ndarray,  # EN: Damp grid to sweep (will be sorted ascending).
    precond_kind: PrecondKind,  # EN: Preconditioner kind.
    randqr_policy: RandQRPolicy,  # EN: Policy for rand-QR (ignored unless precond_kind == "randqr").
    warm_start: bool,  # EN: Whether to use continuation/warm-start across the damp grid within each fold.
    max_iters: int,  # EN: Iteration cap.
    atol: float,  # EN: Absolute tolerance.
    btol: float,  # EN: Relative tolerance.
    estimate_norm_steps: int,  # EN: Norm-estimation steps (costly but stabilizes stopping tests).
    rng: np.random.Generator,  # EN: RNG for sketches and norm estimation.
) -> tuple[list[CVPoint], CVTotals, CVPoint]:  # EN: Return (points, totals, best_point).
    damps_sorted = np.array(sorted([float(d) for d in damps]), dtype=float)  # EN: Sort damps ascending for continuation.
    n_damps = int(damps_sorted.size)  # EN: Number of damp values.
    n_folds = int(len(splits))  # EN: Number of folds.
    n_features = int(A.shape[1])  # EN: Feature count.

    order = np.arange(n_damps, dtype=int)  # EN: Default evaluation order is ascending (good for printing).
    if warm_start:  # EN: For continuation, it is often cheaper/more stable to start from large damp (strong regularization).
        order = order[::-1]  # EN: Evaluate from largest damp down to smallest.

    train_rmse_mat = np.zeros((n_damps, n_folds), dtype=float)  # EN: Store train RMSE per (damp, fold).
    val_rmse_mat = np.zeros((n_damps, n_folds), dtype=float)  # EN: Store val RMSE per (damp, fold).
    xnorm_mat = np.zeros((n_damps, n_folds), dtype=float)  # EN: Store ||x|| per (damp, fold).
    iters_mat = np.zeros((n_damps, n_folds), dtype=float)  # EN: Store iterations per (damp, fold).

    total_build = 0.0  # EN: Accumulate preconditioner build time across all fits in the curve.
    total_solve = 0.0  # EN: Accumulate solver time across all fits in the curve.
    total_iters = 0  # EN: Accumulate iterations across all fits in the curve.

    # EN: Build a human label for the chosen configuration (used in summaries).  # EN: Explain labeling.
    if precond_kind == "none":  # EN: Baseline no-preconditioner label.
        label = "none"  # EN: Label.
    elif precond_kind == "col":  # EN: Column-scaling label.
        label = "col-scaling"  # EN: Label.
    else:  # EN: randqr label depends on policy.
        policy_tag = {"rebuild": "rand-QR(4n)", "shared_sketch": "rand-QR(shared-sketch)", "fixed_R": "rand-QR(fixed-R)"}[randqr_policy]  # EN: Map policy to a readable tag.
        label = policy_tag  # EN: Assign label.
    if warm_start:  # EN: Annotate label when warm-start is enabled.
        label = f"{label}+ws"  # EN: Append warm-start suffix.

    # EN: Precompute a reference damp for fixed-R policy (one preconditioner reused for the whole curve).  # EN: Explain reference damp.
    damp_ref = choose_fixed_randqr_reference_damp(damps_sorted)  # EN: Reference damp for fixed-R builds.

    for fold_id, (train_idx, val_idx) in enumerate(splits):  # EN: Loop folds (outer) to enable continuation within each fold.
        A_tr = A[train_idx, :]  # EN: Training design matrix.
        b_tr = b[train_idx]  # EN: Training targets.
        A_va = A[val_idx, :]  # EN: Validation design matrix.
        b_va = b[val_idx]  # EN: Validation targets.

        x_prev = np.zeros((n_features,), dtype=float)  # EN: Initialize continuation with x=0 for the first damp.

        # EN: For rand-QR reuse policies, build per-fold cached objects here (since A_tr changes per fold).  # EN: Explain caching scope.
        randqr_sketch: RandQRSketch | None = None  # EN: Shared sketch S for this fold (optional).
        randqr_fixed_precond: tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]] | None = None  # EN: Fixed preconditioner closures (optional).

        if precond_kind == "randqr" and randqr_policy in {"shared_sketch", "fixed_R"}:  # EN: Build shared sketch once per fold when requested.
            t0 = perf_counter()  # EN: Start sketch build timer.
            randqr_sketch = build_randqr_sketch(A=A_tr, sketch_factor=4.0, rng=rng)  # EN: Build sketch blocks for this fold.
            total_build += float(perf_counter() - t0)  # EN: Account for one-time sketch build cost.

        if precond_kind == "randqr" and randqr_policy == "fixed_R":  # EN: Build a fixed R once per fold (then reuse across damps).
            if randqr_sketch is None:  # EN: Defensive: fixed-R requires a sketch.
                raise RuntimeError("internal error: randqr_sketch is None for fixed_R policy")  # EN: Fail fast.
            t0 = perf_counter()  # EN: Start R build timer.
            R_ref = randqr_R_from_sketch(sketch=randqr_sketch, damp=float(damp_ref))  # EN: Build R at the reference damp.
            total_build += float(perf_counter() - t0)  # EN: Account for one-time QR cost.
            randqr_fixed_precond = upper_triangular_preconditioner_from_R(R=R_ref, label="rand-QR(fixed-R)")  # EN: Build preconditioner closures.

        for step, idx in enumerate(order):  # EN: Sweep damp values in the chosen continuation order.
            damp = float(damps_sorted[idx])  # EN: Current damp value.
            x_init = x_prev if (warm_start and step > 0) else None  # EN: Use previous x as warm-start except for the first step.

            # EN: Select how to supply the preconditioner to solve_one based on the chosen rand-QR policy.  # EN: Explain branching.
            if precond_kind != "randqr":  # EN: Non-randQR paths use the standard builder.
                rep = solve_one(  # EN: Fit on the training fold.
                    A=A_tr,  # EN: Training A.
                    b=b_tr,  # EN: Training b.
                    damp=float(damp),  # EN: Damp value.
                    precond_kind=precond_kind,  # EN: Preconditioner kind.
                    max_iters=max_iters,  # EN: Iter cap.
                    atol=atol,  # EN: atol.
                    btol=btol,  # EN: btol.
                    estimate_norm_steps=estimate_norm_steps,  # EN: Norm estimation steps.
                    rng=rng,  # EN: RNG.
                    x_init=x_init,  # EN: Warm-start (optional).
                )  # EN: End solve.
            else:  # EN: randqr path supports the three policies.
                if randqr_policy == "rebuild":  # EN: Baseline: rebuild rand-QR (fresh sketch) for every damp.
                    rep = solve_one(  # EN: Fit with per-call preconditioner build.
                        A=A_tr,  # EN: Training A.
                        b=b_tr,  # EN: Training b.
                        damp=float(damp),  # EN: Damp.
                        precond_kind="randqr",  # EN: rand-QR.
                        max_iters=max_iters,  # EN: Cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        estimate_norm_steps=estimate_norm_steps,  # EN: Norm estimation.
                        rng=rng,  # EN: RNG.
                        x_init=x_init,  # EN: Warm-start (optional).
                    )  # EN: End solve.
                elif randqr_policy == "shared_sketch":  # EN: Rebuild R per damp, but reuse the expensive sketch S_top A.
                    if randqr_sketch is None:  # EN: Defensive check.
                        raise RuntimeError("internal error: randqr_sketch is None for shared_sketch policy")  # EN: Fail fast.
                    t0 = perf_counter()  # EN: Time only the R-from-sketch build for this damp.
                    R = randqr_R_from_sketch(sketch=randqr_sketch, damp=float(damp))  # EN: Build R using shared sketch.
                    build_s = float(perf_counter() - t0)  # EN: Record build time for this damp.
                    precond = upper_triangular_preconditioner_from_R(R=R, label="rand-QR(shared-sketch)")  # EN: Create apply closures from R.
                    rep = solve_one(  # EN: Fit with an externally built preconditioner.
                        A=A_tr,  # EN: Training A.
                        b=b_tr,  # EN: Training b.
                        damp=float(damp),  # EN: Damp.
                        precond_kind="randqr",  # EN: Value ignored because we pass precond_override.
                        max_iters=max_iters,  # EN: Cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        estimate_norm_steps=estimate_norm_steps,  # EN: Norm estimation.
                        rng=rng,  # EN: RNG.
                        x_init=x_init,  # EN: Warm-start (optional).
                        precond_override=precond,  # EN: Reuse prebuilt closures for this damp.
                        build_seconds_override=build_s,  # EN: Charge only the QR cost here (sketch cost charged once per fold above).
                    )  # EN: End solve.
                elif randqr_policy == "fixed_R":  # EN: Reuse the SAME preconditioner R for all damp values in the curve.
                    if randqr_fixed_precond is None:  # EN: Defensive check.
                        raise RuntimeError("internal error: randqr_fixed_precond is None for fixed_R policy")  # EN: Fail fast.
                    rep = solve_one(  # EN: Fit with the reused preconditioner.
                        A=A_tr,  # EN: Training A.
                        b=b_tr,  # EN: Training b.
                        damp=float(damp),  # EN: Damp (objective still changes, only M is fixed).
                        precond_kind="randqr",  # EN: Value ignored because we pass precond_override.
                        max_iters=max_iters,  # EN: Cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        estimate_norm_steps=estimate_norm_steps,  # EN: Norm estimation.
                        rng=rng,  # EN: RNG.
                        x_init=x_init,  # EN: Warm-start (optional).
                        precond_override=randqr_fixed_precond,  # EN: Reuse the fixed-R closures.
                        build_seconds_override=0.0,  # EN: No per-damp build cost when R is fixed.
                    )  # EN: End solve.
                else:  # EN: Guard against unknown policy strings.
                    raise ValueError("Unknown randqr_policy")  # EN: Reject invalid policy.

            x_prev = rep.x_hat  # EN: Update continuation state (used only when warm_start=True).

            train_rmse_mat[idx, fold_id] = rmse(y_true=b_tr, y_pred=A_tr @ rep.x_hat)  # EN: Compute train RMSE for this fit.
            val_rmse_mat[idx, fold_id] = rmse(y_true=b_va, y_pred=A_va @ rep.x_hat)  # EN: Compute validation RMSE for this fit.
            xnorm_mat[idx, fold_id] = float(rep.xnorm)  # EN: Record ||x||.
            iters_mat[idx, fold_id] = float(rep.n_iters)  # EN: Record iterations.

            total_build += float(rep.build_seconds)  # EN: Add per-fit build time (0 when reused or already charged separately).
            total_solve += float(rep.solve_seconds)  # EN: Add per-fit solve time.
            total_iters += int(rep.n_iters)  # EN: Add per-fit iterations.

    points: list[CVPoint] = []  # EN: Collect per-damp CV summaries.
    for i, damp in enumerate(damps_sorted):  # EN: Summarize each damp across folds.
        train_row = train_rmse_mat[i, :]  # EN: Train RMSE values across folds.
        val_row = val_rmse_mat[i, :]  # EN: Val RMSE values across folds.
        x_row = xnorm_mat[i, :]  # EN: ||x|| values across folds.
        it_row = iters_mat[i, :]  # EN: Iteration counts across folds.

        key = f"d={damp:.0e}" if damp != 0.0 else "d=0"  # EN: Format parameter label.
        points.append(  # EN: Append a CVPoint for this damp.
            CVPoint(
                key=key,  # EN: Label.
                damp=float(damp),  # EN: Damp value.
                train_mean=float(np.mean(train_row)),  # EN: Mean train RMSE.
                train_std=float(np.std(train_row)),  # EN: Std train RMSE.
                val_mean=float(np.mean(val_row)),  # EN: Mean val RMSE.
                val_std=float(np.std(val_row)),  # EN: Std val RMSE.
                x_norm_mean=float(np.mean(x_row)),  # EN: Mean ||x||.
                iters_mean=float(np.mean(it_row)),  # EN: Mean iterations.
            )
        )  # EN: End append.

    best = min(points, key=lambda p: p.val_mean)  # EN: Pick damp with lowest mean validation RMSE.
    totals = CVTotals(precond=label, total_build_seconds=float(total_build), total_solve_seconds=float(total_solve), total_iters=int(total_iters))  # EN: Package totals.
    return points, totals, best  # EN: Return curve, totals, and best point.


def summarize_cv_for_damp(  # EN: Compute k-fold CV summary for one damp and one preconditioner, also accumulating cost.
    splits: list[tuple[np.ndarray, np.ndarray]],  # EN: CV splits.
    A: np.ndarray,  # EN: Full design matrix.
    b: np.ndarray,  # EN: Full targets.
    damp: float,  # EN: Candidate damp.
    precond_kind: PrecondKind,  # EN: Preconditioner kind.
    max_iters: int,  # EN: Iteration cap.
    atol: float,  # EN: Absolute tolerance.
    btol: float,  # EN: Relative tolerance.
    estimate_norm_steps: int,  # EN: Norm-estimation steps.
    rng: np.random.Generator,  # EN: RNG.
) -> tuple[CVPoint, float, float, int]:  # EN: Return (CVPoint, build_seconds_sum, solve_seconds_sum, iters_sum).
    train_scores: list[float] = []  # EN: Collect train RMSE per fold.
    val_scores: list[float] = []  # EN: Collect val RMSE per fold.
    x_norms: list[float] = []  # EN: Collect ||x|| per fold.
    iters_list: list[int] = []  # EN: Collect iterations per fold.

    total_build = 0.0  # EN: Accumulate build time across folds.
    total_solve = 0.0  # EN: Accumulate solve time across folds.
    total_iters = 0  # EN: Accumulate iterations across folds.

    for train_idx, val_idx in splits:  # EN: Iterate folds.
        A_tr = A[train_idx, :]  # EN: Training design matrix.
        b_tr = b[train_idx]  # EN: Training targets.
        A_va = A[val_idx, :]  # EN: Validation design matrix.
        b_va = b[val_idx]  # EN: Validation targets.

        rep = solve_one(  # EN: Fit model on training fold.
            A=A_tr,  # EN: Training A.
            b=b_tr,  # EN: Training b.
            damp=damp,  # EN: Damp.
            precond_kind=precond_kind,  # EN: Preconditioner.
            max_iters=max_iters,  # EN: Cap.
            atol=atol,  # EN: atol.
            btol=btol,  # EN: btol.
            estimate_norm_steps=estimate_norm_steps,  # EN: Norm estimate steps.
            rng=rng,  # EN: RNG.
        )  # EN: End fit.

        x_hat = rep.x_hat  # EN: Extract coefficients for prediction.

        train_scores.append(rmse(y_true=b_tr, y_pred=A_tr @ x_hat))  # EN: Compute train RMSE for this fold.
        val_scores.append(rmse(y_true=b_va, y_pred=A_va @ x_hat))  # EN: Compute validation RMSE for this fold.
        x_norms.append(l2_norm(x_hat))  # EN: Record ||x|| for shrinkage diagnostics.
        iters_list.append(int(rep.n_iters))  # EN: Record iteration count as a cost proxy.

        total_build += rep.build_seconds  # EN: Accumulate build time.
        total_solve += rep.solve_seconds  # EN: Accumulate solve time.
        total_iters += int(rep.n_iters)  # EN: Accumulate iterations.

    train_arr = np.array(train_scores, dtype=float)  # EN: Convert train scores to array for stats.
    val_arr = np.array(val_scores, dtype=float)  # EN: Convert val scores to array for stats.
    x_arr = np.array(x_norms, dtype=float)  # EN: Convert x norms to array for stats.
    it_arr = np.array(iters_list, dtype=float)  # EN: Convert iterations to float array for mean.

    key = f"d={damp:.0e}" if damp != 0.0 else "d=0"  # EN: Build key label.
    point = CVPoint(  # EN: Construct CVPoint summary for this damp.
        key=key,  # EN: Label.
        damp=float(damp),  # EN: Damp value.
        train_mean=float(np.mean(train_arr)),  # EN: Mean train RMSE.
        train_std=float(np.std(train_arr)),  # EN: Std train RMSE.
        val_mean=float(np.mean(val_arr)),  # EN: Mean val RMSE.
        val_std=float(np.std(val_arr)),  # EN: Std val RMSE.
        x_norm_mean=float(np.mean(x_arr)),  # EN: Mean ||x||.
        iters_mean=float(np.mean(it_arr)),  # EN: Mean iterations.
    )  # EN: End CVPoint.

    return point, float(total_build), float(total_solve), int(total_iters)  # EN: Return summary and per-damp cost totals.


def main() -> None:  # EN: Run per-damp comparisons and then k-fold CV cost comparisons.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    # EN: Build a classic ill-conditioned regression dataset (polynomial Vandermonde).  # EN: Explain dataset choice.
    n_samples = 80  # EN: Number of samples.
    degree = 12  # EN: Polynomial degree (n_features = degree+1).
    noise_std = 0.05  # EN: Noise level.
    n_folds = 5  # EN: Number of CV folds.

    t = np.linspace(-1.0, 1.0, n_samples)  # EN: Input locations.
    A = build_vandermonde(t, degree=degree)  # EN: Design matrix.
    n_features = A.shape[1]  # EN: Feature dimension.

    x_true = np.zeros((n_features,), dtype=float)  # EN: Sparse ground-truth coefficients.
    x_true[0] = 0.5  # EN: Intercept.
    x_true[1] = 1.0  # EN: Linear.
    x_true[2] = -2.0  # EN: Quadratic.
    x_true[3] = 0.7  # EN: Cubic.

    b_clean = A @ x_true  # EN: Clean targets.
    b = b_clean + noise_std * rng.standard_normal(b_clean.shape)  # EN: Add noise.

    splits = k_fold_splits(rng=rng, n_samples=n_samples, n_folds=n_folds)  # EN: Build CV splits.

    print_separator("Dataset Summary")  # EN: Print dataset diagnostics.
    cond_A = float(np.linalg.cond(A))  # EN: Condition number (small dense only).
    rank_A = int(np.linalg.matrix_rank(A))  # EN: Numerical rank.
    print(f"n_samples={n_samples}, degree={degree}, n_features={n_features}, n_folds={n_folds}")  # EN: Print sizes.
    print(f"rank(A)={rank_A}, cond(A)={cond_A:.3e}, noise_std={noise_std}")  # EN: Print conditioning and noise.

    # EN: Solver settings (small problem): a few dozen iterations are enough for demonstration.  # EN: Explain settings.
    max_iters = min(2 * n_features, 50)  # EN: Iteration cap.
    atol = 1e-10  # EN: Absolute tolerance.
    btol = 1e-10  # EN: Relative tolerance.
    estimate_norm_steps = 10  # EN: Norm-estimation steps (small for speed).

    # EN: Per-damp comparisons on the full dataset (not CV).  # EN: Explain section purpose.
    demo_damps = [0.0, 1e-6, 1e-3, 1e-1]  # EN: A small set of damp values to compare.
    preconds: list[PrecondKind] = ["none", "col", "randqr"]  # EN: Preconditioner list.

    print_separator("Per-damp solver comparison (full data)")  # EN: Announce comparison section.
    reports: list[SolveReport] = []  # EN: Collect reports.
    for d in demo_damps:  # EN: Loop demo damp values.
        for pk in preconds:  # EN: Loop preconditioners.
            reports.append(  # EN: Append run report.
                solve_one(  # EN: Solve one configuration.
                    A=A,  # EN: A.
                    b=b,  # EN: b.
                    damp=float(d),  # EN: damp.
                    precond_kind=pk,  # EN: preconditioner.
                    max_iters=max_iters,  # EN: cap.
                    atol=atol,  # EN: atol.
                    btol=btol,  # EN: btol.
                    estimate_norm_steps=estimate_norm_steps,  # EN: norm estimate steps.
                    rng=rng,  # EN: RNG.
                )  # EN: End solve.
            )  # EN: End append.
    print_solver_table(reports)  # EN: Print table.

    # EN: k-fold CV sweep for damp and cost totals.  # EN: Explain next section.
    damps = np.concatenate(([0.0], np.logspace(-8, 1, num=15)))  # EN: Damp grid for CV.

    # EN: Baseline: no warm-start; rand-QR preconditioner rebuilt for every damp (fresh sketch each time).  # EN: Explain baseline.
    print_separator("k-fold CV sweep (baseline): rebuild per damp, no warm-start")  # EN: Announce baseline CV section.

    rng_cv_baseline = np.random.default_rng(SEED + 1)  # EN: Dedicated RNG so baseline is reproducible and independent.
    baseline_rows: dict[PrecondKind, tuple[str, float, float, float, float, int]] = {}  # EN: Map kind -> (label, best_damp, best_val, build_s, solve_s, iters).

    for pk in preconds:  # EN: Sweep preconditioners for baseline comparison.
        points, totals, best = cv_sweep_curve(  # EN: Run a full CV curve sweep for this preconditioner.
            splits=splits,  # EN: Provide splits.
            A=A,  # EN: Provide full A.
            b=b,  # EN: Provide full b.
            damps=damps,  # EN: Provide damp grid.
            precond_kind=pk,  # EN: Preconditioner kind.
            randqr_policy="rebuild",  # EN: Baseline policy (ignored unless pk=="randqr").
            warm_start=False,  # EN: No continuation for baseline.
            max_iters=max_iters,  # EN: Iteration cap.
            atol=atol,  # EN: atol.
            btol=btol,  # EN: btol.
            estimate_norm_steps=estimate_norm_steps,  # EN: Norm-estimation steps.
            rng=rng_cv_baseline,  # EN: RNG stream.
        )  # EN: End sweep.

        print_separator(f"CV results (baseline): {totals.precond}")  # EN: Print per-preconditioner header.
        print_cv_table(points=points, best=best)  # EN: Print the CV curve table.
        total_seconds = float(totals.total_build_seconds + totals.total_solve_seconds)  # EN: Total curve cost.
        print(  # EN: Print cost summary line.
            f"\nTotal curve cost (baseline, {totals.precond}): build={totals.total_build_seconds:.3f}s, solve={totals.total_solve_seconds:.3f}s, total={total_seconds:.3f}s, iters={totals.total_iters}"  # EN: Summary string.
        )  # EN: End print.

        baseline_rows[pk] = (totals.precond, float(best.damp), float(best.val_mean), float(totals.total_build_seconds), float(totals.total_solve_seconds), int(totals.total_iters))  # EN: Store summary.

    print_separator("Baseline CV summary across preconditioners")  # EN: Print final baseline comparison header.
    header = "precond        | best_damp | best_val_rmse | build_s | solve_s | total_s | total_iters"  # EN: Header columns.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Divider.
    for pk in preconds:  # EN: Print rows in consistent precond order.
        label, best_d, best_val, build_s, solve_s, iters = baseline_rows[pk]  # EN: Unpack summary.
        total_s = float(build_s + solve_s)  # EN: Total seconds.
        damp_str = f"{best_d:.0e}" if best_d != 0.0 else "0"  # EN: Format damp.
        print(f"{label:13} | {damp_str:9} | {best_val:13.3e} | {build_s:7.3f} | {solve_s:7.3f} | {total_s:7.3f} | {iters:10d}")  # EN: Print row.

    # EN: Speedups: continuation (warm-start) + rand-QR reuse (shared sketch), to reduce the cost of sweeping the whole curve.  # EN: Explain speedups.
    print_separator("CV sweep speedups: continuation + rand-QR reuse")  # EN: Announce speedup section.

    rng_cv_speed = np.random.default_rng(SEED + 2)  # EN: Separate RNG stream for the speedup experiment.
    speed_rows: dict[PrecondKind, tuple[str, float, float, float, float, int]] = {}  # EN: Map kind -> (label, best_damp, best_val, build_s, solve_s, iters).

    for pk in preconds:  # EN: Sweep preconditioners with speedups enabled.
        policy = "shared_sketch" if pk == "randqr" else "rebuild"  # EN: Only rand-QR uses special reuse; others ignore policy.
        points, totals, best = cv_sweep_curve(  # EN: Run CV sweep with continuation and chosen reuse policy.
            splits=splits,  # EN: Provide splits.
            A=A,  # EN: Provide full A.
            b=b,  # EN: Provide full b.
            damps=damps,  # EN: Provide damp grid.
            precond_kind=pk,  # EN: Preconditioner kind.
            randqr_policy=policy,  # EN: Reuse policy for rand-QR.
            warm_start=True,  # EN: Enable continuation across the damp grid.
            max_iters=max_iters,  # EN: Iteration cap.
            atol=atol,  # EN: atol.
            btol=btol,  # EN: btol.
            estimate_norm_steps=estimate_norm_steps,  # EN: Norm-estimation steps.
            rng=rng_cv_speed,  # EN: RNG stream.
        )  # EN: End sweep.

        total_seconds = float(totals.total_build_seconds + totals.total_solve_seconds)  # EN: Total curve cost.
        damp_str = f"{best.damp:.0e}" if best.damp != 0.0 else "0"  # EN: Format best damp for printing.
        print(  # EN: Print a compact summary line (tables already printed above for baseline).
            f"{totals.precond:18} best_damp={damp_str:>9} best_val={best.val_mean:.3e} total={total_seconds:.3f}s (build={totals.total_build_seconds:.3f}s, solve={totals.total_solve_seconds:.3f}s) iters={totals.total_iters}"  # EN: Summary string.
        )  # EN: End print.
        speed_rows[pk] = (totals.precond, float(best.damp), float(best.val_mean), float(totals.total_build_seconds), float(totals.total_solve_seconds), int(totals.total_iters))  # EN: Store summary.

    # EN: Compare baseline vs speedups in one table (time + iterations + best validation RMSE).  # EN: Explain comparison.
    print_separator("Baseline vs speedups (time/iters/quality)")  # EN: Announce comparison table.
    header = "precond        | base_total_s | sped_total_s | speedup | base_iters | sped_iters | base_best_val | sped_best_val"  # EN: Header columns.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Divider.
    for pk in preconds:  # EN: Loop in consistent order.
        b_label, _, b_best_val, b_build, b_solve, b_iters = baseline_rows[pk]  # EN: Unpack baseline.
        s_label, _, s_best_val, s_build, s_solve, s_iters = speed_rows[pk]  # EN: Unpack speedup.
        _ = s_label  # EN: Keep labels aligned; baseline label is shown in the first column.
        base_total = float(b_build + b_solve)  # EN: Baseline total seconds.
        sped_total = float(s_build + s_solve)  # EN: Speedup total seconds.
        speedup = (base_total / max(sped_total, EPS))  # EN: Compute speedup factor.
        print(  # EN: Print comparison row.
            f"{b_label:13} | {base_total:11.3f} | {sped_total:11.3f} | {speedup:7.2f} | {b_iters:9d} | {s_iters:9d} | {b_best_val:12.3e} | {s_best_val:12.3e}"  # EN: Row.
        )  # EN: End print.

    # EN: Extra: for rand-QR, compare shared-sketch vs fixed-R reuse (both with warm-start).  # EN: Explain extra experiment.
    print_separator("rand-QR reuse variants (warm-start): shared-sketch vs fixed-R")  # EN: Announce extra experiment.
    rng_cv_fixed = np.random.default_rng(SEED + 3)  # EN: Separate RNG for fixed-R variant.
    points_fixed, totals_fixed, best_fixed = cv_sweep_curve(  # EN: Run fixed-R variant.
        splits=splits,  # EN: Provide splits.
        A=A,  # EN: Provide full A.
        b=b,  # EN: Provide full b.
        damps=damps,  # EN: Provide damp grid.
        precond_kind="randqr",  # EN: rand-QR only.
        randqr_policy="fixed_R",  # EN: Fixed-R reuse across the whole curve.
        warm_start=True,  # EN: Continuation across damps.
        max_iters=max_iters,  # EN: Iteration cap.
        atol=atol,  # EN: atol.
        btol=btol,  # EN: btol.
        estimate_norm_steps=estimate_norm_steps,  # EN: Norm estimation.
        rng=rng_cv_fixed,  # EN: RNG stream.
    )  # EN: End sweep.
    _ = points_fixed  # EN: Keep points available for optional debugging/printing.
    total_fixed = float(totals_fixed.total_build_seconds + totals_fixed.total_solve_seconds)  # EN: Total seconds for fixed-R.
    damp_str = f"{best_fixed.damp:.0e}" if best_fixed.damp != 0.0 else "0"  # EN: Format best damp for printing.
    print(  # EN: Print fixed-R summary.
        f"{totals_fixed.precond:18} best_damp={damp_str:>9} best_val={best_fixed.val_mean:.3e} total={total_fixed:.3f}s (build={totals_fixed.total_build_seconds:.3f}s, solve={totals_fixed.total_solve_seconds:.3f}s) iters={totals_fixed.total_iters}"  # EN: Summary line.
    )  # EN: End print.


if __name__ == "__main__":  # EN: Standard Python entrypoint guard.
    main()  # EN: Execute main.
