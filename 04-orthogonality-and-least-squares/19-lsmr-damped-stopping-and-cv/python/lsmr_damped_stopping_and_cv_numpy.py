"""  # EN: Start module docstring.
Damped LSMR (Ridge) with practical stopping criteria + k-fold CV to choose damp (NumPy).  # EN: Summarize what this script demonstrates.

We solve the damped least-squares (Ridge) objective:  # EN: State the optimization problem.
  min_x  ||A x - b||_2^2 + damp^2 ||x||_2^2,   damp >= 0.  # EN: Define damp and the objective.

Key equivalence (augmentation trick):  # EN: Explain the standard reduction to an ordinary least-squares problem.
  min ||Ax-b||^2 + damp^2||x||^2  ==  min || [A; damp I]x - [b; 0] ||^2.  # EN: Show augmented formulation.

LSMR view:  # EN: Explain how LSMR fits.
  LSMR is algebraically equivalent to MINRES applied to the normal equations  # EN: Define LSMR via MINRES.
    (A^T A + damp^2 I) x = A^T b,  # EN: State ridge normal equations.
  but it avoids forming A^T A by using matvecs with A and A^T (and the damp term).  # EN: Explain matvec-only benefit.

Stopping criteria (teaching-friendly, LSQR-style):  # EN: Describe the stopping criteria used.
  - Augmented residual: ||[Ax-b; damp x]||  (controls data fit + regularization)  # EN: Residual in augmented system.
  - Normal residual / gradient: ||A^T(Ax-b) + damp^2 x||  (optimality condition)  # EN: Gradient norm.
  - Mixed absolute/relative thresholds using atol/btol and an estimate of ||A_aug||_2.  # EN: Explain mixed tolerances.

We also do k-fold CV to pick damp by validation RMSE, printing an ASCII curve.  # EN: Explain CV output.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured records.
from typing import Callable  # EN: Use Callable for matvec typing.

import numpy as np  # EN: Import NumPy for numerical linear algebra.


EPS = 1e-12  # EN: Small epsilon for safe divisions.
SEED = 0  # EN: RNG seed for deterministic demos.
PRINT_PRECISION = 6  # EN: Console float precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias for a matrix-vector product function.


@dataclass(frozen=True)  # EN: Immutable record for a damped LSMR run.
class LSMRRun:  # EN: Store solution, diagnostics, and iteration history.
    x_hat: np.ndarray  # EN: Estimated solution vector (n,).
    n_iters: int  # EN: Iterations performed.
    stop_reason: str  # EN: Human-readable termination reason.
    rnorm_data: float  # EN: Data residual norm ||Ax-b||_2.
    rnorm_aug: float  # EN: Augmented residual norm sqrt(||Ax-b||^2 + damp^2||x||^2).
    arnorm: float  # EN: Gradient/optimality norm ||A^T(Ax-b) + damp^2 x||_2.
    xnorm: float  # EN: Solution norm ||x||_2.
    anorm_est: float  # EN: Estimated spectral norm ||A_aug||_2 used in stopping tests.
    history_rnorm_aug: np.ndarray  # EN: History of augmented residual norms (including iter 0).
    history_arnorm: np.ndarray  # EN: History of gradient norms (including iter 0).


@dataclass(frozen=True)  # EN: Immutable record for one CV hyperparameter point.
class CVPoint:  # EN: Store mean/std metrics across folds for one damp value.
    key: str  # EN: Display label (e.g., "d=1e-3").
    param: float  # EN: Numeric damp value.
    train_mean: float  # EN: Mean training RMSE.
    train_std: float  # EN: Std training RMSE.
    val_mean: float  # EN: Mean validation RMSE.
    val_std: float  # EN: Std validation RMSE.
    x_norm_mean: float  # EN: Mean ||x||_2 across folds.


def print_separator(title: str) -> None:  # EN: Print a section separator for console readability.
    print()  # EN: Add a blank line.
    print("=" * 78)  # EN: Print divider line.
    print(title)  # EN: Print title.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:  # EN: Compute root-mean-square error.
    err = y_pred - y_true  # EN: Compute prediction errors.
    return float(np.sqrt(np.mean(err**2)))  # EN: Return sqrt(mean squared error).


def build_vandermonde(t: np.ndarray, degree: int) -> np.ndarray:  # EN: Build polynomial design matrix with powers 0..degree.
    if t.ndim != 1:  # EN: Validate input shape.
        raise ValueError("t must be 1D")  # EN: Reject invalid t.
    if degree < 0:  # EN: Validate degree.
        raise ValueError("degree must be non-negative")  # EN: Reject invalid degree.
    return np.vander(t, N=degree + 1, increasing=True).astype(float)  # EN: Return Vandermonde matrix.


def k_fold_splits(  # EN: Build deterministic k-fold splits (train_idx, val_idx).
    rng: np.random.Generator,  # EN: RNG for shuffling.
    n_samples: int,  # EN: Total number of samples.
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
        val_idx = folds[i]  # EN: Validation indices.
        train_idx = np.concatenate([folds[j] for j in range(n_folds) if j != i])  # EN: Training indices.
        splits.append((train_idx, val_idx))  # EN: Store split.
    return splits  # EN: Return list.


def ascii_bar(value: float, vmin: float, vmax: float, width: int = 30) -> str:  # EN: Render a simple ASCII bar (lower is better).
    if width <= 0:  # EN: Validate width.
        return ""  # EN: Return empty.
    if vmax <= vmin + EPS:  # EN: Handle near-constant values.
        return "#" * width  # EN: Return full bar.
    score = (vmax - value) / (vmax - vmin)  # EN: Map lower RMSE -> higher score.
    score = float(np.clip(score, 0.0, 1.0))  # EN: Clamp to [0,1].
    n = int(round(score * width))  # EN: Convert score to bar length.
    return "#" * n  # EN: Return bar string.


def print_cv_table(points: list[CVPoint], best: CVPoint) -> None:  # EN: Print a CV table with an ASCII curve.
    points_sorted = sorted(points, key=lambda p: p.param)  # EN: Sort by damp.
    vals = [p.val_mean for p in points_sorted]  # EN: Collect val means.
    vmin = min(vals)  # EN: Minimum.
    vmax = max(vals)  # EN: Maximum.

    header = (  # EN: Build table header.
        "param        | train_rmse(mean±std) | val_rmse(mean±std) | ||x||_2(mean) | curve (lower val -> longer bar)"  # EN: Column names.
    )  # EN: End header.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Print divider.

    for p in points_sorted:  # EN: Print each row.
        mark = " <== best" if p.key == best.key else ""  # EN: Mark best damp.
        bar = ascii_bar(value=p.val_mean, vmin=vmin, vmax=vmax, width=32)  # EN: Build curve bar.
        print(  # EN: Print formatted row.
            f"{p.key:12} | "  # EN: Param label.
            f"{p.train_mean:.3e}±{p.train_std:.1e} | "  # EN: Train RMSE mean±std.
            f"{p.val_mean:.3e}±{p.val_std:.1e} | "  # EN: Val RMSE mean±std.
            f"{p.x_norm_mean:.3e} | "  # EN: Mean ||x||.
            f"{bar}{mark}"  # EN: Curve bar + marker.
        )  # EN: End print.


def normalize(v: np.ndarray) -> np.ndarray:  # EN: Normalize a vector to unit 2-norm.
    n = l2_norm(v)  # EN: Compute norm.
    if n < EPS:  # EN: Guard against near-zero vectors.
        raise ValueError("Cannot normalize near-zero vector")  # EN: Reject invalid input.
    return v / n  # EN: Return normalized vector.


def estimate_spectral_norm(  # EN: Estimate ||A||_2 via power iteration on A^T A (matvec-only).
    matvec_A: Matvec,  # EN: Function computing A v.
    matvec_AT: Matvec,  # EN: Function computing A^T u.
    n: int,  # EN: Domain dimension.
    n_steps: int,  # EN: Power-iteration steps.
    rng: np.random.Generator,  # EN: RNG for initialization.
) -> float:  # EN: Return an estimate of ||A||_2.
    if n_steps <= 0:  # EN: Validate step count.
        raise ValueError("n_steps must be positive")  # EN: Reject invalid n_steps.
    v = normalize(rng.standard_normal(n))  # EN: Random initial unit vector.
    for _ in range(n_steps):  # EN: Iterate v <- normalize(A^T A v).
        w = matvec_A(v)  # EN: Compute w = A v.
        v = matvec_AT(w)  # EN: Compute v = A^T w = A^T A v.
        v = normalize(v)  # EN: Normalize to control growth.
    sigma_sq = l2_norm(matvec_A(v)) ** 2  # EN: Rayleigh quotient estimate for A^T A: ||A v||^2.
    return float(np.sqrt(max(sigma_sq, 0.0)))  # EN: Return sqrt of estimated top eigenvalue.


def build_tridiagonal_T_from_golub_kahan(  # EN: Build T_k = B_k^T B_k from GK bidiagonalization coefficients.
    alphas: np.ndarray,  # EN: Alpha sequence (length >= k).
    betas: np.ndarray,  # EN: Beta sequence including beta_1 (length >= k+1).
    k: int,  # EN: Desired tridiagonal size.
) -> np.ndarray:  # EN: Return dense k×k tridiagonal matrix.
    if k <= 0:  # EN: Validate k.
        raise ValueError("k must be positive")  # EN: Reject invalid k.
    diag = (alphas[:k] ** 2) + (betas[1 : k + 1] ** 2)  # EN: diag_i = alpha_i^2 + beta_{i+1}^2.
    off = alphas[1:k] * betas[1:k]  # EN: off_i = alpha_{i+1} * beta_{i+1}.
    T = np.diag(diag.astype(float))  # EN: Start with diagonal.
    if k > 1:  # EN: Add symmetric off-diagonals.
        T = T + np.diag(off.astype(float), 1) + np.diag(off.astype(float), -1)  # EN: Add off-diagonals.
    return T  # EN: Return tridiagonal.


def lsmr_damped_with_stopping(  # EN: Solve damped least squares with LSMR-style MINRES on normal equations (teaching version).
    A: np.ndarray,  # EN: Design matrix (m×n).
    b: np.ndarray,  # EN: RHS vector (m,).
    damp: float,  # EN: Damping parameter (>=0).
    max_iters: int,  # EN: Iteration cap (use <= n for small problems).
    atol: float,  # EN: Absolute tolerance for mixed stopping tests.
    btol: float,  # EN: Relative tolerance for mixed stopping tests.
    estimate_norm_steps: int,  # EN: Steps for ||A||_2 power iteration (used to estimate ||A_aug||_2).
    rng: np.random.Generator,  # EN: RNG for norm estimation and internal random choices.
) -> LSMRRun:  # EN: Return LSMRRun with diagnostics and history.
    if damp < 0.0:  # EN: Validate damp.
        raise ValueError("damp must be non-negative")  # EN: Reject invalid damp.
    if b.ndim != 1:  # EN: Validate b shape.
        raise ValueError("b must be a 1D vector")  # EN: Reject invalid b.
    m, n = A.shape  # EN: Extract dimensions.
    if b.size != m:  # EN: Validate b length.
        raise ValueError("b length must match rows of A")  # EN: Reject mismatch.
    if max_iters <= 0:  # EN: Validate iteration cap.
        raise ValueError("max_iters must be positive")  # EN: Reject invalid max_iters.

    bnorm = l2_norm(b)  # EN: Compute ||b|| for stopping tests.
    x = np.zeros((n,), dtype=float)  # EN: Initialize x0=0.

    r0 = A @ x - b  # EN: Initial data residual r0 = -b.
    grad0 = (A.T @ r0) + (damp * damp) * x  # EN: Initial gradient A^T r + damp^2 x.

    xnorm0 = l2_norm(x)  # EN: Initial ||x||.
    rnorm_data0 = l2_norm(r0)  # EN: Initial ||Ax-b||.
    rnorm_aug0 = float(np.sqrt(rnorm_data0 * rnorm_data0 + (damp * xnorm0) ** 2))  # EN: Initial augmented residual norm.
    arnorm0 = l2_norm(grad0)  # EN: Initial ||A^T(Ax-b)+damp^2 x||.

    hist_r_aug: list[float] = [rnorm_aug0]  # EN: Track augmented residual norms (iter 0 included).
    hist_ar: list[float] = [arnorm0]  # EN: Track gradient norms (iter 0 included).

    if bnorm < EPS:  # EN: Trivial b=0 case.
        return LSMRRun(  # EN: Return early.
            x_hat=x,  # EN: x=0.
            n_iters=0,  # EN: No iterations.
            stop_reason="b is zero (trivial)",  # EN: Reason.
            rnorm_data=rnorm_data0,  # EN: Data residual.
            rnorm_aug=rnorm_aug0,  # EN: Aug residual.
            arnorm=arnorm0,  # EN: Gradient norm.
            xnorm=xnorm0,  # EN: x norm.
            anorm_est=float("nan"),  # EN: No norm estimate needed.
            history_rnorm_aug=np.array(hist_r_aug, dtype=float),  # EN: History.
            history_arnorm=np.array(hist_ar, dtype=float),  # EN: History.
        )  # EN: End return.

    # EN: Estimate ||A||_2 and convert to ||A_aug||_2 = sqrt(||A||_2^2 + damp^2).  # EN: Explain norm estimate.
    def matvec_A(v: np.ndarray) -> np.ndarray:  # EN: Dense matvec for A.
        return A @ v  # EN: Compute A v.

    def matvec_AT(u: np.ndarray) -> np.ndarray:  # EN: Dense matvec for A^T.
        return A.T @ u  # EN: Compute A^T u.

    if estimate_norm_steps > 0:  # EN: Only estimate when requested (can speed up CV).
        anorm_A = estimate_spectral_norm(matvec_A=matvec_A, matvec_AT=matvec_AT, n=n, n_steps=estimate_norm_steps, rng=rng)  # EN: Estimate ||A||_2.
        anorm_est = float(np.sqrt(anorm_A * anorm_A + damp * damp))  # EN: Convert to ||A_aug||_2.
    else:  # EN: Fallback when skipping norm estimation.
        anorm_est = float(np.sqrt(float(np.linalg.norm(A, ord=2)) ** 2 + damp * damp))  # EN: Use exact ||A||_2 (small dense only).

    # EN: Build augmented matvecs for Golub–Kahan bidiagonalization on A_aug.  # EN: Explain operator construction.
    def matvec_A_aug(v: np.ndarray) -> np.ndarray:  # EN: Compute A_aug v = [A v; damp v].
        top = A @ v  # EN: Top block A v.
        bottom = damp * v  # EN: Bottom block damp v.
        return np.concatenate([top, bottom])  # EN: Concatenate into length (m+n).

    def matvec_AT_aug(u_aug: np.ndarray) -> np.ndarray:  # EN: Compute A_aug^T u_aug = A^T u_top + damp u_bottom.
        u_top = u_aug[:m]  # EN: Extract top part (m,).
        u_bottom = u_aug[m:]  # EN: Extract bottom part (n,).
        return (A.T @ u_top) + (damp * u_bottom)  # EN: Combine contributions.

    b_aug = np.concatenate([b, np.zeros((n,), dtype=float)])  # EN: Build augmented RHS [b; 0].
    g0 = A.T @ b  # EN: g = A^T b in ridge normal equations.
    gnorm = l2_norm(g0)  # EN: ||A^T b|| for relative stopping tests.

    # EN: Initialize Golub–Kahan bidiagonalization with u1 = b_aug / ||b_aug||.  # EN: Explain initialization.
    u = b_aug.copy()  # EN: Start u from b_aug.
    beta1 = l2_norm(u)  # EN: beta1 = ||b_aug|| (=||b||).
    if beta1 < EPS:  # EN: Handle b_aug=0 (already handled above, but be defensive).
        return LSMRRun(  # EN: Return early.
            x_hat=x,  # EN: x=0.
            n_iters=0,  # EN: No iterations.
            stop_reason="b is zero (trivial)",  # EN: Reason.
            rnorm_data=rnorm_data0,  # EN: Data residual.
            rnorm_aug=rnorm_aug0,  # EN: Aug residual.
            arnorm=arnorm0,  # EN: Gradient norm.
            xnorm=xnorm0,  # EN: x norm.
            anorm_est=anorm_est,  # EN: Norm estimate.
            history_rnorm_aug=np.array(hist_r_aug, dtype=float),  # EN: History.
            history_arnorm=np.array(hist_ar, dtype=float),  # EN: History.
        )  # EN: End return.
    u = u / beta1  # EN: Normalize u1.

    v = matvec_AT_aug(u)  # EN: v1 = A_aug^T u1.
    alpha1 = l2_norm(v)  # EN: alpha1 = ||v1||.
    if alpha1 < EPS:  # EN: Handle degenerate A_aug^T b_aug = 0.
        return LSMRRun(  # EN: Return early with x=0.
            x_hat=x,  # EN: x=0.
            n_iters=0,  # EN: No iterations.
            stop_reason="A^T b is zero (degenerate)",  # EN: Reason.
            rnorm_data=rnorm_data0,  # EN: Data residual.
            rnorm_aug=rnorm_aug0,  # EN: Aug residual.
            arnorm=arnorm0,  # EN: Gradient norm.
            xnorm=xnorm0,  # EN: x norm.
            anorm_est=anorm_est,  # EN: Norm estimate.
            history_rnorm_aug=np.array(hist_r_aug, dtype=float),  # EN: History.
            history_arnorm=np.array(hist_ar, dtype=float),  # EN: History.
        )  # EN: End return.
    v = v / alpha1  # EN: Normalize v1.

    V_basis = np.zeros((n, min(max_iters, n) + 1), dtype=float)  # EN: Store v basis vectors (cap storage for small problems).
    V_basis[:, 0] = v  # EN: Store v1.

    alphas: list[float] = [float(alpha1)]  # EN: Store alpha_1.
    betas: list[float] = [float(beta1)]  # EN: Store beta_1 (beta_{k+1} appended each iteration).

    stop_reason = "max_iters reached"  # EN: Default stop reason if we hit the cap.
    n_done = 0  # EN: Track iterations completed.

    # EN: Main loop: build the Krylov subspace and compute the MINRES iterate for the ridge normal equations.  # EN: Explain loop purpose.
    for k in range(1, min(max_iters, n) + 1):  # EN: Limit k to <= n for small dense problems.
        u_next = matvec_A_aug(v) - alphas[-1] * u  # EN: u_{k+1} = A_aug v_k - alpha_k u_k.
        beta_next = l2_norm(u_next)  # EN: beta_{k+1} = ||u_{k+1}||.
        if beta_next >= EPS:  # EN: Normalize when possible.
            u_next = u_next / beta_next  # EN: Normalize u_{k+1}.

        v_next = matvec_AT_aug(u_next) - beta_next * v  # EN: v_{k+1} = A_aug^T u_{k+1} - beta_{k+1} v_k.
        alpha_next = l2_norm(v_next)  # EN: alpha_{k+1} = ||v_{k+1}||.
        if alpha_next >= EPS:  # EN: Normalize when possible.
            v_next = v_next / alpha_next  # EN: Normalize v_{k+1}.

        betas.append(float(beta_next))  # EN: Append beta_{k+1}.
        alphas.append(float(alpha_next))  # EN: Append alpha_{k+1}.
        V_basis[:, k] = v_next  # EN: Store v_{k+1}.

        alpha_arr = np.array(alphas, dtype=float)  # EN: Convert alphas to array.
        beta_arr = np.array(betas, dtype=float)  # EN: Convert betas to array.
        T_k = build_tridiagonal_T_from_golub_kahan(alphas=alpha_arr, betas=beta_arr, k=k)  # EN: Build T_k = B_k^T B_k.

        rhs = np.zeros((k,), dtype=float)  # EN: Build right-hand side in Krylov basis.
        rhs[0] = gnorm  # EN: Place ||A^T b|| on e1 (normal equations RHS magnitude).
        y_k, *_ = np.linalg.lstsq(T_k, rhs, rcond=None)  # EN: Solve small LS for MINRES iterate in subspace.
        x = V_basis[:, :k] @ y_k  # EN: Lift subspace solution to x-space.

        # EN: Compute ridge diagnostics in original coordinates.  # EN: Explain diagnostic compute.
        r = A @ x - b  # EN: Data residual r = Ax-b.
        grad = (A.T @ r) + (damp * damp) * x  # EN: Ridge gradient A^T r + damp^2 x.

        xnorm = l2_norm(x)  # EN: Compute ||x||.
        rnorm_data = l2_norm(r)  # EN: Compute ||Ax-b||.
        rnorm_aug = float(np.sqrt(rnorm_data * rnorm_data + (damp * xnorm) ** 2))  # EN: Compute augmented residual norm.
        arnorm = l2_norm(grad)  # EN: Compute ||A^T r + damp^2 x||.

        hist_r_aug.append(float(rnorm_aug))  # EN: Record rnorm_aug.
        hist_ar.append(float(arnorm))  # EN: Record arnorm.
        n_done = k  # EN: Update completed iteration count.

        # EN: Stopping test 1 (LSQR-style): ||r_aug|| <= btol*||b|| + atol*||A_aug||*||x||.  # EN: Explain residual bound.
        rnorm_bound = (btol * bnorm) + (atol * anorm_est * xnorm)  # EN: Mixed absolute/relative bound.
        if rnorm_aug <= rnorm_bound:  # EN: Stop when residual is below bound.
            stop_reason = "residual bound satisfied"  # EN: Record reason.
            break  # EN: Exit loop.

        # EN: Stopping test 2 (normal residual): ||A_aug^T r_aug|| <= atol*||A_aug||*||r_aug||.  # EN: Explain gradient bound.
        if arnorm <= atol * anorm_est * max(rnorm_aug, EPS):  # EN: Stop when gradient is small relative to residual.
            stop_reason = "normal residual bound satisfied"  # EN: Record reason.
            break  # EN: Exit loop.

        # EN: Stopping test 3 (relative to ||A^T b||): ||grad|| <= btol * ||A^T b||.  # EN: Explain relative gradient test.
        if gnorm >= EPS and arnorm <= btol * gnorm:  # EN: Stop when gradient is small relative to initial gradient scale.
            stop_reason = "relative normal residual satisfied"  # EN: Record reason.
            break  # EN: Exit loop.

        if beta_next < EPS and alpha_next < EPS:  # EN: Breakdown: bidiagonalization cannot proceed.
            stop_reason = "breakdown (beta and alpha near zero)"  # EN: Record breakdown reason.
            break  # EN: Exit loop.

        u = u_next  # EN: Advance u.
        v = v_next  # EN: Advance v.

    # EN: Final diagnostics in original coordinates.  # EN: Explain final compute.
    r_final = A @ x - b  # EN: Final data residual.
    grad_final = (A.T @ r_final) + (damp * damp) * x  # EN: Final ridge gradient.
    xnorm_final = l2_norm(x)  # EN: Final ||x||.
    rnorm_data_final = l2_norm(r_final)  # EN: Final ||Ax-b||.
    rnorm_aug_final = float(np.sqrt(rnorm_data_final * rnorm_data_final + (damp * xnorm_final) ** 2))  # EN: Final augmented residual.
    arnorm_final = l2_norm(grad_final)  # EN: Final ||A^T r + damp^2 x||.

    return LSMRRun(  # EN: Package result.
        x_hat=x,  # EN: Solution.
        n_iters=int(n_done),  # EN: Iterations.
        stop_reason=stop_reason,  # EN: Stop reason.
        rnorm_data=float(rnorm_data_final),  # EN: Data residual.
        rnorm_aug=float(rnorm_aug_final),  # EN: Aug residual.
        arnorm=float(arnorm_final),  # EN: Gradient norm.
        xnorm=float(xnorm_final),  # EN: x norm.
        anorm_est=float(anorm_est),  # EN: Norm estimate.
        history_rnorm_aug=np.array(hist_r_aug, dtype=float),  # EN: History.
        history_arnorm=np.array(hist_ar, dtype=float),  # EN: History.
    )  # EN: End return.


def summarize_cv_for_damp(  # EN: Compute k-fold CV summary for one damp value using damped LSMR.
    splits: list[tuple[np.ndarray, np.ndarray]],  # EN: List of (train_idx, val_idx).
    A: np.ndarray,  # EN: Full design matrix.
    b: np.ndarray,  # EN: Full target vector.
    damp: float,  # EN: Damping parameter to evaluate.
    max_iters: int,  # EN: Iteration cap for LSMR.
    atol: float,  # EN: Absolute tolerance for stopping.
    btol: float,  # EN: Relative tolerance for stopping.
    estimate_norm_steps: int,  # EN: Steps for norm estimation.
    rng: np.random.Generator,  # EN: RNG for the solver.
) -> CVPoint:  # EN: Return aggregated CV metrics.
    train_scores: list[float] = []  # EN: Collect train RMSE per fold.
    val_scores: list[float] = []  # EN: Collect val RMSE per fold.
    x_norms: list[float] = []  # EN: Collect ||x|| per fold.

    for train_idx, val_idx in splits:  # EN: Iterate folds.
        A_tr = A[train_idx, :]  # EN: Training matrix.
        b_tr = b[train_idx]  # EN: Training targets.
        A_va = A[val_idx, :]  # EN: Validation matrix.
        b_va = b[val_idx]  # EN: Validation targets.

        run = lsmr_damped_with_stopping(  # EN: Fit on training fold with damped LSMR.
            A=A_tr,  # EN: Provide A_train.
            b=b_tr,  # EN: Provide b_train.
            damp=damp,  # EN: Provide damp.
            max_iters=max_iters,  # EN: Provide cap.
            atol=atol,  # EN: Provide atol.
            btol=btol,  # EN: Provide btol.
            estimate_norm_steps=estimate_norm_steps,  # EN: Provide norm-estimation steps.
            rng=rng,  # EN: Provide RNG.
        )  # EN: End fit.
        x_hat = run.x_hat  # EN: Extract fitted coefficients.

        train_scores.append(rmse(y_true=b_tr, y_pred=A_tr @ x_hat))  # EN: Train RMSE.
        val_scores.append(rmse(y_true=b_va, y_pred=A_va @ x_hat))  # EN: Validation RMSE.
        x_norms.append(l2_norm(x_hat))  # EN: Record ||x||.

    train_arr = np.array(train_scores, dtype=float)  # EN: Convert to array for stats.
    val_arr = np.array(val_scores, dtype=float)  # EN: Convert to array for stats.
    x_arr = np.array(x_norms, dtype=float)  # EN: Convert to array for stats.

    key = f"d={damp:.0e}" if damp != 0.0 else "d=0"  # EN: Build label.
    return CVPoint(  # EN: Package CVPoint.
        key=key,  # EN: Label.
        param=float(damp),  # EN: Numeric damp.
        train_mean=float(np.mean(train_arr)),  # EN: Mean train RMSE.
        train_std=float(np.std(train_arr)),  # EN: Std train RMSE.
        val_mean=float(np.mean(val_arr)),  # EN: Mean val RMSE.
        val_std=float(np.std(val_arr)),  # EN: Std val RMSE.
        x_norm_mean=float(np.mean(x_arr)),  # EN: Mean ||x||.
    )  # EN: End return.


def main() -> None:  # EN: Run a full demo: stopping criteria + k-fold CV to choose damp.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    n_samples = 80  # EN: Number of samples.
    degree = 12  # EN: Polynomial degree (higher => more ill-conditioning).
    noise_std = 0.05  # EN: Noise level.
    n_folds = 5  # EN: Number of CV folds.

    t = np.linspace(-1.0, 1.0, n_samples)  # EN: Sample locations.
    A = build_vandermonde(t, degree=degree)  # EN: Build design matrix.
    n_features = A.shape[1]  # EN: Feature dimension.

    x_true = np.zeros((n_features,), dtype=float)  # EN: Ground-truth coefficients (sparse).
    x_true[0] = 0.5  # EN: Intercept.
    x_true[1] = 1.0  # EN: Linear term.
    x_true[2] = -2.0  # EN: Quadratic term.
    x_true[3] = 0.7  # EN: Cubic term.

    b_clean = A @ x_true  # EN: Clean targets.
    b = b_clean + noise_std * rng.standard_normal(b_clean.shape)  # EN: Add noise.

    splits = k_fold_splits(rng=rng, n_samples=n_samples, n_folds=n_folds)  # EN: Create CV splits.

    print_separator("Dataset Summary")  # EN: Print dataset diagnostics.
    cond_A = float(np.linalg.cond(A))  # EN: Condition number (small dense only).
    rank_A = int(np.linalg.matrix_rank(A))  # EN: Numerical rank.
    print(f"n_samples={n_samples}, degree={degree}, n_features={n_features}, n_folds={n_folds}")  # EN: Print sizes.
    print(f"rank(A)={rank_A}, cond(A)={cond_A:.3e}, noise_std={noise_std}")  # EN: Print conditioning and noise.

    # EN: LSMR settings: cap iterations near n_features and use strict tolerances.  # EN: Explain solver settings.
    max_iters = min(2 * n_features, 50)  # EN: Cap iterations (small problems converge in <= n steps).
    atol = 1e-10  # EN: Absolute tolerance.
    btol = 1e-10  # EN: Relative tolerance.
    estimate_norm_steps = 12  # EN: Norm-estimation steps (keep small for CV speed).

    # EN: Candidate damp values (include 0 baseline).  # EN: Explain grid choice.
    damps = np.concatenate(([0.0], np.logspace(-8, 1, num=15)))  # EN: Sweep damp over many magnitudes.

    print_separator("k-fold CV: damp (Ridge λ = damp^2) using Damped LSMR")  # EN: Announce CV section.
    points: list[CVPoint] = []  # EN: Collect CV summaries.
    for d in damps:  # EN: Loop candidate damps.
        points.append(  # EN: Append summary for this damp.
            summarize_cv_for_damp(  # EN: Compute CV stats.
                splits=splits,  # EN: Splits.
                A=A,  # EN: A.
                b=b,  # EN: b.
                damp=float(d),  # EN: damp.
                max_iters=max_iters,  # EN: iters.
                atol=atol,  # EN: atol.
                btol=btol,  # EN: btol.
                estimate_norm_steps=estimate_norm_steps,  # EN: norm steps.
                rng=rng,  # EN: RNG.
            )  # EN: End summarize call.
        )  # EN: End append.

    best = min(points, key=lambda p: p.val_mean)  # EN: Pick damp with smallest mean validation RMSE.
    print_cv_table(points=points, best=best)  # EN: Print table + ASCII curve.
    print(f"\nSelected damp = {best.param:.3e}  (λ = damp^2 ≈ {best.param**2:.3e})")  # EN: Print chosen damp and equivalent λ.

    # EN: Fit on full data with the selected damp and print stopping/optimality diagnostics.  # EN: Explain final fit.
    run_full = lsmr_damped_with_stopping(  # EN: Train on full dataset.
        A=A,  # EN: Full A.
        b=b,  # EN: Full b.
        damp=best.param,  # EN: Selected damp.
        max_iters=max_iters,  # EN: Iteration cap.
        atol=atol,  # EN: atol.
        btol=btol,  # EN: btol.
        estimate_norm_steps=estimate_norm_steps,  # EN: Norm steps.
        rng=rng,  # EN: RNG.
    )  # EN: End full fit.

    print_separator("Final Fit Diagnostics (full data)")  # EN: Announce diagnostics section.
    print(f"stop={run_full.stop_reason}, iters={run_full.n_iters}")  # EN: Print stop reason and iters.
    print(f"||Ax-b||_2          = {run_full.rnorm_data:.3e}")  # EN: Print data residual norm.
    print(f"||[Ax-b;damp x]||_2  = {run_full.rnorm_aug:.3e}")  # EN: Print augmented residual norm.
    print(f"||A^T r + damp^2 x|| = {run_full.arnorm:.3e}")  # EN: Print gradient norm.
    print(f"||x||_2             = {run_full.xnorm:.3e}")  # EN: Print x norm.
    print(f"||A_aug||_2 (est)    = {run_full.anorm_est:.3e}")  # EN: Print spectral norm estimate used in stopping.

    print_separator("Done")  # EN: End marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main.

