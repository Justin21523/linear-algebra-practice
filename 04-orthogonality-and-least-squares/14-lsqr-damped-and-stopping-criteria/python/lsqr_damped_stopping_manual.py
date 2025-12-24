"""  # EN: Start module docstring.
Damped LSQR (manual) with practical stopping criteria: solve least squares and Ridge without forming A^T A.  # EN: Summarize what this script does.

We solve two related problems:  # EN: Introduce the two problem types.
  (1) Least squares:   min_x ||A x - b||_2  (equivalently min 1/2||Ax-b||^2)  # EN: State LS problem.
  (2) Damped (Ridge):  min_x ||A x - b||_2^2 + damp^2 ||x||_2^2  (damp >= 0)  # EN: State damped LS / Ridge.

Key identity for damped LSQR:  # EN: Explain augmentation trick.
  min ||A x - b||^2 + damp^2||x||^2  ==  min || [A; damp I] x - [b; 0] ||^2.  # EN: Show augmented system.

This allows us to reuse the same LSQR core with only matvecs:  # EN: Explain algorithmic advantage.
  - For A:     y = A v,   z = A^T u  # EN: Base matvecs.
  - For damp:  y_bottom = damp * v,  z_add = damp * u_bottom  # EN: Added terms for augmented operator.

Stopping criteria:  # EN: Explain what we check to stop.
  - We monitor both the residual ||r|| = ||Ax-b|| and the normal residual ||A^T r||.  # EN: LS criteria.
  - For damped (Ridge), the optimality condition becomes ||A^T(Ax-b) + damp^2 x||.  # EN: Ridge gradient condition.
  - We use atol/btol-style mixed absolute/relative thresholds inspired by standard LSQR implementations.  # EN: Mention tolerance style.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured results.
from typing import Callable  # EN: Use Callable for typing matvec functions.

import numpy as np  # EN: Import NumPy for numerical operations.


EPS = 1e-12  # EN: Small epsilon for safe divisions.
RCOND = 1e-12  # EN: Relative cutoff for pseudo-inverse computations.
SEED = 0  # EN: RNG seed for deterministic examples.
PRINT_PRECISION = 6  # EN: Console float precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias for a matrix-vector product function.


@dataclass(frozen=True)  # EN: Immutable record for LSQR run results.
class LSQRRun:  # EN: Store solution and diagnostics for one LSQR call.
    x_hat: np.ndarray  # EN: Estimated solution vector (n,).
    n_iters: int  # EN: Iterations performed.
    stop_reason: str  # EN: Human-readable termination reason.
    rnorm_data: float  # EN: Data residual norm ||Ax-b||_2.
    rnorm_aug: float  # EN: Augmented residual norm sqrt(||Ax-b||^2 + damp^2||x||^2).
    arnorm: float  # EN: Normal-equation residual norm ||A^T(Ax-b) + damp^2 x||_2.
    xnorm: float  # EN: Solution norm ||x||_2.
    anorm_est: float  # EN: Estimated spectral norm ||A_aug||_2 used in stopping criteria.
    history_rnorm_data: np.ndarray  # EN: History of ||Ax-b|| per iteration (including iter 0).
    history_arnorm: np.ndarray  # EN: History of ||A^T(Ax-b)+damp^2 x|| per iteration.


def print_separator(title: str) -> None:  # EN: Print a section separator for readability.
    print()  # EN: Add a blank line.
    print("=" * 78)  # EN: Print divider line.
    print(title)  # EN: Print title.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def normalize(v: np.ndarray) -> np.ndarray:  # EN: Normalize a vector to unit length.
    n = l2_norm(v)  # EN: Compute norm.
    if n < EPS:  # EN: Guard against near-zero vectors.
        raise ValueError("Cannot normalize near-zero vector")  # EN: Raise for invalid input.
    return v / n  # EN: Return normalized vector.


def estimate_spectral_norm(  # EN: Estimate ||A||_2 using power iteration on A^T A (matvec-only).
    matvec_A: Matvec,  # EN: Function computing A v.
    matvec_AT: Matvec,  # EN: Function computing A^T u.
    n: int,  # EN: Dimension of domain vectors (columns).
    n_steps: int = 20,  # EN: Power-iteration steps.
    rng: np.random.Generator | None = None,  # EN: RNG for initialization.
) -> float:  # EN: Return an estimate of spectral norm ||A||_2.
    if n_steps <= 0:  # EN: Validate step count.
        raise ValueError("n_steps must be positive")  # EN: Reject invalid n_steps.
    if rng is None:  # EN: Provide default RNG.
        rng = np.random.default_rng(SEED)  # EN: Use fixed seed.
    v = normalize(rng.standard_normal(n))  # EN: Random unit vector in domain.
    for _ in range(n_steps):  # EN: Iterate on A^T A.
        w = matvec_A(v)  # EN: Compute w = A v.
        v = matvec_AT(w)  # EN: Compute v = A^T w = A^T A v.
        v = normalize(v)  # EN: Normalize to prevent blow-up.
    sigma_sq = l2_norm(matvec_A(v)) ** 2  # EN: Rayleigh quotient for A^T A: ||A v||^2.
    return float(np.sqrt(max(sigma_sq, 0.0)))  # EN: Return sqrt of estimated largest eigenvalue.


def solve_ridge_closed_form_svd(A: np.ndarray, b: np.ndarray, damp: float) -> np.ndarray:  # EN: Solve ridge exactly via SVD (reference only).
    if damp < 0.0:  # EN: Validate damp.
        raise ValueError("damp must be non-negative")  # EN: Reject invalid damp.
    U, s, Vt = np.linalg.svd(A, full_matrices=False)  # EN: Economy SVD A = U diag(s) V^T.
    if s.size == 0:  # EN: Handle empty case.
        return np.zeros((A.shape[1],), dtype=float)  # EN: Return zero vector.
    if damp == 0.0:  # EN: For damp=0, return pseudo-inverse LS solution.
        cutoff = RCOND * float(s.max())  # EN: Cutoff for small singular values.
        keep = s > cutoff  # EN: Keep significant singular values.
        if not np.any(keep):  # EN: If rank is zero, min-norm solution is zero.
            return np.zeros((A.shape[1],), dtype=float)  # EN: Return zeros.
        U_r = U[:, keep]  # EN: Kept U.
        s_r = s[keep]  # EN: Kept singular values.
        Vt_r = Vt[keep, :]  # EN: Kept V^T.
        return Vt_r.T @ ((U_r.T @ b) / s_r)  # EN: x = V diag(1/s) U^T b.
    lam = damp * damp  # EN: Convert damp to ridge lambda.
    factors = s / (s**2 + lam)  # EN: Ridge filter factors σ/(σ^2+λ).
    return Vt.T @ (factors * (U.T @ b))  # EN: x = V diag(factors) U^T b.


def lsqr_core(  # EN: Core LSQR solver for min ||A x - b|| on a generic matvec operator A.
    matvec_A: Matvec,  # EN: Function computing A v for v in R^n.
    matvec_AT: Matvec,  # EN: Function computing A^T u for u in R^m.
    b: np.ndarray,  # EN: Right-hand side vector in R^m.
    n: int,  # EN: Number of unknowns (domain dimension).
    max_iters: int,  # EN: Iteration cap.
    atol: float,  # EN: Absolute tolerance for stopping.
    btol: float,  # EN: Relative tolerance for stopping.
    anorm_est: float,  # EN: Estimated ||A||_2 for mixed stopping tests.
) -> tuple[np.ndarray, int, str]:  # EN: Return (x_hat, iters, stop_reason).
    # EN: This is a lightly adapted LSQR core (Paige–Saunders style) without all internal estimates.  # EN: Explain scope.
    m = b.size  # EN: Number of rows / equations.
    x = np.zeros((n,), dtype=float)  # EN: Start from x0 = 0.

    u = b.copy()  # EN: Initialize u = b.
    beta = l2_norm(u)  # EN: beta = ||b||.
    if beta < EPS:  # EN: Trivial b=0 case.
        return x, 0, "b is zero (trivial solution)"  # EN: Return x=0 immediately.
    u = u / beta  # EN: Normalize u.

    v = matvec_AT(u)  # EN: v = A^T u.
    alpha = l2_norm(v)  # EN: alpha = ||v||.
    if alpha < EPS:  # EN: Degenerate case A^T u = 0.
        return x, 0, "A^T b is zero (degenerate)"  # EN: Return x=0.
    v = v / alpha  # EN: Normalize v.

    w = v.copy()  # EN: Initialize w = v.
    phi_bar = beta  # EN: Initialize phi_bar.
    rho_bar = alpha  # EN: Initialize rho_bar.

    bnorm = beta  # EN: Store ||b|| for stopping tests.

    for it in range(1, max_iters + 1):  # EN: Main LSQR loop.
        u = matvec_A(v) - alpha * u  # EN: u <- A v - alpha u.
        beta = l2_norm(u)  # EN: beta = ||u||.
        if beta > EPS:  # EN: Normalize u if possible.
            u = u / beta  # EN: Normalize u.
        else:  # EN: If beta is zero, bidiagonalization breaks down.
            beta = 0.0  # EN: Keep beta for safe math.

        v = matvec_AT(u) - beta * v  # EN: v <- A^T u - beta v.
        alpha = l2_norm(v)  # EN: alpha = ||v||.
        if alpha > EPS:  # EN: Normalize v if possible.
            v = v / alpha  # EN: Normalize v.
        else:  # EN: If alpha is zero, stop expansion.
            alpha = 0.0  # EN: Keep alpha as zero.

        rho = float(np.sqrt(rho_bar * rho_bar + beta * beta))  # EN: rho = sqrt(rho_bar^2 + beta^2).
        if rho < EPS:  # EN: Guard against breakdown.
            break  # EN: Stop.
        c = rho_bar / rho  # EN: c = rho_bar / rho.
        s = beta / rho  # EN: s = beta / rho.
        theta = s * alpha  # EN: theta = s * alpha.
        rho_bar = -c * alpha  # EN: Update rho_bar.
        phi = c * phi_bar  # EN: phi = c * phi_bar.
        phi_bar = s * phi_bar  # EN: Update phi_bar.

        x = x + (phi / rho) * w  # EN: Update x.
        w = v - (theta / rho) * w  # EN: Update w.

        # EN: LSQR has cheap internal estimates: rnorm ≈ |phi_bar| and arnorm ≈ alpha*|phi_bar|.  # EN: Explain proxies.
        rnorm_est = abs(phi_bar)  # EN: Proxy for ||r||_2.
        arnorm_est = abs(alpha * phi_bar)  # EN: Proxy for ||A^T r||_2 (or augmented normal residual).

        # EN: Mixed stopping test: ||r|| <= btol*||b|| + atol*||A||*||x||.  # EN: Explain the main residual test.
        xnorm = l2_norm(x)  # EN: Compute ||x||.
        rnorm_bound = btol * bnorm + atol * anorm_est * xnorm  # EN: Compute mixed residual bound.
        if rnorm_est <= rnorm_bound:  # EN: Stop when residual proxy is below the mixed bound.
            return x, it, "residual bound satisfied"  # EN: Return with current iterate.

        # EN: Normal residual test: ||A^T r|| <= atol * ||A|| * ||r|| (common LSQR criterion).  # EN: Explain test.
        if arnorm_est <= atol * anorm_est * max(rnorm_est, EPS):  # EN: Stop when normal residual proxy is small.
            return x, it, "normal residual bound satisfied"  # EN: Return with current iterate.

        if beta == 0.0 and alpha == 0.0:  # EN: Stop if bidiagonalization cannot proceed.
            break  # EN: Exit loop.

    if max_iters > 0 and it >= max_iters:  # EN: If we exhausted the iteration cap, report that explicitly.
        return x, max_iters, "max_iters reached"  # EN: Return best effort due to iteration cap.
    return x, it, "stopped (breakdown)"  # EN: Return best effort due to numerical breakdown.


def lsqr_with_damping_and_stopping(  # EN: Solve (possibly damped) least squares with explicit diagnostics and stopping criteria.
    A: np.ndarray,  # EN: Design matrix (m×n) for this teaching demo (dense).
    b: np.ndarray,  # EN: Right-hand side vector (m,).
    damp: float = 0.0,  # EN: Damping parameter (>=0), ridge lambda = damp^2.
    max_iters: int = 200,  # EN: Iteration cap.
    atol: float = 1e-10,  # EN: Absolute tolerance for stopping tests.
    btol: float = 1e-10,  # EN: Relative tolerance for stopping tests.
    estimate_norm_steps: int = 20,  # EN: Steps to estimate ||A_aug||_2 for stopping tests.
    rng: np.random.Generator | None = None,  # EN: RNG for norm estimation and potential random choices.
) -> LSQRRun:  # EN: Return LSQRRun with diagnostics and history.
    if rng is None:  # EN: Provide default RNG.
        rng = np.random.default_rng(SEED)  # EN: Use deterministic seed.
    if damp < 0.0:  # EN: Validate damp.
        raise ValueError("damp must be non-negative")  # EN: Reject invalid damp.
    if b.ndim != 1:  # EN: Validate b shape.
        raise ValueError("b must be a 1D vector")  # EN: Reject invalid b.
    m, n = A.shape  # EN: Extract A shape.
    if b.size != m:  # EN: Validate b length.
        raise ValueError("b length must match number of rows of A")  # EN: Reject mismatch.

    # EN: Build matvecs for the augmented operator A_aug = [A; damp I] without forming it.  # EN: Explain wrapper.
    def matvec_A_aug(v: np.ndarray) -> np.ndarray:  # EN: Compute A_aug v = [A v; damp v].
        top = A @ v  # EN: Compute top block A v.
        bottom = damp * v  # EN: Compute bottom block damp * v.
        return np.concatenate([top, bottom])  # EN: Concatenate to length (m+n).

    def matvec_AT_aug(u_aug: np.ndarray) -> np.ndarray:  # EN: Compute A_aug^T u_aug = A^T u_top + damp u_bottom.
        u_top = u_aug[:m]  # EN: Extract top part of u.
        u_bottom = u_aug[m:]  # EN: Extract bottom part of u.
        return (A.T @ u_top) + (damp * u_bottom)  # EN: Return combined transpose product.

    b_aug = np.concatenate([b, np.zeros((n,), dtype=float)])  # EN: Build augmented RHS [b; 0].

    anorm_est = estimate_spectral_norm(  # EN: Estimate ||A_aug||_2 for mixed stopping tests.
        matvec_A=matvec_A_aug,  # EN: Provide A_aug matvec.
        matvec_AT=matvec_AT_aug,  # EN: Provide A_aug^T matvec.
        n=n,  # EN: Domain dimension is still n.
        n_steps=estimate_norm_steps,  # EN: Power-iteration steps.
        rng=rng,  # EN: RNG.
    )  # EN: End norm estimation.

    # EN: Run LSQR core on the augmented system to get x_hat.  # EN: Explain solver call.
    x_hat, iters, stop_reason_core = lsqr_core(  # EN: Solve min ||A_aug x - b_aug|| using LSQR.
        matvec_A=matvec_A_aug,  # EN: Provide A_aug matvec.
        matvec_AT=matvec_AT_aug,  # EN: Provide A_aug^T matvec.
        b=b_aug,  # EN: Provide augmented RHS.
        n=n,  # EN: Unknown dimension.
        max_iters=max_iters,  # EN: Iteration cap.
        atol=atol,  # EN: Absolute tolerance.
        btol=btol,  # EN: Relative tolerance.
        anorm_est=anorm_est,  # EN: Spectral norm estimate.
    )  # EN: End LSQR call.

    # EN: Build explicit history by re-running and recording at each iteration for teaching (small n only).  # EN: Explain why.
    # EN: This is not how you'd implement large-scale LSQR, but it makes stopping criteria transparent.  # EN: Note about performance.
    history_r: list[float] = []  # EN: Track ||Ax-b|| over iterations.
    history_ar: list[float] = []  # EN: Track ridge optimality norm ||A^T(Ax-b)+damp^2 x||.

    # EN: Replay the LSQR core but record diagnostics each iteration (deterministic given A,b,damp).  # EN: Explain approach.
    x = np.zeros((n,), dtype=float)  # EN: Start from x0=0 for replay.
    u = b_aug.copy()  # EN: Initialize u from augmented RHS.
    beta = l2_norm(u)  # EN: beta = ||b_aug||.
    if beta >= EPS:  # EN: Proceed only if non-trivial.
        u = u / beta  # EN: Normalize u.
        v = matvec_AT_aug(u)  # EN: v = A_aug^T u.
        alpha = l2_norm(v)  # EN: alpha = ||v||.
        if alpha >= EPS:  # EN: Proceed if alpha is non-zero.
            v = v / alpha  # EN: Normalize v.
            w = v.copy()  # EN: Initialize w.
            phi_bar = beta  # EN: Initialize phi_bar.
            rho_bar = alpha  # EN: Initialize rho_bar.

            def record(x_cur: np.ndarray) -> None:  # EN: Record explicit diagnostics for current x.
                r_data = (A @ x_cur) - b  # EN: Data residual Ax-b.
                grad = (A.T @ r_data) + (damp * damp) * x_cur  # EN: Ridge gradient A^T(Ax-b)+damp^2 x.
                history_r.append(l2_norm(r_data))  # EN: Store ||Ax-b||.
                history_ar.append(l2_norm(grad))  # EN: Store ridge optimality norm.

            record(x)  # EN: Record iteration 0.

            for _it in range(1, min(max_iters, iters) + 1):  # EN: Replay up to the actual iteration count.
                u = matvec_A_aug(v) - alpha * u  # EN: u <- A_aug v - alpha u.
                beta = l2_norm(u)  # EN: beta = ||u||.
                if beta > EPS:  # EN: Normalize u.
                    u = u / beta  # EN: Normalize.
                else:  # EN: Break on breakdown.
                    break  # EN: Stop replay.

                v = matvec_AT_aug(u) - beta * v  # EN: v <- A_aug^T u - beta v.
                alpha = l2_norm(v)  # EN: alpha = ||v||.
                if alpha > EPS:  # EN: Normalize v.
                    v = v / alpha  # EN: Normalize.
                else:  # EN: Break on breakdown.
                    break  # EN: Stop replay.

                rho = float(np.sqrt(rho_bar * rho_bar + beta * beta))  # EN: rho = sqrt(rho_bar^2 + beta^2).
                if rho < EPS:  # EN: Break on breakdown.
                    break  # EN: Stop replay.
                c = rho_bar / rho  # EN: c = rho_bar/rho.
                s = beta / rho  # EN: s = beta/rho.
                theta = s * alpha  # EN: theta = s*alpha.
                rho_bar = -c * alpha  # EN: Update rho_bar.
                phi = c * phi_bar  # EN: phi = c*phi_bar.
                phi_bar = s * phi_bar  # EN: Update phi_bar.

                x = x + (phi / rho) * w  # EN: Update x.
                w = v - (theta / rho) * w  # EN: Update w.

                record(x)  # EN: Record diagnostics at this iteration.

    # EN: Compute final diagnostics for the returned x_hat.  # EN: Explain final metrics.
    r_data = (A @ x_hat) - b  # EN: Data residual.
    rnorm_data = l2_norm(r_data)  # EN: ||Ax-b||.
    rnorm_aug = float(np.sqrt(rnorm_data * rnorm_data + (damp * l2_norm(x_hat)) ** 2))  # EN: Augmented residual norm.
    ar = (A.T @ r_data) + (damp * damp) * x_hat  # EN: Ridge normal residual (gradient).
    arnorm = l2_norm(ar)  # EN: ||A^T(Ax-b)+damp^2 x||.
    xnorm = l2_norm(x_hat)  # EN: ||x||.

    # EN: Determine a human-readable stop reason based on the mixed stopping conditions.  # EN: Explain.
    bnorm = l2_norm(b)  # EN: ||b|| for bound.
    rnorm_bound = btol * bnorm + atol * anorm_est * xnorm  # EN: Mixed bound.
    if stop_reason_core == "residual bound satisfied":  # EN: Prefer the core stop reason when it is a standard criterion.
        stop_reason = stop_reason_core  # EN: Store stop reason.
    elif stop_reason_core == "normal residual bound satisfied":  # EN: Prefer the core normal-residual reason.
        stop_reason = stop_reason_core  # EN: Store stop reason.
    elif stop_reason_core == "max_iters reached":  # EN: Prefer the core max-iter reason.
        stop_reason = stop_reason_core  # EN: Store stop reason.
    elif rnorm_aug <= rnorm_bound:  # EN: Otherwise, compute a reason from explicit diagnostics.
        stop_reason = "residual within (btol*||b|| + atol*||A||*||x||) bound"  # EN: Record stop reason.
    elif arnorm <= atol * anorm_est * max(rnorm_aug, EPS):  # EN: Check explicit normal residual bound.
        stop_reason = "normal residual within atol*||A||*||r|| bound"  # EN: Record stop reason.
    else:  # EN: Fallback to core message.
        stop_reason = stop_reason_core  # EN: Record fallback reason.

    return LSQRRun(  # EN: Return the full run report.
        x_hat=x_hat,  # EN: Store solution.
        n_iters=iters,  # EN: Store iteration count.
        stop_reason=stop_reason,  # EN: Store stop reason.
        rnorm_data=rnorm_data,  # EN: Store data residual norm.
        rnorm_aug=rnorm_aug,  # EN: Store augmented residual norm.
        arnorm=arnorm,  # EN: Store normal residual norm.
        xnorm=xnorm,  # EN: Store solution norm.
        anorm_est=anorm_est,  # EN: Store norm estimate.
        history_rnorm_data=np.array(history_r, dtype=float),  # EN: Store history arrays.
        history_arnorm=np.array(history_ar, dtype=float),  # EN: Store history arrays.
    )  # EN: End return.


def build_design_matrix_multicollinear(  # EN: Build a multicollinear design matrix to stress least squares.
    rng: np.random.Generator,  # EN: RNG for reproducibility.
    m: int,  # EN: Number of samples.
    col_eps: float,  # EN: Collinearity level (smaller -> more collinear).
) -> np.ndarray:  # EN: Return A with columns [1, x1, x2≈x1].
    x1 = rng.standard_normal(m)  # EN: First feature.
    x2 = x1 + col_eps * rng.standard_normal(m)  # EN: Second feature nearly equal to x1.
    return np.column_stack([np.ones(m), x1, x2]).astype(float)  # EN: Build design matrix.


def run_demo_case(name: str, A: np.ndarray, b: np.ndarray, damp_values: list[float]) -> None:  # EN: Run LSQR for multiple damp values.
    print_separator(f"Case: {name}")  # EN: Print case title.
    print(f"A shape: {A.shape[0]}×{A.shape[1]}, cond(A)={float(np.linalg.cond(A)):.3e}")  # EN: Print diagnostics.

    for damp in damp_values:  # EN: Loop over damp values.
        res = lsqr_with_damping_and_stopping(  # EN: Run damped LSQR.
            A=A,  # EN: Provide A.
            b=b,  # EN: Provide b.
            damp=damp,  # EN: Provide damp.
            max_iters=200,  # EN: Iteration cap.
            atol=1e-10,  # EN: atol.
            btol=1e-10,  # EN: btol.
            estimate_norm_steps=15,  # EN: Norm estimation steps.
        )  # EN: End LSQR call.

        if damp == 0.0:  # EN: Reference is standard least squares when damp=0.
            x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)  # EN: Reference solution using NumPy.
        else:  # EN: Reference is ridge solution for damp>0.
            x_ref = solve_ridge_closed_form_svd(A, b, damp=damp)  # EN: Reference ridge solution via SVD filter.

        rel_err = l2_norm(res.x_hat - x_ref) / max(l2_norm(x_ref), EPS)  # EN: Relative error vs reference.

        print_separator(f"LSQR run (damp={damp:.3e})")  # EN: Print per-run header.
        print(f"iters={res.n_iters}, stop_reason={res.stop_reason}")  # EN: Print termination info.
        print(f"||Ax-b||_2 = {res.rnorm_data:.3e}")  # EN: Print data residual.
        print(f"augmented ||r||_2 = {res.rnorm_aug:.3e}")  # EN: Print augmented residual.
        print(f"||A^T(Ax-b)+damp^2 x||_2 = {res.arnorm:.3e}")  # EN: Print ridge optimality norm.
        print(f"||x||_2 = {res.xnorm:.3e}")  # EN: Print coefficient norm.
        print(f"||A_aug||_2 estimate = {res.anorm_est:.3e}")  # EN: Print norm estimate.
        print(f"rel_err_vs_ref = {rel_err:.3e}")  # EN: Print relative error vs reference.
        print("x_hat =", res.x_hat)  # EN: Print solution vector.

        # EN: Print a small convergence snapshot from the recorded history.  # EN: Explain why.
        if res.history_rnorm_data.size:  # EN: Ensure non-empty history.
            last = int(res.history_rnorm_data.size - 1)  # EN: Last index.
            for idx in sorted({0, 1, min(5, last), min(10, last), last}):  # EN: Choose a few checkpoints.
                r_i = float(res.history_rnorm_data[idx])  # EN: Data residual at checkpoint.
                ar_i = float(res.history_arnorm[idx])  # EN: Normal residual at checkpoint.
                print(f"iter {idx:3d}: ||Ax-b||={r_i:.3e}, ||normal||={ar_i:.3e}")  # EN: Print checkpoint metrics.


def main() -> None:  # EN: Run two demo cases showing damped LSQR and stopping behavior.
    rng = np.random.default_rng(SEED)  # EN: Deterministic RNG.

    m = 250  # EN: Number of samples.
    x_true = np.array([1.0, 2.0, -1.0], dtype=float)  # EN: Ground-truth coefficients for synthetic regression.
    noise_std = 1e-3  # EN: Noise level.

    A_good = build_design_matrix_multicollinear(rng=rng, m=m, col_eps=1e0)  # EN: Weak collinearity (better conditioned).
    b_good = A_good @ x_true + noise_std * rng.standard_normal(m)  # EN: Build targets.

    A_bad = build_design_matrix_multicollinear(rng=rng, m=m, col_eps=1e-8)  # EN: Strong collinearity (ill-conditioned).
    b_bad = A_bad @ x_true + noise_std * rng.standard_normal(m)  # EN: Build targets.

    damp_values = [0.0, 1e-6, 1e-2]  # EN: Compare no damping vs mild vs stronger damping.

    run_demo_case(name="Well-conditioned-ish", A=A_good, b=b_good, damp_values=damp_values)  # EN: Run good case.
    run_demo_case(name="Ill-conditioned (multicollinearity)", A=A_bad, b=b_bad, damp_values=damp_values)  # EN: Run bad case.

    print_separator("Done")  # EN: End marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main.
