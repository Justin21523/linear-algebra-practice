"""  # EN: Start module docstring.
LSMR (teaching version): solve least squares via MINRES on the normal equations, without forming A^T A.  # EN: Summarize what this script does.

Problem:  # EN: State the mathematical goal.
  x_hat = argmin_x ||A x - b||_2, for (possibly) overdetermined A (m×n).  # EN: Describe least-squares setting.

Normal equations view:  # EN: Explain the normal equation connection.
  The first-order optimality condition is A^T(Ax-b) = 0.  # EN: State optimality.
  When A has full column rank, the LS solution satisfies (A^T A) x = A^T b.  # EN: State normal equations.

LSMR idea (core concept):  # EN: Explain what LSMR is.
  LSMR is algebraically equivalent to applying MINRES to the normal equations  # EN: Connect to MINRES.
    (A^T A) x = A^T b,  # EN: Explicit normal system.
  but it avoids forming A^T A by using matvecs with A and A^T.  # EN: Explain matvec-only advantage.

This implementation is "teaching-first":  # EN: Clarify design goals and limitations.
  - We use Golub–Kahan bidiagonalization to build the small tridiagonal matrix T_k = B_k^T B_k.  # EN: Describe the internal reduction.
  - We compute the MINRES iterate by solving a small least-squares problem with T_k each iteration.  # EN: Explain the approach.
  - This is not the most memory/time efficient way to implement LSMR, but it is easy to verify and understand.  # EN: State trade-off.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured results.
from typing import Callable  # EN: Use Callable for matvec typing.

import numpy as np  # EN: Import NumPy for numerical computation.


EPS = 1e-12  # EN: Small epsilon to avoid division-by-zero.
SEED = 0  # EN: RNG seed for reproducible demos.
PRINT_PRECISION = 6  # EN: Console float precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias for a matrix-vector product function.


@dataclass(frozen=True)  # EN: Immutable record for an LSMR run.
class LSMRRun:  # EN: Store the solution and key diagnostic histories.
    x_hat: np.ndarray  # EN: Estimated solution vector (n,).
    n_iters: int  # EN: Number of iterations performed (subspace dimension).
    stop_reason: str  # EN: Human-readable termination reason.
    rnorm: float  # EN: Final data residual norm ||Ax-b||_2.
    arnorm: float  # EN: Final normal residual norm ||A^T(Ax-b)||_2.
    xnorm: float  # EN: Final solution norm ||x||_2.
    history_rnorm: np.ndarray  # EN: History of ||Ax-b||_2 over iterations (including iter 0).
    history_arnorm: np.ndarray  # EN: History of ||A^T(Ax-b)||_2 over iterations (including iter 0).


@dataclass(frozen=True)  # EN: Immutable record for a simple LSQR run (baseline for comparison).
class LSQRRun:  # EN: Store LSQR solution and diagnostics.
    x_hat: np.ndarray  # EN: Estimated solution vector (n,).
    n_iters: int  # EN: Iterations performed.
    rnorm: float  # EN: Final ||Ax-b||_2.
    arnorm: float  # EN: Final ||A^T(Ax-b)||_2.
    history_rnorm: np.ndarray  # EN: History of ||Ax-b||_2.
    history_arnorm: np.ndarray  # EN: History of ||A^T(Ax-b)||_2.


def print_separator(title: str) -> None:  # EN: Print a section separator for readable console output.
    print()  # EN: Add a blank line.
    print("=" * 78)  # EN: Print divider.
    print(title)  # EN: Print title.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm).
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def orthonormalize_columns(W: np.ndarray) -> np.ndarray:  # EN: Orthonormalize columns via QR.
    Q, _ = np.linalg.qr(W, mode="reduced")  # EN: Reduced QR yields orthonormal Q.
    return Q  # EN: Return Q.


def lsqr(  # EN: Minimal LSQR baseline (stops on normal residual ||A^T r||).
    matvec_A: Matvec,  # EN: Function computing A @ v.
    matvec_AT: Matvec,  # EN: Function computing A^T @ u.
    b: np.ndarray,  # EN: RHS vector (m,).
    n: int,  # EN: Unknown dimension (columns of A).
    max_iters: int = 200,  # EN: Iteration cap.
    tol_normal: float = 1e-10,  # EN: Stop when ||A^T(Ax-b)|| <= tol_normal.
) -> LSQRRun:  # EN: Return LSQRRun.
    x = np.zeros((n,), dtype=float)  # EN: Start from x0=0.

    r0 = matvec_A(x) - b  # EN: Initial residual r0 = Ax0 - b = -b.
    nr0 = matvec_AT(r0)  # EN: Initial normal residual A^T r0.
    hist_r: list[float] = [l2_norm(r0)]  # EN: Record ||r|| at iter 0.
    hist_nr: list[float] = [l2_norm(nr0)]  # EN: Record ||A^T r|| at iter 0.

    u = b.copy()  # EN: Initialize u from b.
    beta = l2_norm(u)  # EN: beta = ||b||.
    if beta < EPS:  # EN: Handle b=0 -> x=0 is optimal.
        return LSQRRun(  # EN: Return early.
            x_hat=x,  # EN: Solution.
            n_iters=0,  # EN: Iterations.
            rnorm=hist_r[-1],  # EN: Final rnorm.
            arnorm=hist_nr[-1],  # EN: Final arnorm.
            history_rnorm=np.array(hist_r, dtype=float),  # EN: History.
            history_arnorm=np.array(hist_nr, dtype=float),  # EN: History.
        )  # EN: End return.
    u = u / beta  # EN: Normalize u.

    v = matvec_AT(u)  # EN: v = A^T u.
    alpha = l2_norm(v)  # EN: alpha = ||v||.
    if alpha < EPS:  # EN: Handle A^T u = 0 breakdown.
        return LSQRRun(  # EN: Return early.
            x_hat=x,  # EN: Solution.
            n_iters=0,  # EN: Iterations.
            rnorm=hist_r[-1],  # EN: Final rnorm.
            arnorm=hist_nr[-1],  # EN: Final arnorm.
            history_rnorm=np.array(hist_r, dtype=float),  # EN: History.
            history_arnorm=np.array(hist_nr, dtype=float),  # EN: History.
        )  # EN: End return.
    v = v / alpha  # EN: Normalize v.

    w = v.copy()  # EN: Initialize w for recurrences.
    phi_bar = beta  # EN: Initialize φ̄.
    rho_bar = alpha  # EN: Initialize ρ̄.

    for it in range(1, max_iters + 1):  # EN: Main LSQR loop.
        u = matvec_A(v) - alpha * u  # EN: u_{k+1} = A v_k - α_k u_k.
        beta = l2_norm(u)  # EN: β_{k+1} = ||u_{k+1}||.
        if beta >= EPS:  # EN: Normalize u when possible.
            u = u / beta  # EN: Normalize u.

        v = matvec_AT(u) - beta * v  # EN: v_{k+1} = A^T u_{k+1} - β_{k+1} v_k.
        alpha = l2_norm(v)  # EN: α_{k+1} = ||v_{k+1}||.
        if alpha >= EPS:  # EN: Normalize v when possible.
            v = v / alpha  # EN: Normalize v.

        rho = float(np.hypot(rho_bar, beta))  # EN: ρ = sqrt(ρ̄^2 + β^2).
        c = rho_bar / max(rho, EPS)  # EN: c = ρ̄ / ρ.
        s = beta / max(rho, EPS)  # EN: s = β / ρ.
        theta = s * alpha  # EN: θ = s α.
        rho_bar = -c * alpha  # EN: Update ρ̄ = -c α.
        phi = c * phi_bar  # EN: φ = c φ̄.
        phi_bar = s * phi_bar  # EN: Update φ̄ = s φ̄.

        x = x + (phi / max(rho, EPS)) * w  # EN: Update solution estimate.
        w = v - (theta / max(rho, EPS)) * w  # EN: Update direction vector.

        r = matvec_A(x) - b  # EN: Compute residual for diagnostics.
        nr = matvec_AT(r)  # EN: Compute normal residual for stopping.
        hist_r.append(l2_norm(r))  # EN: Record ||Ax-b||.
        hist_nr.append(l2_norm(nr))  # EN: Record ||A^T(Ax-b)||.

        if hist_nr[-1] <= tol_normal:  # EN: Stop when normal residual is small.
            break  # EN: Exit loop.
        if beta < EPS and alpha < EPS:  # EN: Detect breakdown.
            break  # EN: Exit on breakdown.

    return LSQRRun(  # EN: Package results.
        x_hat=x,  # EN: Final x.
        n_iters=len(hist_r) - 1,  # EN: Iterations (history includes iter 0).
        rnorm=hist_r[-1],  # EN: Final rnorm.
        arnorm=hist_nr[-1],  # EN: Final arnorm.
        history_rnorm=np.array(hist_r, dtype=float),  # EN: rnorm history.
        history_arnorm=np.array(hist_nr, dtype=float),  # EN: arnorm history.
    )  # EN: End return.


def build_tridiagonal_T_from_golub_kahan(  # EN: Build T_k = B_k^T B_k given alpha[1..k], beta[2..k+1].
    alphas: np.ndarray,  # EN: Array of alpha values (length >= k), alpha_i = ||A^T u_i - beta_i v_{i-1}||.
    betas: np.ndarray,  # EN: Array of beta values (length >= k+1), beta_1 = ||b||, beta_{i+1} from bidiagonalization.
    k: int,  # EN: Desired tridiagonal size.
) -> np.ndarray:  # EN: Return dense k×k tridiagonal matrix T_k.
    if k <= 0:  # EN: Validate k.
        raise ValueError("k must be positive")  # EN: Reject invalid k.
    if alphas.size < k:  # EN: Validate alpha length.
        raise ValueError("alphas must have length >= k")  # EN: Reject invalid inputs.
    if betas.size < k + 1:  # EN: Validate beta length (needs beta_1..beta_{k+1}).
        raise ValueError("betas must have length >= k+1")  # EN: Reject invalid inputs.

    diag = (alphas[:k] ** 2) + (betas[1 : k + 1] ** 2)  # EN: diag_i = alpha_i^2 + beta_{i+1}^2.
    off = alphas[1:k] * betas[1:k]  # EN: off_i = alpha_{i+1} * beta_{i+1} for i=1..k-1.

    T = np.diag(diag.astype(float))  # EN: Start with diagonal matrix.
    if k > 1:  # EN: Add off-diagonals when k>=2.
        T = T + np.diag(off.astype(float), k=1) + np.diag(off.astype(float), k=-1)  # EN: Add symmetric off-diagonals.
    return T  # EN: Return T_k.


def lsmr_teaching_minres_normal_equations(  # EN: Teaching LSMR: MINRES iterate via solving small LS with T_k each iteration.
    matvec_A: Matvec,  # EN: Function computing A @ v.
    matvec_AT: Matvec,  # EN: Function computing A^T @ u.
    b: np.ndarray,  # EN: RHS vector (m,).
    n: int,  # EN: Unknown dimension.
    max_iters: int = 200,  # EN: Maximum subspace dimension.
    tol_rel_arnorm: float = 1e-10,  # EN: Relative tolerance on ||A^T(Ax-b)|| (normal residual).
) -> LSMRRun:  # EN: Return LSMRRun with histories.
    if b.ndim != 1:  # EN: Validate b shape.
        raise ValueError("b must be a 1D vector")  # EN: Reject invalid b.
    if n <= 0:  # EN: Validate n.
        raise ValueError("n must be positive")  # EN: Reject invalid n.
    if max_iters <= 0:  # EN: Validate max_iters.
        raise ValueError("max_iters must be positive")  # EN: Reject invalid max_iters.
    if tol_rel_arnorm <= 0.0:  # EN: Validate tolerance.
        raise ValueError("tol_rel_arnorm must be positive")  # EN: Reject invalid tol.

    x = np.zeros((n,), dtype=float)  # EN: Start from x0=0.
    r0 = matvec_A(x) - b  # EN: Initial residual r0 = -b.
    ar0 = matvec_AT(r0)  # EN: Initial normal residual A^T r0.

    hist_r: list[float] = [l2_norm(r0)]  # EN: Record ||Ax-b|| at iter 0.
    hist_ar: list[float] = [l2_norm(ar0)]  # EN: Record ||A^T(Ax-b)|| at iter 0.

    # EN: Initialize Golub–Kahan bidiagonalization.  # EN: Explain initialization.
    u = b.copy()  # EN: Start u from b.
    beta1 = l2_norm(u)  # EN: beta1 = ||b||.
    if beta1 < EPS:  # EN: Handle trivial b=0.
        return LSMRRun(  # EN: Return early with x=0.
            x_hat=x,  # EN: Solution.
            n_iters=0,  # EN: Iterations.
            stop_reason="b is zero",  # EN: Reason.
            rnorm=hist_r[-1],  # EN: Final rnorm.
            arnorm=hist_ar[-1],  # EN: Final arnorm.
            xnorm=l2_norm(x),  # EN: x norm.
            history_rnorm=np.array(hist_r, dtype=float),  # EN: History.
            history_arnorm=np.array(hist_ar, dtype=float),  # EN: History.
        )  # EN: End return.
    u = u / beta1  # EN: Normalize u1.

    v = matvec_AT(u)  # EN: v1 = A^T u1.
    alpha1 = l2_norm(v)  # EN: alpha1 = ||A^T u1||.
    if alpha1 < EPS:  # EN: Handle A^T b = 0 -> x=0 is LS solution.
        return LSMRRun(  # EN: Return early.
            x_hat=x,  # EN: Solution.
            n_iters=0,  # EN: Iterations.
            stop_reason="A^T b is zero",  # EN: Reason.
            rnorm=hist_r[-1],  # EN: Final rnorm.
            arnorm=hist_ar[-1],  # EN: Final arnorm.
            xnorm=l2_norm(x),  # EN: x norm.
            history_rnorm=np.array(hist_r, dtype=float),  # EN: History.
            history_arnorm=np.array(hist_ar, dtype=float),  # EN: History.
        )  # EN: End return.
    v = v / alpha1  # EN: Normalize v1.

    # EN: Prepare storage for v-basis vectors (so we can form x_k = V_k y_k).  # EN: Explain memory use.
    V_basis = np.zeros((n, max_iters + 1), dtype=float)  # EN: Store v1..v_{max_iters+1} as columns.
    V_basis[:, 0] = v  # EN: Store v1.

    # EN: Store alpha and beta sequences for building T_k = B_k^T B_k.  # EN: Explain sequences.
    alphas: list[float] = [float(alpha1)]  # EN: alpha_1 at index 0.
    betas: list[float] = [float(beta1)]  # EN: beta_1 at index 0 (beta_{i+1} will be appended each iteration).

    g_norm = float(alpha1 * beta1)  # EN: ||A^T b|| equals alpha1*beta1 (since v1 is unit).
    if g_norm < EPS:  # EN: Handle degenerate g_norm defensively.
        g_norm = float(l2_norm(matvec_AT(b)))  # EN: Fallback to direct compute when needed.

    stop_reason = "max_iters reached"  # EN: Default stop reason.
    n_done = 0  # EN: Track actual iterations performed (k).

    # EN: Main loop: each iteration expands the Krylov subspace by 1 dimension.  # EN: Explain loop meaning.
    for k in range(1, max_iters + 1):  # EN: k = current subspace dimension.
        # EN: One Golub–Kahan step computes beta_{k+1}, alpha_{k+1}, and v_{k+1}.  # EN: Explain step outputs.
        u_next = matvec_A(v) - alphas[-1] * u  # EN: u_{k+1} = A v_k - alpha_k u_k.
        beta_next = l2_norm(u_next)  # EN: beta_{k+1} = ||u_{k+1}||.
        if beta_next >= EPS:  # EN: Normalize when non-zero.
            u_next = u_next / beta_next  # EN: Normalize u_{k+1}.

        v_next = matvec_AT(u_next) - beta_next * v  # EN: v_{k+1} = A^T u_{k+1} - beta_{k+1} v_k.
        alpha_next = l2_norm(v_next)  # EN: alpha_{k+1} = ||v_{k+1}||.
        if alpha_next >= EPS:  # EN: Normalize when non-zero.
            v_next = v_next / alpha_next  # EN: Normalize v_{k+1}.

        betas.append(float(beta_next))  # EN: Append beta_{k+1}.
        alphas.append(float(alpha_next))  # EN: Append alpha_{k+1}.
        V_basis[:, k] = v_next  # EN: Store v_{k+1} for future iterations.

        # EN: Build T_k using alpha_1..alpha_k and beta_1..beta_{k+1}.  # EN: Explain T_k build.
        alpha_arr = np.array(alphas, dtype=float)  # EN: Convert alpha list to array.
        beta_arr = np.array(betas, dtype=float)  # EN: Convert beta list to array.
        T_k = build_tridiagonal_T_from_golub_kahan(alphas=alpha_arr, betas=beta_arr, k=k)  # EN: Build k×k tridiagonal.

        rhs = np.zeros((k,), dtype=float)  # EN: Right-hand side in Krylov basis for normal equations.
        rhs[0] = g_norm  # EN: Initial residual magnitude for Hx=g is ||g||, placed on e1.

        # EN: MINRES iterate solves y_k = argmin ||rhs - T_k y||_2.  # EN: Define the small LS problem.
        y_k, *_ = np.linalg.lstsq(T_k, rhs, rcond=None)  # EN: Solve the small LS problem (stable for small k).

        # EN: Map back to x-space: x_k = V_k y_k, where V_k = [v1..vk].  # EN: Explain reconstruction.
        x = V_basis[:, :k] @ y_k  # EN: Compute x_k in R^n.

        # EN: Normal-residual norm equals ||g - Hx|| = ||A^T(Ax-b)||; in Krylov basis it is ||rhs - T_k y||.  # EN: Explain arnorm compute.
        arnorm = l2_norm(rhs - (T_k @ y_k))  # EN: Compute normal residual norm from small system residual.

        r = matvec_A(x) - b  # EN: Compute data residual r = Ax-b for diagnostics.
        rnorm = l2_norm(r)  # EN: Compute ||Ax-b||.

        hist_r.append(float(rnorm))  # EN: Record data residual norm.
        hist_ar.append(float(arnorm))  # EN: Record normal residual norm.

        n_done = k  # EN: Update completed iteration count.

        # EN: Relative stopping test on normal residual: ||A^T r|| <= tol * ||A^T b||.  # EN: Explain stopping criterion.
        if arnorm <= tol_rel_arnorm * max(g_norm, EPS):  # EN: Stop when normal residual is sufficiently small.
            stop_reason = "normal residual tolerance satisfied"  # EN: Record stop reason.
            break  # EN: Exit loop.

        if beta_next < EPS and alpha_next < EPS:  # EN: Breakdown: bidiagonalization cannot proceed.
            stop_reason = "breakdown (beta and alpha near zero)"  # EN: Record breakdown reason.
            break  # EN: Exit loop.

        # EN: Advance GK states to the next iteration.  # EN: Explain state update.
        u = u_next  # EN: Set u_k <- u_{k+1}.
        v = v_next  # EN: Set v_k <- v_{k+1}.

    # EN: Final diagnostics computed from the last x.  # EN: Explain post-loop diagnostics.
    r_final = matvec_A(x) - b  # EN: Final data residual.
    ar_final = matvec_AT(r_final)  # EN: Final normal residual vector.

    return LSMRRun(  # EN: Package outputs into LSMRRun.
        x_hat=x,  # EN: Final solution.
        n_iters=int(n_done),  # EN: Iterations completed.
        stop_reason=stop_reason,  # EN: Termination reason.
        rnorm=l2_norm(r_final),  # EN: Final ||Ax-b||.
        arnorm=l2_norm(ar_final),  # EN: Final ||A^T(Ax-b)||.
        xnorm=l2_norm(x),  # EN: Final ||x||.
        history_rnorm=np.array(hist_r, dtype=float),  # EN: rnorm history.
        history_arnorm=np.array(hist_ar, dtype=float),  # EN: arnorm history.
    )  # EN: End return.


def make_conditioned_matrix(  # EN: Build A with controlled singular values to create an ill-conditioned LS problem.
    rng: np.random.Generator,  # EN: RNG for reproducibility.
    m: int,  # EN: Number of rows.
    n: int,  # EN: Number of columns.
    singular_values: np.ndarray,  # EN: Desired singular values (length n).
) -> np.ndarray:  # EN: Return dense A ≈ Q1 diag(s) Q2^T.
    G1 = rng.standard_normal((m, n))  # EN: Random Gaussian matrix for left factor.
    Q1, _ = np.linalg.qr(G1, mode="reduced")  # EN: Orthonormalize columns for Q1 (m×n).
    G2 = rng.standard_normal((n, n))  # EN: Random Gaussian matrix for right factor.
    Q2, _ = np.linalg.qr(G2)  # EN: Orthonormalize for Q2 (n×n).
    S = np.diag(singular_values.astype(float))  # EN: Diagonal matrix of singular values.
    return Q1 @ S @ Q2.T  # EN: Construct A.


def main() -> None:  # EN: Run a small demo comparing LSQR vs LSMR on an ill-conditioned least-squares problem.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    m = 300  # EN: Number of samples/rows.
    n = 60  # EN: Number of features/columns.
    noise_std = 1e-3  # EN: Noise level for b.

    # EN: Build a strongly ill-conditioned A using a wide singular spectrum.  # EN: Explain design choice.
    s = np.logspace(0.0, -10.0, num=n, base=10.0)  # EN: Singular values from 1 to 1e-10.
    A = make_conditioned_matrix(rng=rng, m=m, n=n, singular_values=s)  # EN: Construct design matrix.

    x_true = rng.standard_normal(n)  # EN: Choose a random ground-truth x.
    b = A @ x_true + noise_std * rng.standard_normal(m)  # EN: Build noisy targets.

    cond_A = float(np.linalg.cond(A))  # EN: Compute cond(A) (small n only) for intuition.
    print_separator("Problem Summary")  # EN: Print summary header.
    print(f"A shape: {m}×{n}, cond(A)≈{cond_A:.3e}, noise_std={noise_std:.1e}")  # EN: Print problem diagnostics.

    def matvec_A(v: np.ndarray) -> np.ndarray:  # EN: Dense matvec for A.
        return A @ v  # EN: Compute A v.

    def matvec_AT(u: np.ndarray) -> np.ndarray:  # EN: Dense matvec for A^T.
        return A.T @ u  # EN: Compute A^T u.

    max_iters = 200  # EN: Iteration cap.
    tol_normal = 1e-8  # EN: Normal residual tolerance for both methods.

    print_separator("Run: LSQR (baseline)")  # EN: Announce LSQR run.
    lsqr_res = lsqr(matvec_A=matvec_A, matvec_AT=matvec_AT, b=b, n=n, max_iters=max_iters, tol_normal=tol_normal)  # EN: Run LSQR.
    print(f"iters={lsqr_res.n_iters}, ||Ax-b||={lsqr_res.rnorm:.3e}, ||A^T r||={lsqr_res.arnorm:.3e}")  # EN: Print LSQR summary.

    print_separator("Run: LSMR (teaching MINRES-on-normal-equations)")  # EN: Announce LSMR run.
    lsmr_res = lsmr_teaching_minres_normal_equations(  # EN: Run LSMR-style solver.
        matvec_A=matvec_A,  # EN: Provide A matvec.
        matvec_AT=matvec_AT,  # EN: Provide A^T matvec.
        b=b,  # EN: Provide RHS.
        n=n,  # EN: Provide n.
        max_iters=max_iters,  # EN: Provide max_iters.
        tol_rel_arnorm=tol_normal,  # EN: Use same tolerance for comparability.
    )  # EN: End call.
    print(f"iters={lsmr_res.n_iters}, stop={lsmr_res.stop_reason}")  # EN: Print LSMR stop info.
    print(f"||Ax-b||={lsmr_res.rnorm:.3e}, ||A^T r||={lsmr_res.arnorm:.3e}, ||x||={lsmr_res.xnorm:.3e}")  # EN: Print LSMR summary.

    # EN: Reference comparison (small dense only): NumPy least squares.  # EN: Explain reference.
    x_ref, *_ = np.linalg.lstsq(A, b, rcond=None)  # EN: Compute stable least-squares solution.
    r_ref = A @ x_ref - b  # EN: Residual for reference solution.
    ar_ref = A.T @ r_ref  # EN: Normal residual for reference solution.

    print_separator("Reference: numpy.linalg.lstsq")  # EN: Announce reference section.
    print(f"||Ax-b||={l2_norm(r_ref):.3e}, ||A^T r||={l2_norm(ar_ref):.3e}, ||x||={l2_norm(x_ref):.3e}")  # EN: Print reference norms.
    print(f"rel_err(lsqr, ref) = {l2_norm(lsqr_res.x_hat - x_ref)/max(l2_norm(x_ref), EPS):.3e}")  # EN: Compare LSQR x to reference.
    print(f"rel_err(lsmr, ref) = {l2_norm(lsmr_res.x_hat - x_ref)/max(l2_norm(x_ref), EPS):.3e}")  # EN: Compare LSMR x to reference.

    # EN: Show a few normal-residual checkpoints to observe convergence behavior.  # EN: Explain checkpoint printing.
    print_separator("Normal-residual checkpoints (||A^T r||)")  # EN: Announce checkpoints.
    checkpoints = [0, 1, 2, 5, 10, 20, 50, 100]  # EN: Chosen checkpoint indices.
    for c in checkpoints:  # EN: Iterate checkpoint list.
        lsqr_val = lsqr_res.history_arnorm[c] if c < lsqr_res.history_arnorm.size else None  # EN: LSQR value if available.
        lsmr_val = lsmr_res.history_arnorm[c] if c < lsmr_res.history_arnorm.size else None  # EN: LSMR value if available.
        lsqr_str = "n/a" if lsqr_val is None else f"{float(lsqr_val):.3e}"  # EN: Format LSQR value.
        lsmr_str = "n/a" if lsmr_val is None else f"{float(lsmr_val):.3e}"  # EN: Format LSMR value.
        print(f"iter {c:3d}: LSQR {lsqr_str:>11} | LSMR {lsmr_str:>11}")  # EN: Print checkpoint row.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Run the demo.
