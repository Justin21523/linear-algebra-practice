"""  # EN: Start module docstring.
Matrix-free damped LSMR on a sparse (CSR) matrix (NumPy-only).  # EN: Summarize this script in one line.

We solve Ridge / damped least squares:  # EN: State the optimization problem.
  min_x ||A x - b||_2^2 + damp^2 ||x||_2^2,  damp >= 0.  # EN: Define the objective and damping parameter.

Key teaching points:  # EN: Describe what this unit demonstrates.
  1) Sparse + matrix-free: the solver only needs matvecs (A@x, A^T@y) and never forms A^T A.  # EN: Emphasize large-scale pattern.
  2) Right preconditioning via column scaling: x = D^{-1} y with D_j = sqrt(||A[:,j]||^2 + damp^2).  # EN: Explain preconditioner used.
  3) Continuation / warm-start along a damp path: reuse the previous solution to reduce iterations.  # EN: Explain practical speedup.

This is a small NumPy teaching demo (not an optimized production sparse solver).  # EN: Set expectations about performance.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured records.
from time import perf_counter  # EN: Use perf_counter for wall-clock timing.
from typing import Callable, Literal  # EN: Use typing helpers for clearer interfaces.

import numpy as np  # EN: Import NumPy for numerical computing.


EPS = 1e-12  # EN: Small epsilon for safe divisions and numeric guards.
SEED = 0  # EN: RNG seed for reproducible experiments.
PRINT_PRECISION = 6  # EN: Console float precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias for a matrix-vector product function.
PrecondKind = Literal["none", "col"]  # EN: Preconditioners supported in this sparse unit.


@dataclass(frozen=True)  # EN: Immutable representation of a sparse matrix in CSR format.
class CSRMatrix:  # EN: Store CSR arrays needed for matvec and transpose-matvec.
    data: np.ndarray  # EN: Nonzero values (length nnz).
    indices: np.ndarray  # EN: Column indices for each nonzero (length nnz).
    indptr: np.ndarray  # EN: Row pointer array (length m+1).
    shape: tuple[int, int]  # EN: Matrix shape (m, n).


@dataclass(frozen=True)  # EN: Immutable record for one solver run.
class SolveReport:  # EN: Store diagnostics and timing for one (damp, precond) run.
    precond: str  # EN: Preconditioner label.
    damp: float  # EN: Damping parameter used.
    n_iters: int  # EN: Iterations performed by the iterative solver.
    stop_reason: str  # EN: Human-readable termination reason.
    build_seconds: float  # EN: Time spent building preconditioner.
    solve_seconds: float  # EN: Time spent in the iterative solve.
    rnorm_data: float  # EN: ||Ax-b||_2 (data residual).
    grad_norm: float  # EN: ||A^T(Ax-b)+damp^2 x||_2 (ridge optimality residual).
    xnorm: float  # EN: ||x||_2 (solution norm).
    x_hat: np.ndarray  # EN: Estimated solution vector (stored for warm-start paths).


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


def generate_random_sparse_csr(  # EN: Generate a random sparse matrix with fixed nnz per row (CSR).
    rng: np.random.Generator,  # EN: RNG for reproducibility.
    m: int,  # EN: Number of rows.
    n: int,  # EN: Number of columns.
    nnz_per_row: int,  # EN: Nonzeros per row (must be <= n).
) -> CSRMatrix:  # EN: Return CSRMatrix with shape (m, n).
    if m <= 0 or n <= 0:  # EN: Validate dimensions.
        raise ValueError("m and n must be positive")  # EN: Reject invalid sizes.
    if nnz_per_row <= 0 or nnz_per_row > n:  # EN: Validate sparsity level.
        raise ValueError("nnz_per_row must be in [1, n]")  # EN: Reject invalid nnz_per_row.

    nnz = int(m * nnz_per_row)  # EN: Total number of nonzeros.
    data = np.empty((nnz,), dtype=float)  # EN: Allocate value array.
    indices = np.empty((nnz,), dtype=int)  # EN: Allocate column index array.
    indptr = (np.arange(m + 1, dtype=int) * nnz_per_row).astype(int)  # EN: Fixed-stride row pointers.

    for i in range(m):  # EN: Fill each row independently.
        start = int(indptr[i])  # EN: Row start pointer.
        end = int(indptr[i + 1])  # EN: Row end pointer.
        cols = rng.choice(n, size=nnz_per_row, replace=False)  # EN: Choose unique column indices.
        cols = np.sort(cols.astype(int))  # EN: Sort indices (not required, but makes debugging easier).
        vals = rng.standard_normal(nnz_per_row).astype(float)  # EN: Draw random values for the row.
        indices[start:end] = cols  # EN: Store column indices.
        data[start:end] = vals  # EN: Store values.

    return CSRMatrix(data=data, indices=indices, indptr=indptr, shape=(int(m), int(n)))  # EN: Return CSR container.


def csr_matvec(A: CSRMatrix, x: np.ndarray) -> np.ndarray:  # EN: Compute y = A x for CSR matrix A.
    m, n = A.shape  # EN: Extract matrix dimensions.
    x1 = np.asarray(x, dtype=float).reshape(-1)  # EN: Convert x to 1D float array.
    if x1.size != n:  # EN: Validate x length.
        raise ValueError("x has incompatible dimension")  # EN: Reject dimension mismatch.

    y = np.zeros((m,), dtype=float)  # EN: Allocate output vector.
    for i in range(m):  # EN: Loop rows.
        start = int(A.indptr[i])  # EN: Row start pointer.
        end = int(A.indptr[i + 1])  # EN: Row end pointer.
        cols = A.indices[start:end]  # EN: Column indices for this row.
        vals = A.data[start:end]  # EN: Nonzero values for this row.
        y[i] = float(np.dot(vals, x1[cols]))  # EN: Row dot-product against selected x entries.
    return y  # EN: Return y = A x.


def csr_rmatvec(A: CSRMatrix, y: np.ndarray) -> np.ndarray:  # EN: Compute x = A^T y for CSR matrix A.
    m, n = A.shape  # EN: Extract dimensions.
    y1 = np.asarray(y, dtype=float).reshape(-1)  # EN: Convert y to 1D float array.
    if y1.size != m:  # EN: Validate y length.
        raise ValueError("y has incompatible dimension")  # EN: Reject dimension mismatch.

    x = np.zeros((n,), dtype=float)  # EN: Allocate output x in R^n.
    for i in range(m):  # EN: Loop rows of A (entries contribute to columns of A^T).
        start = int(A.indptr[i])  # EN: Row start pointer.
        end = int(A.indptr[i + 1])  # EN: Row end pointer.
        cols = A.indices[start:end]  # EN: Column indices for this row.
        vals = A.data[start:end]  # EN: Values for this row.
        x[cols] += vals * y1[i]  # EN: Accumulate contributions into x at the referenced columns.
    return x  # EN: Return x = A^T y.


def csr_column_norms_sq(A: CSRMatrix) -> np.ndarray:  # EN: Compute column squared norms ||A[:,j]||_2^2 for a CSR matrix.
    _, n = A.shape  # EN: Extract n for allocating the output.
    col_sq = np.zeros((n,), dtype=float)  # EN: Allocate accumulator for column squared norms.
    np.add.at(col_sq, A.indices, A.data * A.data)  # EN: Add each nonzero's square into its column bin.
    return col_sq  # EN: Return column squared norms.


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


def column_scaling_D_aug_from_col_sq(col_sq: np.ndarray, damp: float) -> np.ndarray:  # EN: Build D_j = sqrt(||A[:,j]||^2 + damp^2).
    return np.sqrt(np.maximum(col_sq + (float(damp) * float(damp)), EPS)).astype(float)  # EN: Compute and return D.


def build_preconditioner(  # EN: Build right preconditioner for the sparse operator (none or column scaling).
    kind: PrecondKind,  # EN: Preconditioner kind.
    col_sq: np.ndarray,  # EN: Column squared norms of A (precomputed once).
    damp: float,  # EN: Damping parameter (affects augmented column norms).
) -> tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray], float]:  # EN: Return (label, Minv, Minv_T, build_seconds).
    n = int(col_sq.size)  # EN: Feature dimension.
    _ = n  # EN: Keep n visible for readability (used in closures).

    if kind == "none":  # EN: No preconditioning.
        label = "none"  # EN: Human label.

        def apply_Minv(y: np.ndarray) -> np.ndarray:  # EN: Identity M^{-1}.
            return y  # EN: Return y unchanged.

        def apply_Minv_T(z: np.ndarray) -> np.ndarray:  # EN: Identity M^{-T}.
            return z  # EN: Return z unchanged.

        return label, apply_Minv, apply_Minv_T, 0.0  # EN: Return with zero build time.

    if kind == "col":  # EN: Column scaling preconditioner based on augmented column norms.
        t0 = perf_counter()  # EN: Start build timer.
        D = column_scaling_D_aug_from_col_sq(col_sq=col_sq, damp=float(damp))  # EN: Build diagonal scaling D.
        build_seconds = float(perf_counter() - t0)  # EN: Stop build timer.
        label = "col-scaling"  # EN: Human label.

        def apply_Minv(y: np.ndarray, D: np.ndarray = D) -> np.ndarray:  # EN: Apply D^{-1} (right preconditioning).
            return y / D  # EN: Elementwise divide by diagonal.

        def apply_Minv_T(z: np.ndarray, D: np.ndarray = D) -> np.ndarray:  # EN: Apply D^{-T} = D^{-1} (diagonal).
            return z / D  # EN: Elementwise divide.

        return label, apply_Minv, apply_Minv_T, build_seconds  # EN: Return preconditioner and timing.

    raise ValueError("Unknown preconditioner kind")  # EN: Guard against invalid kind.


def lsmr_damped_minres_teaching_operator(  # EN: Teaching LSMR for ridge using only matvecs (MINRES on normal equations).
    matvec_A: Matvec,  # EN: Function computing A x.
    matvec_AT: Matvec,  # EN: Function computing A^T y.
    m: int,  # EN: Row dimension of A.
    n: int,  # EN: Column dimension of A.
    b: np.ndarray,  # EN: Top RHS (length m).
    damp: float,  # EN: Damping parameter.
    apply_Minv: Callable[[np.ndarray], np.ndarray],  # EN: Function applying M^{-1} (maps y -> x).
    apply_Minv_T: Callable[[np.ndarray], np.ndarray],  # EN: Function applying M^{-T}.
    max_iters: int,  # EN: Iteration cap.
    atol: float,  # EN: Absolute tolerance for stopping.
    btol: float,  # EN: Relative tolerance for stopping.
    anorm_est: float,  # EN: Estimated ||A_aug|| (we use a Frobenius upper bound in this unit).
    b_aug_bottom: np.ndarray | None = None,  # EN: Optional bottom RHS for augmented system (length n); default is zeros.
) -> tuple[np.ndarray, int, str, float, float, float, float]:  # EN: Return (x, iters, reason, rnorm_data, grad_norm, xnorm, rnorm_aug).
    if b_aug_bottom is None:  # EN: Default bottom RHS is zero (standard ridge).
        b_bottom = np.zeros((n,), dtype=float)  # EN: Use zeros for the regularization rows.
    else:  # EN: Accept a non-zero bottom RHS (used for warm-start correction solves).
        b_bottom = np.asarray(b_aug_bottom, dtype=float).reshape(-1)  # EN: Convert to 1D float array.
        if b_bottom.size != n:  # EN: Validate length.
            raise ValueError("b_aug_bottom must have length n")  # EN: Reject mismatched bottom RHS.

    b_aug = np.concatenate([np.asarray(b, dtype=float).reshape(-1), b_bottom])  # EN: Build augmented RHS [b; b_bottom].
    if b_aug.size != (m + n):  # EN: Validate augmented RHS length.
        raise ValueError("b must have length m")  # EN: Reject invalid b length.
    bnorm = l2_norm(b_aug)  # EN: Use ||b_aug|| in residual-bound stopping tests.
    gnorm_x = l2_norm(matvec_AT(b_aug[:m]) + (float(damp) * b_bottom))  # EN: Scale for relative-gradient stopping tests.

    # EN: Define augmented operator B = [A; damp I] M^{-1} via matvecs.  # EN: Explain operator design.
    def matvec_B(y: np.ndarray) -> np.ndarray:  # EN: Compute B y = [A x; damp x] where x = M^{-1} y.
        x = apply_Minv(y)  # EN: Map y -> x.
        top = matvec_A(x)  # EN: Compute A x (top block).
        bottom = float(damp) * x  # EN: Compute damp x (bottom block).
        return np.concatenate([top, bottom])  # EN: Return concatenated vector of length (m+n).

    def matvec_BT(u_aug: np.ndarray) -> np.ndarray:  # EN: Compute B^T u_aug = M^{-T}(A^T u_top + damp u_bottom).
        u_top = u_aug[:m]  # EN: Extract top part.
        u_bottom = u_aug[m:]  # EN: Extract bottom part.
        z = matvec_AT(u_top) + (float(damp) * u_bottom)  # EN: Compute A_aug^T u_aug in x-space.
        return apply_Minv_T(z)  # EN: Apply M^{-T} to map to y-space.

    # EN: Initialize Golub–Kahan bidiagonalization with u1 = b_aug / ||b_aug||.  # EN: Explain initialization.
    u = b_aug.copy()  # EN: Start u from b_aug.
    beta1 = l2_norm(u)  # EN: beta1 = ||b_aug||.
    if beta1 < EPS:  # EN: Trivial b_aug=0 case.
        x0 = np.zeros((n,), dtype=float)  # EN: Solution is x=0.
        r0_top = matvec_A(x0) - b_aug[:m]  # EN: Top residual is -b.
        r0_bottom = (float(damp) * x0) - b_bottom  # EN: Bottom residual is -b_bottom.
        grad0 = matvec_AT(r0_top) + (float(damp) * r0_bottom)  # EN: Gradient for augmented objective.
        rnorm_data0 = l2_norm(r0_top)  # EN: ||Ax-b||.
        rnorm_bottom0 = l2_norm(r0_bottom)  # EN: ||damp x - b_bottom||.
        rnorm_aug0 = float(np.sqrt(rnorm_data0 * rnorm_data0 + rnorm_bottom0 * rnorm_bottom0))  # EN: ||[r_top;r_bottom]||.
        return x0, 0, "b is zero (trivial)", rnorm_data0, l2_norm(grad0), 0.0, rnorm_aug0  # EN: Return.
    u = u / beta1  # EN: Normalize u1.

    v = matvec_BT(u)  # EN: v1 = B^T u1.
    alpha1 = l2_norm(v)  # EN: alpha1 = ||v1||.
    if alpha1 < EPS:  # EN: Degenerate case B^T b_aug = 0.
        x0 = np.zeros((n,), dtype=float)  # EN: Use x=0.
        r0_top = matvec_A(x0) - b_aug[:m]  # EN: Top residual.
        r0_bottom = (float(damp) * x0) - b_bottom  # EN: Bottom residual.
        grad0 = matvec_AT(r0_top) + (float(damp) * r0_bottom)  # EN: Gradient for augmented objective.
        rnorm_data0 = l2_norm(r0_top)  # EN: ||Ax-b||.
        rnorm_bottom0 = l2_norm(r0_bottom)  # EN: ||damp x - b_bottom||.
        rnorm_aug0 = float(np.sqrt(rnorm_data0 * rnorm_data0 + rnorm_bottom0 * rnorm_bottom0))  # EN: ||[r_top;r_bottom]||.
        return x0, 0, "B^T b is zero (degenerate)", rnorm_data0, l2_norm(grad0), 0.0, rnorm_aug0  # EN: Return.
    v = v / alpha1  # EN: Normalize v1.

    # EN: Store v-basis vectors in y-space so we can reconstruct y_k and map to x.  # EN: Explain storage.
    V_basis_y = np.zeros((n, min(max_iters, n) + 1), dtype=float)  # EN: Store v vectors as columns.
    V_basis_y[:, 0] = v  # EN: Store v1.

    alphas: list[float] = [float(alpha1)]  # EN: Store alpha_1.
    betas: list[float] = [float(beta1)]  # EN: Store beta_1 (append beta_{k+1} each iteration).
    gnorm_y = float(alpha1 * beta1)  # EN: ||B^T b_aug|| in y-space coordinates.

    x_hat = np.zeros((n,), dtype=float)  # EN: Initialize x estimate (mapped from y).
    stop_reason = "max_iters reached"  # EN: Default stop reason.
    n_done = 0  # EN: Iterations completed.

    for k in range(1, min(max_iters, n) + 1):  # EN: Expand Krylov subspace up to the cap.
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

        r_top = matvec_A(x_hat) - b_aug[:m]  # EN: Top residual.
        r_bottom = (float(damp) * x_hat) - b_bottom  # EN: Bottom residual.
        grad = matvec_AT(r_top) + (float(damp) * r_bottom)  # EN: Gradient A^T r_top + damp r_bottom.

        xnorm = l2_norm(x_hat)  # EN: ||x||.
        rnorm_data = l2_norm(r_top)  # EN: ||Ax-b||.
        rnorm_bottom = l2_norm(r_bottom)  # EN: ||damp x - b_bottom||.
        rnorm_aug = float(np.sqrt(rnorm_data * rnorm_data + rnorm_bottom * rnorm_bottom))  # EN: ||[r_top;r_bottom]||.
        grad_norm = l2_norm(grad)  # EN: ||A^T r + damp^2 x - damp b_bottom|| (optimality residual).

        n_done = k  # EN: Update completed iteration count.

        r_bound = (btol * bnorm) + (atol * float(anorm_est) * xnorm)  # EN: Mixed absolute/relative residual bound.
        if rnorm_aug <= r_bound:  # EN: Stop when residual is small enough.
            stop_reason = "residual bound satisfied"  # EN: Record stop reason.
            break  # EN: Exit loop.

        if grad_norm <= atol * float(anorm_est) * max(rnorm_aug, EPS):  # EN: Stop when gradient is small relative to residual.
            stop_reason = "normal residual bound satisfied"  # EN: Record stop reason.
            break  # EN: Exit loop.

        if gnorm_x >= EPS and grad_norm <= btol * gnorm_x:  # EN: Stop when gradient is small relative to initial scale.
            stop_reason = "relative normal residual satisfied"  # EN: Record stop reason.
            break  # EN: Exit loop.

        if beta_next < EPS and alpha_next < EPS:  # EN: Breakdown: cannot expand Krylov basis further.
            stop_reason = "breakdown (beta and alpha near zero)"  # EN: Record breakdown reason.
            break  # EN: Exit loop.

        u = u_next  # EN: Advance u.
        v = v_next  # EN: Advance v.

    r_final_top = matvec_A(x_hat) - b_aug[:m]  # EN: Final top residual.
    r_final_bottom = (float(damp) * x_hat) - b_bottom  # EN: Final bottom residual.
    grad_final = matvec_AT(r_final_top) + (float(damp) * r_final_bottom)  # EN: Final gradient for augmented objective.

    rnorm_data_final = l2_norm(r_final_top)  # EN: Final ||Ax-b||.
    rnorm_bottom_final = l2_norm(r_final_bottom)  # EN: Final ||damp x - b_bottom||.
    rnorm_aug_final = float(np.sqrt(rnorm_data_final * rnorm_data_final + rnorm_bottom_final * rnorm_bottom_final))  # EN: Final ||[r_top;r_bottom]||.
    grad_norm_final = l2_norm(grad_final)  # EN: Final ||grad||.
    xnorm_final = l2_norm(x_hat)  # EN: Final ||x||.

    return x_hat, int(n_done), stop_reason, float(rnorm_data_final), float(grad_norm_final), float(xnorm_final), float(rnorm_aug_final)  # EN: Return final values.


def solve_one_sparse(  # EN: Solve one sparse ridge problem (optionally warm-started) and return a SolveReport.
    A: CSRMatrix,  # EN: Sparse design matrix in CSR.
    matvec_A: Matvec,  # EN: Matvec for A.
    matvec_AT: Matvec,  # EN: Matvec for A^T.
    col_sq: np.ndarray,  # EN: Column squared norms of A (precomputed once).
    fro_sq: float,  # EN: Frobenius norm squared of A (precomputed once).
    b: np.ndarray,  # EN: RHS targets (length m).
    damp: float,  # EN: Damping parameter.
    precond_kind: PrecondKind,  # EN: Preconditioner kind.
    max_iters: int,  # EN: Iteration cap.
    atol: float,  # EN: Absolute tolerance.
    btol: float,  # EN: Relative tolerance.
    x_init: np.ndarray | None = None,  # EN: Optional warm-start initial guess in x-space.
) -> SolveReport:  # EN: Return structured diagnostics for this run.
    m, n = A.shape  # EN: Extract dimensions.
    damp_f = float(damp)  # EN: Ensure damp is a float.

    # EN: Use a deterministic upper bound for ||A_aug||_2 via Frobenius: ||A_aug||_2 <= ||A_aug||_F.  # EN: Explain estimate choice.
    anorm_est = float(np.sqrt(float(fro_sq) + (damp_f * damp_f) * float(n)))  # EN: Compute ||[A;damp I]||_F.

    label, apply_Minv, apply_Minv_T, build_seconds = build_preconditioner(kind=precond_kind, col_sq=col_sq, damp=damp_f)  # EN: Build preconditioner.

    # EN: Warm-start via a correction solve with RHS [b-Ax0; -damp x0].  # EN: Explain technique.
    if x_init is None:  # EN: No warm-start.
        x0 = np.zeros((n,), dtype=float)  # EN: Use x0=0.
        b_top = np.asarray(b, dtype=float).reshape(-1)  # EN: Top RHS is original b.
        b_bottom = None  # EN: Bottom RHS is zero by default.
    else:  # EN: Warm-start enabled.
        x0 = np.asarray(x_init, dtype=float).reshape(-1)  # EN: Convert x_init to 1D float array.
        if x0.size != n:  # EN: Validate x_init length.
            raise ValueError("x_init must have length n")  # EN: Reject invalid warm-start.
        b_top = np.asarray(b, dtype=float).reshape(-1) - matvec_A(x0)  # EN: Top RHS becomes residual b - A x0.
        b_bottom = -damp_f * x0  # EN: Bottom RHS becomes -damp x0.

    t0 = perf_counter()  # EN: Start solve timer.
    x_delta, iters, stop_reason, _, _, _, _ = lsmr_damped_minres_teaching_operator(  # EN: Solve for delta (or x when x0=0).
        matvec_A=matvec_A,  # EN: Provide matvec A.
        matvec_AT=matvec_AT,  # EN: Provide matvec A^T.
        m=int(m),  # EN: Provide m.
        n=int(n),  # EN: Provide n.
        b=b_top,  # EN: Provide top RHS.
        damp=damp_f,  # EN: Provide damp.
        apply_Minv=apply_Minv,  # EN: Provide M^{-1}.
        apply_Minv_T=apply_Minv_T,  # EN: Provide M^{-T}.
        max_iters=max_iters,  # EN: Provide cap.
        atol=atol,  # EN: Provide atol.
        btol=btol,  # EN: Provide btol.
        anorm_est=anorm_est,  # EN: Provide norm estimate.
        b_aug_bottom=b_bottom,  # EN: Provide bottom RHS (or None).
    )  # EN: End solver call.
    solve_seconds = float(perf_counter() - t0)  # EN: Stop solve timer.

    x_hat = (x0 + x_delta).astype(float)  # EN: Combine x0 and delta.
    r_final = matvec_A(x_hat) - np.asarray(b, dtype=float).reshape(-1)  # EN: Final data residual in original coordinates.
    grad_final = matvec_AT(r_final) + (damp_f * damp_f) * x_hat  # EN: Final ridge gradient.

    return SolveReport(  # EN: Package report.
        precond=str(label),  # EN: Label.
        damp=float(damp_f),  # EN: Damp.
        n_iters=int(iters),  # EN: Iterations.
        stop_reason=str(stop_reason),  # EN: Stop reason.
        build_seconds=float(build_seconds),  # EN: Build time.
        solve_seconds=float(solve_seconds),  # EN: Solve time.
        rnorm_data=float(l2_norm(r_final)),  # EN: ||Ax-b||.
        grad_norm=float(l2_norm(grad_final)),  # EN: ||A^T r + damp^2 x||.
        xnorm=float(l2_norm(x_hat)),  # EN: ||x||.
        x_hat=x_hat,  # EN: Store solution for warm-start.
    )  # EN: End report.


def print_solver_table(reports: list[SolveReport]) -> None:  # EN: Print a compact solver comparison table.
    if not reports:  # EN: Handle empty list.
        print("(no reports)")  # EN: Print message.
        return  # EN: Exit.

    header = "damp      | precond      | iters | build_s | solve_s | ||Ax-b||   | ||grad||   | ||x||"  # EN: Header columns.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Divider.

    reps = sorted(reports, key=lambda r: (r.damp, r.precond))  # EN: Sort by damp then precond.
    for r in reps:  # EN: Print each row.
        damp_key = f"{r.damp:.0e}" if r.damp != 0.0 else "0"  # EN: Compact damp formatting.
        print(  # EN: Print formatted row.
            f"{damp_key:9} | "  # EN: Damp column.
            f"{r.precond:11} | "  # EN: Preconditioner label.
            f"{r.n_iters:5d} | "  # EN: Iterations.
            f"{r.build_seconds:7.3f} | "  # EN: Build time.
            f"{r.solve_seconds:7.3f} | "  # EN: Solve time.
            f"{r.rnorm_data:9.2e} | "  # EN: Data residual norm.
            f"{r.grad_norm:9.2e} | "  # EN: Gradient norm.
            f"{r.xnorm:9.2e}"  # EN: x norm.
        )  # EN: End print.


def main() -> None:  # EN: Run a sparse matrix-free ridge demo with preconditioning and warm-start.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    # EN: Build a "recommendation-like" sparse design matrix: many rows, few nonzeros per row.  # EN: Explain dataset design.
    m = 2000  # EN: Number of samples (rows).
    n = 200  # EN: Number of features (columns).
    nnz_per_row = 10  # EN: Sparsity level per row.
    noise_std = 0.05  # EN: Noise level for b.

    A = generate_random_sparse_csr(rng=rng, m=m, n=n, nnz_per_row=nnz_per_row)  # EN: Generate sparse CSR matrix.
    nnz = int(A.data.size)  # EN: Total number of nonzeros.
    density = float(nnz) / float(m * n)  # EN: Sparsity density.

    col_sq = csr_column_norms_sq(A)  # EN: Precompute column squared norms (reused across damps).
    fro_sq = float(np.sum(A.data * A.data))  # EN: Precompute ||A||_F^2 for cheap norm bounds.

    # EN: Define matvecs so the solver never needs the dense matrix explicitly.  # EN: Explain matrix-free interface.
    def matvec_A(x: np.ndarray) -> np.ndarray:  # EN: CSR matvec for A.
        return csr_matvec(A=A, x=x)  # EN: Compute A x.

    def matvec_AT(y: np.ndarray) -> np.ndarray:  # EN: CSR transpose matvec for A^T.
        return csr_rmatvec(A=A, y=y)  # EN: Compute A^T y.

    # EN: Create a sparse-ish ground-truth coefficient vector and synthetic targets b.  # EN: Explain target construction.
    x_true = np.zeros((n,), dtype=float)  # EN: Initialize x_true.
    support = rng.choice(n, size=12, replace=False)  # EN: Choose a small support set.
    x_true[support] = rng.standard_normal(support.size)  # EN: Fill support with random values.
    b_clean = matvec_A(x_true)  # EN: Compute noiseless targets.
    b = b_clean + noise_std * rng.standard_normal((m,)).astype(float)  # EN: Add Gaussian noise.

    print_separator("Sparse / matrix-free dataset summary")  # EN: Announce dataset summary section.
    print(f"m={m}, n={n}, nnz={nnz}, nnz_per_row={nnz_per_row}, density={density:.3e}")  # EN: Print sparsity stats.
    print(f"||A||_F={np.sqrt(fro_sq):.3e}, noise_std={noise_std}")  # EN: Print Frobenius norm and noise.

    # EN: Solver settings: keep the cap moderate (teaching demo).  # EN: Explain settings.
    max_iters = 80  # EN: Iteration cap.
    atol = 1e-10  # EN: Absolute tolerance.
    btol = 1e-10  # EN: Relative tolerance.

    # EN: Compare cold-start solves for a few damp values.  # EN: Explain experiment.
    demo_damps = [1e-2, 1e-1, 1.0]  # EN: Damp values to compare.
    preconds: list[PrecondKind] = ["none", "col"]  # EN: Preconditioner list.

    print_separator("Cold-start comparison (each damp starts from x=0)")  # EN: Announce cold-start comparison.
    reports: list[SolveReport] = []  # EN: Collect reports.
    for d in demo_damps:  # EN: Loop damp values.
        for pk in preconds:  # EN: Loop preconditioners.
            reports.append(  # EN: Append report.
                solve_one_sparse(  # EN: Solve one configuration.
                    A=A,  # EN: Provide A.
                    matvec_A=matvec_A,  # EN: Provide matvec A.
                    matvec_AT=matvec_AT,  # EN: Provide matvec A^T.
                    col_sq=col_sq,  # EN: Provide column norms.
                    fro_sq=fro_sq,  # EN: Provide Frobenius squared.
                    b=b,  # EN: Provide b.
                    damp=float(d),  # EN: Provide damp.
                    precond_kind=pk,  # EN: Provide preconditioner.
                    max_iters=max_iters,  # EN: Cap.
                    atol=atol,  # EN: atol.
                    btol=btol,  # EN: btol.
                    x_init=None,  # EN: Cold-start.
                )  # EN: End solve.
            )  # EN: End append.
    print_solver_table(reports)  # EN: Print table.

    # EN: Demonstrate continuation / warm-start along a damp path (from large to small).  # EN: Explain continuation experiment.
    damp_path = [10.0, 1.0, 1e-1, 1e-2]  # EN: A decreasing damp path (often more stable for continuation).
    print_separator("Continuation / warm-start along a damp path (per preconditioner)")  # EN: Announce continuation section.

    for pk in preconds:  # EN: Loop preconditioners.
        # EN: Cold-start total cost for this path.  # EN: Explain baseline within path.
        cold_total_iters = 0  # EN: Accumulate cold-start iterations.
        cold_total_solve = 0.0  # EN: Accumulate cold-start solve time.
        for d in damp_path:  # EN: Loop damps.
            rep = solve_one_sparse(  # EN: Cold-start solve.
                A=A,  # EN: A.
                matvec_A=matvec_A,  # EN: matvec A.
                matvec_AT=matvec_AT,  # EN: matvec A^T.
                col_sq=col_sq,  # EN: col norms.
                fro_sq=fro_sq,  # EN: Frobenius squared.
                b=b,  # EN: b.
                damp=float(d),  # EN: damp.
                precond_kind=pk,  # EN: preconditioner.
                max_iters=max_iters,  # EN: cap.
                atol=atol,  # EN: atol.
                btol=btol,  # EN: btol.
                x_init=None,  # EN: cold-start.
            )  # EN: End solve.
            cold_total_iters += int(rep.n_iters)  # EN: Add iterations.
            cold_total_solve += float(rep.solve_seconds)  # EN: Add solve time.

        # EN: Warm-start total cost for this path.  # EN: Explain warm-start.
        warm_total_iters = 0  # EN: Accumulate warm-start iterations.
        warm_total_solve = 0.0  # EN: Accumulate warm-start solve time.
        x_prev: np.ndarray | None = None  # EN: Store previous solution for continuation.
        for i, d in enumerate(damp_path):  # EN: Loop damps in path order.
            rep = solve_one_sparse(  # EN: Warm-start solve (x_init from previous damp).
                A=A,  # EN: A.
                matvec_A=matvec_A,  # EN: matvec A.
                matvec_AT=matvec_AT,  # EN: matvec A^T.
                col_sq=col_sq,  # EN: col norms.
                fro_sq=fro_sq,  # EN: Frobenius squared.
                b=b,  # EN: b.
                damp=float(d),  # EN: damp.
                precond_kind=pk,  # EN: preconditioner.
                max_iters=max_iters,  # EN: cap.
                atol=atol,  # EN: atol.
                btol=btol,  # EN: btol.
                x_init=x_prev if i > 0 else None,  # EN: continuation (skip first).
            )  # EN: End solve.
            x_prev = rep.x_hat  # EN: Update continuation state.
            warm_total_iters += int(rep.n_iters)  # EN: Add iterations.
            warm_total_solve += float(rep.solve_seconds)  # EN: Add solve time.

        speedup = float(cold_total_solve / max(warm_total_solve, EPS))  # EN: Compute solve-time speedup factor.
        print(  # EN: Print one-line comparison for this preconditioner.
            f"{pk:10} cold: iters={cold_total_iters:4d}, solve_s={cold_total_solve:.3f} | warm: iters={warm_total_iters:4d}, solve_s={warm_total_solve:.3f} | speedup={speedup:.2f}x"  # EN: Summary line.
        )  # EN: End print.


if __name__ == "__main__":  # EN: Standard Python entrypoint guard.
    main()  # EN: Execute main.
