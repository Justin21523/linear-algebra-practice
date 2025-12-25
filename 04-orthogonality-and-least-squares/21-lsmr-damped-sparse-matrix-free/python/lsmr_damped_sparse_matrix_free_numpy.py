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
PrecondKind = Literal["none", "col", "randqr"]  # EN: Preconditioners supported in this sparse unit.
RandQRPolicy = Literal["rebuild", "shared_sketch", "fixed_R"]  # EN: Policies for building/reusing sparse rand-QR preconditioners along a damp sweep.


@dataclass(frozen=True)  # EN: Immutable representation of a sparse matrix in CSR format.
class CSRMatrix:  # EN: Store CSR arrays needed for matvec and transpose-matvec.
    data: np.ndarray  # EN: Nonzero values (length nnz).
    indices: np.ndarray  # EN: Column indices for each nonzero (length nnz).
    indptr: np.ndarray  # EN: Row pointer array (length m+1).
    shape: tuple[int, int]  # EN: Matrix shape (m, n).


@dataclass(frozen=True)  # EN: Immutable "view" of a CSR matrix restricted to a row subset (no CSR copying).
class CSRRowSubset:  # EN: Represent A[row_ids, :] as an operator using only row indices and CSR pointers.
    A: CSRMatrix  # EN: Reference to the full CSR matrix (data/indices/indptr are not copied).
    row_ids: np.ndarray  # EN: Selected full-matrix row indices in the desired order (length m_subset).
    starts: np.ndarray  # EN: Cached row start pointers indptr[row_ids] (length m_subset).
    ends: np.ndarray  # EN: Cached row end pointers indptr[row_ids+1] (length m_subset).
    shape: tuple[int, int]  # EN: Shape of the row-subset operator (m_subset, n).


@dataclass(frozen=True)  # EN: Immutable container for a reusable CountSketch of the augmented matrix [A; damp I].
class CountSketchAug:  # EN: Store SA_top plus the identity-row hashing so SA_aug(damp) can be formed cheaply.
    SA_top: np.ndarray  # EN: Dense sketch of the data block, SA_top = S_top A (shape s×n).
    h_bottom: np.ndarray  # EN: Hash bucket for each identity row (length n).
    sign_bottom: np.ndarray  # EN: Random sign for each identity row (length n).
    scale: float  # EN: Sketch scaling factor (typically 1/sqrt(s)).
    s: int  # EN: Sketch row count.


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


@dataclass(frozen=True)  # EN: Immutable record for one CV point (one damp value).
class CVPoint:  # EN: Store mean/std metrics across folds for one damp.
    key: str  # EN: Display label (e.g., "d=1e-2").
    damp: float  # EN: Numeric damp.
    train_mean: float  # EN: Mean training RMSE across folds.
    train_std: float  # EN: Std training RMSE across folds.
    val_mean: float  # EN: Mean validation RMSE across folds.
    val_std: float  # EN: Std validation RMSE across folds.
    x_norm_mean: float  # EN: Mean ||x|| across folds (shrinkage proxy).
    iters_mean: float  # EN: Mean iterations across folds (cost proxy).


@dataclass(frozen=True)  # EN: Immutable record for CV sweep totals (whole curve cost).
class CVTotals:  # EN: Store total cost for sweeping all damps for one preconditioner configuration.
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


def ascii_bar(value: float, vmin: float, vmax: float, width: int = 30) -> str:  # EN: Render an ASCII bar where lower values -> longer bars.
    if width <= 0:  # EN: Validate width.
        return ""  # EN: Return empty bar.
    if vmax <= vmin + EPS:  # EN: Handle near-constant series.
        return "#" * width  # EN: Return full bar.
    score = (vmax - value) / (vmax - vmin)  # EN: Map lower -> higher score in [0,1].
    score = float(np.clip(score, 0.0, 1.0))  # EN: Clamp to [0,1].
    n = int(round(score * width))  # EN: Convert score to character count.
    return "#" * n  # EN: Return bar string.


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


def csr_matvec(A: CSRMatrix, x: np.ndarray) -> np.ndarray:  # EN: Compute y = A x for CSR matrix A (vectorized NumPy implementation).
    m, n = A.shape  # EN: Extract matrix dimensions.
    x1 = np.asarray(x, dtype=float).reshape(-1)  # EN: Convert x to 1D float array.
    if x1.size != n:  # EN: Validate x length.
        raise ValueError("x has incompatible dimension")  # EN: Reject dimension mismatch.
    if A.data.size == 0:  # EN: Handle empty matrix fast.
        return np.zeros((m,), dtype=float)  # EN: Return zeros.

    # EN: For CSR, nonzeros are stored row-by-row, so we can multiply per-nnz and then reduce within each row segment.  # EN: Explain vectorization.
    prod = A.data * x1[A.indices]  # EN: Multiply each nonzero by its corresponding x entry.
    y = np.add.reduceat(prod, A.indptr[:-1])  # EN: Sum products within each row segment.
    return y.astype(float)  # EN: Return y as float.


def csr_rmatvec(  # EN: Compute x = A^T y for CSR matrix A (vectorized with optional precomputed row indices).
    A: CSRMatrix,  # EN: Sparse matrix in CSR.
    y: np.ndarray,  # EN: Input vector in R^m.
    row_indices: np.ndarray | None = None,  # EN: Optional per-nnz row index array to avoid recomputing np.repeat each call.
) -> np.ndarray:  # EN: Return x in R^n.
    m, n = A.shape  # EN: Extract dimensions.
    y1 = np.asarray(y, dtype=float).reshape(-1)  # EN: Convert y to 1D float array.
    if y1.size != m:  # EN: Validate y length.
        raise ValueError("y has incompatible dimension")  # EN: Reject dimension mismatch.
    if A.data.size == 0:  # EN: Handle empty matrix fast.
        return np.zeros((n,), dtype=float)  # EN: Return zeros.

    if row_indices is None:  # EN: Build row index per nonzero when not provided.
        row_indices = np.repeat(np.arange(m, dtype=int), np.diff(A.indptr).astype(int))  # EN: Map each nnz to its row id.
    row_idx = np.asarray(row_indices, dtype=int).reshape(-1)  # EN: Ensure row_indices is a 1D int array.
    if row_idx.size != A.data.size:  # EN: Validate that row_indices matches nnz length.
        raise ValueError("row_indices length must match nnz")  # EN: Reject mismatched helper array.

    contrib = A.data * y1[row_idx]  # EN: Each nnz contributes data[p] * y[row(p)].
    x = np.zeros((n,), dtype=float)  # EN: Allocate output x.
    np.add.at(x, A.indices, contrib)  # EN: Accumulate contributions into columns (transpose matvec).
    return x.astype(float)  # EN: Return x as float.


def csr_column_norms_sq(A: CSRMatrix) -> np.ndarray:  # EN: Compute column squared norms ||A[:,j]||_2^2 for a CSR matrix.
    _, n = A.shape  # EN: Extract n for allocating the output.
    col_sq = np.zeros((n,), dtype=float)  # EN: Allocate accumulator for column squared norms.
    np.add.at(col_sq, A.indices, A.data * A.data)  # EN: Add each nonzero's square into its column bin.
    return col_sq  # EN: Return column squared norms.


def make_csr_row_subset(A: CSRMatrix, row_ids: np.ndarray) -> CSRRowSubset:  # EN: Create a CSRRowSubset view A[row_ids, :] without copying CSR data.
    m_full, n = A.shape  # EN: Extract full matrix dimensions.
    rows = np.asarray(row_ids, dtype=int).reshape(-1)  # EN: Convert row ids to a 1D int array.
    if rows.size == 0:  # EN: Handle empty selection.
        empty = np.array([], dtype=int)  # EN: Create an empty int array.
        return CSRRowSubset(A=A, row_ids=empty, starts=empty, ends=empty, shape=(0, int(n)))  # EN: Return an empty subset view.
    if np.any(rows < 0) or np.any(rows >= int(m_full)):  # EN: Validate bounds.
        raise ValueError("row_ids out of bounds")  # EN: Reject invalid row indices.
    starts = np.asarray(A.indptr[rows], dtype=int).reshape(-1)  # EN: Cache row start pointers for each selected row.
    ends = np.asarray(A.indptr[rows + 1], dtype=int).reshape(-1)  # EN: Cache row end pointers for each selected row.
    return CSRRowSubset(A=A, row_ids=rows, starts=starts, ends=ends, shape=(int(rows.size), int(n)))  # EN: Return subset view.


def csr_matvec_subset(subset: CSRRowSubset, x: np.ndarray) -> np.ndarray:  # EN: Compute y = A[row_ids,:] x using only CSR pointers and row indices.
    A = subset.A  # EN: Unpack the full CSR matrix reference.
    _, n = A.shape  # EN: Extract feature dimension.
    x1 = np.asarray(x, dtype=float).reshape(-1)  # EN: Convert x to 1D float array.
    if x1.size != int(n):  # EN: Validate x length.
        raise ValueError("x has incompatible dimension")  # EN: Reject dimension mismatch.
    m_subset = int(subset.shape[0])  # EN: Number of selected rows.
    y = np.zeros((m_subset,), dtype=float)  # EN: Allocate output vector for the selected rows.
    for k in range(m_subset):  # EN: Loop selected rows in the requested order.
        start = int(subset.starts[k])  # EN: Row start pointer in the full CSR arrays.
        end = int(subset.ends[k])  # EN: Row end pointer in the full CSR arrays.
        if end <= start:  # EN: Skip empty rows.
            continue  # EN: Move to next row.
        cols = A.indices[start:end]  # EN: Column indices for this row segment (view into full arrays).
        vals = A.data[start:end]  # EN: Nonzero values for this row segment (view into full arrays).
        y[k] = float(np.dot(vals, x1[cols]))  # EN: Dot product of the sparse row with x.
    return y.astype(float)  # EN: Return y as float.


def csr_rmatvec_subset(subset: CSRRowSubset, y: np.ndarray) -> np.ndarray:  # EN: Compute x = A[row_ids,:]^T y using only CSR pointers and row indices.
    A = subset.A  # EN: Unpack the full CSR matrix reference.
    m_subset, n = subset.shape  # EN: Extract subset dimensions.
    y1 = np.asarray(y, dtype=float).reshape(-1)  # EN: Convert y to 1D float array.
    if y1.size != int(m_subset):  # EN: Validate y length against subset row count.
        raise ValueError("y has incompatible dimension")  # EN: Reject dimension mismatch.
    x = np.zeros((int(n),), dtype=float)  # EN: Allocate output vector in R^n.
    for k in range(int(m_subset)):  # EN: Loop selected rows.
        weight = float(y1[k])  # EN: Scalar weight for this row in the transpose product.
        if weight == 0.0:  # EN: Skip work when y[k] is zero.
            continue  # EN: Move to next row.
        start = int(subset.starts[k])  # EN: Row start pointer.
        end = int(subset.ends[k])  # EN: Row end pointer.
        if end <= start:  # EN: Skip empty rows.
            continue  # EN: Move to next row.
        cols = A.indices[start:end]  # EN: Column indices for the row segment.
        vals = A.data[start:end]  # EN: Nonzero values for the row segment.
        np.add.at(x, cols, vals * weight)  # EN: Accumulate v_ij * y_k into x_j.
    return x.astype(float)  # EN: Return x as float.


def csr_column_norms_sq_and_fro_sq_subset(subset: CSRRowSubset) -> tuple[np.ndarray, float]:  # EN: Compute col norms and Frobenius^2 for A[row_ids,:] without copying CSR.
    A = subset.A  # EN: Unpack full CSR matrix reference.
    _, n = A.shape  # EN: Extract feature dimension.
    col_sq = np.zeros((int(n),), dtype=float)  # EN: Allocate accumulator for column squared norms.
    fro_sq = 0.0  # EN: Accumulate Frobenius norm squared over selected rows.
    m_subset = int(subset.shape[0])  # EN: Selected row count.
    for k in range(m_subset):  # EN: Loop selected rows.
        start = int(subset.starts[k])  # EN: Row start pointer.
        end = int(subset.ends[k])  # EN: Row end pointer.
        if end <= start:  # EN: Skip empty rows.
            continue  # EN: Move to next row.
        cols = A.indices[start:end]  # EN: Column indices for the row segment.
        vals = A.data[start:end]  # EN: Values for the row segment.
        sq = vals * vals  # EN: Square values (contributes to both fro_sq and col_sq).
        fro_sq += float(np.sum(sq))  # EN: Add row contribution to ||A||_F^2.
        np.add.at(col_sq, cols, sq)  # EN: Add squared values into corresponding columns.
    return col_sq.astype(float), float(fro_sq)  # EN: Return (col_sq, fro_sq).


def csr_select_rows(A: CSRMatrix, row_ids: np.ndarray) -> CSRMatrix:  # EN: Build a CSR submatrix consisting of selected rows (in the given order).
    m, n = A.shape  # EN: Extract original shape.
    rows = np.asarray(row_ids, dtype=int).reshape(-1)  # EN: Convert row ids to a 1D int array.
    if rows.size == 0:  # EN: Handle empty selection.
        return CSRMatrix(data=np.array([], dtype=float), indices=np.array([], dtype=int), indptr=np.zeros((1,), dtype=int), shape=(0, int(n)))  # EN: Return empty CSR matrix.
    if np.any(rows < 0) or np.any(rows >= m):  # EN: Validate bounds.
        raise ValueError("row_ids out of bounds")  # EN: Reject invalid row indices.

    nnz_per_row = (A.indptr[rows + 1] - A.indptr[rows]).astype(int)  # EN: Compute nnz in each selected row.
    nnz_new = int(np.sum(nnz_per_row))  # EN: Total nnz in the new matrix.

    data_new = np.empty((nnz_new,), dtype=float)  # EN: Allocate new data array.
    indices_new = np.empty((nnz_new,), dtype=int)  # EN: Allocate new indices array.
    indptr_new = np.zeros((rows.size + 1,), dtype=int)  # EN: Allocate new indptr array.

    cursor = 0  # EN: Write cursor into the new data/indices arrays.
    for i_new, i_old in enumerate(rows):  # EN: Copy rows one-by-one into the new CSR.
        start_old = int(A.indptr[int(i_old)])  # EN: Old row start pointer.
        end_old = int(A.indptr[int(i_old) + 1])  # EN: Old row end pointer.
        row_nnz = int(end_old - start_old)  # EN: Number of nonzeros in this row.
        indptr_new[i_new] = int(cursor)  # EN: Record row start in new CSR.
        if row_nnz > 0:  # EN: Copy row slices when non-empty.
            data_new[cursor : cursor + row_nnz] = A.data[start_old:end_old]  # EN: Copy nonzero values.
            indices_new[cursor : cursor + row_nnz] = A.indices[start_old:end_old]  # EN: Copy column indices.
        cursor += row_nnz  # EN: Advance cursor by row nnz.
    indptr_new[rows.size] = int(cursor)  # EN: Final pointer equals nnz_new.

    return CSRMatrix(data=data_new.astype(float), indices=indices_new.astype(int), indptr=indptr_new.astype(int), shape=(int(rows.size), int(n)))  # EN: Return the row-selected CSR matrix.


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


def choose_sketch_rows(m_aug: int, n: int, sketch_factor: float) -> int:  # EN: Choose sketch row count s for an oversampled rand-QR preconditioner.
    s_target = int(round(float(sketch_factor) * float(n)))  # EN: Oversampled target like 4n.
    s = int(max(n, min(m_aug, s_target)))  # EN: Clamp s into [n, m_aug] so QR is well-posed.
    return int(s)  # EN: Return as Python int.


def build_countsketch_aug(  # EN: Build a CountSketch for the augmented matrix [A; damp I] that can be reused across damp values.
    A: CSRMatrix,  # EN: Sparse design matrix in CSR (shape m×n).
    sketch_factor: float,  # EN: Oversampling factor for sketch rows (e.g., 4.0 => s≈4n).
    rng: np.random.Generator,  # EN: RNG for sketch construction.
) -> CountSketchAug:  # EN: Return CountSketchAug with SA_top and identity hashing.
    m, n = A.shape  # EN: Extract dimensions.
    m_aug = int(m + n)  # EN: Augmented row count for [A; damp I].
    s = choose_sketch_rows(m_aug=m_aug, n=int(n), sketch_factor=float(sketch_factor))  # EN: Choose sketch row count.
    scale = float(1.0 / np.sqrt(max(s, 1)))  # EN: Use 1/sqrt(s) scaling (keeps norms comparable).

    # EN: CountSketch: each original row is mapped to one sketch row with a random sign.  # EN: Explain sketch structure.
    h_top = rng.integers(low=0, high=s, size=int(m), dtype=int)  # EN: Hash bucket for each data row.
    sign_top = rng.choice(np.array([-1.0, 1.0]), size=int(m)).astype(float)  # EN: Random sign for each data row.

    h_bottom = rng.integers(low=0, high=s, size=int(n), dtype=int)  # EN: Hash bucket for each identity row (one per feature).
    sign_bottom = rng.choice(np.array([-1.0, 1.0]), size=int(n)).astype(float)  # EN: Random sign for each identity row.

    SA_top = np.zeros((int(s), int(n)), dtype=float)  # EN: Allocate dense sketch matrix for S_top A.

    # EN: Compute SA_top = S_top A in O(nnz) by accumulating each sparse row into its hashed sketch row.  # EN: Explain loop.
    for i in range(int(m)):  # EN: Loop over data rows.
        row_bucket = int(h_top[i])  # EN: Sketch row index for this data row.
        row_sign = float(sign_top[i])  # EN: Random sign for this data row.
        start = int(A.indptr[i])  # EN: Row start pointer.
        end = int(A.indptr[i + 1])  # EN: Row end pointer.
        cols = A.indices[start:end]  # EN: Column indices for this row.
        vals = A.data[start:end]  # EN: Nonzero values for this row.
        if cols.size > 0:  # EN: Skip empty rows.
            SA_top[row_bucket, cols] += (scale * row_sign) * vals  # EN: Add signed row into the sketch.

    return CountSketchAug(  # EN: Return sketch container.
        SA_top=SA_top.astype(float),  # EN: Store SA_top.
        h_bottom=h_bottom.astype(int),  # EN: Store identity hashes.
        sign_bottom=sign_bottom.astype(float),  # EN: Store identity signs.
        scale=float(scale),  # EN: Store scaling factor.
        s=int(s),  # EN: Store sketch row count.
    )  # EN: End return.


def build_countsketch_aug_subset(  # EN: Build a CountSketch for [A[row_ids,:]; damp I] without copying CSR rows.
    subset: CSRRowSubset,  # EN: Row-subset view (A[row_ids,:]) to sketch.
    sketch_factor: float,  # EN: Oversampling factor for sketch rows (e.g., 4.0 => s≈4n).
    rng: np.random.Generator,  # EN: RNG for sketch construction.
) -> CountSketchAug:  # EN: Return CountSketchAug with SA_top and identity hashing.
    A = subset.A  # EN: Unpack full CSR matrix reference.
    m_sub, n = subset.shape  # EN: Extract subset dimensions.
    m_aug = int(int(m_sub) + int(n))  # EN: Augmented row count for [A_sub; damp I].
    s = choose_sketch_rows(m_aug=m_aug, n=int(n), sketch_factor=float(sketch_factor))  # EN: Choose sketch row count.
    scale = float(1.0 / np.sqrt(max(s, 1)))  # EN: Use 1/sqrt(s) scaling (keeps norms comparable).

    # EN: For the subset operator, "top" hashing is per subset-row (k=0..m_sub-1), not per full row id.  # EN: Explain mapping.
    h_top = rng.integers(low=0, high=s, size=int(m_sub), dtype=int)  # EN: Hash bucket for each selected data row.
    sign_top = rng.choice(np.array([-1.0, 1.0]), size=int(m_sub)).astype(float)  # EN: Random sign for each selected data row.

    h_bottom = rng.integers(low=0, high=s, size=int(n), dtype=int)  # EN: Hash bucket for each identity row (one per feature).
    sign_bottom = rng.choice(np.array([-1.0, 1.0]), size=int(n)).astype(float)  # EN: Random sign for each identity row.

    SA_top = np.zeros((int(s), int(n)), dtype=float)  # EN: Allocate dense sketch matrix for S_top A_sub.

    # EN: Compute SA_top = S_top A_sub in O(nnz_selected) by scanning only the selected CSR rows.  # EN: Explain loop.
    for k in range(int(m_sub)):  # EN: Loop subset rows (in subset order).
        row_bucket = int(h_top[k])  # EN: Sketch row index for this subset row.
        row_sign = float(sign_top[k])  # EN: Random sign for this subset row.
        start = int(subset.starts[k])  # EN: Row start pointer in full CSR arrays.
        end = int(subset.ends[k])  # EN: Row end pointer in full CSR arrays.
        if end <= start:  # EN: Skip empty rows.
            continue  # EN: Move to next row.
        cols = A.indices[start:end]  # EN: Column indices for this row segment.
        vals = A.data[start:end]  # EN: Nonzero values for this row segment.
        SA_top[row_bucket, cols] += (scale * row_sign) * vals  # EN: Add signed row into the sketch.

    return CountSketchAug(  # EN: Return sketch container.
        SA_top=SA_top.astype(float),  # EN: Store SA_top.
        h_bottom=h_bottom.astype(int),  # EN: Store identity hashes.
        sign_bottom=sign_bottom.astype(float),  # EN: Store identity signs.
        scale=float(scale),  # EN: Store scaling factor.
        s=int(s),  # EN: Store sketch row count.
    )  # EN: End return.


def randqr_R_from_countsketch(sketch: CountSketchAug, damp: float) -> np.ndarray:  # EN: Compute R from QR(S [A; damp I]) using a prebuilt CountSketch.
    n = int(sketch.SA_top.shape[1])  # EN: Extract n from SA_top.
    A_sketch = sketch.SA_top.copy()  # EN: Start from S_top A (data contribution).
    j = np.arange(n, dtype=int)  # EN: Column indices for the identity contribution.
    # EN: Since A_aug=[A;damp I], the identity rows contribute damp*(scale*sign_bottom) at (h_bottom[j], j).  # EN: Explain update.
    np.add.at(A_sketch, (sketch.h_bottom, j), (float(damp) * float(sketch.scale)) * sketch.sign_bottom)  # EN: Add hashed identity contribution.

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

    return str(label), apply_Minv, apply_Minv_T  # EN: Return preconditioner functions.


def build_preconditioner(  # EN: Build a right preconditioner for the sparse operator (none / col-scaling / rand-QR).
    kind: PrecondKind,  # EN: Preconditioner kind.
    A: CSRMatrix | None,  # EN: Sparse design matrix (needed only for rand-QR; may be None for none/col).
    col_sq: np.ndarray,  # EN: Column squared norms of A (used by col-scaling).
    damp: float,  # EN: Damping parameter (affects augmented column norms and rand-QR).
    rng: np.random.Generator,  # EN: RNG for randomized preconditioners (ignored for deterministic kinds).
    sketch_factor: float = 4.0,  # EN: Oversampling factor for rand-QR sketches (e.g., 4.0 => s≈4n).
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

    if kind == "randqr":  # EN: Randomized QR preconditioner built from a CountSketch of the augmented matrix.
        if A is None:  # EN: rand-QR needs access to the CSR rows to build a sketch.
            raise ValueError("A must be provided when kind='randqr'")  # EN: Fail fast on missing A.
        t0 = perf_counter()  # EN: Start build timer.
        sketch = build_countsketch_aug(A=A, sketch_factor=float(sketch_factor), rng=rng)  # EN: Build CountSketch blocks for this matrix.
        R = randqr_R_from_countsketch(sketch=sketch, damp=float(damp))  # EN: Build R from QR of the sketched augmented matrix.
        build_seconds = float(perf_counter() - t0)  # EN: Stop build timer.
        label, apply_Minv, apply_Minv_T = upper_triangular_preconditioner_from_R(R=R, label="rand-QR(countsketch)")  # EN: Create apply closures.
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


def solve_one_operator(  # EN: Solve one ridge problem given only matvecs (matrix-free), returning a SolveReport.
    m: int,  # EN: Row dimension of the operator A (number of samples).
    n: int,  # EN: Column dimension of the operator A (number of features).
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
    rng: np.random.Generator,  # EN: RNG for randomized preconditioners (rand-QR).
    x_init: np.ndarray | None = None,  # EN: Optional warm-start initial guess in x-space.
    precond_override: tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]] | None = None,  # EN: Optional prebuilt (label, Minv, Minv_T).
    build_seconds_override: float | None = None,  # EN: Optional build-time override (used when precond_override is provided).
    A_for_precond: CSRMatrix | None = None,  # EN: Optional CSR matrix for building rand-QR internally (not needed when precond_override is provided).
) -> SolveReport:  # EN: Return structured diagnostics for this run.
    m_i = int(m)  # EN: Ensure m is an int.
    n_i = int(n)  # EN: Ensure n is an int.
    damp_f = float(damp)  # EN: Ensure damp is a float.

    # EN: Use a deterministic upper bound for ||A_aug||_2 via Frobenius: ||A_aug||_2 <= ||A_aug||_F.  # EN: Explain estimate choice.
    anorm_est = float(np.sqrt(float(fro_sq) + (damp_f * damp_f) * float(n_i)))  # EN: Compute ||[A;damp I]||_F.

    # EN: Choose the preconditioner: either build it here, or reuse a prebuilt one supplied by the caller.  # EN: Explain override.
    if precond_override is None:  # EN: Default behavior: build preconditioner per call.
        label, apply_Minv, apply_Minv_T, build_seconds = build_preconditioner(  # EN: Build preconditioner.
            kind=precond_kind,  # EN: Preconditioner kind.
            A=A_for_precond,  # EN: Provide CSR rows only when needed (rand-QR); None is OK for none/col.
            col_sq=col_sq,  # EN: Provide column norms (used by col-scaling).
            damp=damp_f,  # EN: Provide damp.
            rng=rng,  # EN: Provide RNG.
        )  # EN: End build call.
    else:  # EN: Reuse an externally built preconditioner (e.g., shared sketch across a damp sweep).
        label, apply_Minv, apply_Minv_T = precond_override  # EN: Unpack prebuilt closures.
        build_seconds = float(build_seconds_override) if build_seconds_override is not None else 0.0  # EN: Charge explicit build time (or 0).

    # EN: Warm-start via a correction solve with RHS [b-Ax0; -damp x0].  # EN: Explain technique.
    if x_init is None:  # EN: No warm-start.
        x0 = np.zeros((n_i,), dtype=float)  # EN: Use x0=0.
        b_top = np.asarray(b, dtype=float).reshape(-1)  # EN: Top RHS is original b.
        b_bottom = None  # EN: Bottom RHS is zero by default.
    else:  # EN: Warm-start enabled.
        x0 = np.asarray(x_init, dtype=float).reshape(-1)  # EN: Convert x_init to 1D float array.
        if x0.size != n_i:  # EN: Validate x_init length.
            raise ValueError("x_init must have length n")  # EN: Reject invalid warm-start.
        b_top = np.asarray(b, dtype=float).reshape(-1) - matvec_A(x0)  # EN: Top RHS becomes residual b - A x0.
        b_bottom = -damp_f * x0  # EN: Bottom RHS becomes -damp x0.

    t0 = perf_counter()  # EN: Start solve timer.
    x_delta, iters, stop_reason, _, _, _, _ = lsmr_damped_minres_teaching_operator(  # EN: Solve for delta (or x when x0=0).
        matvec_A=matvec_A,  # EN: Provide matvec A.
        matvec_AT=matvec_AT,  # EN: Provide matvec A^T.
        m=int(m_i),  # EN: Provide m.
        n=int(n_i),  # EN: Provide n.
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
    rng: np.random.Generator,  # EN: RNG for randomized preconditioners (rand-QR).
    x_init: np.ndarray | None = None,  # EN: Optional warm-start initial guess in x-space.
    precond_override: tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]] | None = None,  # EN: Optional prebuilt (label, Minv, Minv_T).
    build_seconds_override: float | None = None,  # EN: Optional build-time override (used when precond_override is provided).
) -> SolveReport:  # EN: Return structured diagnostics for this run.
    m, n = A.shape  # EN: Extract dimensions from the CSR matrix.
    return solve_one_operator(  # EN: Delegate to the matrix-free solver core.
        m=int(m),  # EN: Row count.
        n=int(n),  # EN: Feature count.
        matvec_A=matvec_A,  # EN: Matvec A.
        matvec_AT=matvec_AT,  # EN: Matvec A^T.
        col_sq=col_sq,  # EN: Column norms.
        fro_sq=float(fro_sq),  # EN: Frobenius squared.
        b=b,  # EN: Targets.
        damp=float(damp),  # EN: Damp.
        precond_kind=precond_kind,  # EN: Preconditioner kind.
        max_iters=max_iters,  # EN: Iteration cap.
        atol=atol,  # EN: atol.
        btol=btol,  # EN: btol.
        rng=rng,  # EN: RNG.
        x_init=x_init,  # EN: Warm-start (optional).
        precond_override=precond_override,  # EN: Optional prebuilt preconditioner.
        build_seconds_override=build_seconds_override,  # EN: Optional build-time override.
        A_for_precond=A,  # EN: Provide CSR rows for internal rand-QR builds when needed.
    )  # EN: End delegation.


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


def cv_sweep_curve_sparse(  # EN: Sweep a damp grid with k-fold CV for a sparse CSR matrix, accumulating total cost.
    splits: list[tuple[np.ndarray, np.ndarray]],  # EN: CV splits (train_idx, val_idx).
    A_full: CSRMatrix,  # EN: Full sparse design matrix (rows will be selected per fold).
    b_full: np.ndarray,  # EN: Full targets vector (length m_full).
    damps: np.ndarray,  # EN: Damp grid to sweep (will be sorted ascending for reporting).
    precond_kind: PrecondKind,  # EN: Preconditioner kind (none/col/randqr).
    randqr_policy: RandQRPolicy,  # EN: Policy for rand-QR (ignored unless precond_kind == "randqr").
    warm_start: bool,  # EN: Whether to use continuation/warm-start across the damp grid within each fold.
    max_iters: int,  # EN: Iteration cap.
    atol: float,  # EN: Absolute tolerance.
    btol: float,  # EN: Relative tolerance.
    sketch_factor: float,  # EN: Oversampling factor for rand-QR sketches.
    rng: np.random.Generator,  # EN: RNG for sketches and split-specific randomness.
) -> tuple[list[CVPoint], CVTotals, CVPoint]:  # EN: Return (points, totals, best_point).
    damps_sorted = np.array(sorted([float(d) for d in damps]), dtype=float)  # EN: Sort damps ascending for reporting.
    n_damps = int(damps_sorted.size)  # EN: Number of damp values.
    n_folds = int(len(splits))  # EN: Number of folds.
    n_features = int(A_full.shape[1])  # EN: Feature count.

    order = np.arange(n_damps, dtype=int)  # EN: Default evaluation order is ascending.
    if warm_start:  # EN: For continuation, starting from large damp is often more stable (solution near 0).
        order = order[::-1]  # EN: Evaluate from largest damp down to smallest.

    train_rmse_mat = np.zeros((n_damps, n_folds), dtype=float)  # EN: Store train RMSE per (damp, fold).
    val_rmse_mat = np.zeros((n_damps, n_folds), dtype=float)  # EN: Store val RMSE per (damp, fold).
    xnorm_mat = np.zeros((n_damps, n_folds), dtype=float)  # EN: Store ||x|| per (damp, fold).
    iters_mat = np.zeros((n_damps, n_folds), dtype=float)  # EN: Store iterations per (damp, fold).

    total_build = 0.0  # EN: Accumulate preconditioner build time across all fits.
    total_solve = 0.0  # EN: Accumulate solver time across all fits.
    total_iters = 0  # EN: Accumulate iterations across all fits.

    # EN: Build a readable label for this configuration (used in summaries).  # EN: Explain labeling.
    if precond_kind == "none":  # EN: Baseline no-preconditioner label.
        label = "none"  # EN: Label.
    elif precond_kind == "col":  # EN: Column-scaling label.
        label = "col-scaling"  # EN: Label.
    else:  # EN: rand-QR label depends on policy.
        policy_tag = {"rebuild": "rand-QR(rebuild)", "shared_sketch": "rand-QR(shared)", "fixed_R": "rand-QR(fixed-R)"}[randqr_policy]  # EN: Map policy to a readable tag.
        label = policy_tag  # EN: Assign label.
    if warm_start:  # EN: Annotate label when warm-start is enabled.
        label = f"{label}+ws"  # EN: Append warm-start suffix.

    damp_ref = choose_fixed_randqr_reference_damp(damps_sorted)  # EN: Reference damp for fixed-R policy.

    b_full_1d = np.asarray(b_full, dtype=float).reshape(-1)  # EN: Normalize b_full to a 1D float array once.

    for fold_id, (train_idx, val_idx) in enumerate(splits):  # EN: Loop folds (outer) to enable continuation within each fold.
        # EN: IMPORTANT: do NOT build A_tr/A_va as new CSR matrices; instead, keep A_full and use row-index operators.  # EN: Explain memory goal.
        subset_tr = make_csr_row_subset(A=A_full, row_ids=train_idx)  # EN: Training operator view A_full[train_idx, :].
        subset_va = make_csr_row_subset(A=A_full, row_ids=val_idx)  # EN: Validation operator view A_full[val_idx, :].

        m_tr = int(subset_tr.shape[0])  # EN: Training sample count for this fold.
        b_tr = b_full_1d[subset_tr.row_ids]  # EN: Slice training targets (aligned with subset order).
        b_va = b_full_1d[subset_va.row_ids]  # EN: Slice validation targets (aligned with subset order).

        col_sq_tr, fro_sq_tr = csr_column_norms_sq_and_fro_sq_subset(subset_tr)  # EN: Column norms + ||A_tr||_F^2 for norm bounds.

        def matvec_A_tr(x: np.ndarray, subset_tr: CSRRowSubset = subset_tr) -> np.ndarray:  # EN: Training matvec A_tr x.
            return csr_matvec_subset(subset=subset_tr, x=x)  # EN: Compute A_full[train_idx,:] x.

        def matvec_AT_tr(y: np.ndarray, subset_tr: CSRRowSubset = subset_tr) -> np.ndarray:  # EN: Training transpose matvec A_tr^T y.
            return csr_rmatvec_subset(subset=subset_tr, y=y)  # EN: Compute A_full[train_idx,:]^T y.

        x_prev = np.zeros((n_features,), dtype=float)  # EN: Initialize continuation with x=0 for the first step.

        # EN: Build per-fold cached objects for rand-QR reuse policies (since A_tr changes per fold).  # EN: Explain caching scope.
        shared_sketch: CountSketchAug | None = None  # EN: Shared sketch for this fold (optional).
        fixed_precond: tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]] | None = None  # EN: Fixed-R preconditioner closures (optional).

        if precond_kind == "randqr" and randqr_policy in {"shared_sketch", "fixed_R"}:  # EN: Build shared CountSketch once per fold.
            t0 = perf_counter()  # EN: Start sketch build timer.
            shared_sketch = build_countsketch_aug_subset(subset=subset_tr, sketch_factor=float(sketch_factor), rng=rng)  # EN: Build CountSketch blocks from selected rows.
            total_build += float(perf_counter() - t0)  # EN: Account for one-time sketch build cost.

        if precond_kind == "randqr" and randqr_policy == "fixed_R":  # EN: Build a fixed R once per fold (then reuse across all damps).
            if shared_sketch is None:  # EN: Defensive: fixed-R requires a sketch.
                raise RuntimeError("internal error: shared_sketch is None for fixed_R policy")  # EN: Fail fast.
            t0 = perf_counter()  # EN: Start R build timer.
            R_ref = randqr_R_from_countsketch(sketch=shared_sketch, damp=float(damp_ref))  # EN: Build R at reference damp.
            total_build += float(perf_counter() - t0)  # EN: Account for one-time QR cost.
            fixed_precond = upper_triangular_preconditioner_from_R(R=R_ref, label="rand-QR(fixed-R)")  # EN: Build apply closures.

        for step, idx in enumerate(order):  # EN: Sweep damp values in the chosen continuation order.
            damp = float(damps_sorted[idx])  # EN: Current damp value.
            x_init = x_prev if (warm_start and step > 0) else None  # EN: Use previous x as warm-start except for the first step.

            # EN: Choose how to build/reuse the preconditioner for this fit.  # EN: Explain branching.
            if precond_kind != "randqr":  # EN: None/col use internal builders that do not require CSR copying.
                rep = solve_one_operator(  # EN: Fit on the training fold with matrix-free operators.
                    m=int(m_tr),  # EN: Training row count.
                    n=int(n_features),  # EN: Feature count.
                    matvec_A=matvec_A_tr,  # EN: Training matvec A.
                    matvec_AT=matvec_AT_tr,  # EN: Training matvec A^T.
                    col_sq=col_sq_tr,  # EN: Column norms for training operator.
                    fro_sq=float(fro_sq_tr),  # EN: Frobenius squared for norm bound.
                    b=b_tr,  # EN: Training targets.
                    damp=float(damp),  # EN: Damp.
                    precond_kind=precond_kind,  # EN: Preconditioner kind.
                    max_iters=max_iters,  # EN: Cap.
                    atol=atol,  # EN: atol.
                    btol=btol,  # EN: btol.
                    rng=rng,  # EN: RNG (unused by deterministic kinds).
                    x_init=x_init,  # EN: Warm-start (optional).
                    A_for_precond=None,  # EN: No CSR needed for none/col builds.
                )  # EN: End solve.
            else:  # EN: rand-QR path supports rebuild/shared/fixed policies.
                if randqr_policy == "rebuild":  # EN: Baseline: rebuild sketch and QR for every damp.
                    t0 = perf_counter()  # EN: Time sketch+QR build for baseline rebuild.
                    sketch = build_countsketch_aug_subset(subset=subset_tr, sketch_factor=float(sketch_factor), rng=rng)  # EN: Build CountSketch from selected rows.
                    R = randqr_R_from_countsketch(sketch=sketch, damp=float(damp))  # EN: Build R from QR of the sketched augmented matrix.
                    build_s = float(perf_counter() - t0)  # EN: Total build time for rebuild policy.
                    precond = upper_triangular_preconditioner_from_R(R=R, label="rand-QR(rebuild)")  # EN: Create apply closures from R.
                    rep = solve_one_operator(  # EN: Fit with externally built rand-QR preconditioner.
                        m=int(m_tr),  # EN: Training row count.
                        n=int(n_features),  # EN: Feature count.
                        matvec_A=matvec_A_tr,  # EN: Training matvec A.
                        matvec_AT=matvec_AT_tr,  # EN: Training matvec A^T.
                        col_sq=col_sq_tr,  # EN: Column norms (still used for metadata / consistency).
                        fro_sq=float(fro_sq_tr),  # EN: Frobenius squared.
                        b=b_tr,  # EN: Training targets.
                        damp=float(damp),  # EN: Damp.
                        precond_kind="randqr",  # EN: Kind ignored because we pass precond_override.
                        max_iters=max_iters,  # EN: Cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        rng=rng,  # EN: RNG (unused due to override).
                        x_init=x_init,  # EN: Warm-start (optional).
                        precond_override=precond,  # EN: Use the prebuilt preconditioner closures.
                        build_seconds_override=build_s,  # EN: Charge full build time for this fit.
                        A_for_precond=None,  # EN: No CSR needed because precond_override is used.
                    )  # EN: End solve.
                elif randqr_policy == "shared_sketch":  # EN: Rebuild R per damp, but reuse the expensive sketch SA_top.
                    if shared_sketch is None:  # EN: Defensive check.
                        raise RuntimeError("internal error: shared_sketch is None for shared_sketch policy")  # EN: Fail fast.
                    t0 = perf_counter()  # EN: Time only the R-from-sketch build for this damp.
                    R = randqr_R_from_countsketch(sketch=shared_sketch, damp=float(damp))  # EN: Build R using shared sketch.
                    build_s = float(perf_counter() - t0)  # EN: Record per-damp QR build time.
                    precond = upper_triangular_preconditioner_from_R(R=R, label="rand-QR(shared)")  # EN: Create apply closures from R.
                    rep = solve_one_operator(  # EN: Fit with an externally built preconditioner.
                        m=int(m_tr),  # EN: Training row count.
                        n=int(n_features),  # EN: Feature count.
                        matvec_A=matvec_A_tr,  # EN: Training matvec A.
                        matvec_AT=matvec_AT_tr,  # EN: Training matvec A^T.
                        col_sq=col_sq_tr,  # EN: Column norms.
                        fro_sq=float(fro_sq_tr),  # EN: Frobenius squared.
                        b=b_tr,  # EN: Training targets.
                        damp=float(damp),  # EN: Damp.
                        precond_kind="randqr",  # EN: Value ignored because we pass precond_override.
                        max_iters=max_iters,  # EN: Cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        rng=rng,  # EN: RNG (unused due to override).
                        x_init=x_init,  # EN: Warm-start (optional).
                        precond_override=precond,  # EN: Use the prebuilt preconditioner closures.
                        build_seconds_override=build_s,  # EN: Charge only QR time (sketch build charged once per fold above).
                        A_for_precond=None,  # EN: No CSR needed because precond_override is used.
                    )  # EN: End solve.
                elif randqr_policy == "fixed_R":  # EN: Reuse the SAME R preconditioner for all damp values in the curve.
                    if fixed_precond is None:  # EN: Defensive check.
                        raise RuntimeError("internal error: fixed_precond is None for fixed_R policy")  # EN: Fail fast.
                    rep = solve_one_operator(  # EN: Fit with the reused preconditioner.
                        m=int(m_tr),  # EN: Training row count.
                        n=int(n_features),  # EN: Feature count.
                        matvec_A=matvec_A_tr,  # EN: Training matvec A.
                        matvec_AT=matvec_AT_tr,  # EN: Training matvec A^T.
                        col_sq=col_sq_tr,  # EN: Column norms.
                        fro_sq=float(fro_sq_tr),  # EN: Frobenius squared.
                        b=b_tr,  # EN: Training targets.
                        damp=float(damp),  # EN: Damp (objective still changes, only M is fixed).
                        precond_kind="randqr",  # EN: Value ignored because we pass precond_override.
                        max_iters=max_iters,  # EN: Cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        rng=rng,  # EN: RNG (unused due to override).
                        x_init=x_init,  # EN: Warm-start (optional).
                        precond_override=fixed_precond,  # EN: Use the fixed-R preconditioner closures.
                        build_seconds_override=0.0,  # EN: No per-damp build cost when R is fixed.
                        A_for_precond=None,  # EN: No CSR needed because precond_override is used.
                    )  # EN: End solve.
                else:  # EN: Guard against unknown policy values.
                    raise ValueError("Unknown randqr_policy")  # EN: Reject invalid policy.

            x_prev = rep.x_hat  # EN: Update continuation state for the next damp (used only when warm_start=True).

            # EN: Training RMSE can be computed from ||A_tr x - b_tr||_2 without another matvec: RMSE = ||r||_2 / sqrt(m_tr).  # EN: Explain shortcut.
            train_rmse = float(rep.rnorm_data) / float(np.sqrt(max(m_tr, 1)))  # EN: Compute train RMSE cheaply.
            val_pred = csr_matvec_subset(subset=subset_va, x=rep.x_hat)  # EN: Compute validation predictions A_full[val_idx,:] x_hat.

            train_rmse_mat[idx, fold_id] = float(train_rmse)  # EN: Store train RMSE.
            val_rmse_mat[idx, fold_id] = rmse(y_true=b_va, y_pred=val_pred)  # EN: Store validation RMSE.
            xnorm_mat[idx, fold_id] = float(rep.xnorm)  # EN: Store ||x||.
            iters_mat[idx, fold_id] = float(rep.n_iters)  # EN: Store iterations.

            total_build += float(rep.build_seconds)  # EN: Add per-fit build time.
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
    totals = CVTotals(precond=str(label), total_build_seconds=float(total_build), total_solve_seconds=float(total_solve), total_iters=int(total_iters))  # EN: Package totals.
    return points, totals, best  # EN: Return curve, totals, and best point.


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
    row_idx_A = np.repeat(np.arange(m, dtype=int), np.diff(A.indptr).astype(int))  # EN: Precompute row index per nnz for fast A^T matvec.

    # EN: Define matvecs so the solver never needs the dense matrix explicitly.  # EN: Explain matrix-free interface.
    def matvec_A(x: np.ndarray) -> np.ndarray:  # EN: CSR matvec for A.
        return csr_matvec(A=A, x=x)  # EN: Compute A x.

    def matvec_AT(y: np.ndarray) -> np.ndarray:  # EN: CSR transpose matvec for A^T.
        return csr_rmatvec(A=A, y=y, row_indices=row_idx_A)  # EN: Compute A^T y using precomputed row indices.

    # EN: Create a sparse-ish ground-truth coefficient vector and synthetic targets b.  # EN: Explain target construction.
    x_true = np.zeros((n,), dtype=float)  # EN: Initialize x_true.
    support = rng.choice(n, size=12, replace=False)  # EN: Choose a small support set.
    x_true[support] = rng.standard_normal(support.size)  # EN: Fill support with random values.
    b_clean = matvec_A(x_true)  # EN: Compute noiseless targets.
    b = b_clean + noise_std * rng.standard_normal((m,)).astype(float)  # EN: Add Gaussian noise.

    print_separator("Sparse / matrix-free dataset summary")  # EN: Announce dataset summary section.
    print(f"m={m}, n={n}, nnz={nnz}, nnz_per_row={nnz_per_row}, density={density:.3e}")  # EN: Print sparsity stats.
    print(f"||A||_F={np.sqrt(fro_sq):.3e}, noise_std={noise_std}")  # EN: Print Frobenius norm and noise.

    # EN: Solver settings used across experiments.  # EN: Explain settings.
    max_iters = min(2 * n, 150)  # EN: Iteration cap (keep it moderate but allow convergence).
    atol = 1e-10  # EN: Absolute tolerance.
    btol = 1e-10  # EN: Relative tolerance.
    sketch_factor = 4.0  # EN: Oversampling factor for rand-QR sketches (s≈4n).

    # EN: Per-damp solver comparison on the full dataset (not CV).  # EN: Explain purpose.
    demo_damps = [0.0, 1e-2, 1e-1, 1.0]  # EN: Damp values to compare.
    preconds: list[PrecondKind] = ["none", "col", "randqr"]  # EN: Preconditioner list.
    rng_demo = np.random.default_rng(SEED + 10)  # EN: Dedicated RNG for demo preconditioners (rand-QR).

    print_separator("Per-damp solver comparison (full data)")  # EN: Announce comparison section.
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
                    rng=rng_demo,  # EN: RNG (used by rand-QR).
                    x_init=None,  # EN: Cold-start.
                )  # EN: End solve.
            )  # EN: End append.
    print_solver_table(reports)  # EN: Print table.

    # EN: Build CV splits once so baseline/speedups compare on the same folds.  # EN: Explain split policy.
    n_folds = 5  # EN: Number of folds for k-fold CV.
    rng_split = np.random.default_rng(SEED + 20)  # EN: Dedicated RNG for deterministic splits.
    splits = k_fold_splits(rng=rng_split, n_samples=m, n_folds=n_folds)  # EN: Build k-fold splits.

    # EN: Damp grid for CV (include 0 plus a log-grid).  # EN: Explain hyperparameter grid.
    damps = np.concatenate(([0.0], np.logspace(-6, 1, num=12)))  # EN: Candidate damps for CV.

    # EN: Baseline CV: no warm-start; rand-QR is rebuilt from scratch for every damp.  # EN: Explain baseline.
    print_separator("k-fold CV sweep (baseline): rebuild per damp, no warm-start")  # EN: Announce baseline section.
    rng_cv_baseline = np.random.default_rng(SEED + 30)  # EN: Dedicated RNG stream for baseline.
    baseline_rows: dict[str, tuple[str, float, float, float, float, int]] = {}  # EN: Map precond -> summary tuple.

    for pk in preconds:  # EN: Run baseline CV for each preconditioner.
        points, totals, best = cv_sweep_curve_sparse(  # EN: Run one full CV curve sweep.
            splits=splits,  # EN: Provide splits.
            A_full=A,  # EN: Provide full A.
            b_full=b,  # EN: Provide full b.
            damps=damps,  # EN: Provide damp grid.
            precond_kind=pk,  # EN: Preconditioner kind.
            randqr_policy="rebuild",  # EN: Baseline policy (ignored unless pk=="randqr").
            warm_start=False,  # EN: No continuation.
            max_iters=max_iters,  # EN: Iteration cap.
            atol=atol,  # EN: atol.
            btol=btol,  # EN: btol.
            sketch_factor=sketch_factor,  # EN: Sketch factor for rand-QR.
            rng=rng_cv_baseline,  # EN: RNG stream.
        )  # EN: End sweep.

        print_separator(f"CV results (baseline): {totals.precond}")  # EN: Announce per-preconditioner baseline results.
        print_cv_table(points=points, best=best)  # EN: Print CV curve table.
        total_seconds = float(totals.total_build_seconds + totals.total_solve_seconds)  # EN: Total curve cost.
        print(  # EN: Print cost summary line.
            f"\nTotal curve cost (baseline, {totals.precond}): build={totals.total_build_seconds:.3f}s, solve={totals.total_solve_seconds:.3f}s, total={total_seconds:.3f}s, iters={totals.total_iters}"  # EN: Summary string.
        )  # EN: End print.

        baseline_rows[str(pk)] = (totals.precond, float(best.damp), float(best.val_mean), float(totals.total_build_seconds), float(totals.total_solve_seconds), int(totals.total_iters))  # EN: Store baseline summary.

    print_separator("Baseline CV summary across preconditioners")  # EN: Announce baseline summary table.
    header = "precond              | best_damp | best_val_rmse | build_s | solve_s | total_s | total_iters"  # EN: Header columns.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Divider.
    for pk in preconds:  # EN: Print in consistent order.
        label, best_d, best_val, build_s, solve_s, iters = baseline_rows[str(pk)]  # EN: Unpack baseline summary.
        total_s = float(build_s + solve_s)  # EN: Total seconds.
        damp_str = f"{best_d:.0e}" if best_d != 0.0 else "0"  # EN: Format damp.
        print(f"{label:19} | {damp_str:9} | {best_val:13.3e} | {build_s:7.3f} | {solve_s:7.3f} | {total_s:7.3f} | {iters:10d}")  # EN: Print row.

    # EN: Speedups: continuation (warm-start) + rand-QR shared sketch, to reduce full-curve CV cost.  # EN: Explain speedups.
    print_separator("CV sweep speedups: warm-start + rand-QR shared sketch")  # EN: Announce speedup section.
    rng_cv_speed = np.random.default_rng(SEED + 40)  # EN: Dedicated RNG stream for speedups.
    speed_rows: dict[str, tuple[str, float, float, float, float, int]] = {}  # EN: Map precond -> summary tuple.

    for pk in preconds:  # EN: Run speedup CV for each preconditioner.
        policy = "shared_sketch" if pk == "randqr" else "rebuild"  # EN: Only rand-QR uses special reuse; others ignore policy.
        points, totals, best = cv_sweep_curve_sparse(  # EN: Run CV sweep with speedups enabled.
            splits=splits,  # EN: Provide splits.
            A_full=A,  # EN: Provide full A.
            b_full=b,  # EN: Provide full b.
            damps=damps,  # EN: Provide damp grid.
            precond_kind=pk,  # EN: Preconditioner kind.
            randqr_policy=policy,  # EN: rand-QR reuse policy.
            warm_start=True,  # EN: Enable continuation across damps.
            max_iters=max_iters,  # EN: Iteration cap.
            atol=atol,  # EN: atol.
            btol=btol,  # EN: btol.
            sketch_factor=sketch_factor,  # EN: Sketch factor.
            rng=rng_cv_speed,  # EN: RNG stream.
        )  # EN: End sweep.

        total_seconds = float(totals.total_build_seconds + totals.total_solve_seconds)  # EN: Total curve cost.
        damp_str = f"{best.damp:.0e}" if best.damp != 0.0 else "0"  # EN: Format best damp for printing.
        print(  # EN: Print a compact summary line (baseline tables already printed above).
            f"{totals.precond:22} best_damp={damp_str:>9} best_val={best.val_mean:.3e} total={total_seconds:.3f}s (build={totals.total_build_seconds:.3f}s, solve={totals.total_solve_seconds:.3f}s) iters={totals.total_iters}"  # EN: Summary string.
        )  # EN: End print.

        speed_rows[str(pk)] = (totals.precond, float(best.damp), float(best.val_mean), float(totals.total_build_seconds), float(totals.total_solve_seconds), int(totals.total_iters))  # EN: Store speed summary.

    # EN: Compare baseline vs speedups in one table (time + iterations + best validation RMSE).  # EN: Explain comparison.
    print_separator("Baseline vs speedups (time/iters/quality)")  # EN: Announce comparison table.
    header = "precond              | base_total_s | sped_total_s | speedup | base_iters | sped_iters | base_best_val | sped_best_val"  # EN: Header columns.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Divider.
    for pk in preconds:  # EN: Loop in consistent order.
        b_label, _, b_best_val, b_build, b_solve, b_iters = baseline_rows[str(pk)]  # EN: Unpack baseline.
        s_label, _, s_best_val, s_build, s_solve, s_iters = speed_rows[str(pk)]  # EN: Unpack speedup.
        _ = s_label  # EN: Keep labels aligned; baseline label is shown in the first column.
        base_total = float(b_build + b_solve)  # EN: Baseline total seconds.
        sped_total = float(s_build + s_solve)  # EN: Speedup total seconds.
        speedup = (base_total / max(sped_total, EPS))  # EN: Compute speedup factor.
        print(  # EN: Print comparison row.
            f"{b_label:19} | {base_total:11.3f} | {sped_total:11.3f} | {speedup:7.2f} | {b_iters:9d} | {s_iters:9d} | {b_best_val:12.3e} | {s_best_val:12.3e}"  # EN: Row.
        )  # EN: End print.

    # EN: Extra: for rand-QR, compare shared-sketch vs fixed-R reuse (both with warm-start).  # EN: Explain extra experiment.
    print_separator("rand-QR reuse variants (warm-start): shared-sketch vs fixed-R")  # EN: Announce extra experiment.
    rng_cv_fixed = np.random.default_rng(SEED + 50)  # EN: Separate RNG for fixed-R variant.
    _points_fixed, totals_fixed, best_fixed = cv_sweep_curve_sparse(  # EN: Run fixed-R variant.
        splits=splits,  # EN: Provide splits.
        A_full=A,  # EN: Provide full A.
        b_full=b,  # EN: Provide full b.
        damps=damps,  # EN: Provide damp grid.
        precond_kind="randqr",  # EN: rand-QR only.
        randqr_policy="fixed_R",  # EN: Fixed-R reuse across the whole curve.
        warm_start=True,  # EN: Continuation across damps.
        max_iters=max_iters,  # EN: Iteration cap.
        atol=atol,  # EN: atol.
        btol=btol,  # EN: btol.
        sketch_factor=sketch_factor,  # EN: Sketch factor.
        rng=rng_cv_fixed,  # EN: RNG stream.
    )  # EN: End sweep.
    _ = _points_fixed  # EN: Keep points available for optional debugging.
    total_fixed = float(totals_fixed.total_build_seconds + totals_fixed.total_solve_seconds)  # EN: Total seconds for fixed-R.
    damp_str = f"{best_fixed.damp:.0e}" if best_fixed.damp != 0.0 else "0"  # EN: Format best damp for printing.
    print(  # EN: Print fixed-R summary line.
        f"{totals_fixed.precond:22} best_damp={damp_str:>9} best_val={best_fixed.val_mean:.3e} total={total_fixed:.3f}s (build={totals_fixed.total_build_seconds:.3f}s, solve={totals_fixed.total_solve_seconds:.3f}s) iters={totals_fixed.total_iters}"  # EN: Summary line.
    )  # EN: End print.


if __name__ == "__main__":  # EN: Standard Python entrypoint guard.
    main()  # EN: Execute main.
