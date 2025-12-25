"""  # EN: Start module docstring.
Operator-only sparse Ridge demo: matrix-free damped LSMR without storing CSR arrays.  # EN: Summarize this script in one line.

This file is the "next step" after the CSR-based unit script:  # EN: Explain why this variant exists.
  - The CSR version is already matrix-free in the *solver* sense (only matvec/rmatvec, never forms A^T A).  # EN: Contrast CSR vs operator-only.
  - This variant is matrix-free in the *data* sense: we never materialize A in memory.  # EN: Emphasize memory motivation.

We model a sparse design matrix A (m×n) where each row has nnz_per_row non-zeros,  # EN: Describe the synthetic sparse matrix model.
but the (col,value) pairs are generated deterministically from (row_id, seed) via hashing.  # EN: Explain deterministic generation.

We then run the same Ridge / damped least squares pipeline:  # EN: Outline what we do next.
  min_x ||A x - b||_2^2 + damp^2 ||x||_2^2,  damp >= 0,  # EN: State the objective.
with the same comparisons: none / col-scaling / rand-QR(CountSketch), k-fold CV,  # EN: Describe comparisons.
and warm-start + preconditioner reuse to reduce full-curve cost.  # EN: Mention speedups.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for small immutable records.
from time import perf_counter  # EN: Use perf_counter for build/solve timing.
from typing import Callable  # EN: Use Callable for typed matvec closures.

import numpy as np  # EN: Import NumPy for arrays, math, and linear algebra.

# EN: Reuse solver + reporting utilities from the CSR-based unit implementation (keeps this file focused on the data/operator change).  # EN: Explain import strategy.
from lsmr_damped_sparse_matrix_free_numpy import (  # EN: Import shared utilities from the sibling script in this unit.
    EPS,  # EN: Small epsilon for safe divisions.
    SEED,  # EN: Default RNG seed used across units for reproducibility.
    CountSketchAug,  # EN: Dataclass holding a reusable CountSketch for [A; damp I].
    CVPoint,  # EN: Dataclass for one CV curve point summary.
    CVTotals,  # EN: Dataclass for whole-curve cost totals.
    PrecondKind,  # EN: Literal type for supported preconditioners ("none"/"col"/"randqr").
    RandQRPolicy,  # EN: Literal type for rand-QR reuse policies.
    SolveReport,  # EN: Dataclass for one solver run report.
    choose_fixed_randqr_reference_damp,  # EN: Helper to pick damp_ref for fixed-R reuse.
    choose_sketch_rows,  # EN: Helper to choose CountSketch row count s.
    k_fold_assignments,  # EN: Fold-id assignment helper (O(m) memory, avoids storing k splits).
    print_cv_table,  # EN: Pretty-print CV curve tables.
    print_separator,  # EN: Print section separators.
    print_solver_table,  # EN: Print per-damp solver comparison tables.
    randqr_R_from_countsketch,  # EN: Build R from QR of the sketched augmented matrix.
    solve_one_operator,  # EN: Matrix-free ridge solver core (LSMR teaching variant).
    upper_triangular_preconditioner_from_R,  # EN: Turn R into right-preconditioner closures.
)  # EN: End import list.


MASK64 = (1 << 64) - 1  # EN: Bitmask to keep integers in unsigned 64-bit range.
SM64_GOLDEN = 0x9E3779B97F4A7C15  # EN: SplitMix64 increment constant (2^64 / golden ratio).
SM64_M1 = 0xBF58476D1CE4E5B9  # EN: SplitMix64 mixing constant 1.
SM64_M2 = 0x94D049BB133111EB  # EN: SplitMix64 mixing constant 2.
ROW_STRIDE = 0xD6E8FEB86659FD93  # EN: Extra stride constant to decorrelate (row,k) hashing.
NNZ_STRIDE = 0xA5A3564E27F5A9B1  # EN: Extra stride constant for per-nonzero hashing.


def splitmix64(x: int) -> int:  # EN: Deterministic 64-bit mixing function (fast hash / PRNG primitive).
    z = (int(x) + SM64_GOLDEN) & MASK64  # EN: Add the increment constant and wrap to 64 bits.
    z = (z ^ (z >> 30)) * SM64_M1 & MASK64  # EN: First xor-shift-multiply mix.
    z = (z ^ (z >> 27)) * SM64_M2 & MASK64  # EN: Second xor-shift-multiply mix.
    z = (z ^ (z >> 31)) & MASK64  # EN: Final xor-shift and wrap.
    return int(z)  # EN: Return as a Python int.


def hash_to_signed_value(h: int, scale: float) -> float:  # EN: Map a 64-bit hash to a reproducible signed float value.
    sign = 1.0 if (int(h) & 1) == 1 else -1.0  # EN: Use the lowest bit as a random sign.
    u53 = (int(h) >> 11) & ((1 << 53) - 1)  # EN: Take 53 bits to build a uniform float mantissa.
    u = float(u53) / float(1 << 53)  # EN: Convert to U[0,1) in double precision.
    mag = 0.5 + u  # EN: Keep magnitude in [0.5,1.5) to avoid near-zero values.
    return float(scale) * float(sign) * float(mag)  # EN: Return a scaled signed value.


@dataclass(frozen=True)  # EN: Immutable description of an implicit sparse design matrix.
class ImplicitHashedSparseMatrix:  # EN: Generate sparse rows on-the-fly via hashing (no CSR storage).
    m: int  # EN: Number of rows (samples).
    n: int  # EN: Number of columns (features).
    nnz_per_row: int  # EN: Non-zeros per row (sparsity level).
    seed_u64: int  # EN: 64-bit seed for deterministic row generation.
    val_scale: float  # EN: Per-entry value scaling (keeps row norms reasonable).


def make_implicit_sparse_matrix(m: int, n: int, nnz_per_row: int, seed: int) -> ImplicitHashedSparseMatrix:  # EN: Factory for an implicit sparse matrix with sensible scaling.
    if m <= 0 or n <= 0:  # EN: Validate dimensions.
        raise ValueError("m and n must be positive")  # EN: Reject invalid shapes.
    if nnz_per_row <= 0 or nnz_per_row > n:  # EN: Validate sparsity level.
        raise ValueError("nnz_per_row must be in [1, n]")  # EN: Reject invalid nnz_per_row.
    seed_u64 = int(seed) & MASK64  # EN: Normalize seed into unsigned 64-bit space.
    val_scale = float(1.0 / np.sqrt(max(int(nnz_per_row), 1)))  # EN: Scale values so expected row norms stay O(1).
    return ImplicitHashedSparseMatrix(m=int(m), n=int(n), nnz_per_row=int(nnz_per_row), seed_u64=int(seed_u64), val_scale=float(val_scale))  # EN: Return matrix descriptor.


def implicit_row_dot(A: ImplicitHashedSparseMatrix, row_id: int, x: np.ndarray) -> float:  # EN: Compute (A[row_id,:] @ x) without materializing the row.
    n = int(A.n)  # EN: Cache n locally for speed.
    nnz = int(A.nnz_per_row)  # EN: Cache nnz_per_row locally for speed.
    seed = int(A.seed_u64)  # EN: Cache seed locally for speed.
    acc = 0.0  # EN: Accumulate the sparse dot product.
    base = (seed ^ (int(row_id) * ROW_STRIDE)) & MASK64  # EN: Base hash state for this row.
    for k in range(nnz):  # EN: Loop synthetic non-zeros in this row.
        h = splitmix64((base + (int(k) * NNZ_STRIDE)) & MASK64)  # EN: Deterministically derive a hash for (row,k).
        col = int(h % n)  # EN: Map hash to a column index in [0, n).
        val = hash_to_signed_value(h=h, scale=float(A.val_scale))  # EN: Map hash to a signed float value.
        acc += float(val) * float(x[col])  # EN: Accumulate val * x[col].
    return float(acc)  # EN: Return dot product result.


def implicit_row_axpy_to(A: ImplicitHashedSparseMatrix, row_id: int, weight: float, out: np.ndarray) -> None:  # EN: Accumulate out += weight * A[row_id,:]^T (scatter-add).
    n = int(A.n)  # EN: Cache n locally.
    nnz = int(A.nnz_per_row)  # EN: Cache nnz_per_row locally.
    seed = int(A.seed_u64)  # EN: Cache seed locally.
    w = float(weight)  # EN: Ensure weight is float.
    base = (seed ^ (int(row_id) * ROW_STRIDE)) & MASK64  # EN: Base hash state for this row.
    for k in range(nnz):  # EN: Loop synthetic non-zeros in this row.
        h = splitmix64((base + (int(k) * NNZ_STRIDE)) & MASK64)  # EN: Deterministically derive a hash for (row,k).
        col = int(h % n)  # EN: Column index.
        val = hash_to_signed_value(h=h, scale=float(A.val_scale))  # EN: Entry value.
        out[col] += float(val) * w  # EN: Scatter-add into out at the chosen column.


def implicit_matvec_rows(A: ImplicitHashedSparseMatrix, row_ids: np.ndarray, x: np.ndarray) -> np.ndarray:  # EN: Compute y = A[row_ids,:] x for a selected row set.
    x1 = np.asarray(x, dtype=float).reshape(-1)  # EN: Normalize x to 1D float array.
    if x1.size != int(A.n):  # EN: Validate x length.
        raise ValueError("x has incompatible dimension")  # EN: Reject dimension mismatch.
    rows = np.asarray(row_ids, dtype=int).reshape(-1)  # EN: Normalize row ids to a 1D int array.
    y = np.zeros((int(rows.size),), dtype=float)  # EN: Allocate output vector.
    for i, rid in enumerate(rows):  # EN: Loop selected rows in order.
        y[i] = implicit_row_dot(A=A, row_id=int(rid), x=x1)  # EN: Compute dot for this row.
    return y.astype(float)  # EN: Return y as float.


def implicit_rmatvec_rows(A: ImplicitHashedSparseMatrix, row_ids: np.ndarray, y: np.ndarray) -> np.ndarray:  # EN: Compute x = A[row_ids,:]^T y for a selected row set.
    rows = np.asarray(row_ids, dtype=int).reshape(-1)  # EN: Normalize row ids to 1D int array.
    y1 = np.asarray(y, dtype=float).reshape(-1)  # EN: Normalize y to 1D float array.
    if y1.size != int(rows.size):  # EN: Validate y length.
        raise ValueError("y has incompatible dimension")  # EN: Reject dimension mismatch.
    x = np.zeros((int(A.n),), dtype=float)  # EN: Allocate output vector in R^n.
    for i, rid in enumerate(rows):  # EN: Loop selected rows in order.
        weight = float(y1[i])  # EN: Weight for this row.
        if weight == 0.0:  # EN: Skip zero weights to save work.
            continue  # EN: Move to next row.
        implicit_row_axpy_to(A=A, row_id=int(rid), weight=weight, out=x)  # EN: Accumulate A[row]^T * weight.
    return x.astype(float)  # EN: Return x as float.


def implicit_col_norms_sq_and_fro_sq_rows(A: ImplicitHashedSparseMatrix, row_ids: np.ndarray) -> tuple[np.ndarray, float]:  # EN: Compute column norm squares and Frobenius^2 for selected rows.
    rows = np.asarray(row_ids, dtype=int).reshape(-1)  # EN: Normalize row ids.
    n = int(A.n)  # EN: Feature count.
    nnz = int(A.nnz_per_row)  # EN: Non-zeros per row.
    seed = int(A.seed_u64)  # EN: Seed.
    col_sq = np.zeros((n,), dtype=float)  # EN: Allocate accumulator for column squared norms.
    fro_sq = 0.0  # EN: Accumulate Frobenius norm squared.
    for rid in rows:  # EN: Loop selected rows.
        base = (seed ^ (int(rid) * ROW_STRIDE)) & MASK64  # EN: Base hash for this row.
        for k in range(nnz):  # EN: Loop synthetic non-zeros in the row.
            h = splitmix64((base + (int(k) * NNZ_STRIDE)) & MASK64)  # EN: Hash for (row,k).
            col = int(h % n)  # EN: Column index.
            val = hash_to_signed_value(h=h, scale=float(A.val_scale))  # EN: Entry value.
            sq = float(val) * float(val)  # EN: Square value for norm contributions.
            fro_sq += sq  # EN: Add to Frobenius^2.
            col_sq[col] += sq  # EN: Add to column squared norm.
    return col_sq.astype(float), float(fro_sq)  # EN: Return (col_sq, fro_sq).


def build_countsketch_aug_implicit_rows(  # EN: Build CountSketchAug for [A_rows; damp I] using only row generation.
    A: ImplicitHashedSparseMatrix,  # EN: Implicit sparse matrix descriptor.
    row_ids: np.ndarray,  # EN: Selected row ids (subset).
    sketch_factor: float,  # EN: Oversampling factor for sketch rows (e.g., 4.0).
    rng: np.random.Generator,  # EN: RNG for sketch hashing/signs.
) -> CountSketchAug:  # EN: Return CountSketchAug with SA_top and identity hashing.
    rows = np.asarray(row_ids, dtype=int).reshape(-1)  # EN: Normalize row ids.
    m_sub = int(rows.size)  # EN: Row count in the subset.
    n = int(A.n)  # EN: Feature count.
    m_aug = int(m_sub + n)  # EN: Augmented row count for [A_sub; damp I].
    s = choose_sketch_rows(m_aug=m_aug, n=int(n), sketch_factor=float(sketch_factor))  # EN: Choose sketch row count.
    scale = float(1.0 / np.sqrt(max(int(s), 1)))  # EN: Scale factor (keeps norms comparable).

    h_top = rng.integers(low=0, high=int(s), size=int(m_sub), dtype=int)  # EN: Hash bucket for each selected data row.
    sign_top = rng.choice(np.array([-1.0, 1.0]), size=int(m_sub)).astype(float)  # EN: Random sign for each selected data row.
    h_bottom = rng.integers(low=0, high=int(s), size=int(n), dtype=int)  # EN: Hash bucket for each identity row.
    sign_bottom = rng.choice(np.array([-1.0, 1.0]), size=int(n)).astype(float)  # EN: Random sign for each identity row.

    SA_top = np.zeros((int(s), int(n)), dtype=float)  # EN: Allocate dense sketch matrix for S_top A_sub.

    nnz = int(A.nnz_per_row)  # EN: Cache nnz_per_row.
    seed = int(A.seed_u64)  # EN: Cache seed.
    for i, rid in enumerate(rows):  # EN: Loop subset rows in order.
        bucket = int(h_top[i])  # EN: Sketch row index for this subset row.
        sgn = float(sign_top[i])  # EN: Random sign for this subset row.
        base = (seed ^ (int(rid) * ROW_STRIDE)) & MASK64  # EN: Base hash for this row.
        for k in range(nnz):  # EN: Loop synthetic non-zeros in the row.
            h = splitmix64((base + (int(k) * NNZ_STRIDE)) & MASK64)  # EN: Hash for (row,k).
            col = int(h % n)  # EN: Column index.
            val = hash_to_signed_value(h=h, scale=float(A.val_scale))  # EN: Entry value.
            SA_top[bucket, col] += (scale * sgn) * float(val)  # EN: Accumulate into the sketch matrix.

    return CountSketchAug(  # EN: Package sketch container.
        SA_top=SA_top.astype(float),  # EN: Store SA_top.
        h_bottom=h_bottom.astype(int),  # EN: Store identity hashes.
        sign_bottom=sign_bottom.astype(float),  # EN: Store identity signs.
        scale=float(scale),  # EN: Store scaling factor.
        s=int(s),  # EN: Store sketch row count.
    )  # EN: End return.


def rmse_on_rows(A: ImplicitHashedSparseMatrix, row_ids: np.ndarray, x: np.ndarray, b_full: np.ndarray) -> float:  # EN: Compute RMSE on a row subset without storing predictions.
    rows = np.asarray(row_ids, dtype=int).reshape(-1)  # EN: Normalize row ids.
    if rows.size == 0:  # EN: Handle empty validation fold (should not happen, but be defensive).
        return 0.0  # EN: Define RMSE as 0 for empty set.
    x1 = np.asarray(x, dtype=float).reshape(-1)  # EN: Normalize x to 1D float array.
    b1 = np.asarray(b_full, dtype=float).reshape(-1)  # EN: Normalize b to 1D float array.
    se = 0.0  # EN: Accumulate squared error.
    for rid in rows:  # EN: Loop validation rows.
        pred = implicit_row_dot(A=A, row_id=int(rid), x=x1)  # EN: Compute prediction for this row.
        err = float(pred) - float(b1[int(rid)])  # EN: Compute prediction error.
        se += err * err  # EN: Accumulate squared error.
    return float(np.sqrt(se / float(rows.size)))  # EN: Return RMSE.


Matvec = Callable[[np.ndarray], np.ndarray]  # EN: Alias for a matvec function type.


def cv_sweep_curve_operator_only(  # EN: Sweep a damp grid with k-fold CV using an implicit sparse operator (no CSR storage).
    fold_ids: np.ndarray,  # EN: Per-sample fold assignment array (length m).
    n_folds: int,  # EN: Number of folds (k).
    A: ImplicitHashedSparseMatrix,  # EN: Implicit sparse matrix descriptor.
    b_full: np.ndarray,  # EN: Full targets vector (length m).
    damps: np.ndarray,  # EN: Damp grid to sweep.
    precond_kind: PrecondKind,  # EN: Preconditioner kind.
    randqr_policy: RandQRPolicy,  # EN: rand-QR reuse policy (ignored unless precond_kind == "randqr").
    warm_start: bool,  # EN: Whether to use continuation/warm-start across the damp grid within each fold.
    max_iters: int,  # EN: Iteration cap.
    atol: float,  # EN: Absolute tolerance.
    btol: float,  # EN: Relative tolerance.
    sketch_factor: float,  # EN: Oversampling factor for rand-QR sketches.
    rng: np.random.Generator,  # EN: RNG for sketch construction.
) -> tuple[list[CVPoint], CVTotals, CVPoint]:  # EN: Return (points, totals, best_point).
    damps_sorted = np.array(sorted([float(d) for d in damps]), dtype=float)  # EN: Sort damps for reporting.
    n_damps = int(damps_sorted.size)  # EN: Number of damp values.
    n_folds_i = int(n_folds)  # EN: Normalize n_folds to int.
    if n_folds_i < 2:  # EN: Require at least 2 folds.
        raise ValueError("n_folds must be >= 2")  # EN: Reject invalid fold count.

    m = int(A.m)  # EN: Sample count.
    n = int(A.n)  # EN: Feature count.
    fold_ids_1d = np.asarray(fold_ids, dtype=int).reshape(-1)  # EN: Normalize fold ids to 1D int array.
    if fold_ids_1d.size != m:  # EN: Validate fold id length.
        raise ValueError("fold_ids must have length m")  # EN: Reject mismatched fold ids.
    if np.any(fold_ids_1d < 0) or np.any(fold_ids_1d >= n_folds_i):  # EN: Validate fold id range.
        raise ValueError("fold_ids entries must be in [0, n_folds)")  # EN: Reject invalid assignments.

    b1 = np.asarray(b_full, dtype=float).reshape(-1)  # EN: Normalize b to 1D float array.
    if b1.size != m:  # EN: Validate b length.
        raise ValueError("b_full must have length m")  # EN: Reject invalid b.

    order = np.arange(n_damps, dtype=int)  # EN: Default evaluation order is ascending.
    if warm_start:  # EN: For continuation, start from large damp (solution closer to 0).
        order = order[::-1]  # EN: Evaluate from largest damp down to smallest.

    train_rmse_mat = np.zeros((n_damps, n_folds_i), dtype=float)  # EN: Store train RMSE per (damp, fold).
    val_rmse_mat = np.zeros((n_damps, n_folds_i), dtype=float)  # EN: Store val RMSE per (damp, fold).
    xnorm_mat = np.zeros((n_damps, n_folds_i), dtype=float)  # EN: Store ||x|| per (damp, fold).
    iters_mat = np.zeros((n_damps, n_folds_i), dtype=float)  # EN: Store iterations per (damp, fold).

    total_build = 0.0  # EN: Accumulate preconditioner build time across all fits.
    total_solve = 0.0  # EN: Accumulate solver time across all fits.
    total_iters = 0  # EN: Accumulate iterations across all fits.

    if precond_kind == "none":  # EN: Baseline label.
        label = "none"  # EN: Label string.
    elif precond_kind == "col":  # EN: Column scaling label.
        label = "col-scaling"  # EN: Label string.
    else:  # EN: rand-QR label depends on policy.
        policy_tag = {"rebuild": "rand-QR(rebuild)", "shared_sketch": "rand-QR(shared)", "fixed_R": "rand-QR(fixed-R)"}[randqr_policy]  # EN: Map policy to a label.
        label = policy_tag  # EN: Use policy label.
    if warm_start:  # EN: Annotate label when warm-start is enabled.
        label = f"{label}+ws"  # EN: Append warm-start suffix.

    damp_ref = choose_fixed_randqr_reference_damp(damps_sorted)  # EN: Reference damp for fixed-R policy.

    # EN: Build an int32 "all rows" index array so boolean masking does not create int64 indices (reduces memory peaks).  # EN: Explain indexing.
    row_dtype = np.int32 if m <= int(np.iinfo(np.int32).max) else np.int64  # EN: Choose smallest safe integer dtype.
    all_rows = np.arange(m, dtype=row_dtype)  # EN: Create [0,1,...,m-1] in compact dtype.

    for fold_id in range(n_folds_i):  # EN: Loop folds so continuation happens within each fold.
        is_val = (fold_ids_1d == int(fold_id))  # EN: Boolean mask for validation rows.
        val_rows = all_rows[is_val]  # EN: Validation row ids (compact dtype).
        train_rows = all_rows[~is_val]  # EN: Training row ids (compact dtype).

        m_tr = int(train_rows.size)  # EN: Training sample count.
        b_tr = b1[train_rows]  # EN: Training targets (copy; solver needs a dense RHS vector).

        col_sq_tr, fro_sq_tr = implicit_col_norms_sq_and_fro_sq_rows(A=A, row_ids=train_rows)  # EN: Compute training col norms + ||A_tr||_F^2.

        def matvec_A_tr(x: np.ndarray, A: ImplicitHashedSparseMatrix = A, rows: np.ndarray = train_rows) -> np.ndarray:  # EN: Training matvec closure.
            return implicit_matvec_rows(A=A, row_ids=rows, x=x)  # EN: Compute A_tr x.

        def matvec_AT_tr(y: np.ndarray, A: ImplicitHashedSparseMatrix = A, rows: np.ndarray = train_rows) -> np.ndarray:  # EN: Training transpose-matvec closure.
            return implicit_rmatvec_rows(A=A, row_ids=rows, y=y)  # EN: Compute A_tr^T y.

        x_prev = np.zeros((n,), dtype=float)  # EN: Initialize continuation state with x=0.
        shared_sketch: CountSketchAug | None = None  # EN: Shared sketch for rand-QR reuse within this fold.
        fixed_precond: tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]] | None = None  # EN: Fixed-R preconditioner closures.

        if precond_kind == "randqr" and randqr_policy in {"shared_sketch", "fixed_R"}:  # EN: Build shared CountSketch once per fold.
            t0 = perf_counter()  # EN: Start timing sketch build.
            shared_sketch = build_countsketch_aug_implicit_rows(A=A, row_ids=train_rows, sketch_factor=float(sketch_factor), rng=rng)  # EN: Build SA_top for training rows.
            total_build += float(perf_counter() - t0)  # EN: Charge one-time sketch build cost.

        if precond_kind == "randqr" and randqr_policy == "fixed_R":  # EN: Build a fixed R once per fold.
            if shared_sketch is None:  # EN: Defensive check.
                raise RuntimeError("internal error: shared_sketch is None for fixed_R policy")  # EN: Fail fast.
            t0 = perf_counter()  # EN: Start timing QR build.
            R_ref = randqr_R_from_countsketch(sketch=shared_sketch, damp=float(damp_ref))  # EN: Build R at damp_ref.
            total_build += float(perf_counter() - t0)  # EN: Charge one-time QR cost.
            fixed_precond = upper_triangular_preconditioner_from_R(R=R_ref, label="rand-QR(fixed-R)")  # EN: Build apply closures.

        for step, idx in enumerate(order):  # EN: Sweep damps in the chosen continuation order.
            damp = float(damps_sorted[int(idx)])  # EN: Current damp value.
            x_init = x_prev if (warm_start and step > 0) else None  # EN: Use previous x as warm-start after the first step.

            if precond_kind != "randqr":  # EN: none/col paths do not need an explicit sketch build.
                rep = solve_one_operator(  # EN: Solve using the shared matrix-free solver.
                    m=int(m_tr),  # EN: Training row count.
                    n=int(n),  # EN: Feature count.
                    matvec_A=matvec_A_tr,  # EN: Training matvec.
                    matvec_AT=matvec_AT_tr,  # EN: Training transpose matvec.
                    col_sq=col_sq_tr,  # EN: Column squared norms for training operator.
                    fro_sq=float(fro_sq_tr),  # EN: Frobenius^2 for norm estimate.
                    b=b_tr,  # EN: Training targets.
                    damp=float(damp),  # EN: Damp.
                    precond_kind=precond_kind,  # EN: none or col.
                    max_iters=max_iters,  # EN: Iteration cap.
                    atol=atol,  # EN: atol.
                    btol=btol,  # EN: btol.
                    rng=rng,  # EN: RNG (unused by deterministic kinds).
                    x_init=x_init,  # EN: Warm-start (optional).
                    precond_override=None,  # EN: Build preconditioner internally.
                    build_seconds_override=None,  # EN: No override.
                    A_for_precond=None,  # EN: No CSR needed for none/col.
                )  # EN: End solve.
            else:  # EN: rand-QR uses CountSketch+QR builds, with optional reuse policies.
                if randqr_policy == "rebuild":  # EN: Baseline: rebuild sketch + QR for every damp.
                    t0 = perf_counter()  # EN: Time sketch+QR build.
                    sketch = build_countsketch_aug_implicit_rows(A=A, row_ids=train_rows, sketch_factor=float(sketch_factor), rng=rng)  # EN: Build CountSketch for this damp.
                    R = randqr_R_from_countsketch(sketch=sketch, damp=float(damp))  # EN: Build R from QR of sketched augmented matrix.
                    build_s = float(perf_counter() - t0)  # EN: Total build seconds for this fit.
                    precond = upper_triangular_preconditioner_from_R(R=R, label="rand-QR(rebuild)")  # EN: Create apply closures.
                    rep = solve_one_operator(  # EN: Solve with externally built preconditioner.
                        m=int(m_tr),  # EN: Training row count.
                        n=int(n),  # EN: Feature count.
                        matvec_A=matvec_A_tr,  # EN: Training matvec.
                        matvec_AT=matvec_AT_tr,  # EN: Training transpose matvec.
                        col_sq=col_sq_tr,  # EN: Column norms (kept for consistency).
                        fro_sq=float(fro_sq_tr),  # EN: Frobenius^2.
                        b=b_tr,  # EN: Training targets.
                        damp=float(damp),  # EN: Damp.
                        precond_kind="randqr",  # EN: Ignored because we pass precond_override.
                        max_iters=max_iters,  # EN: Iter cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        rng=rng,  # EN: RNG (unused due to override).
                        x_init=x_init,  # EN: Warm-start (optional).
                        precond_override=precond,  # EN: Use the prebuilt preconditioner.
                        build_seconds_override=float(build_s),  # EN: Charge full build time.
                        A_for_precond=None,  # EN: No CSR needed because precond is prebuilt.
                    )  # EN: End solve.
                elif randqr_policy == "shared_sketch":  # EN: Rebuild R per damp but reuse the expensive sketch SA_top.
                    if shared_sketch is None:  # EN: Defensive check.
                        raise RuntimeError("internal error: shared_sketch is None for shared_sketch policy")  # EN: Fail fast.
                    t0 = perf_counter()  # EN: Time only the QR step (sketch already built).
                    R = randqr_R_from_countsketch(sketch=shared_sketch, damp=float(damp))  # EN: Build R from shared sketch.
                    build_s = float(perf_counter() - t0)  # EN: Per-damp QR build time.
                    precond = upper_triangular_preconditioner_from_R(R=R, label="rand-QR(shared)")  # EN: Create apply closures.
                    rep = solve_one_operator(  # EN: Solve with externally built preconditioner.
                        m=int(m_tr),  # EN: Training row count.
                        n=int(n),  # EN: Feature count.
                        matvec_A=matvec_A_tr,  # EN: Training matvec.
                        matvec_AT=matvec_AT_tr,  # EN: Training transpose matvec.
                        col_sq=col_sq_tr,  # EN: Column norms.
                        fro_sq=float(fro_sq_tr),  # EN: Frobenius^2.
                        b=b_tr,  # EN: Training targets.
                        damp=float(damp),  # EN: Damp.
                        precond_kind="randqr",  # EN: Ignored because we pass precond_override.
                        max_iters=max_iters,  # EN: Iter cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        rng=rng,  # EN: RNG (unused due to override).
                        x_init=x_init,  # EN: Warm-start (optional).
                        precond_override=precond,  # EN: Use the prebuilt preconditioner.
                        build_seconds_override=float(build_s),  # EN: Charge only QR time (sketch cost charged once per fold).
                        A_for_precond=None,  # EN: No CSR needed because precond is prebuilt.
                    )  # EN: End solve.
                elif randqr_policy == "fixed_R":  # EN: Reuse the same R for all damps in the curve.
                    if fixed_precond is None:  # EN: Defensive check.
                        raise RuntimeError("internal error: fixed_precond is None for fixed_R policy")  # EN: Fail fast.
                    rep = solve_one_operator(  # EN: Solve with reused fixed-R preconditioner.
                        m=int(m_tr),  # EN: Training row count.
                        n=int(n),  # EN: Feature count.
                        matvec_A=matvec_A_tr,  # EN: Training matvec.
                        matvec_AT=matvec_AT_tr,  # EN: Training transpose matvec.
                        col_sq=col_sq_tr,  # EN: Column norms.
                        fro_sq=float(fro_sq_tr),  # EN: Frobenius^2.
                        b=b_tr,  # EN: Training targets.
                        damp=float(damp),  # EN: Damp (objective changes; only M is fixed).
                        precond_kind="randqr",  # EN: Ignored because we pass precond_override.
                        max_iters=max_iters,  # EN: Iter cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        rng=rng,  # EN: RNG (unused due to override).
                        x_init=x_init,  # EN: Warm-start (optional).
                        precond_override=fixed_precond,  # EN: Use fixed-R closures.
                        build_seconds_override=0.0,  # EN: No per-damp build cost.
                        A_for_precond=None,  # EN: No CSR needed because precond is prebuilt.
                    )  # EN: End solve.
                else:  # EN: Guard against unknown policy strings.
                    raise ValueError("Unknown randqr_policy")  # EN: Reject invalid policy.

            x_prev = rep.x_hat  # EN: Update continuation state for the next damp.

            train_rmse = float(rep.rnorm_data) / float(np.sqrt(max(m_tr, 1)))  # EN: Compute train RMSE from solver residual norm.
            val_rmse = rmse_on_rows(A=A, row_ids=val_rows, x=rep.x_hat, b_full=b1)  # EN: Compute validation RMSE streaming (no val_pred stored).

            train_rmse_mat[int(idx), fold_id] = float(train_rmse)  # EN: Store train RMSE.
            val_rmse_mat[int(idx), fold_id] = float(val_rmse)  # EN: Store val RMSE.
            xnorm_mat[int(idx), fold_id] = float(rep.xnorm)  # EN: Store ||x||.
            iters_mat[int(idx), fold_id] = float(rep.n_iters)  # EN: Store iterations.

            total_build += float(rep.build_seconds)  # EN: Accumulate build time.
            total_solve += float(rep.solve_seconds)  # EN: Accumulate solve time.
            total_iters += int(rep.n_iters)  # EN: Accumulate iterations.

    points: list[CVPoint] = []  # EN: Collect per-damp CV summaries.
    for i, damp in enumerate(damps_sorted):  # EN: Summarize each damp across folds.
        train_row = train_rmse_mat[i, :]  # EN: Train RMSE values across folds.
        val_row = val_rmse_mat[i, :]  # EN: Val RMSE values across folds.
        x_row = xnorm_mat[i, :]  # EN: ||x|| values across folds.
        it_row = iters_mat[i, :]  # EN: Iteration counts across folds.

        key = f"d={float(damp):.0e}" if float(damp) != 0.0 else "d=0"  # EN: Format parameter label.
        points.append(  # EN: Append a CVPoint for this damp.
            CVPoint(
                key=str(key),  # EN: Label.
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


def main() -> None:  # EN: Run the operator-only sparse ridge demo.
    rng = np.random.default_rng(SEED)  # EN: Deterministic RNG for reproducible experiments.

    # EN: Problem sizes are kept moderate so the teaching demo runs quickly, but the memory pattern matches huge-m pipelines.  # EN: Explain defaults.
    m = 2000  # EN: Sample count.
    n = 200  # EN: Feature count.
    nnz_per_row = 10  # EN: Sparse nnz per row (hash-generated).
    noise_std = 0.05  # EN: Noise level for b.

    A = make_implicit_sparse_matrix(m=m, n=n, nnz_per_row=nnz_per_row, seed=SEED + 123)  # EN: Build implicit sparse operator (no CSR arrays).
    density = float(nnz_per_row) / float(n)  # EN: Expected density per row in this model.

    print_separator("Operator-only sparse dataset summary (A is not stored)")  # EN: Announce dataset summary.
    print(f"m={m}, n={n}, nnz_per_row={nnz_per_row}, expected_density={density:.3e}")  # EN: Print basic stats.

    # EN: Create a sparse-ish ground-truth x_true and synthetic targets b = A x_true + noise.  # EN: Explain target generation.
    x_true = np.zeros((n,), dtype=float)  # EN: Initialize x_true.
    support = rng.choice(n, size=12, replace=False)  # EN: Choose sparse support set.
    x_true[support] = rng.standard_normal(support.size).astype(float)  # EN: Fill support with random values.

    all_rows = np.arange(m, dtype=int)  # EN: Row ids for full-data matvec.
    b_clean = implicit_matvec_rows(A=A, row_ids=all_rows, x=x_true)  # EN: Compute noiseless targets b_clean = A x_true.
    b = b_clean + float(noise_std) * rng.standard_normal((m,)).astype(float)  # EN: Add Gaussian noise.

    # EN: Solver settings are matched to the CSR-based unit to keep comparisons consistent.  # EN: Explain solver settings.
    max_iters = min(2 * n, 150)  # EN: Iteration cap.
    atol = 1e-10  # EN: Absolute tolerance.
    btol = 1e-10  # EN: Relative tolerance.
    sketch_factor = 4.0  # EN: Oversampling factor for rand-QR.

    # EN: Compute full-data column norms and Frobenius^2 once (used for col-scaling and norm estimates).  # EN: Explain preprocessing.
    col_sq_full, fro_sq_full = implicit_col_norms_sq_and_fro_sq_rows(A=A, row_ids=all_rows)  # EN: Compute col norms + ||A||_F^2.

    def matvec_A_full(x: np.ndarray, A: ImplicitHashedSparseMatrix = A, rows: np.ndarray = all_rows) -> np.ndarray:  # EN: Full-data matvec closure.
        return implicit_matvec_rows(A=A, row_ids=rows, x=x)  # EN: Compute A x.

    def matvec_AT_full(y: np.ndarray, A: ImplicitHashedSparseMatrix = A, rows: np.ndarray = all_rows) -> np.ndarray:  # EN: Full-data transpose matvec closure.
        return implicit_rmatvec_rows(A=A, row_ids=rows, y=y)  # EN: Compute A^T y.

    # EN: Quick per-damp solver comparison (full data) for none/col/rand-QR.  # EN: Explain purpose.
    demo_damps = [0.0, 1e-2, 1e-1, 1.0]  # EN: Damp values to compare.
    preconds: list[PrecondKind] = ["none", "col", "randqr"]  # EN: Preconditioner list.
    rng_demo = np.random.default_rng(SEED + 10)  # EN: Dedicated RNG for rand-QR sketches.

    print_separator("Per-damp solver comparison (operator-only, full data)")  # EN: Announce solver comparison section.
    reports: list[SolveReport] = []  # EN: Collect reports.
    for d in demo_damps:  # EN: Loop damps.
        for pk in preconds:  # EN: Loop preconditioners.
            if pk != "randqr":  # EN: none/col can be built internally (no CSR needed).
                reports.append(  # EN: Append report.
                    solve_one_operator(  # EN: Solve one configuration.
                        m=int(m),  # EN: Row count.
                        n=int(n),  # EN: Feature count.
                        matvec_A=matvec_A_full,  # EN: Matvec A.
                        matvec_AT=matvec_AT_full,  # EN: Matvec A^T.
                        col_sq=col_sq_full,  # EN: Column norms.
                        fro_sq=float(fro_sq_full),  # EN: Frobenius^2.
                        b=b,  # EN: Targets.
                        damp=float(d),  # EN: Damp.
                        precond_kind=pk,  # EN: none or col.
                        max_iters=max_iters,  # EN: Iter cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        rng=rng_demo,  # EN: RNG (unused for deterministic kinds).
                        x_init=None,  # EN: Cold-start.
                        precond_override=None,  # EN: No override.
                        build_seconds_override=None,  # EN: No override.
                        A_for_precond=None,  # EN: No CSR needed.
                    )
                )
            else:  # EN: rand-QR must be built externally from a CountSketch (since we have no CSR to pass in).
                t0 = perf_counter()  # EN: Time sketch+QR build.
                sketch = build_countsketch_aug_implicit_rows(A=A, row_ids=all_rows, sketch_factor=float(sketch_factor), rng=rng_demo)  # EN: Build CountSketch once per (damp,run).
                R = randqr_R_from_countsketch(sketch=sketch, damp=float(d))  # EN: Build R for this damp.
                build_s = float(perf_counter() - t0)  # EN: Record build time.
                precond = upper_triangular_preconditioner_from_R(R=R, label="rand-QR(countsketch)")  # EN: Create apply closures.
                reports.append(  # EN: Append report.
                    solve_one_operator(  # EN: Solve with prebuilt preconditioner.
                        m=int(m),  # EN: Row count.
                        n=int(n),  # EN: Feature count.
                        matvec_A=matvec_A_full,  # EN: Matvec A.
                        matvec_AT=matvec_AT_full,  # EN: Matvec A^T.
                        col_sq=col_sq_full,  # EN: Column norms.
                        fro_sq=float(fro_sq_full),  # EN: Frobenius^2.
                        b=b,  # EN: Targets.
                        damp=float(d),  # EN: Damp.
                        precond_kind="randqr",  # EN: Ignored due to override.
                        max_iters=max_iters,  # EN: Iter cap.
                        atol=atol,  # EN: atol.
                        btol=btol,  # EN: btol.
                        rng=rng_demo,  # EN: RNG (unused due to override).
                        x_init=None,  # EN: Cold-start.
                        precond_override=precond,  # EN: Use prebuilt preconditioner.
                        build_seconds_override=float(build_s),  # EN: Charge build time.
                        A_for_precond=None,  # EN: No CSR needed.
                    )
                )
    print_solver_table(reports)  # EN: Print table.

    # EN: Build fold assignments once so baseline/speedups compare on identical folds.  # EN: Explain split policy.
    n_folds = 5  # EN: Number of folds for CV.
    rng_split = np.random.default_rng(SEED + 20)  # EN: RNG for fold assignment.
    fold_ids = k_fold_assignments(rng=rng_split, n_samples=m, n_folds=n_folds)  # EN: Per-sample fold ids.

    damps = np.concatenate(([0.0], np.logspace(-6, 1, num=12)))  # EN: Candidate damp grid.

    print_separator("k-fold CV sweep (baseline, operator-only): rebuild per damp, no warm-start")  # EN: Announce baseline CV.
    rng_cv_baseline = np.random.default_rng(SEED + 30)  # EN: RNG stream for baseline CV.
    baseline_rows: dict[str, tuple[str, float, float, float, float, int]] = {}  # EN: Map precond -> summary tuple.

    for pk in preconds:  # EN: Run baseline CV for each preconditioner.
        points, totals, best = cv_sweep_curve_operator_only(  # EN: Run CV sweep.
            fold_ids=fold_ids,  # EN: Fold ids.
            n_folds=n_folds,  # EN: Fold count.
            A=A,  # EN: Implicit operator.
            b_full=b,  # EN: Targets.
            damps=damps,  # EN: Damp grid.
            precond_kind=pk,  # EN: Preconditioner.
            randqr_policy="rebuild",  # EN: Baseline policy (ignored unless pk=="randqr").
            warm_start=False,  # EN: No continuation.
            max_iters=max_iters,  # EN: Iter cap.
            atol=atol,  # EN: atol.
            btol=btol,  # EN: btol.
            sketch_factor=sketch_factor,  # EN: Sketch factor.
            rng=rng_cv_baseline,  # EN: RNG stream.
        )  # EN: End sweep.

        print_separator(f"CV results (baseline, operator-only): {totals.precond}")  # EN: Announce per-preconditioner results.
        print_cv_table(points=points, best=best)  # EN: Print CV curve.
        total_seconds = float(totals.total_build_seconds + totals.total_solve_seconds)  # EN: Total curve cost.
        print(  # EN: Print cost summary.
            f"\nTotal curve cost (baseline, {totals.precond}): build={totals.total_build_seconds:.3f}s, solve={totals.total_solve_seconds:.3f}s, total={total_seconds:.3f}s, iters={totals.total_iters}"  # EN: Summary line.
        )  # EN: End print.
        baseline_rows[str(pk)] = (totals.precond, float(best.damp), float(best.val_mean), float(totals.total_build_seconds), float(totals.total_solve_seconds), int(totals.total_iters))  # EN: Store summary row.

    print_separator("CV sweep speedups (operator-only): warm-start + rand-QR shared sketch")  # EN: Announce speedup CV.
    rng_cv_speed = np.random.default_rng(SEED + 40)  # EN: RNG stream for speedups.
    speed_rows: dict[str, tuple[str, float, float, float, float, int]] = {}  # EN: Map precond -> summary tuple.

    for pk in preconds:  # EN: Run speedup CV for each preconditioner.
        policy = "shared_sketch" if pk == "randqr" else "rebuild"  # EN: Only rand-QR uses reuse; others ignore.
        points, totals, best = cv_sweep_curve_operator_only(  # EN: Run CV sweep with speedups enabled.
            fold_ids=fold_ids,  # EN: Fold ids.
            n_folds=n_folds,  # EN: Fold count.
            A=A,  # EN: Implicit operator.
            b_full=b,  # EN: Targets.
            damps=damps,  # EN: Damp grid.
            precond_kind=pk,  # EN: Preconditioner.
            randqr_policy=policy,  # EN: Policy (shared for rand-QR).
            warm_start=True,  # EN: Enable continuation.
            max_iters=max_iters,  # EN: Iter cap.
            atol=atol,  # EN: atol.
            btol=btol,  # EN: btol.
            sketch_factor=sketch_factor,  # EN: Sketch factor.
            rng=rng_cv_speed,  # EN: RNG stream.
        )  # EN: End sweep.
        _ = points  # EN: Keep points available for optional debugging.

        total_seconds = float(totals.total_build_seconds + totals.total_solve_seconds)  # EN: Total curve cost.
        damp_str = f"{best.damp:.0e}" if best.damp != 0.0 else "0"  # EN: Format best damp.
        print(  # EN: Print summary line.
            f"{totals.precond:22} best_damp={damp_str:>9} best_val={best.val_mean:.3e} total={total_seconds:.3f}s (build={totals.total_build_seconds:.3f}s, solve={totals.total_solve_seconds:.3f}s) iters={totals.total_iters}"  # EN: Summary.
        )  # EN: End print.
        speed_rows[str(pk)] = (totals.precond, float(best.damp), float(best.val_mean), float(totals.total_build_seconds), float(totals.total_solve_seconds), int(totals.total_iters))  # EN: Store summary row.

    print_separator("Baseline vs speedups (operator-only)")  # EN: Announce comparison table.
    header = "precond              | base_total_s | sped_total_s | speedup | base_iters | sped_iters | base_best_val | sped_best_val"  # EN: Header.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Divider.
    for pk in preconds:  # EN: Print rows in consistent order.
        b_label, _, b_best_val, b_build, b_solve, b_iters = baseline_rows[str(pk)]  # EN: Unpack baseline row.
        s_label, _, s_best_val, s_build, s_solve, s_iters = speed_rows[str(pk)]  # EN: Unpack speedup row.
        _ = s_label  # EN: Keep labels aligned; baseline label is printed.
        base_total = float(b_build + b_solve)  # EN: Baseline total seconds.
        sped_total = float(s_build + s_solve)  # EN: Speedup total seconds.
        speedup = base_total / max(sped_total, EPS)  # EN: Speedup factor.
        print(f"{b_label:19} | {base_total:11.3f} | {sped_total:11.3f} | {speedup:7.2f} | {b_iters:9d} | {s_iters:9d} | {b_best_val:12.3e} | {s_best_val:12.3e}")  # EN: Print row.


if __name__ == "__main__":  # EN: Standard Python entrypoint guard.
    main()  # EN: Execute main.

