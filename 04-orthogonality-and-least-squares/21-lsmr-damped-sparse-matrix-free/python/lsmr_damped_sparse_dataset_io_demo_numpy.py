"""  # EN: Start module docstring.
Demo: Damped LSMR + k-fold CV (choose damp) on file-backed matrix-free datasets (NumPy-only).  # EN: Summarize the purpose.

This script extends unit 21 with "realistic data plumbing":  # EN: Explain why this demo exists.
  - The solver stays matrix-free (only matvec/rmatvec).  # EN: Remind matrix-free solver idea.
  - The dataset can also be matrix-free (read from files / streams).  # EN: Highlight new capability.
  - We keep the same ML-flavored evaluation: k-fold CV sweeping a damp grid, and we track total curve cost.  # EN: Explain evaluation framing.

Backends supported (see --backend):  # EN: List supported backends.
  - csr_npz: sparse CSR stored in NPZ (fastest for sparse without SciPy).  # EN: Mention CSR NPZ.
  - libsvm: sparse LibSVM text (row-based, on-demand parsing).  # EN: Mention LibSVM.
  - coo: sparse COO text triples (streaming, no CSR stored).  # EN: Mention COO.
  - csv: dense CSV (loaded into memory for simplicity).  # EN: Mention CSV.
  - memmap: dense .npy memmap (np.load(..., mmap_mode='r')).  # EN: Mention memmap.

Task modes:  # EN: Describe task support.
  - regression: standard ridge / damped least squares.  # EN: Mention regression.
  - binary_classification: least-squares classifier on labels in {-1,+1}; we still CV on RMSE but also report accuracy.  # EN: Mention classification.

We also support weighted least squares via sample weights w_i >= 0:  # EN: Explain weighting.
  - We solve on A_w = diag(sqrt(w)) A and b_w = sqrt(w) b, which is equivalent to minimizing ||W^{1/2}(Ax-b)||^2 + damp^2||x||^2.  # EN: Give equivalence.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

import argparse  # EN: Parse command-line arguments.
from dataclasses import replace  # EN: Create modified frozen dataclass copies (for label preprocessing).
from pathlib import Path  # EN: Use Path for filesystem paths.
from tempfile import TemporaryDirectory  # EN: Create temp directories for generated sample datasets.
from time import perf_counter  # EN: Time build/solve cost for CV curves.
from typing import Literal  # EN: Use Literal for small enumerations.

import numpy as np  # EN: Import NumPy for arrays and linear algebra.

from lsmr_damped_sparse_matrix_free_numpy import (  # EN: Import solver and printing helpers from unit 21.
    EPS,  # EN: Small epsilon used across unit utilities.
    SEED,  # EN: Default RNG seed for reproducibility.
    CVPoint,  # EN: CV curve point dataclass (RMSE mean/std, iters, ||x||).
    CVTotals,  # EN: Whole-curve totals dataclass.
    PrecondKind,  # EN: Supported preconditioner kinds.
    RandQRPolicy,  # EN: Supported rand-QR reuse policies.
    choose_fixed_randqr_reference_damp,  # EN: Helper to pick damp_ref for fixed-R reuse.
    k_fold_assignments,  # EN: O(m) fold-id builder (avoids storing k splits).
    l2_norm,  # EN: Euclidean norm helper.
    print_cv_table,  # EN: Pretty CV table printer.
    print_separator,  # EN: Section separator printer.
    randqr_R_from_countsketch,  # EN: Compute R from QR(S [A; damp I]) using a reusable CountSketch.
    solve_one_operator,  # EN: Matrix-free damped LSMR teaching solver.
    upper_triangular_preconditioner_from_R,  # EN: Convert R into right-preconditioner closures.
)  # EN: End imports.

from matrix_free_dataset_io_numpy import (  # EN: Import dataset backends and operator wrappers.
    COOTextDataset,  # EN: COO text backend.
    CSRInMemoryDataset,  # EN: CSR NPZ backend.
    DenseArrayDataset,  # EN: Dense CSV / memmap backend.
    LibSVMTextDataset,  # EN: LibSVM text backend.
    MatrixFreeDataset,  # EN: Base dataset type.
    RowSubsetOperator,  # EN: Row-subset operator wrapper.
    TaskKind,  # EN: Task kind literal.
    default_weights,  # EN: Helper for weights=1.
    ensure_1d_float,  # EN: Helper to normalize vectors.
)  # EN: End imports.


BackendKind = Literal["csr_npz", "libsvm", "coo", "csv", "memmap"]  # EN: Backends supported by this demo CLI.


def normalize_binary_labels(b: np.ndarray) -> np.ndarray:  # EN: Map common binary label encodings to {-1,+1}.
    y = np.asarray(b, dtype=float).reshape(-1)  # EN: Normalize to 1D float array.
    uniq = set(np.unique(y).tolist())  # EN: Get unique values for heuristic mapping.
    if uniq.issubset({-1.0, 1.0}):  # EN: Already in {-1,+1}.
        return y.astype(float)  # EN: Return as-is.
    if uniq.issubset({0.0, 1.0}):  # EN: Map {0,1} -> {-1,+1}.
        return np.where(y > 0.0, 1.0, -1.0).astype(float)  # EN: Convert 1->+1, 0->-1.
    return np.where(y >= 0.0, 1.0, -1.0).astype(float)  # EN: Fallback: sign threshold at 0.


def weighted_rmse_from_scaled_residual(r_scaled: np.ndarray, w: np.ndarray) -> float:  # EN: Compute weighted RMSE given r_scaled = sqrt(w)*(pred-b).
    r = ensure_1d_float(r_scaled)  # EN: Normalize residual vector.
    w1 = ensure_1d_float(w, length=int(r.size))  # EN: Normalize weights.
    denom = float(np.sum(w1))  # EN: Weighted count (sum of weights).
    if denom <= 0.0:  # EN: Handle all-zero weights defensively.
        return 0.0  # EN: Define RMSE as 0 when no effective samples exist.
    return float(l2_norm(r) / np.sqrt(denom))  # EN: RMSE = ||sqrt(w)*err|| / sqrt(sum(w)).


def binary_accuracy_from_raw_scores(scores: np.ndarray, y_true: np.ndarray, w: np.ndarray) -> float:  # EN: Compute (optionally weighted) accuracy from raw scores.
    s = ensure_1d_float(scores)  # EN: Normalize scores.
    y = ensure_1d_float(y_true, length=int(s.size))  # EN: Normalize labels.
    w1 = ensure_1d_float(w, length=int(s.size))  # EN: Normalize weights.
    mask = w1 > 0.0  # EN: Ignore zero-weight samples for accuracy.
    if not bool(np.any(mask)):  # EN: Handle degenerate case.
        return 0.0  # EN: Define accuracy as 0 when no effective samples exist.
    pred = np.where(s >= 0.0, 1.0, -1.0).astype(float)  # EN: Predict sign(score).
    correct = (pred[mask] == y[mask]).astype(float)  # EN: 1.0 for correct predictions.
    return float(np.mean(correct))  # EN: Return mean accuracy (unweighted among nonzero-weight samples).


def damp_grid(default_count: int = 12) -> np.ndarray:  # EN: Build a default damp grid including 0 and a log-grid.
    return np.concatenate(([0.0], np.logspace(-6, 1, num=int(default_count))))  # EN: Include 0 plus logspace.


def build_precond_override(  # EN: Build a right-preconditioner override for solve_one_operator from fold-local stats/sketches.
    precond_kind: PrecondKind,  # EN: Preconditioner kind requested.
    col_sq: np.ndarray,  # EN: Column squared norms of A_w on the training fold.
    damp: float,  # EN: Current damp value.
    op_tr: RowSubsetOperator,  # EN: Training operator (provides CountSketch builder for rand-QR).
    randqr_policy: RandQRPolicy,  # EN: rand-QR reuse policy.
    sketch_factor: float,  # EN: Oversampling factor for CountSketch.
    rng: np.random.Generator,  # EN: RNG for randomized sketches.
    shared_sketch: dict[str, object],  # EN: Mutable cache for sketch/R reuse within a fold.
    damps_sorted: np.ndarray,  # EN: Full damp grid (sorted) used to choose fixed-R reference.
) -> tuple[tuple[str, Callable[[np.ndarray], np.ndarray], Callable[[np.ndarray], np.ndarray]], float]:  # EN: Return (precond_override, build_seconds).
    if precond_kind == "none":  # EN: No preconditioning.
        label = "none"  # EN: Human label.

        def apply_Minv(y: np.ndarray) -> np.ndarray:  # EN: Identity M^{-1}.
            return ensure_1d_float(y)  # EN: Return y unchanged (normalized).

        def apply_Minv_T(z: np.ndarray) -> np.ndarray:  # EN: Identity M^{-T}.
            return ensure_1d_float(z)  # EN: Return z unchanged (normalized).

        return (label, apply_Minv, apply_Minv_T), 0.0  # EN: Return with zero build time.

    if precond_kind == "col":  # EN: Column scaling preconditioner.
        t0 = perf_counter()  # EN: Start build timer.
        D = np.sqrt(np.asarray(col_sq, dtype=float).reshape(-1) + float(damp) * float(damp)).astype(float)  # EN: D_j = sqrt(||A_w[:,j]||^2 + damp^2).
        D[D < float(EPS)] = float(EPS)  # EN: Avoid division by ~0 for numerically empty columns.
        build_seconds = float(perf_counter() - t0)  # EN: Stop build timer.
        label = "col-scaling"  # EN: Human label.

        def apply_Minv(y: np.ndarray, D: np.ndarray = D) -> np.ndarray:  # EN: Apply D^{-1} (right preconditioning).
            return ensure_1d_float(y) / D  # EN: Elementwise divide by D.

        def apply_Minv_T(z: np.ndarray, D: np.ndarray = D) -> np.ndarray:  # EN: Apply D^{-T} = D^{-1}.
            return ensure_1d_float(z) / D  # EN: Same divide.

        return (label, apply_Minv, apply_Minv_T), float(build_seconds)  # EN: Return diagonal preconditioner.

    # EN: rand-QR preconditioner built from CountSketch of the augmented matrix [A_w; damp I].  # EN: Explain branch.
    if precond_kind != "randqr":  # EN: Defensive: reject unknown kinds.
        raise ValueError("unknown precond_kind")  # EN: Fail fast.

    # EN: Cache keys used inside shared_sketch dict.  # EN: Explain cache layout.
    SKETCH_KEY = "sketch"  # EN: Key for CountSketchAug.
    FIXED_R_KEY = "fixed_R"  # EN: Key for fixed-R reuse.
    FIXED_REF_KEY = "fixed_ref_damp"  # EN: Key for damp_ref used to build fixed R.

    if randqr_policy == "rebuild":  # EN: Baseline: rebuild sketch + QR at every damp.
        t0 = perf_counter()  # EN: Start build timer.
        sketch = op_tr.build_countsketch_aug_weighted(float(sketch_factor), rng)  # EN: Build CountSketchAug by scanning training data.
        R = randqr_R_from_countsketch(sketch=sketch, damp=float(damp))  # EN: Compute R from QR of sketched augmented matrix.
        precond_override = upper_triangular_preconditioner_from_R(R=R, label="rand-QR(rebuild)")  # EN: Turn R into Minv/Minv_T closures.
        build_seconds = float(perf_counter() - t0)  # EN: Stop timer.
        return precond_override, float(build_seconds)  # EN: Return.

    if randqr_policy == "shared_sketch":  # EN: Speedup: build sketch once, QR per damp.
        t0 = perf_counter()  # EN: Start build timer.
        if SKETCH_KEY not in shared_sketch:  # EN: Build and cache sketch on first use.
            shared_sketch[SKETCH_KEY] = op_tr.build_countsketch_aug_weighted(float(sketch_factor), rng)  # EN: Build sketch once per fold.
        sketch = shared_sketch[SKETCH_KEY]  # EN: Fetch cached sketch.
        R = randqr_R_from_countsketch(sketch=sketch, damp=float(damp))  # EN: Compute R for current damp (QR is per damp).
        precond_override = upper_triangular_preconditioner_from_R(R=R, label="rand-QR(shared)")  # EN: Build closures.
        build_seconds = float(perf_counter() - t0)  # EN: Stop timer.
        return precond_override, float(build_seconds)  # EN: Return.

    if randqr_policy == "fixed_R":  # EN: Speedup: compute one R_ref and reuse it for all damps.
        if FIXED_R_KEY not in shared_sketch:  # EN: Build fixed R on first use in this fold.
            t0 = perf_counter()  # EN: Start build timer.
            if SKETCH_KEY not in shared_sketch:  # EN: Build sketch if missing.
                shared_sketch[SKETCH_KEY] = op_tr.build_countsketch_aug_weighted(float(sketch_factor), rng)  # EN: Build sketch once.
            sketch = shared_sketch[SKETCH_KEY]  # EN: Fetch cached sketch.
            damp_ref = float(choose_fixed_randqr_reference_damp(damps_sorted=damps_sorted))  # EN: Choose reference damp.
            R_ref = randqr_R_from_countsketch(sketch=sketch, damp=float(damp_ref))  # EN: Compute R at reference damp.
            shared_sketch[FIXED_R_KEY] = R_ref  # EN: Cache fixed R.
            shared_sketch[FIXED_REF_KEY] = float(damp_ref)  # EN: Cache reference damp for reporting.
            build_seconds = float(perf_counter() - t0)  # EN: Stop timer.
        else:  # EN: Reuse existing fixed R; build time is 0 for this damp.
            build_seconds = 0.0  # EN: No build work.
        R = shared_sketch[FIXED_R_KEY]  # EN: Use cached R for all damps.
        damp_ref = float(shared_sketch.get(FIXED_REF_KEY, 0.0))  # EN: Fetch damp_ref for labeling.
        precond_override = upper_triangular_preconditioner_from_R(R=R, label=f"rand-QR(fixed@{damp_ref:.0e})")  # EN: Build closures.
        return precond_override, float(build_seconds)  # EN: Return.

    raise ValueError("unknown randqr_policy")  # EN: Reject unsupported policy.


def cv_sweep_curve_dataset(  # EN: Sweep a damp grid with k-fold CV for any MatrixFreeDataset backend.
    dataset: MatrixFreeDataset,  # EN: Dataset backend (file-backed or in-memory) providing row subsets.
    fold_ids: np.ndarray,  # EN: Per-sample fold assignment vector (length m).
    n_folds: int,  # EN: Number of folds.
    damps: np.ndarray,  # EN: Damp grid to sweep.
    precond_kind: PrecondKind,  # EN: Preconditioner kind.
    randqr_policy: RandQRPolicy,  # EN: rand-QR reuse policy (ignored unless precond_kind == "randqr").
    warm_start: bool,  # EN: Whether to use continuation along the damp path.
    max_iters: int,  # EN: Iteration cap.
    atol: float,  # EN: Absolute tolerance.
    btol: float,  # EN: Relative tolerance.
    sketch_factor: float,  # EN: Oversampling factor for CountSketch rand-QR.
    rng: np.random.Generator,  # EN: RNG stream (split into fold-local RNGs).
    task: TaskKind,  # EN: Task kind (classification adds extra reporting).
) -> tuple[list[CVPoint], CVTotals, CVPoint, dict[float, float]]:  # EN: Return (points, totals, best, acc_by_damp).
    damps_sorted = np.array(sorted([float(d) for d in np.asarray(damps, dtype=float).reshape(-1)]), dtype=float)  # EN: Sort damps ascending.
    n_damps = int(damps_sorted.size)  # EN: Number of damp points.
    m = int(dataset.m)  # EN: Dataset row count.
    n = int(dataset.n)  # EN: Dataset feature count.

    fold_ids_1d = np.asarray(fold_ids, dtype=int).reshape(-1)  # EN: Normalize fold ids.
    if int(fold_ids_1d.size) != int(m):  # EN: Validate fold id length.
        raise ValueError("fold_ids length must match dataset.m")  # EN: Reject mismatch.

    order = np.arange(n_damps, dtype=int)  # EN: Default evaluation order (ascending).
    if warm_start:  # EN: For continuation, it is usually better to go from large damp to small damp.
        order = order[::-1]  # EN: Reverse order to start from the largest damp.

    # EN: Accumulators: one list per damp, each list stores per-fold values.  # EN: Explain storage layout.
    train_rmse_per_d = [[] for _ in range(n_damps)]  # EN: Train RMSE per damp per fold.
    val_rmse_per_d = [[] for _ in range(n_damps)]  # EN: Val RMSE per damp per fold.
    xnorm_per_d = [[] for _ in range(n_damps)]  # EN: ||x|| per damp per fold.
    iters_per_d = [[] for _ in range(n_damps)]  # EN: Iterations per damp per fold.
    acc_per_d = [[] for _ in range(n_damps)]  # EN: Accuracy per damp per fold (classification only).

    total_build = 0.0  # EN: Sum of preconditioner build time across all fits.
    total_solve = 0.0  # EN: Sum of solver time across all fits.
    total_iters = 0  # EN: Sum of iterations across all fits.

    for fold in range(int(n_folds)):  # EN: Loop folds.
        train_ids = np.flatnonzero(fold_ids_1d != int(fold)).astype(int)  # EN: Training rows are all except this fold.
        val_ids = np.flatnonzero(fold_ids_1d == int(fold)).astype(int)  # EN: Validation rows are this fold.

        op_tr = dataset.subset(train_ids)  # EN: Build training operator (row-indexed, no matrix copying).
        op_va = dataset.subset(val_ids)  # EN: Build validation operator.

        col_sq, fro_sq = op_tr.col_norms_sq_and_fro_sq_weighted()  # EN: Precompute fold-local stats for A_w (reused across damps).
        w_train_sum = float(np.sum(op_tr.w))  # EN: Sum of train weights (for weighted RMSE normalization).
        w_val_sum = float(np.sum(op_va.w))  # EN: Sum of val weights.
        if w_train_sum <= 0.0 or w_val_sum <= 0.0:  # EN: Guard against degenerate all-zero weights.
            raise ValueError("each fold must have positive total weight")  # EN: Reject invalid weight configuration.

        # EN: Fold-local cache for rand-QR reuse policies (shared sketch / fixed R).  # EN: Explain cache.
        reuse_cache: dict[str, object] = {}  # EN: Mutable cache survives across damps within the fold.

        x_prev: np.ndarray | None = None  # EN: Warm-start state for this fold.
        for di in order:  # EN: Sweep damps in chosen order.
            damp = float(damps_sorted[int(di)])  # EN: Current damp.
            precond_override, build_seconds = build_precond_override(  # EN: Build (or reuse) preconditioner for this damp.
                precond_kind=precond_kind,  # EN: Kind.
                col_sq=col_sq,  # EN: Fold col norms.
                damp=float(damp),  # EN: Damp.
                op_tr=op_tr,  # EN: Training operator.
                randqr_policy=randqr_policy,  # EN: rand-QR policy.
                sketch_factor=float(sketch_factor),  # EN: Sketch factor.
                rng=rng,  # EN: RNG.
                shared_sketch=reuse_cache,  # EN: Cache for reuse.
                damps_sorted=damps_sorted,  # EN: Damps for damp_ref selection.
            )  # EN: End build.

            report = solve_one_operator(  # EN: Fit on training fold (weighted operator).
                m=int(op_tr.m),  # EN: m for A_tr.
                n=int(n),  # EN: n for A_tr.
                matvec_A=op_tr.matvec,  # EN: Weighted matvec A_w x.
                matvec_AT=op_tr.rmatvec,  # EN: Weighted transpose matvec A_w^T y.
                col_sq=np.asarray(col_sq, dtype=float).reshape(-1),  # EN: Fold col norms.
                fro_sq=float(fro_sq),  # EN: Fold Frobenius^2.
                b=op_tr.b_scaled,  # EN: Weighted targets b_w.
                damp=float(damp),  # EN: Damp.
                precond_kind=precond_kind,  # EN: Kind (ignored when precond_override is provided).
                max_iters=int(max_iters),  # EN: Cap.
                atol=float(atol),  # EN: atol.
                btol=float(btol),  # EN: btol.
                rng=rng,  # EN: RNG (unused when we override preconditioner).
                x_init=x_prev if bool(warm_start) else None,  # EN: Warm-start in x-space.
                precond_override=precond_override,  # EN: Provide prebuilt preconditioner.
                build_seconds_override=float(build_seconds),  # EN: Charge the build time we measured.
                A_for_precond=None,  # EN: Not needed; we build externally.
            )  # EN: End solve.

            total_build += float(report.build_seconds)  # EN: Accumulate build time.
            total_solve += float(report.solve_seconds)  # EN: Accumulate solve time.
            total_iters += int(report.n_iters)  # EN: Accumulate iterations.

            train_rmse = float(report.rnorm_data / np.sqrt(w_train_sum))  # EN: Weighted train RMSE from solver residual.
            r_val_scaled = op_va.matvec(report.x_hat) - op_va.b_scaled  # EN: Validation scaled residual sqrt(w)*(Ax-b).
            val_rmse = weighted_rmse_from_scaled_residual(r_scaled=r_val_scaled, w=op_va.w)  # EN: Weighted val RMSE.

            train_rmse_per_d[int(di)].append(float(train_rmse))  # EN: Store train RMSE for this fold.
            val_rmse_per_d[int(di)].append(float(val_rmse))  # EN: Store val RMSE for this fold.
            xnorm_per_d[int(di)].append(float(report.xnorm))  # EN: Store ||x||.
            iters_per_d[int(di)].append(float(report.n_iters))  # EN: Store iterations.

            if task == "binary_classification":  # EN: Compute accuracy as extra diagnostic in classification mode.
                # EN: op_va.matvec(x) returns sqrt(w)*(A x); divide by sqrt(w) to recover raw scores.  # EN: Explain conversion.
                scores_scaled = op_va.matvec(report.x_hat)  # EN: Scaled scores.
                scores = np.divide(  # EN: Safe divide by sqrt_w with zeros handled.
                    scores_scaled,  # EN: Numerator.
                    np.where(op_va.sqrt_w > 0.0, op_va.sqrt_w, 1.0),  # EN: Denominator (avoid zero).
                )  # EN: End divide.
                acc = binary_accuracy_from_raw_scores(scores=scores, y_true=op_va.b_raw, w=op_va.w)  # EN: Compute accuracy.
                acc_per_d[int(di)].append(float(acc))  # EN: Store accuracy.

            if warm_start:  # EN: Update warm-start state when enabled.
                x_prev = report.x_hat.astype(float)  # EN: Use this solution as the next initial guess.

    # EN: Aggregate per-damp stats across folds (mean/std).  # EN: Explain aggregation.
    points: list[CVPoint] = []  # EN: Collect CV points.
    acc_by_damp: dict[float, float] = {}  # EN: Store accuracy mean per damp (classification).
    for di, d in enumerate(damps_sorted.tolist()):  # EN: Iterate damps in ascending order for reporting.
        tr = np.asarray(train_rmse_per_d[di], dtype=float)  # EN: Train RMSE across folds.
        va = np.asarray(val_rmse_per_d[di], dtype=float)  # EN: Val RMSE across folds.
        xn = np.asarray(xnorm_per_d[di], dtype=float)  # EN: ||x|| across folds.
        it = np.asarray(iters_per_d[di], dtype=float)  # EN: iters across folds.

        key = f"d={float(d):.0e}" if float(d) != 0.0 else "d=0"  # EN: Compact label.
        points.append(  # EN: Append CV point summary.
            CVPoint(  # EN: Build CVPoint.
                key=str(key),  # EN: Label.
                damp=float(d),  # EN: Damp.
                train_mean=float(np.mean(tr)),  # EN: Mean train RMSE.
                train_std=float(np.std(tr)),  # EN: Std train RMSE.
                val_mean=float(np.mean(va)),  # EN: Mean val RMSE.
                val_std=float(np.std(va)),  # EN: Std val RMSE.
                x_norm_mean=float(np.mean(xn)),  # EN: Mean ||x||.
                iters_mean=float(np.mean(it)),  # EN: Mean iters.
            )  # EN: End CVPoint.
        )  # EN: End append.

        if task == "binary_classification":  # EN: Store accuracy mean for printing later.
            acc = np.asarray(acc_per_d[di], dtype=float)  # EN: Acc across folds.
            acc_by_damp[float(d)] = float(np.mean(acc))  # EN: Mean accuracy.

    best = min(points, key=lambda p: p.val_mean)  # EN: Choose damp with lowest mean validation RMSE.
    totals = CVTotals(  # EN: Package totals.
        precond=str(precond_kind) if precond_kind != "randqr" else f"randqr/{randqr_policy}",  # EN: Label totals.
        total_build_seconds=float(total_build),  # EN: Total build time.
        total_solve_seconds=float(total_solve),  # EN: Total solve time.
        total_iters=int(total_iters),  # EN: Total iterations.
    )  # EN: End totals.
    return points, totals, best, acc_by_damp  # EN: Return all results.


def print_curve_totals(totals: CVTotals, best: CVPoint) -> None:  # EN: Print a compact summary line for whole-curve totals.
    total = float(totals.total_build_seconds + totals.total_solve_seconds)  # EN: Total wall time for the curve.
    damp_str = f"{best.damp:.0e}" if float(best.damp) != 0.0 else "0"  # EN: Format best damp.
    print(  # EN: Print summary.
        f"{totals.precond:18} best_damp={damp_str:>8} best_val={best.val_mean:.3e} total={total:.3f}s (build={totals.total_build_seconds:.3f}s, solve={totals.total_solve_seconds:.3f}s) iters={totals.total_iters}"  # EN: Summary row.
    )  # EN: End print.


def compare_baseline_vs_speedups(  # EN: Print a small comparison table for baseline vs speedups totals.
    baseline: dict[str, tuple[CVTotals, CVPoint]],  # EN: Map label -> (totals,best) for baseline.
    speedups: dict[str, tuple[CVTotals, CVPoint]],  # EN: Map label -> (totals,best) for speedups.
) -> None:  # EN: Print comparison rows.
    header = "precond            | baseline_total | speed_total | speedup | base_iters | sped_iters | base_best_val | sped_best_val"  # EN: Column names.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Divider.
    for k in baseline.keys():  # EN: Iterate in baseline insertion order.
        b_tot, b_best = baseline[k]  # EN: Unpack baseline.
        s_tot, s_best = speedups[k]  # EN: Unpack speedups.
        b_total = float(b_tot.total_build_seconds + b_tot.total_solve_seconds)  # EN: Baseline seconds.
        s_total = float(s_tot.total_build_seconds + s_tot.total_solve_seconds)  # EN: Speed seconds.
        sp = float(b_total / max(s_total, float(EPS)))  # EN: Speedup factor.
        print(  # EN: Print row.
            f"{k:18} | {b_total:13.3f} | {s_total:11.3f} | {sp:7.2f} | {b_tot.total_iters:9d} | {s_tot.total_iters:9d} | {b_best.val_mean:12.3e} | {s_best.val_mean:12.3e}"  # EN: Row.
        )  # EN: End print.


def generate_sample_dataset_to_dir(out_dir: Path, m: int = 800, n: int = 120, nnz_per_row: int = 10, task: TaskKind = "regression") -> dict[str, Path]:  # EN: Generate a small synthetic dataset and write multiple formats.
    rng = np.random.default_rng(SEED + 123)  # EN: Deterministic RNG for data generation.

    # EN: Generate a sparse design matrix in CSR by explicitly sampling nnz per row (same style as unit 21).  # EN: Explain generation.
    data = []  # EN: Collect nonzero values.
    indices = []  # EN: Collect column indices.
    indptr = [0]  # EN: CSR row pointer starts at 0.
    for _ in range(int(m)):  # EN: Loop rows.
        cols = rng.choice(int(n), size=int(nnz_per_row), replace=False)  # EN: Pick unique columns per row.
        cols = np.sort(cols.astype(int))  # EN: Sort for determinism.
        vals = rng.standard_normal(int(nnz_per_row)).astype(float)  # EN: Random values.
        indices.extend(cols.tolist())  # EN: Append indices.
        data.extend(vals.tolist())  # EN: Append values.
        indptr.append(int(indptr[-1] + int(nnz_per_row)))  # EN: Advance row pointer.
    data_arr = np.asarray(data, dtype=float)  # EN: CSR data array.
    indices_arr = np.asarray(indices, dtype=int)  # EN: CSR indices array.
    indptr_arr = np.asarray(indptr, dtype=int)  # EN: CSR indptr array.

    # EN: Create a sparse-ish ground-truth x_true and compute labels.  # EN: Explain target creation.
    x_true = np.zeros((int(n),), dtype=float)  # EN: Initialize coefficients.
    support = rng.choice(int(n), size=max(5, int(n // 10)), replace=False)  # EN: Choose support set.
    x_true[support] = rng.standard_normal(int(support.size)).astype(float)  # EN: Fill support.

    # EN: Compute b = A x_true by scanning CSR once (no need for dense A).  # EN: Explain matvec.
    b = np.zeros((int(m),), dtype=float)  # EN: Allocate b.
    for i in range(int(m)):  # EN: Loop rows.
        start = int(indptr_arr[int(i)])  # EN: Row start.
        end = int(indptr_arr[int(i) + 1])  # EN: Row end.
        cols = indices_arr[start:end]  # EN: Columns.
        vals = data_arr[start:end]  # EN: Values.
        b[int(i)] = float(np.dot(vals, x_true[cols]))  # EN: Row dot product.
    b += 0.05 * rng.standard_normal(int(m)).astype(float)  # EN: Add small noise.

    if task == "binary_classification":  # EN: Convert to {-1,+1} labels for classification.
        b = np.where(b >= 0.0, 1.0, -1.0).astype(float)  # EN: Threshold at 0.

    # EN: Sample weights in [0.2, 2.0] to demonstrate weighted least squares.  # EN: Explain weights.
    w = (0.2 + 1.8 * rng.random(int(m))).astype(float)  # EN: Positive weights.

    out_dir.mkdir(parents=True, exist_ok=True)  # EN: Ensure output directory exists.

    # EN: Write CSR NPZ with targets and weights.  # EN: Explain output.
    csr_npz = out_dir / "sample_csr.npz"  # EN: NPZ path.
    np.savez(  # EN: Save arrays into NPZ.
        csr_npz,  # EN: Output path.
        data=data_arr,  # EN: CSR data.
        indices=indices_arr,  # EN: CSR indices.
        indptr=indptr_arr,  # EN: CSR indptr.
        shape=np.asarray([int(m), int(n)], dtype=int),  # EN: Shape.
        b=b.astype(float),  # EN: Targets.
        weights=w.astype(float),  # EN: Weights.
    )  # EN: End save.

    # EN: Write LibSVM (include an explicit weight token after label to exercise weight parsing).  # EN: Explain output format.
    libsvm_path = out_dir / "sample_libsvm.txt"  # EN: LibSVM path.
    with libsvm_path.open("w", encoding="utf-8") as f:  # EN: Open for writing.
        for i in range(int(m)):  # EN: Loop rows.
            start = int(indptr_arr[int(i)])  # EN: Row start.
            end = int(indptr_arr[int(i) + 1])  # EN: Row end.
            cols = indices_arr[start:end]  # EN: Columns (0-based).
            vals = data_arr[start:end]  # EN: Values.
            parts = [f"{float(b[int(i)]):.12g}", f"{float(w[int(i)]):.12g}"]  # EN: Start with label and weight.
            for c, v in zip(cols.tolist(), vals.tolist()):  # EN: Append sparse features.
                parts.append(f"{int(c) + 1}:{float(v):.12g}")  # EN: Use 1-based indices for LibSVM.
            f.write(" ".join(parts) + "\n")  # EN: Write line.

    # EN: Write COO triples (row col val) + separate .npy targets/weights.  # EN: Explain output format.
    coo_path = out_dir / "sample_coo.txt"  # EN: COO path.
    with coo_path.open("w", encoding="utf-8") as f:  # EN: Open for writing.
        for i in range(int(m)):  # EN: Loop rows.
            start = int(indptr_arr[int(i)])  # EN: Row start.
            end = int(indptr_arr[int(i) + 1])  # EN: Row end.
            cols = indices_arr[start:end]  # EN: Cols.
            vals = data_arr[start:end]  # EN: Vals.
            for c, v in zip(cols.tolist(), vals.tolist()):  # EN: Loop nnz.
                f.write(f"{int(i) + 1} {int(c) + 1} {float(v):.12g}\n")  # EN: Write 1-based triple.
    targets_npy = out_dir / "sample_targets.npy"  # EN: Targets .npy path.
    weights_npy = out_dir / "sample_weights.npy"  # EN: Weights .npy path.
    np.save(targets_npy, b.astype(float))  # EN: Save targets.
    np.save(weights_npy, w.astype(float))  # EN: Save weights.

    # EN: Also write a dense CSV and a dense memmap version by densifying the CSR (small sizes only).  # EN: Explain dense outputs.
    A_dense = np.zeros((int(m), int(n)), dtype=float)  # EN: Allocate dense matrix.
    for i in range(int(m)):  # EN: Fill dense rows.
        start = int(indptr_arr[int(i)])  # EN: Start.
        end = int(indptr_arr[int(i) + 1])  # EN: End.
        A_dense[int(i), indices_arr[start:end]] = data_arr[start:end]  # EN: Scatter nnz values.

    csv_path = out_dir / "sample_dense.csv"  # EN: CSV path.
    dense_with_yw = np.hstack([A_dense, b.reshape(-1, 1), w.reshape(-1, 1)]).astype(float)  # EN: Build [X | y | w] matrix.
    np.savetxt(csv_path, dense_with_yw, delimiter=",", fmt="%.10g")  # EN: Write numeric CSV without header.

    X_npy = out_dir / "sample_X.npy"  # EN: Dense X .npy path for memmap.
    b_npy = out_dir / "sample_b.npy"  # EN: Dense b .npy path.
    w_npy = out_dir / "sample_w.npy"  # EN: Dense w .npy path.
    np.save(X_npy, A_dense.astype(float))  # EN: Save X to .npy.
    np.save(b_npy, b.astype(float))  # EN: Save b to .npy.
    np.save(w_npy, w.astype(float))  # EN: Save w to .npy.

    return {  # EN: Return paths to all generated artifacts.
        "csr_npz": csr_npz,  # EN: CSR NPZ.
        "libsvm": libsvm_path,  # EN: LibSVM.
        "coo": coo_path,  # EN: COO triples.
        "coo_targets": targets_npy,  # EN: COO targets.
        "coo_weights": weights_npy,  # EN: COO weights.
        "csv": csv_path,  # EN: CSV.
        "memmap_X": X_npy,  # EN: Memmap X.
        "memmap_b": b_npy,  # EN: Memmap b.
        "memmap_w": w_npy,  # EN: Memmap w.
    }  # EN: End return.


def load_dataset_from_args(args: argparse.Namespace) -> MatrixFreeDataset:  # EN: Load the selected backend dataset according to CLI flags.
    backend: BackendKind = str(args.backend)  # EN: Normalize backend.
    if backend == "csr_npz":  # EN: CSR NPZ backend.
        return CSRInMemoryDataset.load_npz(Path(args.npz))  # EN: Load CSR NPZ dataset.
    if backend == "libsvm":  # EN: LibSVM backend.
        return LibSVMTextDataset.load(  # EN: Load LibSVM dataset.
            path=Path(args.libsvm),  # EN: File path.
            n_features=int(args.n_features) if args.n_features is not None else None,  # EN: Optional n.
            one_based=bool(args.one_based),  # EN: Indexing convention.
            feature_hashing=bool(args.feature_hashing),  # EN: Hashing flag.
            hash_seed=int(args.hash_seed),  # EN: Hash seed.
            signed_hash=bool(args.signed_hash),  # EN: Signed hashing.
            has_weight=None if args.has_weight == "auto" else bool(args.has_weight == "yes"),  # EN: Weight token mode.
        )  # EN: End load.
    if backend == "coo":  # EN: COO backend.
        shape = None  # EN: Default: infer shape.
        if args.shape is not None:  # EN: Parse shape when provided.
            m_s, n_s = (int(x) for x in str(args.shape).split(","))  # EN: Parse "m,n".
            shape = (int(m_s), int(n_s))  # EN: Build tuple.
        return COOTextDataset.load(  # EN: Load COO dataset.
            path=Path(args.coo),  # EN: COO path.
            targets_npy=Path(args.targets_npy),  # EN: Targets path.
            weights_npy=Path(args.weights_npy) if args.weights_npy is not None else None,  # EN: Optional weights path.
            shape=shape,  # EN: Shape or None.
            one_based=bool(args.one_based),  # EN: Indexing.
            feature_hashing=bool(args.feature_hashing),  # EN: Hashing.
            n_features=int(args.n_features) if args.n_features is not None else None,  # EN: Hash dim.
            hash_seed=int(args.hash_seed),  # EN: Seed.
            signed_hash=bool(args.signed_hash),  # EN: Signed hashing.
        )  # EN: End load.
    if backend == "csv":  # EN: CSV backend.
        return DenseArrayDataset.load_csv(  # EN: Load CSV into memory.
            path=Path(args.csv),  # EN: CSV path.
            delimiter=str(args.delimiter),  # EN: Delimiter.
            has_header=bool(args.has_header),  # EN: Header flag.
            target_col=int(args.target_col),  # EN: Target col.
            weight_col=int(args.weight_col) if args.weight_col is not None else None,  # EN: Weight col.
        )  # EN: End load.
    if backend == "memmap":  # EN: Memmap backend.
        return DenseArrayDataset.load_memmap_npy(  # EN: Load memmap X and arrays b/w.
            X_npy=Path(args.X_npy),  # EN: X path.
            b_npy=Path(args.b_npy),  # EN: b path.
            w_npy=Path(args.w_npy) if args.w_npy is not None else None,  # EN: w path.
        )  # EN: End load.
    raise ValueError("unknown backend")  # EN: Reject unsupported backend.


def main() -> None:  # EN: CLI entrypoint for this demo.
    parser = argparse.ArgumentParser(description="Unit 21 demo: damped LSMR + CV on file-backed matrix-free datasets")  # EN: Build CLI parser.
    parser.add_argument("--backend", type=str, default="csr_npz", choices=["csr_npz", "libsvm", "coo", "csv", "memmap"], help="Dataset backend kind")  # EN: Select backend.
    parser.add_argument("--task", type=str, default="regression", choices=["regression", "binary_classification"], help="Task kind (labels for classification should be ±1)")  # EN: Select task.

    # EN: Common CV/solver flags.  # EN: Explain grouping.
    parser.add_argument("--folds", type=int, default=5, help="Number of k-fold CV folds")  # EN: CV folds.
    parser.add_argument("--max_iters", type=int, default=150, help="Iteration cap for the teaching LSMR solver")  # EN: Iteration cap.
    parser.add_argument("--atol", type=float, default=1e-10, help="Absolute tolerance")  # EN: atol.
    parser.add_argument("--btol", type=float, default=1e-10, help="Relative tolerance")  # EN: btol.
    parser.add_argument("--sketch_factor", type=float, default=4.0, help="rand-QR CountSketch oversampling factor (s≈factor*(m+n))")  # EN: Sketch factor.
    parser.add_argument("--damps_count", type=int, default=12, help="Number of logspace damp points (plus 0)")  # EN: Damp grid size.

    # EN: Backend-specific file args.  # EN: Explain grouping.
    parser.add_argument("--npz", type=str, default="", help="csr_npz: path to CSR NPZ (data/indices/indptr/shape/b[/weights])")  # EN: CSR NPZ path.
    parser.add_argument("--libsvm", type=str, default="", help="libsvm: path to LibSVM text file")  # EN: LibSVM path.
    parser.add_argument("--coo", type=str, default="", help="coo: path to COO triples text file")  # EN: COO path.
    parser.add_argument("--targets_npy", type=str, default="", help="coo: path to targets .npy (length m)")  # EN: COO targets path.
    parser.add_argument("--weights_npy", type=str, default=None, help="coo: optional path to weights .npy (length m)")  # EN: COO weights path.
    parser.add_argument("--shape", type=str, default=None, help="coo: optional shape 'm,n' (m must match targets length)")  # EN: COO shape.
    parser.add_argument("--csv", type=str, default="", help="csv: path to numeric CSV (features + target [+ optional weight])")  # EN: CSV path.
    parser.add_argument("--delimiter", type=str, default=",", help="csv: delimiter")  # EN: CSV delimiter.
    parser.add_argument("--has_header", action="store_true", help="csv: first row is a header")  # EN: CSV header.
    parser.add_argument("--target_col", type=int, default=-2, help="csv: target column index (default: second last, assuming last is weight)")  # EN: Target col.
    parser.add_argument("--weight_col", type=int, default=-1, help="csv: weight column index (default: last column)")  # EN: Weight col.
    parser.add_argument("--X_npy", type=str, default="", help="memmap: path to X .npy (m×n)")  # EN: Memmap X.
    parser.add_argument("--b_npy", type=str, default="", help="memmap: path to b .npy (m,)")  # EN: Memmap b.
    parser.add_argument("--w_npy", type=str, default=None, help="memmap: optional path to weights .npy (m,)")  # EN: Memmap weights.

    # EN: Feature hashing flags used by libsvm/coo.  # EN: Explain grouping.
    parser.add_argument("--feature_hashing", action="store_true", help="Enable feature hashing (requires --n_features)")  # EN: Hashing flag.
    parser.add_argument("--n_features", type=int, default=None, help="Hash dimension (when --feature_hashing); otherwise optional fixed n")  # EN: Hash dimension.
    parser.add_argument("--hash_seed", type=int, default=0, help="Seed for stable feature hashing")  # EN: Hash seed.
    parser.add_argument("--signed_hash", action="store_true", help="Use signed hashing (value *= sign) when hashing is enabled")  # EN: Signed hashing.
    parser.add_argument("--one_based", action="store_true", help="Treat file indices as 1-based (default for generated sample files)")  # EN: 1-based indexing.
    parser.add_argument("--has_weight", type=str, default="auto", choices=["auto", "yes", "no"], help="libsvm: whether token[1] is a weight (auto/yes/no)")  # EN: LibSVM weight token mode.

    # EN: Convenience flag to generate a sample dataset (in a temp dir) and run all backends on it.  # EN: Explain generator.
    parser.add_argument("--generate_sample", action="store_true", help="Generate a small sample dataset and run the demo on all formats")  # EN: Generator flag.
    args = parser.parse_args()  # EN: Parse CLI arguments.

    task: TaskKind = str(args.task)  # EN: Normalize task kind.
    damps = damp_grid(default_count=int(args.damps_count))  # EN: Build default damp grid.
    n_folds = int(args.folds)  # EN: Fold count.

    rng_split = np.random.default_rng(SEED + 7)  # EN: RNG for fold assignment.
    rng_cv = np.random.default_rng(SEED + 9)  # EN: RNG for sketches (shared across folds for simplicity).

    if args.generate_sample:  # EN: Generate sample dataset and run all backends on it.
        with TemporaryDirectory() as td:  # EN: Create temp directory.
            out_dir = Path(td)  # EN: Convert to Path.
            paths = generate_sample_dataset_to_dir(out_dir=out_dir, task=task)  # EN: Generate files.

            print_separator("Generated sample dataset paths")  # EN: Announce paths.
            for k, v in paths.items():  # EN: Print each artifact.
                print(f"{k}: {v}")  # EN: Path line.

            # EN: Run the same CV suite on each backend (using the same fold ids for comparability).  # EN: Explain loop.
            for backend in ["csr_npz", "libsvm", "coo", "csv", "memmap"]:  # EN: Iterate supported backends.
                print_separator(f"Backend: {backend}")  # EN: Section title per backend.
                if backend == "csr_npz":  # EN: Load CSR NPZ.
                    dataset = CSRInMemoryDataset.load_npz(paths["csr_npz"])  # EN: Load.
                elif backend == "libsvm":  # EN: Load LibSVM.
                    dataset = LibSVMTextDataset.load(path=paths["libsvm"], n_features=None, one_based=True, feature_hashing=False, has_weight=True)  # EN: Load with explicit weight token.
                elif backend == "coo":  # EN: Load COO.
                    dataset = COOTextDataset.load(path=paths["coo"], targets_npy=paths["coo_targets"], weights_npy=paths["coo_weights"], one_based=True)  # EN: Load.
                elif backend == "csv":  # EN: Load CSV.
                    dataset = DenseArrayDataset.load_csv(path=paths["csv"], delimiter=",", has_header=False, target_col=-2, weight_col=-1)  # EN: Load.
                else:  # EN: memmap backend.
                    dataset = DenseArrayDataset.load_memmap_npy(X_npy=paths["memmap_X"], b_npy=paths["memmap_b"], w_npy=paths["memmap_w"])  # EN: Load.

                if task == "binary_classification":  # EN: Normalize labels for classification to {-1,+1}.
                    dataset = replace(dataset, b=normalize_binary_labels(dataset.b))  # EN: Replace b in the frozen dataclass.

                fold_ids = k_fold_assignments(rng=rng_split, n_samples=int(dataset.m), n_folds=int(n_folds))  # EN: Create fold ids.
                run_cv_suite(  # EN: Run baseline + speedups CV experiments and print tables.
                    dataset=dataset,  # EN: Dataset.
                    fold_ids=fold_ids,  # EN: Folds.
                    n_folds=int(n_folds),  # EN: Fold count.
                    damps=damps,  # EN: Damp grid.
                    max_iters=int(args.max_iters),  # EN: Solver cap.
                    atol=float(args.atol),  # EN: atol.
                    btol=float(args.btol),  # EN: btol.
                    sketch_factor=float(args.sketch_factor),  # EN: Sketch factor.
                    rng=rng_cv,  # EN: RNG for sketches.
                    task=task,  # EN: Task kind.
                )  # EN: End run.
        return  # EN: Done.

    # EN: Normal mode: load only the specified backend dataset.  # EN: Explain else branch.
    dataset = load_dataset_from_args(args)  # EN: Load dataset.
    if task == "binary_classification":  # EN: Normalize labels if requested.
        dataset = replace(dataset, b=normalize_binary_labels(dataset.b))  # EN: Replace b vector.

    fold_ids = k_fold_assignments(rng=rng_split, n_samples=int(dataset.m), n_folds=int(n_folds))  # EN: Build fold ids.
    run_cv_suite(  # EN: Run baseline + speedups CV suite.
        dataset=dataset,  # EN: Dataset.
        fold_ids=fold_ids,  # EN: Folds.
        n_folds=int(n_folds),  # EN: Fold count.
        damps=damps,  # EN: Damp grid.
        max_iters=int(args.max_iters),  # EN: Solver cap.
        atol=float(args.atol),  # EN: atol.
        btol=float(args.btol),  # EN: btol.
        sketch_factor=float(args.sketch_factor),  # EN: Sketch factor.
        rng=rng_cv,  # EN: RNG.
        task=task,  # EN: Task.
    )  # EN: End run.


def run_cv_suite(  # EN: Run baseline and speedups CV sweeps for none/col/randqr and print summaries.
    dataset: MatrixFreeDataset,  # EN: Dataset.
    fold_ids: np.ndarray,  # EN: Fold assignments.
    n_folds: int,  # EN: Fold count.
    damps: np.ndarray,  # EN: Damp grid.
    max_iters: int,  # EN: Iter cap.
    atol: float,  # EN: atol.
    btol: float,  # EN: btol.
    sketch_factor: float,  # EN: Sketch factor for rand-QR.
    rng: np.random.Generator,  # EN: RNG.
    task: TaskKind,  # EN: Task kind.
) -> None:  # EN: No return; prints results.
    print(f"m={dataset.m}, n={dataset.n}, folds={n_folds}, damps={len(np.asarray(damps).reshape(-1))}")  # EN: Print dataset summary line.

    # EN: Baseline: cold-start, rand-QR rebuild per damp.  # EN: Explain baseline.
    print_separator("k-fold CV sweep (baseline): cold-start + rand-QR rebuild")  # EN: Announce baseline.
    baseline: dict[str, tuple[CVTotals, CVPoint]] = {}  # EN: Store totals/best for baseline.
    for pk in ["none", "col", "randqr"]:  # EN: Loop preconditioners.
        policy: RandQRPolicy = "rebuild"  # EN: Baseline policy for rand-QR.
        points, totals, best, acc = cv_sweep_curve_dataset(  # EN: Run CV sweep.
            dataset=dataset,  # EN: Dataset.
            fold_ids=fold_ids,  # EN: Fold ids.
            n_folds=int(n_folds),  # EN: Fold count.
            damps=damps,  # EN: Damp grid.
            precond_kind=pk,  # EN: Preconditioner kind.
            randqr_policy=policy,  # EN: rand-QR policy.
            warm_start=False,  # EN: Cold-start.
            max_iters=int(max_iters),  # EN: Cap.
            atol=float(atol),  # EN: atol.
            btol=float(btol),  # EN: btol.
            sketch_factor=float(sketch_factor),  # EN: Sketch factor.
            rng=rng,  # EN: RNG.
            task=task,  # EN: Task.
        )  # EN: End sweep.
        print_separator(f"baseline: {pk}")  # EN: Section per preconditioner.
        print_cv_table(points=points, best=best)  # EN: Print table.
        if task == "binary_classification":  # EN: Print accuracy curve summary.
            best_acc = float(acc.get(float(best.damp), 0.0))  # EN: Best-damp accuracy.
            print(f"best_damp accuracy(mean over folds) = {best_acc:.3f}")  # EN: Print accuracy.
        print_curve_totals(totals=totals, best=best)  # EN: Print totals line.
        baseline[str(pk)] = (totals, best)  # EN: Store.

    # EN: Speedups: warm-start; for rand-QR use shared sketch across damps.  # EN: Explain speedups.
    print_separator("k-fold CV sweep (speedups): warm-start + rand-QR shared-sketch")  # EN: Announce speedups.
    speedups: dict[str, tuple[CVTotals, CVPoint]] = {}  # EN: Store totals/best for speedups.
    for pk in ["none", "col", "randqr"]:  # EN: Loop preconditioners.
        policy: RandQRPolicy = "shared_sketch" if pk == "randqr" else "rebuild"  # EN: Shared sketch only applies to rand-QR.
        points, totals, best, acc = cv_sweep_curve_dataset(  # EN: Run CV sweep.
            dataset=dataset,  # EN: Dataset.
            fold_ids=fold_ids,  # EN: Fold ids.
            n_folds=int(n_folds),  # EN: Fold count.
            damps=damps,  # EN: Damp grid.
            precond_kind=pk,  # EN: Kind.
            randqr_policy=policy,  # EN: Policy.
            warm_start=True,  # EN: Warm-start enabled.
            max_iters=int(max_iters),  # EN: Cap.
            atol=float(atol),  # EN: atol.
            btol=float(btol),  # EN: btol.
            sketch_factor=float(sketch_factor),  # EN: Sketch factor.
            rng=rng,  # EN: RNG.
            task=task,  # EN: Task.
        )  # EN: End sweep.
        print_separator(f"speedups: {pk}")  # EN: Section per preconditioner.
        print_cv_table(points=points, best=best)  # EN: Print table.
        if task == "binary_classification":  # EN: Print accuracy at best damp.
            best_acc = float(acc.get(float(best.damp), 0.0))  # EN: Best accuracy.
            print(f"best_damp accuracy(mean over folds) = {best_acc:.3f}")  # EN: Print.
        print_curve_totals(totals=totals, best=best)  # EN: Totals line.
        speedups[str(pk)] = (totals, best)  # EN: Store.

    print_separator("Baseline vs speedups (whole-curve cost)")  # EN: Announce comparison.
    compare_baseline_vs_speedups(baseline=baseline, speedups=speedups)  # EN: Print comparison table.

    # EN: Extra: for rand-QR, also compare fixed-R reuse (warm-start) to show even lower build cost.  # EN: Explain extra.
    print_separator("rand-QR reuse variants (warm-start): shared-sketch vs fixed-R")  # EN: Announce extra experiment.
    points, totals, best, acc = cv_sweep_curve_dataset(  # EN: Run fixed-R variant.
        dataset=dataset,  # EN: Dataset.
        fold_ids=fold_ids,  # EN: Fold ids.
        n_folds=int(n_folds),  # EN: Fold count.
        damps=damps,  # EN: Damps.
        precond_kind="randqr",  # EN: rand-QR.
        randqr_policy="fixed_R",  # EN: Fixed-R.
        warm_start=True,  # EN: Warm-start.
        max_iters=int(max_iters),  # EN: Cap.
        atol=float(atol),  # EN: atol.
        btol=float(btol),  # EN: btol.
        sketch_factor=float(sketch_factor),  # EN: Sketch factor.
        rng=rng,  # EN: RNG.
        task=task,  # EN: Task.
    )  # EN: End sweep.
    _ = points  # EN: Keep points available for optional debugging; table printing is optional here.
    if task == "binary_classification":  # EN: Print best accuracy if classification.
        best_acc = float(acc.get(float(best.damp), 0.0))  # EN: Best accuracy.
        print(f"best_damp accuracy(mean over folds) = {best_acc:.3f}")  # EN: Print.
    print_curve_totals(totals=totals, best=best)  # EN: Print totals line.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Run CLI.
