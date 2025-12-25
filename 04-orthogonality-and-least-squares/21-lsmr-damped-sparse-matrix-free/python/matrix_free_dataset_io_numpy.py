"""  # EN: Start module docstring.
Matrix-free dataset I/O backends (NumPy-only) for the damped LSMR unit.  # EN: Summarize what this module provides.

This module exists to make the "matrix-free" idea more realistic:  # EN: Explain motivation.
  - In unit 21 we already avoid forming A^T A (solver is matrix-free).  # EN: Recall previous unit goal.
  - Here we additionally support reading A from files / streams (data can be matrix-free too).  # EN: Explain the new capability.

Supported backends (lightweight, dependency-free):  # EN: List what we support.
  - LibSVM text (sparse rows) with optional feature hashing.  # EN: Mention LibSVM support.
  - COO text triples (sparse) with optional feature hashing.  # EN: Mention COO support.
  - CSR stored in NPZ (sparse, in-memory CSR arrays).  # EN: Mention CSR NPZ support.
  - Dense CSV (loads into memory for simplicity).  # EN: Mention dense CSV support.
  - Dense .npy memmap (np.load(..., mmap_mode="r")).  # EN: Mention memmap support.

Each backend can build a row-subset operator that exposes:  # EN: Describe the key interface used by the solver.
  - matvec / rmatvec for the *weighted* operator A_w = diag(sqrt(w)) A  # EN: Explain weighting.
  - scaled targets b_w = sqrt(w) b (for weighted least squares)  # EN: Explain b scaling.
  - column norms / Frobenius^2 (needed for column-scaling and stopping tests)  # EN: Explain required stats.
  - CountSketchAug builder for sparse rand-QR preconditioning.  # EN: Explain preconditioner support.

Note: these backends are written for teaching and small-to-medium demos.  # EN: Set expectations.
File-streaming matvecs can be very slow for large datasets; use memmap/NPZ or operator-only generation when needed.  # EN: Warn about performance.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclasses for small immutable containers.
from pathlib import Path  # EN: Use Path for filesystem paths.
from typing import Callable, Iterable, Literal  # EN: Import typing helpers.

import numpy as np  # EN: Import NumPy for arrays and numerics.

# EN: Reuse a few shared dataclasses/utilities from the unit 21 CSR implementation to stay consistent.  # EN: Explain import choice.
from lsmr_damped_sparse_matrix_free_numpy import (  # EN: Import from sibling unit script (safe: guarded by __main__).
    CountSketchAug,  # EN: Dataclass holding a reusable CountSketch for [A; damp I].
    CSRMatrix,  # EN: Dataclass for CSR matrices (data/indices/indptr).
    CSRRowSubset,  # EN: Dataclass for CSR row-subset operators without copying nnz arrays.
    choose_sketch_rows,  # EN: Helper to pick sketch row count s.
    csr_matvec_subset,  # EN: Matvec for CSRRowSubset.
    csr_rmatvec_subset,  # EN: Transpose-matvec for CSRRowSubset.
)  # EN: End import list.


TaskKind = Literal["regression", "binary_classification"]  # EN: Supported task labels for demos using these backends.

MASK64 = (1 << 64) - 1  # EN: Keep integer arithmetic in unsigned 64-bit range for deterministic hashing.
SM64_GOLDEN = 0x9E3779B97F4A7C15  # EN: SplitMix64 increment constant (2^64 / golden ratio).
SM64_M1 = 0xBF58476D1CE4E5B9  # EN: SplitMix64 mixing constant 1.
SM64_M2 = 0x94D049BB133111EB  # EN: SplitMix64 mixing constant 2.


def splitmix64(x: int) -> int:  # EN: Deterministic 64-bit mixing function (fast stable hash primitive).
    z = (int(x) + int(SM64_GOLDEN)) & int(MASK64)  # EN: Add golden ratio increment and wrap to 64 bits.
    z = (z ^ (z >> 30)) * int(SM64_M1) & int(MASK64)  # EN: First xor-shift-multiply mix.
    z = (z ^ (z >> 27)) * int(SM64_M2) & int(MASK64)  # EN: Second xor-shift-multiply mix.
    z = (z ^ (z >> 31)) & int(MASK64)  # EN: Final xor-shift and wrap.
    return int(z)  # EN: Return hash as Python int.


def fnv1a64(data: bytes, seed: int = 0) -> int:  # EN: Deterministic FNV-1a 64-bit hash for byte strings.
    h = (0xCBF29CE484222325 ^ int(seed)) & int(MASK64)  # EN: Initialize with FNV offset basis xor seed.
    prime = 0x100000001B3  # EN: FNV 64-bit prime.
    for b in data:  # EN: Process each byte.
        h ^= int(b)  # EN: XOR byte into hash.
        h = (h * int(prime)) & int(MASK64)  # EN: Multiply by prime and wrap to 64 bits.
    return int(h)  # EN: Return 64-bit hash.


def hash_int_to_index_and_sign(feature: int, n_features: int, seed: int) -> tuple[int, float]:  # EN: Map an integer feature id to (hashed_index, sign).
    if n_features <= 0:  # EN: Validate hash dimension.
        raise ValueError("n_features must be positive")  # EN: Reject invalid hash dimension.
    h = splitmix64(int(feature) ^ int(seed))  # EN: Mix feature id with seed for stability.
    idx = int(h % int(n_features))  # EN: Map to [0, n_features).
    sign = 1.0 if (int(h) & 1) == 1 else -1.0  # EN: Use low bit for signed hashing.
    return int(idx), float(sign)  # EN: Return hashed index and sign.


def hash_token_to_index_and_sign(token: str, n_features: int, seed: int) -> tuple[int, float]:  # EN: Map a string token to (hashed_index, sign).
    if n_features <= 0:  # EN: Validate hash dimension.
        raise ValueError("n_features must be positive")  # EN: Reject invalid hash dimension.
    h = fnv1a64(token.encode("utf-8"), seed=int(seed))  # EN: Hash bytes deterministically with a seed.
    idx = int(h % int(n_features))  # EN: Map to [0, n_features).
    sign = 1.0 if (int(h) & 1) == 1 else -1.0  # EN: Use low bit for signed hashing.
    return int(idx), float(sign)  # EN: Return hashed index and sign.


def ensure_1d_float(x: np.ndarray | Iterable[float], length: int | None = None) -> np.ndarray:  # EN: Convert an input to a 1D float array with optional length check.
    arr = np.asarray(x, dtype=float).reshape(-1)  # EN: Convert to 1D float NumPy array.
    if length is not None and int(arr.size) != int(length):  # EN: Validate length when requested.
        raise ValueError("array has incompatible length")  # EN: Raise for mismatch.
    return arr  # EN: Return normalized array.


def ensure_1d_int(x: np.ndarray | Iterable[int]) -> np.ndarray:  # EN: Convert an input to a 1D int array.
    return np.asarray(x, dtype=int).reshape(-1)  # EN: Normalize to a 1D int array.


def default_weights(m: int) -> np.ndarray:  # EN: Create an all-ones weight vector of length m.
    if int(m) < 0:  # EN: Validate size.
        raise ValueError("m must be non-negative")  # EN: Reject invalid size.
    return np.ones((int(m),), dtype=float)  # EN: Return weights=1 for all samples.


def sqrt_weights_from_weights(w: np.ndarray) -> np.ndarray:  # EN: Compute elementwise sqrt(w) with guards for tiny negatives.
    w1 = np.asarray(w, dtype=float).reshape(-1)  # EN: Normalize weights vector.
    if np.any(w1 < -1e-12):  # EN: Reject clearly negative weights (not meaningful for WLS).
        raise ValueError("weights must be non-negative")  # EN: Enforce non-negativity.
    w_clamped = np.maximum(w1, 0.0)  # EN: Clamp tiny negatives due to numeric issues.
    return np.sqrt(w_clamped).astype(float)  # EN: Return sqrt weights.


@dataclass(frozen=True)  # EN: Immutable container for a row-subset operator and its associated vectors.
class RowSubsetOperator:  # EN: Represent A[row_ids,:] with optional sample-weights via A_w = diag(sqrt(w)) A.
    n: int  # EN: Number of columns (features) in the operator.
    row_ids: np.ndarray  # EN: Global row ids included in this subset (length m_subset).
    b_raw: np.ndarray  # EN: Raw targets b for this subset (length m_subset).
    w: np.ndarray  # EN: Sample weights for this subset (length m_subset).
    sqrt_w: np.ndarray  # EN: sqrt(w) for this subset (length m_subset).
    matvec_raw: Callable[[np.ndarray], np.ndarray]  # EN: Function computing A_sub @ x (unweighted).
    rmatvec_raw: Callable[[np.ndarray], np.ndarray]  # EN: Function computing A_sub^T @ y (unweighted).
    col_norms_sq_and_fro_sq_weighted: Callable[[], tuple[np.ndarray, float]]  # EN: Compute (col_sq, fro_sq) for A_w.
    build_countsketch_aug_weighted: Callable[[float, np.random.Generator], CountSketchAug]  # EN: Build CountSketchAug for A_w.

    @property  # EN: Expose m (row count) as a derived property.
    def m(self) -> int:  # EN: Return number of rows in the subset.
        return int(self.row_ids.size)  # EN: m = len(row_ids).

    @property  # EN: Provide b scaled by sqrt(w) for weighted least squares.
    def b_scaled(self) -> np.ndarray:  # EN: Return b_w = sqrt(w) * b (length m).
        return (self.sqrt_w * self.b_raw).astype(float)  # EN: Apply row scaling to targets.

    def matvec(self, x: np.ndarray) -> np.ndarray:  # EN: Compute A_w x = diag(sqrt(w)) (A x).
        y_raw = self.matvec_raw(x)  # EN: Compute unweighted predictions.
        return (self.sqrt_w * y_raw).astype(float)  # EN: Apply row scaling.

    def rmatvec(self, y: np.ndarray) -> np.ndarray:  # EN: Compute A_w^T y = A^T (diag(sqrt(w)) y).
        y1 = ensure_1d_float(y, length=self.m)  # EN: Normalize y to 1D float and validate length.
        return self.rmatvec_raw(self.sqrt_w * y1).astype(float)  # EN: Apply sqrt(w) then unweighted rmatvec.


@dataclass(frozen=True)  # EN: Immutable base class for a dataset that can provide matrix-free row-subset operators.
class MatrixFreeDataset:  # EN: Minimal interface: store targets/weights and build RowSubsetOperator objects.
    n: int  # EN: Feature dimension (number of columns).
    b: np.ndarray  # EN: Raw target vector (length m).
    w: np.ndarray  # EN: Raw sample weights (length m).

    @property  # EN: Derive m from the target vector length.
    def m(self) -> int:  # EN: Return number of samples (rows).
        return int(np.asarray(self.b).reshape(-1).size)  # EN: m = len(b).

    def subset(self, row_ids: np.ndarray) -> RowSubsetOperator:  # EN: Build a row-subset operator for the given row ids.
        raise NotImplementedError  # EN: Subclasses must implement backend-specific subset construction.


def _parse_libsvm_line_tokens(line: str) -> list[str]:  # EN: Split a LibSVM line into tokens while stripping inline comments.
    # EN: LibSVM commonly uses '#' for comments; we drop anything after the first '#'.  # EN: Explain comment handling.
    line_no_comment = line.split("#", 1)[0]  # EN: Remove trailing comment part if present.
    return [t for t in line_no_comment.strip().split() if t]  # EN: Return non-empty whitespace-separated tokens.


def _libsvm_detect_weight_token(tokens: list[str], has_weight: bool | None) -> tuple[float, int]:  # EN: Detect optional weight token after the label.
    if len(tokens) < 1:  # EN: Handle empty token lists defensively.
        raise ValueError("empty libsvm line")  # EN: Reject empty lines (unexpected for datasets).
    if len(tokens) == 1:  # EN: Only label is present.
        return 1.0, 1  # EN: weight=1, features start at index 1.

    second = tokens[1]  # EN: Candidate weight token.
    is_feature = ":" in second  # EN: Heuristic: features look like 'idx:val'.

    if has_weight is True:  # EN: Caller explicitly says a weight token exists.
        if is_feature:  # EN: Detect contradiction (weight token cannot contain ':').
            raise ValueError("has_weight=True but token[1] looks like a feature (contains ':')")  # EN: Raise clear error.
        return float(second), 2  # EN: Parse weight and set feature start.

    if has_weight is False:  # EN: Caller explicitly says there is no weight token.
        return 1.0, 1  # EN: weight=1, features start at index 1.

    # EN: Auto-detect mode: treat token[1] as a weight iff it is not a feature and parses as float.  # EN: Explain heuristic.
    if not is_feature:  # EN: Only try parse when token[1] does not look like a feature.
        try:  # EN: Attempt to parse as float.
            w = float(second)  # EN: Parse weight.
            return float(w), 2  # EN: Accept as weight token.
        except ValueError:  # EN: Not parseable as float.
            return 1.0, 1  # EN: Fall back to no weight token.
    return 1.0, 1  # EN: Default: no weight token.


def _parse_libsvm_features(  # EN: Parse feature tokens into (indices, values) arrays, with optional feature hashing.
    tokens: list[str],  # EN: Full token list (label, [weight], features...).
    start: int,  # EN: Feature token start index in the list.
    one_based: bool,  # EN: Whether feature indices are 1-based (LibSVM common default).
    n_features: int | None,  # EN: Hash dimension if hashing is enabled; otherwise None.
    hash_seed: int,  # EN: Seed used for stable hashing.
    signed_hash: bool,  # EN: Whether to use signed feature hashing (value *= sign).
) -> tuple[np.ndarray, np.ndarray, int]:  # EN: Return (idx, val, max_col_seen) for non-hashed mode.
    # EN: We use a dict to combine duplicates (e.g., hashing collisions or repeated indices).  # EN: Explain accumulation.
    acc: dict[int, float] = {}  # EN: Map column index -> accumulated value.
    max_col = -1  # EN: Track max column index encountered (for non-hashed inference).

    for tok in tokens[start:]:  # EN: Iterate feature tokens.
        if ":" not in tok:  # EN: Skip unexpected tokens (robustness for noisy lines).
            continue  # EN: Ignore token.
        k, v = tok.split(":", 1)  # EN: Split "idx:val".
        if not k:  # EN: Skip malformed entries.
            continue  # EN: Ignore malformed token.
        col_raw = int(k)  # EN: Parse column index.
        col0 = int(col_raw - 1) if bool(one_based) else int(col_raw)  # EN: Convert to 0-based if needed.
        val = float(v)  # EN: Parse numeric value.

        if n_features is not None:  # EN: Feature hashing mode.
            col_h, sign = hash_int_to_index_and_sign(feature=int(col0), n_features=int(n_features), seed=int(hash_seed))  # EN: Hash to a fixed dimension.
            val_h = float(val) * (float(sign) if bool(signed_hash) else 1.0)  # EN: Optionally apply signed hashing.
            acc[col_h] = float(acc.get(col_h, 0.0) + val_h)  # EN: Accumulate hashed value.
        else:  # EN: Non-hashed mode: keep original index.
            acc[col0] = float(acc.get(col0, 0.0) + float(val))  # EN: Accumulate by original column index.
            if int(col0) > int(max_col):  # EN: Update max index.
                max_col = int(col0)  # EN: Record new maximum.

    if not acc:  # EN: Handle empty feature rows.
        return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float), int(max_col)  # EN: Return empty arrays.

    cols = np.fromiter(acc.keys(), dtype=int)  # EN: Materialize column indices.
    vals = np.fromiter((acc[c] for c in cols), dtype=float)  # EN: Materialize values aligned with cols.
    order = np.argsort(cols)  # EN: Sort by column for determinism.
    cols = cols[order].astype(int)  # EN: Apply sort order.
    vals = vals[order].astype(float)  # EN: Apply sort order.
    return cols, vals, int(max_col)  # EN: Return parsed sparse row.


@dataclass(frozen=True)  # EN: Immutable LibSVM text dataset backend using line-offset indexing for row access.
class LibSVMTextDataset(MatrixFreeDataset):  # EN: Parse labels/features from a LibSVM file and expose matrix-free row subsets.
    path: Path  # EN: Path to the .svm/.libsvm text file.
    offsets: np.ndarray  # EN: Byte offsets for each line start (length m).
    one_based: bool  # EN: Whether feature indices in file are 1-based.
    use_feature_hashing: bool  # EN: Whether to hash features to a fixed dimension.
    hash_seed: int  # EN: Seed for stable feature hashing.
    signed_hash: bool  # EN: Whether to apply signed hashing (value *= sign).
    has_weight: bool | None  # EN: Whether a weight token exists (True/False) or auto-detect (None).

    @staticmethod  # EN: Provide a constructor-style loader for convenience.
    def load(  # EN: Load a LibSVM file, index line offsets, and parse labels/weights (features are parsed on-demand).
        path: str | Path,  # EN: Path to LibSVM file.
        n_features: int | None = None,  # EN: Feature dimension; required for hashing, inferred otherwise.
        one_based: bool = True,  # EN: Treat indices as 1-based (LibSVM default).
        feature_hashing: bool = False,  # EN: Enable feature hashing.
        hash_seed: int = 0,  # EN: Hash seed.
        signed_hash: bool = True,  # EN: Enable signed hashing (recommended for hashing trick).
        has_weight: bool | None = None,  # EN: Whether a weight token exists; None means auto-detect.
    ) -> LibSVMTextDataset:  # EN: Return a dataset instance.
        p = Path(path)  # EN: Normalize path.
        if not p.exists():  # EN: Validate input file.
            raise FileNotFoundError(str(p))  # EN: Raise for missing file.

        # EN: We scan the file once to record line offsets and read labels/weights;  # EN: Explain single-pass indexing.
        # EN: in non-hashing mode we also track max feature id to infer n.  # EN: Explain n inference.
        offsets: list[int] = []  # EN: Collect offsets in Python list first (then convert to ndarray).
        labels: list[float] = []  # EN: Collect labels.
        weights: list[float] = []  # EN: Collect weights (default 1.0 if not present).
        max_col_seen = -1  # EN: Track maximum column seen (0-based) in non-hashed mode.

        with p.open("rb") as f:  # EN: Open in binary to compute exact byte offsets.
            pos = 0  # EN: Current byte offset at the start of the next line.
            for raw in f:  # EN: Iterate lines as raw bytes.
                offsets.append(int(pos))  # EN: Record the start offset of this line.
                pos += int(len(raw))  # EN: Advance offset by the raw line length (includes newline).

                line = raw.decode("utf-8", errors="ignore")  # EN: Decode line to text for parsing.
                tokens = _parse_libsvm_line_tokens(line)  # EN: Split into tokens (label, optional weight, features...).
                if not tokens:  # EN: Skip empty lines.
                    labels.append(0.0)  # EN: Pad with dummy label to keep indexing consistent.
                    weights.append(1.0)  # EN: Default weight.
                    continue  # EN: Move to next line.

                labels.append(float(tokens[0]))  # EN: Parse and store label.
                w_i, start = _libsvm_detect_weight_token(tokens=tokens, has_weight=has_weight)  # EN: Detect optional weight token.
                weights.append(float(w_i))  # EN: Store weight.

                if not feature_hashing:  # EN: Only infer n from max index when not hashing.
                    _cols, _vals, max_col_line = _parse_libsvm_features(  # EN: Parse features (discard arrays) just to track max col.
                        tokens=tokens,  # EN: Token list.
                        start=int(start),  # EN: Feature start index.
                        one_based=bool(one_based),  # EN: Indexing convention.
                        n_features=None,  # EN: No hashing.
                        hash_seed=int(hash_seed),  # EN: Seed (unused when n_features=None).
                        signed_hash=bool(signed_hash),  # EN: Signed hashing flag (unused when n_features=None).
                    )  # EN: End parse.
                    _ = _cols  # EN: Explicitly ignore parsed arrays.
                    _ = _vals  # EN: Explicitly ignore parsed arrays.
                    if int(max_col_line) > int(max_col_seen):  # EN: Update global maximum.
                        max_col_seen = int(max_col_line)  # EN: Record new max col.

        b = np.asarray(labels, dtype=float).reshape(-1)  # EN: Convert labels to float array.
        w = np.asarray(weights, dtype=float).reshape(-1)  # EN: Convert weights to float array.

        if feature_hashing:  # EN: In hashing mode, n_features must be specified.
            if n_features is None:  # EN: Enforce explicit dimension for hashing.
                raise ValueError("n_features must be provided when feature_hashing=True")  # EN: Reject missing dimension.
            n_final = int(n_features)  # EN: Use provided hash dimension.
        else:  # EN: Non-hashed mode: infer n from max column seen.
            if n_features is not None:  # EN: Allow manual override for n (useful for known fixed dimension).
                n_final = int(n_features)  # EN: Use user-provided n.
            else:  # EN: Infer n as max_col+1 (or 0 if no features).
                n_final = int(max_col_seen + 1) if int(max_col_seen) >= 0 else 0  # EN: Infer dimension.

        return LibSVMTextDataset(  # EN: Construct dataset instance.
            n=int(n_final),  # EN: Feature dimension.
            b=b.astype(float),  # EN: Targets.
            w=w.astype(float),  # EN: Weights.
            path=p,  # EN: File path.
            offsets=np.asarray(offsets, dtype=np.int64),  # EN: Offsets for random row access.
            one_based=bool(one_based),  # EN: Indexing convention.
            use_feature_hashing=bool(feature_hashing),  # EN: Whether hashing is enabled.
            hash_seed=int(hash_seed),  # EN: Hash seed.
            signed_hash=bool(signed_hash),  # EN: Signed hashing flag.
            has_weight=has_weight,  # EN: Weight token mode.
        )  # EN: End construction.

    def _read_line(self, row_id: int) -> str:  # EN: Read a specific line by row id using stored byte offsets.
        rid = int(row_id)  # EN: Normalize row id to int.
        if rid < 0 or rid >= int(self.offsets.size):  # EN: Validate bounds.
            raise IndexError("row_id out of range")  # EN: Raise for invalid row.
        with self.path.open("rb") as f:  # EN: Open file in binary for byte-accurate seek.
            f.seek(int(self.offsets[rid]))  # EN: Seek to the start of the requested line.
            raw = f.readline()  # EN: Read exactly one line.
        return raw.decode("utf-8", errors="ignore")  # EN: Decode to text.

    def row_features(self, row_id: int) -> tuple[np.ndarray, np.ndarray]:  # EN: Parse one row into (cols, vals) arrays.
        line = self._read_line(row_id=int(row_id))  # EN: Read the raw line for this row.
        tokens = _parse_libsvm_line_tokens(line)  # EN: Tokenize line.
        if not tokens:  # EN: Empty line means no features.
            return np.zeros((0,), dtype=int), np.zeros((0,), dtype=float)  # EN: Return empty arrays.
        _w_i, start = _libsvm_detect_weight_token(tokens=tokens, has_weight=self.has_weight)  # EN: Detect optional weight (ignored here; we store w separately).
        n_hash = int(self.n) if bool(self.use_feature_hashing) else None  # EN: Provide hash dimension when hashing.
        cols, vals, _ = _parse_libsvm_features(  # EN: Parse features with or without hashing.
            tokens=tokens,  # EN: Token list.
            start=int(start),  # EN: Feature start index.
            one_based=bool(self.one_based),  # EN: Indexing convention.
            n_features=n_hash,  # EN: Hash dimension or None.
            hash_seed=int(self.hash_seed),  # EN: Hash seed.
            signed_hash=bool(self.signed_hash),  # EN: Signed hashing flag.
        )  # EN: End parse.
        return cols.astype(int), vals.astype(float)  # EN: Return row features.

    def subset(self, row_ids: np.ndarray) -> RowSubsetOperator:  # EN: Build a RowSubsetOperator for selected rows (no data copying, only row id list).
        rows = ensure_1d_int(row_ids)  # EN: Normalize row id list.
        b_sub = ensure_1d_float(self.b[rows], length=int(rows.size))  # EN: Gather raw targets for the subset.
        w_sub = ensure_1d_float(self.w[rows], length=int(rows.size))  # EN: Gather weights for the subset.
        sqrt_w_sub = sqrt_weights_from_weights(w_sub)  # EN: Compute sqrt weights for scaling.

        def matvec_raw(x: np.ndarray, rows: np.ndarray = rows) -> np.ndarray:  # EN: Compute unweighted A_sub @ x by parsing rows on demand.
            x1 = ensure_1d_float(x, length=int(self.n))  # EN: Normalize x vector and validate length.
            y = np.zeros((int(rows.size),), dtype=float)  # EN: Allocate output vector.
            for i, rid in enumerate(rows):  # EN: Loop subset rows in given order.
                cols, vals = self.row_features(int(rid))  # EN: Read and parse sparse row.
                if cols.size == 0:  # EN: Skip empty rows.
                    continue  # EN: Keep y[i]=0.
                y[int(i)] = float(np.dot(x1[cols], vals))  # EN: Dot product for this row.
            return y.astype(float)  # EN: Return predictions.

        def rmatvec_raw(y: np.ndarray, rows: np.ndarray = rows) -> np.ndarray:  # EN: Compute unweighted A_sub^T @ y by parsing rows on demand.
            y1 = ensure_1d_float(y, length=int(rows.size))  # EN: Normalize y and validate length.
            out = np.zeros((int(self.n),), dtype=float)  # EN: Allocate output vector in feature space.
            for i, rid in enumerate(rows):  # EN: Loop subset rows.
                yi = float(y1[int(i)])  # EN: Extract scalar y value for this row.
                if yi == 0.0:  # EN: Skip zeros to save parsing work when possible.
                    continue  # EN: Move to next row.
                cols, vals = self.row_features(int(rid))  # EN: Parse sparse row.
                if cols.size == 0:  # EN: Skip empty rows.
                    continue  # EN: No contribution.
                out[cols] += yi * vals  # EN: Accumulate A^T y contributions.
            return out.astype(float)  # EN: Return transpose product.

        def col_norms_sq_and_fro_sq_weighted(rows: np.ndarray = rows, w_sub: np.ndarray = w_sub) -> tuple[np.ndarray, float]:  # EN: Compute weighted column norms and Frobenius^2 by scanning subset rows.
            col_sq = np.zeros((int(self.n),), dtype=float)  # EN: Allocate column squared norm accumulator.
            fro_sq = 0.0  # EN: Accumulate Frobenius norm squared.
            for i, rid in enumerate(rows):  # EN: Loop subset rows.
                wi = float(w_sub[int(i)])  # EN: Extract weight for this row.
                if wi <= 0.0:  # EN: Skip rows with zero weight (no effect).
                    continue  # EN: Move to next row.
                cols, vals = self.row_features(int(rid))  # EN: Parse sparse row.
                if cols.size == 0:  # EN: Skip empty rows.
                    continue  # EN: No contribution.
                sq = wi * (vals * vals)  # EN: Weighted squared values for this row.
                col_sq[cols] += sq  # EN: Accumulate into column norms.
                fro_sq += float(np.sum(sq))  # EN: Add to Frobenius^2.
            return col_sq.astype(float), float(fro_sq)  # EN: Return stats.

        def build_countsketch_aug_weighted(  # EN: Build CountSketchAug for A_w_sub = diag(sqrt(w)) A_sub by scanning rows.
            sketch_factor: float,  # EN: Oversampling factor (s≈factor*(m_sub+n)).
            rng: np.random.Generator,  # EN: RNG for sketch hashes and signs.
            rows: np.ndarray = rows,  # EN: Capture row ids.
            sqrt_w_sub: np.ndarray = sqrt_w_sub,  # EN: Capture sqrt weights.
        ) -> CountSketchAug:  # EN: Return reusable CountSketchAug.
            m_sub = int(rows.size)  # EN: Number of data rows in subset.
            n = int(self.n)  # EN: Feature dimension.
            m_aug = int(m_sub + n)  # EN: Augmented row count for [A_w; damp I].
            s = choose_sketch_rows(m_aug=m_aug, n=int(n), sketch_factor=float(sketch_factor))  # EN: Choose sketch row count.
            scale = float(1.0 / np.sqrt(max(int(s), 1)))  # EN: Standard CountSketch scaling.

            h_top = rng.integers(low=0, high=int(s), size=int(m_sub), dtype=int)  # EN: Bucket per data row.
            sign_top = rng.choice(np.array([-1.0, 1.0]), size=int(m_sub)).astype(float)  # EN: Sign per data row.
            h_bottom = rng.integers(low=0, high=int(s), size=int(n), dtype=int)  # EN: Bucket per identity row.
            sign_bottom = rng.choice(np.array([-1.0, 1.0]), size=int(n)).astype(float)  # EN: Sign per identity row.

            SA_top = np.zeros((int(s), int(n)), dtype=float)  # EN: Allocate dense sketch for S_top A_w.
            for i, rid in enumerate(rows):  # EN: Loop subset rows in order.
                bucket = int(h_top[int(i)])  # EN: Bucket for this row.
                sgn = float(sign_top[int(i)])  # EN: Sign for this row.
                sw = float(sqrt_w_sub[int(i)])  # EN: sqrt(weight) for this row.
                if sw == 0.0:  # EN: Skip zero-weight rows.
                    continue  # EN: No contribution.
                cols, vals = self.row_features(int(rid))  # EN: Parse sparse row.
                if cols.size == 0:  # EN: Skip empty rows.
                    continue  # EN: No contribution.
                SA_top[bucket, cols] += (scale * sgn * sw) * vals  # EN: Accumulate hashed signed scaled row.

            return CountSketchAug(  # EN: Package sketch container.
                SA_top=SA_top.astype(float),  # EN: Store SA_top.
                h_bottom=h_bottom.astype(int),  # EN: Store identity hashes.
                sign_bottom=sign_bottom.astype(float),  # EN: Store identity signs.
                scale=float(scale),  # EN: Store scaling factor.
                s=int(s),  # EN: Store sketch row count.
            )  # EN: End return.

        return RowSubsetOperator(  # EN: Construct row-subset operator wrapper.
            n=int(self.n),  # EN: Feature dimension.
            row_ids=rows.astype(int),  # EN: Store row ids.
            b_raw=b_sub.astype(float),  # EN: Store raw targets.
            w=w_sub.astype(float),  # EN: Store weights.
            sqrt_w=sqrt_w_sub.astype(float),  # EN: Store sqrt weights.
            matvec_raw=matvec_raw,  # EN: Unweighted matvec.
            rmatvec_raw=rmatvec_raw,  # EN: Unweighted transpose matvec.
            col_norms_sq_and_fro_sq_weighted=col_norms_sq_and_fro_sq_weighted,  # EN: Stats function.
            build_countsketch_aug_weighted=build_countsketch_aug_weighted,  # EN: Sketch builder.
        )  # EN: End return.


@dataclass(frozen=True)  # EN: Immutable dataset for in-memory CSR stored in NPZ (or created programmatically).
class CSRInMemoryDataset(MatrixFreeDataset):  # EN: Store a CSRMatrix and provide row-subset operators without copying nnz arrays.
    A: CSRMatrix  # EN: CSR matrix with shape (m, n).

    @staticmethod  # EN: Load a CSR dataset from a .npz file.
    def load_npz(path: str | Path) -> CSRInMemoryDataset:  # EN: Return a CSRInMemoryDataset loaded from NPZ.
        p = Path(path)  # EN: Normalize path.
        if not p.exists():  # EN: Validate path.
            raise FileNotFoundError(str(p))  # EN: Raise for missing file.

        with np.load(p, allow_pickle=False) as z:  # EN: Load NPZ archive (no pickle).
            data = np.asarray(z["data"], dtype=float).reshape(-1)  # EN: Nonzero values.
            indices = np.asarray(z["indices"], dtype=int).reshape(-1)  # EN: Column indices.
            indptr = np.asarray(z["indptr"], dtype=int).reshape(-1)  # EN: Row pointers.
            shape = tuple(int(x) for x in np.asarray(z["shape"]).reshape(-1)[:2])  # EN: Matrix shape (m,n).

            # EN: Accept a couple of common names for targets in serialized datasets.  # EN: Explain key flexibility.
            if "b" in z:  # EN: Preferred key.
                b = np.asarray(z["b"], dtype=float).reshape(-1)  # EN: Targets.
            elif "y" in z:  # EN: Alternate key.
                b = np.asarray(z["y"], dtype=float).reshape(-1)  # EN: Targets.
            else:  # EN: Missing targets is an error (this unit expects supervised problems).
                raise KeyError("NPZ must contain 'b' (or 'y') targets")  # EN: Fail fast.

            if "weights" in z:  # EN: Optional sample weights.
                w = np.asarray(z["weights"], dtype=float).reshape(-1)  # EN: Weights.
            else:  # EN: Default weights.
                w = default_weights(int(b.size))  # EN: Use weights=1.

        A = CSRMatrix(data=data, indices=indices, indptr=indptr, shape=(int(shape[0]), int(shape[1])))  # EN: Build CSRMatrix container.
        if int(A.shape[0]) != int(b.size):  # EN: Validate target length matches m.
            raise ValueError("targets length does not match CSR shape[0]")  # EN: Reject mismatched dataset.
        if int(w.size) != int(b.size):  # EN: Validate weights length.
            raise ValueError("weights length does not match targets length")  # EN: Reject mismatch.

        return CSRInMemoryDataset(n=int(A.shape[1]), b=b.astype(float), w=w.astype(float), A=A)  # EN: Return dataset instance.

    def subset(self, row_ids: np.ndarray) -> RowSubsetOperator:  # EN: Build a weighted row-subset operator for CSR without copying nnz arrays.
        rows = ensure_1d_int(row_ids)  # EN: Normalize row ids.
        b_sub = ensure_1d_float(self.b[rows], length=int(rows.size))  # EN: Gather targets.
        w_sub = ensure_1d_float(self.w[rows], length=int(rows.size))  # EN: Gather weights.
        sqrt_w_sub = sqrt_weights_from_weights(w_sub)  # EN: Compute sqrt weights.

        subset = CSRRowSubset(  # EN: Build CSR row-subset view (no nnz copying).
            A=self.A,  # EN: Reference full CSR.
            row_ids=rows.astype(int),  # EN: Selected global row ids.
            starts=None,  # EN: Do not cache pointers by default (saves memory on huge folds).
            ends=None,  # EN: Do not cache pointers by default.
            shape=(int(rows.size), int(self.n)),  # EN: Subset shape.
        )  # EN: End subset construction.

        def matvec_raw(x: np.ndarray, subset: CSRRowSubset = subset) -> np.ndarray:  # EN: Compute unweighted A_sub @ x via CSR pointers.
            return csr_matvec_subset(subset=subset, x=x).astype(float)  # EN: Delegate to shared CSR subset matvec.

        def rmatvec_raw(y: np.ndarray, subset: CSRRowSubset = subset) -> np.ndarray:  # EN: Compute unweighted A_sub^T @ y via CSR pointers.
            return csr_rmatvec_subset(subset=subset, y=y).astype(float)  # EN: Delegate to shared CSR subset rmatvec.

        def col_norms_sq_and_fro_sq_weighted(  # EN: Compute weighted column norms and Frobenius^2 for A_w by scanning CSR nnz once.
            subset: CSRRowSubset = subset,  # EN: Capture subset view.
            w_sub: np.ndarray = w_sub,  # EN: Capture weights aligned with subset row order.
        ) -> tuple[np.ndarray, float]:  # EN: Return (col_sq, fro_sq) for A_w.
            A = subset.A  # EN: Extract full CSR reference.
            col_sq = np.zeros((int(self.n),), dtype=float)  # EN: Allocate accumulator.
            fro_sq = 0.0  # EN: Accumulate Frobenius^2.
            for i, rid in enumerate(subset.row_ids):  # EN: Loop subset rows.
                wi = float(w_sub[int(i)])  # EN: Row weight.
                if wi <= 0.0:  # EN: Skip zero-weight rows.
                    continue  # EN: Move on.
                start = int(A.indptr[int(rid)])  # EN: Row start pointer.
                end = int(A.indptr[int(rid) + 1])  # EN: Row end pointer.
                if end <= start:  # EN: Skip empty rows.
                    continue  # EN: No contribution.
                cols = A.indices[start:end]  # EN: Column indices for this row segment.
                vals = A.data[start:end]  # EN: Values for this row segment.
                sq = wi * (vals * vals)  # EN: Weighted squared values.
                col_sq[cols] += sq  # EN: Accumulate column norms.
                fro_sq += float(np.sum(sq))  # EN: Accumulate Frobenius^2.
            return col_sq.astype(float), float(fro_sq)  # EN: Return computed stats.

        def build_countsketch_aug_weighted(  # EN: Build CountSketchAug for A_w_sub by scanning CSR nnz.
            sketch_factor: float,  # EN: Oversampling factor.
            rng: np.random.Generator,  # EN: RNG for hashing/sign.
            subset: CSRRowSubset = subset,  # EN: Capture subset.
            sqrt_w_sub: np.ndarray = sqrt_w_sub,  # EN: Capture sqrt weights.
        ) -> CountSketchAug:  # EN: Return CountSketchAug for A_w_sub.
            m_sub = int(subset.row_ids.size)  # EN: Data row count.
            n = int(self.n)  # EN: Feature dimension.
            m_aug = int(m_sub + n)  # EN: Augmented row count.
            s = choose_sketch_rows(m_aug=m_aug, n=int(n), sketch_factor=float(sketch_factor))  # EN: Choose s.
            scale = float(1.0 / np.sqrt(max(int(s), 1)))  # EN: Scaling factor.

            h_top = rng.integers(low=0, high=int(s), size=int(m_sub), dtype=int)  # EN: Bucket per subset row.
            sign_top = rng.choice(np.array([-1.0, 1.0]), size=int(m_sub)).astype(float)  # EN: Sign per subset row.
            h_bottom = rng.integers(low=0, high=int(s), size=int(n), dtype=int)  # EN: Bucket per identity row.
            sign_bottom = rng.choice(np.array([-1.0, 1.0]), size=int(n)).astype(float)  # EN: Sign per identity row.

            SA_top = np.zeros((int(s), int(n)), dtype=float)  # EN: Allocate S_top A_w.
            A = subset.A  # EN: Extract full CSR.
            for i, rid in enumerate(subset.row_ids):  # EN: Loop subset rows.
                sw = float(sqrt_w_sub[int(i)])  # EN: sqrt(weight) for this row.
                if sw == 0.0:  # EN: Skip zero-weight rows.
                    continue  # EN: No contribution.
                start = int(A.indptr[int(rid)])  # EN: Row start pointer.
                end = int(A.indptr[int(rid) + 1])  # EN: Row end pointer.
                if end <= start:  # EN: Skip empty rows.
                    continue  # EN: Move on.
                bucket = int(h_top[int(i)])  # EN: Bucket for this row.
                sgn = float(sign_top[int(i)])  # EN: Sign for this row.
                cols = A.indices[start:end]  # EN: Column indices.
                vals = A.data[start:end]  # EN: Values.
                SA_top[bucket, cols] += (scale * sgn * sw) * vals  # EN: Accumulate sketched row for A_w.

            return CountSketchAug(  # EN: Package sketch.
                SA_top=SA_top.astype(float),  # EN: Store.
                h_bottom=h_bottom.astype(int),  # EN: Store.
                sign_bottom=sign_bottom.astype(float),  # EN: Store.
                scale=float(scale),  # EN: Store.
                s=int(s),  # EN: Store.
            )  # EN: End return.

        return RowSubsetOperator(  # EN: Construct operator.
            n=int(self.n),  # EN: Feature dimension.
            row_ids=rows.astype(int),  # EN: Row ids.
            b_raw=b_sub.astype(float),  # EN: Targets.
            w=w_sub.astype(float),  # EN: Weights.
            sqrt_w=sqrt_w_sub.astype(float),  # EN: sqrt weights.
            matvec_raw=matvec_raw,  # EN: Unweighted matvec.
            rmatvec_raw=rmatvec_raw,  # EN: Unweighted transpose matvec.
            col_norms_sq_and_fro_sq_weighted=col_norms_sq_and_fro_sq_weighted,  # EN: Stats.
            build_countsketch_aug_weighted=build_countsketch_aug_weighted,  # EN: Sketch builder.
        )  # EN: End return.


@dataclass(frozen=True)  # EN: Immutable COO text dataset backend (row,col,val per line) with optional feature hashing.
class COOTextDataset(MatrixFreeDataset):  # EN: Provide streaming COO matvec/rmatvec without storing CSR arrays.
    path: Path  # EN: Path to COO text file.
    shape: tuple[int, int]  # EN: Matrix shape (m, n).
    one_based: bool  # EN: Whether row/col indices in file are 1-based.
    use_feature_hashing: bool  # EN: Whether to hash column indices to a fixed dimension.
    hash_seed: int  # EN: Hash seed for column hashing.
    signed_hash: bool  # EN: Whether to apply signed hashing to values.

    @staticmethod  # EN: Load a COO dataset from a triples file plus separate targets/weights arrays.
    def load(  # EN: Construct a COOTextDataset.
        path: str | Path,  # EN: Path to COO triples file (row col val per line).
        targets_npy: str | Path,  # EN: Path to .npy containing b (length m).
        weights_npy: str | Path | None = None,  # EN: Optional path to .npy containing weights (length m).
        shape: tuple[int, int] | None = None,  # EN: Optional (m,n); if None we infer by scanning triples.
        one_based: bool = True,  # EN: Whether file indices are 1-based.
        feature_hashing: bool = False,  # EN: Enable hashing of column indices.
        n_features: int | None = None,  # EN: Hash dimension; required when hashing is enabled.
        hash_seed: int = 0,  # EN: Hash seed.
        signed_hash: bool = True,  # EN: Signed hashing for values.
    ) -> COOTextDataset:  # EN: Return dataset instance.
        p = Path(path)  # EN: Normalize path.
        if not p.exists():  # EN: Validate file.
            raise FileNotFoundError(str(p))  # EN: Raise for missing COO file.

        b = np.load(Path(targets_npy), allow_pickle=False).astype(float).reshape(-1)  # EN: Load targets from .npy.
        if weights_npy is None:  # EN: Default weights.
            w = default_weights(int(b.size))  # EN: Use ones.
        else:  # EN: Load weights from file.
            w = np.load(Path(weights_npy), allow_pickle=False).astype(float).reshape(-1)  # EN: Load weights.

        if int(w.size) != int(b.size):  # EN: Validate weights length.
            raise ValueError("weights length does not match targets length")  # EN: Reject mismatch.

        # EN: Infer shape if not provided by scanning the triples file once for max indices.  # EN: Explain inference.
        if shape is None:  # EN: Need inference.
            max_r = -1  # EN: Track max row.
            max_c = -1  # EN: Track max col.
            with p.open("r", encoding="utf-8", errors="ignore") as f:  # EN: Open text for scanning.
                for line in f:  # EN: Scan each triple line.
                    parts = line.strip().split()  # EN: Split whitespace.
                    if len(parts) < 3:  # EN: Skip malformed lines.
                        continue  # EN: Ignore line.
                    r = int(parts[0]) - (1 if bool(one_based) else 0)  # EN: Parse row index (0-based).
                    c = int(parts[1]) - (1 if bool(one_based) else 0)  # EN: Parse col index (0-based).
                    if int(r) > int(max_r):  # EN: Update max row.
                        max_r = int(r)  # EN: Store.
                    if int(c) > int(max_c):  # EN: Update max col.
                        max_c = int(c)  # EN: Store.
            # EN: m is ultimately defined by the supervised target vector length;  # EN: Explain choice.
            # EN: we take max() to allow trailing all-zero rows that never appear in COO triples.  # EN: Explain robustness.
            m_inf = max(int(b.size), int(max_r + 1) if int(max_r) >= 0 else 0)  # EN: Infer m safely.
            n_inf = int(max_c + 1) if int(max_c) >= 0 else 0  # EN: Infer n (before hashing).
            shape = (int(m_inf), int(n_inf))  # EN: Build inferred shape.

        m_shape, n_shape = int(shape[0]), int(shape[1])  # EN: Unpack shape.
        if int(m_shape) != int(b.size):  # EN: Validate m matches targets length.
            raise ValueError("provided/inferred m does not match targets length")  # EN: Reject mismatch.

        if feature_hashing:  # EN: Hashing mode sets n to fixed dimension.
            if n_features is None:  # EN: Require dimension for hashing.
                raise ValueError("n_features must be provided when feature_hashing=True")  # EN: Reject missing dimension.
            n_final = int(n_features)  # EN: Use hash dimension.
        else:  # EN: No hashing: n is from shape.
            n_final = int(n_shape)  # EN: Use inferred/provided n.

        return COOTextDataset(  # EN: Construct dataset.
            n=int(n_final),  # EN: Feature dimension.
            b=b.astype(float),  # EN: Targets.
            w=w.astype(float),  # EN: Weights.
            path=p,  # EN: COO file path.
            shape=(int(m_shape), int(n_final)),  # EN: Shape after hashing (n maybe changed).
            one_based=bool(one_based),  # EN: Indexing.
            use_feature_hashing=bool(feature_hashing),  # EN: Hashing enabled.
            hash_seed=int(hash_seed),  # EN: Hash seed.
            signed_hash=bool(signed_hash),  # EN: Signed hashing.
        )  # EN: End return.

    def subset(self, row_ids: np.ndarray) -> RowSubsetOperator:  # EN: Build a row-subset operator that streams triples and filters by row id.
        rows = ensure_1d_int(row_ids)  # EN: Normalize row ids.
        b_sub = ensure_1d_float(self.b[rows], length=int(rows.size))  # EN: Targets subset.
        w_sub = ensure_1d_float(self.w[rows], length=int(rows.size))  # EN: Weights subset.
        sqrt_w_sub = sqrt_weights_from_weights(w_sub)  # EN: sqrt weights subset.

        # EN: Build a row->local-position map so streaming COO triples can accumulate into subset order.  # EN: Explain mapping.
        row_to_pos = {int(r): int(i) for i, r in enumerate(rows.tolist())}  # EN: Python dict mapping global row -> local row index.

        def _iter_triples(path: Path = self.path) -> Iterable[tuple[int, int, float]]:  # EN: Stream parsed triples from the COO file.
            with path.open("r", encoding="utf-8", errors="ignore") as f:  # EN: Open text file for streaming.
                for line in f:  # EN: Iterate lines.
                    parts = line.strip().split()  # EN: Split by whitespace.
                    if len(parts) < 3:  # EN: Skip malformed lines.
                        continue  # EN: Ignore line.
                    r = int(parts[0]) - (1 if bool(self.one_based) else 0)  # EN: Parse row (0-based).
                    c = int(parts[1]) - (1 if bool(self.one_based) else 0)  # EN: Parse col (0-based).
                    v = float(parts[2])  # EN: Parse value.
                    if self.use_feature_hashing:  # EN: Optionally hash column index.
                        c_h, sgn = hash_int_to_index_and_sign(feature=int(c), n_features=int(self.n), seed=int(self.hash_seed))  # EN: Hash column.
                        v = float(v) * (float(sgn) if bool(self.signed_hash) else 1.0)  # EN: Apply signed hashing if enabled.
                        c = int(c_h)  # EN: Replace with hashed col.
                    yield int(r), int(c), float(v)  # EN: Yield normalized triple.

        def matvec_raw(x: np.ndarray) -> np.ndarray:  # EN: Compute unweighted A_sub @ x by streaming triples once.
            x1 = ensure_1d_float(x, length=int(self.n))  # EN: Normalize x.
            y = np.zeros((int(rows.size),), dtype=float)  # EN: Allocate raw output.
            for r, c, v in _iter_triples():  # EN: Stream all triples.
                pos = row_to_pos.get(int(r))  # EN: Check if this triple's row is in the subset.
                if pos is None:  # EN: Skip rows not in subset.
                    continue  # EN: Move on.
                y[int(pos)] += float(v) * float(x1[int(c)])  # EN: Accumulate dot product contribution.
            return y.astype(float)  # EN: Return raw predictions.

        def rmatvec_raw(y: np.ndarray) -> np.ndarray:  # EN: Compute unweighted A_sub^T @ y by streaming triples once.
            y1 = ensure_1d_float(y, length=int(rows.size))  # EN: Normalize y.
            out = np.zeros((int(self.n),), dtype=float)  # EN: Allocate output in feature space.
            for r, c, v in _iter_triples():  # EN: Stream triples.
                pos = row_to_pos.get(int(r))  # EN: Map global row to subset position.
                if pos is None:  # EN: Skip non-subset rows.
                    continue  # EN: Move on.
                out[int(c)] += float(v) * float(y1[int(pos)])  # EN: Accumulate transpose contribution.
            return out.astype(float)  # EN: Return A^T y.

        def col_norms_sq_and_fro_sq_weighted() -> tuple[np.ndarray, float]:  # EN: Compute weighted col norms and Frobenius^2 by streaming triples.
            col_sq = np.zeros((int(self.n),), dtype=float)  # EN: Allocate col norm accumulator.
            fro_sq = 0.0  # EN: Accumulate Frobenius^2.
            for r, c, v in _iter_triples():  # EN: Stream triples.
                pos = row_to_pos.get(int(r))  # EN: Map row to subset position.
                if pos is None:  # EN: Skip non-subset rows.
                    continue  # EN: Move on.
                wi = float(w_sub[int(pos)])  # EN: Weight for that row.
                if wi <= 0.0:  # EN: Skip zero-weight rows.
                    continue  # EN: No contribution.
                sq = wi * float(v) * float(v)  # EN: Weighted square.
                col_sq[int(c)] += float(sq)  # EN: Add to column norm.
                fro_sq += float(sq)  # EN: Add to Frobenius^2.
            return col_sq.astype(float), float(fro_sq)  # EN: Return stats.

        def build_countsketch_aug_weighted(sketch_factor: float, rng: np.random.Generator) -> CountSketchAug:  # EN: Build CountSketchAug for A_w by streaming triples once.
            m_sub = int(rows.size)  # EN: Subset row count.
            n = int(self.n)  # EN: Feature dimension.
            m_aug = int(m_sub + n)  # EN: Augmented row count.
            s = choose_sketch_rows(m_aug=m_aug, n=int(n), sketch_factor=float(sketch_factor))  # EN: Choose sketch rows.
            scale = float(1.0 / np.sqrt(max(int(s), 1)))  # EN: CountSketch scaling.

            h_top = rng.integers(low=0, high=int(s), size=int(m_sub), dtype=int)  # EN: Bucket per subset row (local index).
            sign_top = rng.choice(np.array([-1.0, 1.0]), size=int(m_sub)).astype(float)  # EN: Sign per subset row.
            h_bottom = rng.integers(low=0, high=int(s), size=int(n), dtype=int)  # EN: Bucket per identity row.
            sign_bottom = rng.choice(np.array([-1.0, 1.0]), size=int(n)).astype(float)  # EN: Sign per identity row.

            SA_top = np.zeros((int(s), int(n)), dtype=float)  # EN: Allocate sketch matrix for S_top A_w.
            for r, c, v in _iter_triples():  # EN: Stream triples.
                pos = row_to_pos.get(int(r))  # EN: Map row to subset position.
                if pos is None:  # EN: Skip non-subset rows.
                    continue  # EN: Move on.
                sw = float(sqrt_w_sub[int(pos)])  # EN: sqrt(weight) for this row.
                if sw == 0.0:  # EN: Skip zero-weight rows.
                    continue  # EN: No contribution.
                bucket = int(h_top[int(pos)])  # EN: Bucket for this row.
                sgn = float(sign_top[int(pos)])  # EN: Sign for this row.
                SA_top[bucket, int(c)] += (scale * sgn * sw) * float(v)  # EN: Accumulate into sketched row.

            return CountSketchAug(  # EN: Package sketch.
                SA_top=SA_top.astype(float),  # EN: Store.
                h_bottom=h_bottom.astype(int),  # EN: Store.
                sign_bottom=sign_bottom.astype(float),  # EN: Store.
                scale=float(scale),  # EN: Store.
                s=int(s),  # EN: Store.
            )  # EN: End return.

        return RowSubsetOperator(  # EN: Construct operator wrapper.
            n=int(self.n),  # EN: Feature dimension.
            row_ids=rows.astype(int),  # EN: Row ids.
            b_raw=b_sub.astype(float),  # EN: Targets.
            w=w_sub.astype(float),  # EN: Weights.
            sqrt_w=sqrt_w_sub.astype(float),  # EN: sqrt weights.
            matvec_raw=matvec_raw,  # EN: Unweighted matvec.
            rmatvec_raw=rmatvec_raw,  # EN: Unweighted transpose matvec.
            col_norms_sq_and_fro_sq_weighted=col_norms_sq_and_fro_sq_weighted,  # EN: Stats.
            build_countsketch_aug_weighted=build_countsketch_aug_weighted,  # EN: Sketch builder.
        )  # EN: End return.


@dataclass(frozen=True)  # EN: Helper that precomputes contiguous chunks for efficient dense subset matvecs.
class DenseRowChunks:  # EN: Represent row_ids via sorted contiguous ranges to avoid fancy-indexing copies.
    ranges: list[tuple[int, int, np.ndarray]]  # EN: List of (start_row, end_row_exclusive, orig_positions_in_sorted_order).
    m: int  # EN: Number of rows in the subset.


def build_dense_row_chunks(row_ids: np.ndarray) -> DenseRowChunks:  # EN: Precompute contiguous chunks for a row id list.
    rows = ensure_1d_int(row_ids)  # EN: Normalize input.
    m_sub = int(rows.size)  # EN: Subset size.
    if m_sub == 0:  # EN: Handle empty subset.
        return DenseRowChunks(ranges=[], m=0)  # EN: Return empty chunks.
    perm = np.argsort(rows)  # EN: Sort rows to group contiguous segments.
    rows_sorted = rows[perm]  # EN: Sorted row ids.

    ranges: list[tuple[int, int, np.ndarray]] = []  # EN: Collect contiguous ranges.
    start = 0  # EN: Start index within rows_sorted.
    while start < m_sub:  # EN: Scan until all rows are chunked.
        end = start + 1  # EN: Candidate end (exclusive).
        while end < m_sub and int(rows_sorted[end]) == int(rows_sorted[end - 1]) + 1:  # EN: Extend while consecutive.
            end += 1  # EN: Advance end.
        r0 = int(rows_sorted[start])  # EN: Range start row id.
        r1 = int(rows_sorted[end - 1]) + 1  # EN: Range end exclusive.
        orig_pos_sorted = perm[start:end].astype(int)  # EN: Original positions for these rows, ordered by sorted row id.
        ranges.append((int(r0), int(r1), orig_pos_sorted))  # EN: Store range triple.
        start = end  # EN: Continue from next segment.
    return DenseRowChunks(ranges=ranges, m=int(m_sub))  # EN: Return computed chunks.


@dataclass(frozen=True)  # EN: Dense dataset wrapper (in-memory or memmap) supporting matrix-free subset operators.
class DenseArrayDataset(MatrixFreeDataset):  # EN: Store dense X plus targets/weights; subset operators use chunked matvecs.
    X: np.ndarray  # EN: Dense feature matrix (m×n), can be memmap or in-memory ndarray.

    @staticmethod  # EN: Load dense dataset from CSV (in-memory).
    def load_csv(  # EN: Read a numeric CSV into X and b.
        path: str | Path,  # EN: CSV file path.
        delimiter: str = ",",  # EN: Column delimiter.
        has_header: bool = False,  # EN: Whether the first row is a header.
        target_col: int = -1,  # EN: Which column is the target b.
        weight_col: int | None = None,  # EN: Optional column for sample weights.
    ) -> DenseArrayDataset:  # EN: Return dataset.
        p = Path(path)  # EN: Normalize path.
        if not p.exists():  # EN: Validate file exists.
            raise FileNotFoundError(str(p))  # EN: Raise for missing CSV.

        skip = 1 if bool(has_header) else 0  # EN: Number of header lines to skip.
        data = np.loadtxt(p, delimiter=str(delimiter), skiprows=int(skip), dtype=float)  # EN: Load CSV into a dense array.
        if data.ndim == 1:  # EN: Handle single-row CSV (loadtxt returns 1D).
            data = data.reshape(1, -1)  # EN: Promote to 2D.

        n_cols = int(data.shape[1])  # EN: Total number of columns in CSV.
        tcol = int(target_col) % int(n_cols)  # EN: Normalize target col (support negative indexing).
        if weight_col is None:  # EN: No weight column.
            wcol = None  # EN: Mark none.
        else:  # EN: Normalize weight column index.
            wcol = int(weight_col) % int(n_cols)  # EN: Support negative indexing.

        b = data[:, tcol].astype(float).reshape(-1)  # EN: Extract targets.
        if wcol is None:  # EN: Default weights.
            w = default_weights(int(b.size))  # EN: Ones.
        else:  # EN: Extract weights.
            w = data[:, wcol].astype(float).reshape(-1)  # EN: Weights.

        # EN: Build X by taking all columns except target and weight.  # EN: Explain feature selection.
        drop = {int(tcol)} | ({int(wcol)} if wcol is not None else set())  # EN: Column indices to drop.
        keep = [j for j in range(n_cols) if j not in drop]  # EN: Columns kept as features.
        X = data[:, keep].astype(float)  # EN: Feature matrix.
        return DenseArrayDataset(n=int(X.shape[1]), b=b.astype(float), w=w.astype(float), X=X)  # EN: Return dataset.

    @staticmethod  # EN: Load dense dataset from .npy using memory mapping.
    def load_memmap_npy(  # EN: Load X from .npy with mmap_mode='r' and load targets/weights from .npy.
        X_npy: str | Path,  # EN: Path to X .npy file (m×n).
        b_npy: str | Path,  # EN: Path to b .npy file (length m).
        w_npy: str | Path | None = None,  # EN: Optional path to weights .npy file (length m).
    ) -> DenseArrayDataset:  # EN: Return dataset.
        X = np.load(Path(X_npy), allow_pickle=False, mmap_mode="r")  # EN: Memory-map X array.
        if X.ndim != 2:  # EN: Validate X is 2D.
            raise ValueError("X_npy must be a 2D array")  # EN: Reject invalid.
        b = np.load(Path(b_npy), allow_pickle=False).astype(float).reshape(-1)  # EN: Load targets.
        if int(b.size) != int(X.shape[0]):  # EN: Validate lengths.
            raise ValueError("b length must match X.shape[0]")  # EN: Reject mismatch.
        if w_npy is None:  # EN: Default weights.
            w = default_weights(int(b.size))  # EN: Ones.
        else:  # EN: Load weights.
            w = np.load(Path(w_npy), allow_pickle=False).astype(float).reshape(-1)  # EN: Load.
            if int(w.size) != int(b.size):  # EN: Validate.
                raise ValueError("weights length must match b length")  # EN: Reject mismatch.
        return DenseArrayDataset(n=int(X.shape[1]), b=b.astype(float), w=w.astype(float), X=np.asarray(X))  # EN: Return dataset.

    def subset(self, row_ids: np.ndarray) -> RowSubsetOperator:  # EN: Build a dense row-subset operator using chunked matvecs (no CSR conversion).
        rows = ensure_1d_int(row_ids)  # EN: Normalize rows.
        b_sub = ensure_1d_float(self.b[rows], length=int(rows.size))  # EN: Targets subset.
        w_sub = ensure_1d_float(self.w[rows], length=int(rows.size))  # EN: Weights subset.
        sqrt_w_sub = sqrt_weights_from_weights(w_sub)  # EN: sqrt weights subset.
        chunks = build_dense_row_chunks(rows)  # EN: Precompute contiguous row chunks for efficient slicing.

        def matvec_raw(x: np.ndarray, chunks: DenseRowChunks = chunks) -> np.ndarray:  # EN: Compute A_sub @ x (unweighted) for dense X using chunked slices.
            x1 = ensure_1d_float(x, length=int(self.n))  # EN: Normalize x.
            y = np.zeros((int(chunks.m),), dtype=float)  # EN: Allocate output in subset order.
            for r0, r1, orig_pos_sorted in chunks.ranges:  # EN: Loop contiguous row ranges.
                X_block = np.asarray(self.X[int(r0) : int(r1), :], dtype=float)  # EN: Read dense block (view for memmap when possible).
                y_block = X_block @ x1  # EN: Compute block matvec.
                y[orig_pos_sorted] = y_block.astype(float)  # EN: Scatter back to original subset order.
            return y.astype(float)  # EN: Return predictions.

        def rmatvec_raw(y: np.ndarray, chunks: DenseRowChunks = chunks) -> np.ndarray:  # EN: Compute A_sub^T @ y (unweighted) using chunked slices.
            y1 = ensure_1d_float(y, length=int(chunks.m))  # EN: Normalize y.
            out = np.zeros((int(self.n),), dtype=float)  # EN: Allocate feature-space output.
            for r0, r1, orig_pos_sorted in chunks.ranges:  # EN: Loop contiguous ranges.
                X_block = np.asarray(self.X[int(r0) : int(r1), :], dtype=float)  # EN: Read dense block.
                y_block = y1[orig_pos_sorted]  # EN: Gather y values in the same order as X_block rows.
                out += X_block.T @ y_block  # EN: Accumulate transpose product.
            return out.astype(float)  # EN: Return A^T y.

        def col_norms_sq_and_fro_sq_weighted(chunks: DenseRowChunks = chunks, w_sub: np.ndarray = w_sub) -> tuple[np.ndarray, float]:  # EN: Compute weighted col norms and Frobenius^2 for dense A_w.
            col_sq = np.zeros((int(self.n),), dtype=float)  # EN: Allocate.
            fro_sq = 0.0  # EN: Accumulate.
            for r0, r1, orig_pos_sorted in chunks.ranges:  # EN: Loop ranges.
                X_block = np.asarray(self.X[int(r0) : int(r1), :], dtype=float)  # EN: Read block.
                w_block = w_sub[orig_pos_sorted].astype(float)  # EN: Weights in block-row order.
                X2 = X_block * X_block  # EN: Elementwise square.
                col_sq += w_block @ X2  # EN: Weighted sum over rows for each column.
                fro_sq += float(np.sum(w_block * np.sum(X2, axis=1)))  # EN: Weighted sum of row norms.
            return col_sq.astype(float), float(fro_sq)  # EN: Return stats.

        def build_countsketch_aug_weighted(sketch_factor: float, rng: np.random.Generator, chunks: DenseRowChunks = chunks, sqrt_w_sub: np.ndarray = sqrt_w_sub) -> CountSketchAug:  # EN: Build CountSketchAug for dense A_w.
            m_sub = int(chunks.m)  # EN: Subset row count.
            n = int(self.n)  # EN: Feature dimension.
            m_aug = int(m_sub + n)  # EN: Augmented row count.
            s = choose_sketch_rows(m_aug=m_aug, n=int(n), sketch_factor=float(sketch_factor))  # EN: Choose sketch rows.
            scale = float(1.0 / np.sqrt(max(int(s), 1)))  # EN: Scaling.

            h_top = rng.integers(low=0, high=int(s), size=int(m_sub), dtype=int)  # EN: Bucket per subset row (subset order).
            sign_top = rng.choice(np.array([-1.0, 1.0]), size=int(m_sub)).astype(float)  # EN: Sign per subset row.
            h_bottom = rng.integers(low=0, high=int(s), size=int(n), dtype=int)  # EN: Bucket per identity row.
            sign_bottom = rng.choice(np.array([-1.0, 1.0]), size=int(n)).astype(float)  # EN: Sign per identity row.

            SA_top = np.zeros((int(s), int(n)), dtype=float)  # EN: Allocate sketch matrix.
            for r0, r1, orig_pos_sorted in chunks.ranges:  # EN: Process chunked blocks to reduce overhead.
                X_block = np.asarray(self.X[int(r0) : int(r1), :], dtype=float)  # EN: Read block.
                idx = orig_pos_sorted  # EN: Local row indices in subset order for this block (sorted by row).
                factors = (scale * sign_top[idx] * sqrt_w_sub[idx]).astype(float).reshape(-1, 1)  # EN: Per-row factors (includes sqrt(w)).
                block_rows = factors * X_block  # EN: Scale rows for A_w and CountSketch signs.
                np.add.at(SA_top, h_top[idx], block_rows)  # EN: Bucket-add each scaled row into SA_top.

            return CountSketchAug(  # EN: Package sketch.
                SA_top=SA_top.astype(float),  # EN: Store.
                h_bottom=h_bottom.astype(int),  # EN: Store.
                sign_bottom=sign_bottom.astype(float),  # EN: Store.
                scale=float(scale),  # EN: Store.
                s=int(s),  # EN: Store.
            )  # EN: End return.

        return RowSubsetOperator(  # EN: Construct operator.
            n=int(self.n),  # EN: Feature dimension.
            row_ids=rows.astype(int),  # EN: Row ids.
            b_raw=b_sub.astype(float),  # EN: Targets.
            w=w_sub.astype(float),  # EN: Weights.
            sqrt_w=sqrt_w_sub.astype(float),  # EN: sqrt weights.
            matvec_raw=matvec_raw,  # EN: Unweighted matvec.
            rmatvec_raw=rmatvec_raw,  # EN: Unweighted transpose matvec.
            col_norms_sq_and_fro_sq_weighted=col_norms_sq_and_fro_sq_weighted,  # EN: Stats.
            build_countsketch_aug_weighted=build_countsketch_aug_weighted,  # EN: Sketch builder.
        )  # EN: End return.
