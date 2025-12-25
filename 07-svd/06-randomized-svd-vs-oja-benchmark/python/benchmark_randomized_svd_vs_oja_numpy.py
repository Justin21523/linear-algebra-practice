"""  # EN: Start module docstring.
Benchmark: Randomized SVD vs Oja's online PCA on the same centered dataset Xc (NumPy).  # EN: Summarize what this script does.

We compare two common "large-scale PCA" approaches:  # EN: Explain the two methods we compare.
  1) Randomized SVD (range finder + small SVD)  # EN: Method 1.
  2) Oja's online / mini-batch PCA (stochastic subspace updates)  # EN: Method 2.

Evaluation metrics (quality):  # EN: List the quality metrics.
  - Principal-angle cosines between the learned top-k subspace and a reference top-k PCA subspace  # EN: Subspace metric.
  - Explained variance ratio (EVR) captured by the learned subspace  # EN: Variance metric.

Cost proxies (not exact flops, but useful for intuition):  # EN: Explain what we measure as "cost".
  - Randomized SVD: number of A/A^T "sketch passes" implied by (k+p) and q  # EN: Cost for randomized method.
  - Oja: number of epochs (full data passes) and update steps, implied by batch size  # EN: Cost for Oja method.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured benchmark records.
from time import perf_counter  # EN: Use perf_counter for wall-clock timing (best effort).

import numpy as np  # EN: Import NumPy for numerical linear algebra.


EPS = 1e-12  # EN: Small epsilon for safe divisions and clipping.
SEED = 0  # EN: RNG seed for reproducible benchmarks.
PRINT_PRECISION = 6  # EN: Console float precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing for readability.


@dataclass(frozen=True)  # EN: Immutable record for one method+hyperparameter run.
class RunResult:  # EN: Store quality metrics and cost proxies for one configuration.
    method: str  # EN: Method name (e.g., "randSVD" or "Oja").
    k: int  # EN: Target rank / number of principal components.
    config: str  # EN: Hyperparameter string for display.
    min_cos: float  # EN: Minimum principal-angle cosine (worst alignment among k angles).
    mean_cos: float  # EN: Mean principal-angle cosine.
    evr: float  # EN: Explained variance ratio captured by the subspace.
    seconds: float  # EN: Wall-clock seconds (best-effort; depends on machine).
    cost_hint: str  # EN: Human-readable cost proxy (passes, matvec counts, steps).


def print_separator(title: str) -> None:  # EN: Print a section separator for readable console output.
    print()  # EN: Add a blank line before sections.
    print("=" * 78)  # EN: Print divider line.
    print(title)  # EN: Print section title.
    print("=" * 78)  # EN: Print closing divider.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm of a vector.
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def orthonormalize_columns(W: np.ndarray) -> np.ndarray:  # EN: Orthonormalize columns of W via thin QR.
    Q, _ = np.linalg.qr(W, mode="reduced")  # EN: Reduced QR returns orthonormal columns spanning the same subspace.
    return Q  # EN: Return Q.


def principal_angle_cosines(U_ref: np.ndarray, U_hat: np.ndarray) -> np.ndarray:  # EN: Compute principal-angle cosines between two subspaces.
    M = U_ref.T @ U_hat  # EN: Cross-Gram matrix for the two orthonormal bases.
    s = np.linalg.svd(M, compute_uv=False)  # EN: Singular values are cosines of principal angles.
    s = np.clip(s, 0.0, 1.0)  # EN: Clip to [0,1] for numerical safety.
    return s  # EN: Return cosines (sorted desc by SVD).


def explained_variance_ratio(C: np.ndarray, W: np.ndarray) -> float:  # EN: Compute EVR captured by subspace span(W).
    num = float(np.trace(W.T @ C @ W))  # EN: Captured variance is trace of projected covariance.
    den = float(np.trace(C))  # EN: Total variance is trace(C).
    return num / max(den, EPS)  # EN: Return EVR with epsilon floor.


def build_synthetic_data(  # EN: Generate a centered dataset with a controlled covariance spectrum.
    n_samples: int,  # EN: Number of samples (rows).
    d: int,  # EN: Feature dimension (columns).
    spectrum: str,  # EN: Spectrum type selector ("slow" or "fast").
    noise_std: float,  # EN: Additive isotropic noise level.
    rng: np.random.Generator,  # EN: RNG for reproducibility.
) -> np.ndarray:  # EN: Return centered data matrix Xc (n×d).
    if n_samples <= 0:  # EN: Validate n_samples.
        raise ValueError("n_samples must be positive")  # EN: Reject invalid inputs.
    if d <= 0:  # EN: Validate d.
        raise ValueError("d must be positive")  # EN: Reject invalid inputs.
    if noise_std < 0.0:  # EN: Validate noise_std.
        raise ValueError("noise_std must be non-negative")  # EN: Reject invalid inputs.

    # EN: Create a covariance eigen-spectrum (controls difficulty).  # EN: Explain spectrum choice.
    if spectrum == "slow":  # EN: Slow decay -> harder for both randomized and stochastic methods.
        eigenvalues = 1.0 / (1.0 + np.arange(d))  # EN: Harmonic-like decay.
    elif spectrum == "fast":  # EN: Fast decay -> easier; top-k dominates quickly.
        eigenvalues = 10.0 ** (-np.linspace(0.0, 4.0, d))  # EN: Exponential decay.
    else:  # EN: Guard against unknown spectrum choices.
        raise ValueError("spectrum must be 'slow' or 'fast'")  # EN: Reject invalid spectrum.

    # EN: Build a random orthonormal eigenvector matrix V using QR.  # EN: Explain eigenvector basis construction.
    V, _ = np.linalg.qr(rng.standard_normal((d, d)))  # EN: Random orthonormal basis in R^d.

    # EN: Sample Z ~ N(0, I) and transform it to have covariance V diag(eigs) V^T.  # EN: Explain sampling.
    Z = rng.standard_normal((n_samples, d))  # EN: Isotropic Gaussian samples.
    X = (Z * np.sqrt(eigenvalues)) @ V.T  # EN: Scale by sqrt(eigs) and rotate.

    X = X + noise_std * rng.standard_normal((n_samples, d))  # EN: Add isotropic noise for realism.
    Xc = X - np.mean(X, axis=0, keepdims=True)  # EN: Center the data (critical for PCA).
    return Xc.astype(float)  # EN: Return centered data.


def reference_pca_from_covariance(Xc: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # EN: Compute covariance and full eigen-decomposition as reference.
    n, d = Xc.shape  # EN: Extract shape.
    C = (Xc.T @ Xc) / max(n - 1, 1)  # EN: Sample covariance (d×d).
    evals, evecs = np.linalg.eigh(C)  # EN: Eigen-decomposition of symmetric covariance.
    order = np.argsort(evals)[::-1]  # EN: Sort eigenvalues descending.
    evals = evals[order]  # EN: Reorder eigenvalues.
    evecs = evecs[:, order]  # EN: Reorder eigenvectors accordingly.
    evecs = orthonormalize_columns(evecs)  # EN: Re-orthonormalize for numerical cleanliness.
    return C, evals, evecs  # EN: Return covariance, eigenvalues, and eigenvectors.


def randomized_pca_components(  # EN: Approximate top-k PCA components using randomized SVD on Xc.
    Xc: np.ndarray,  # EN: Centered data matrix (n×d).
    k: int,  # EN: Target rank.
    p: int,  # EN: Oversampling parameter.
    q: int,  # EN: Power iteration count.
    rng: np.random.Generator,  # EN: RNG for the random test matrix.
) -> np.ndarray:  # EN: Return V_hat (d×k) with orthonormal columns.
    n, d = Xc.shape  # EN: Extract dimensions.
    if k <= 0 or k > min(n, d):  # EN: Validate k.
        raise ValueError("k must be in [1, min(n,d)]")  # EN: Reject invalid k.
    if p < 0:  # EN: Validate p.
        raise ValueError("p must be non-negative")  # EN: Reject invalid p.
    if q < 0:  # EN: Validate q.
        raise ValueError("q must be non-negative")  # EN: Reject invalid q.

    sketch_dim = min(d, k + p)  # EN: Sketch dimension cannot exceed d.
    Omega = rng.standard_normal((d, sketch_dim)).astype(float)  # EN: Gaussian test matrix Ω.

    Y = Xc @ Omega  # EN: Range sketch Y = AΩ, where A=Xc.
    Q = orthonormalize_columns(Y)  # EN: Orthonormal basis for the sketch range.

    for _ in range(q):  # EN: Power iterations improve accuracy for slow spectral decay.
        Z = Xc.T @ Q  # EN: Z = A^T Q.
        Y = Xc @ Z  # EN: Y = A (A^T Q) = (A A^T) Q.
        Q = orthonormalize_columns(Y)  # EN: Re-orthonormalize to control drift.

    B = Q.T @ Xc  # EN: Small matrix B = Q^T A (sketch_dim×d).
    _, _, Vt = np.linalg.svd(B, full_matrices=False)  # EN: SVD of the small matrix; right vectors approximate A's.
    V_hat = Vt[:k, :].T  # EN: Take top-k right singular vectors as PCA directions (d×k).
    V_hat = orthonormalize_columns(V_hat)  # EN: Ensure orthonormal columns.
    return V_hat  # EN: Return the approximate PCA components.


def block_oja_pca(  # EN: Learn top-k PCA subspace via mini-batch block Oja updates.
    Xc: np.ndarray,  # EN: Centered data matrix (n×d).
    k: int,  # EN: Number of components.
    batch_size: int,  # EN: Mini-batch size.
    n_epochs: int,  # EN: Number of passes over data.
    eta0: float,  # EN: Initial learning rate.
    decay_steps: float,  # EN: Learning-rate decay timescale (in steps).
    rng: np.random.Generator,  # EN: RNG for shuffling and initialization.
) -> np.ndarray:  # EN: Return W (d×k) with orthonormal columns.
    n, d = Xc.shape  # EN: Extract dimensions.
    if k <= 0 or k > d:  # EN: Validate k.
        raise ValueError("k must be in [1, d]")  # EN: Reject invalid k.
    if batch_size <= 0:  # EN: Validate batch size.
        raise ValueError("batch_size must be positive")  # EN: Reject invalid batch size.
    if n_epochs <= 0:  # EN: Validate epoch count.
        raise ValueError("n_epochs must be positive")  # EN: Reject invalid epoch count.
    if eta0 <= 0.0:  # EN: Validate learning rate.
        raise ValueError("eta0 must be positive")  # EN: Reject invalid eta0.

    W = orthonormalize_columns(rng.standard_normal((d, k)).astype(float))  # EN: Initialize random orthonormal subspace.
    global_step = 0  # EN: Count update steps for learning-rate schedule.

    for _ in range(n_epochs):  # EN: Loop over epochs (full passes over data).
        perm = rng.permutation(n)  # EN: Shuffle sample order to reduce bias.
        for start in range(0, n, batch_size):  # EN: Iterate over mini-batches.
            idx = perm[start : start + batch_size]  # EN: Indices for this batch.
            Xb = Xc[idx, :]  # EN: Batch matrix (b×d).
            b = int(Xb.shape[0])  # EN: Actual batch size (last batch may be smaller).

            # EN: Use a simple decaying learning rate to stabilize late iterations.  # EN: Explain schedule.
            eta = eta0 / (1.0 + (global_step / max(decay_steps, EPS)))  # EN: 1/(1+t/T) schedule.

            XW = Xb @ W  # EN: Project batch samples onto current subspace (b×k).
            Cw = (Xb.T @ XW) / max(b, 1)  # EN: Approximate covariance-times-W: X^T(XW)/b.
            G = Cw - W @ (W.T @ Cw)  # EN: Tangent-space gradient for orthonormal constraint W^T W = I.

            W = W + eta * G  # EN: Apply Oja-style update step.
            W = orthonormalize_columns(W)  # EN: Re-orthonormalize for numerical stability.

            global_step += 1  # EN: Increment global step counter.

    return W  # EN: Return learned orthonormal basis.


def run_randomized_svd_sweep(  # EN: Sweep (k,p,q) configurations for randomized SVD and collect results.
    Xc: np.ndarray,  # EN: Centered data.
    C: np.ndarray,  # EN: Covariance matrix for EVR computation.
    V_ref: np.ndarray,  # EN: Reference eigenvectors (d×d), sorted by eigenvalue desc.
    k_list: list[int],  # EN: List of k values to test.
    p_list: list[int],  # EN: List of oversampling p values to test.
    q_list: list[int],  # EN: List of power iteration counts to test.
    rng: np.random.Generator,  # EN: RNG for randomized methods.
) -> list[RunResult]:  # EN: Return list of RunResult entries.
    results: list[RunResult] = []  # EN: Collect all results.
    n, d = Xc.shape  # EN: Extract shape for cost hints.

    for k in k_list:  # EN: Sweep k.
        V_ref_k = V_ref[:, :k]  # EN: Reference top-k subspace basis.
        for p in p_list:  # EN: Sweep oversampling.
            for q in q_list:  # EN: Sweep power iterations.
                t0 = perf_counter()  # EN: Start timer.
                V_hat = randomized_pca_components(Xc=Xc, k=k, p=p, q=q, rng=rng)  # EN: Compute approximate PCA subspace.
                seconds = perf_counter() - t0  # EN: Stop timer.

                cos = principal_angle_cosines(U_ref=V_ref_k, U_hat=V_hat)  # EN: Compute subspace alignment.
                evr = explained_variance_ratio(C=C, W=V_hat)  # EN: Compute EVR of learned subspace.

                sketch_dim = min(d, k + p)  # EN: Effective sketch dimension.
                # EN: Cost proxy: each multiply with Ω/Q acts like sketch_dim "matvecs"; count A/A^T blocks: 2*(q+1).  # EN: Explain estimate.
                approx_matvecs = (2 * (q + 1)) * sketch_dim  # EN: Equivalent number of A/A^T multiplies by vectors.
                cost_hint = f"sketch_dim={sketch_dim}, approx_matvecs≈{approx_matvecs}"  # EN: Human-readable cost hint.

                results.append(  # EN: Store run result.
                    RunResult(  # EN: Construct record.
                        method="randSVD",  # EN: Method label.
                        k=k,  # EN: Store k.
                        config=f"p={p}, q={q}",  # EN: Store hyperparameters.
                        min_cos=float(np.min(cos)),  # EN: Store worst principal-angle cosine.
                        mean_cos=float(np.mean(cos)),  # EN: Store mean principal-angle cosine.
                        evr=float(evr),  # EN: Store EVR.
                        seconds=float(seconds),  # EN: Store time.
                        cost_hint=cost_hint,  # EN: Store cost hint.
                    )  # EN: End record construction.
                )  # EN: End append.

    return results  # EN: Return all results.


def run_oja_sweep(  # EN: Sweep learning-rate configurations for Oja and collect results.
    Xc: np.ndarray,  # EN: Centered data.
    C: np.ndarray,  # EN: Covariance matrix for EVR computation.
    V_ref: np.ndarray,  # EN: Reference eigenvectors (d×d), sorted by eigenvalue desc.
    k_list: list[int],  # EN: List of k values to test.
    batch_size: int,  # EN: Mini-batch size for Oja.
    n_epochs: int,  # EN: Number of epochs (passes over data).
    eta0_list: list[float],  # EN: Initial learning rates to test.
    decay_steps_list: list[float],  # EN: Decay timescales to test.
    rng: np.random.Generator,  # EN: RNG for shuffling and initialization.
) -> list[RunResult]:  # EN: Return list of RunResult entries.
    results: list[RunResult] = []  # EN: Collect all results.
    n, d = Xc.shape  # EN: Extract shape for cost hints.
    n_steps_per_epoch = int(np.ceil(n / max(batch_size, 1)))  # EN: Compute number of mini-batch updates per epoch.
    total_steps = n_epochs * n_steps_per_epoch  # EN: Total update steps across epochs.

    for k in k_list:  # EN: Sweep k.
        V_ref_k = V_ref[:, :k]  # EN: Reference top-k subspace basis.
        for eta0 in eta0_list:  # EN: Sweep initial learning rate.
            for decay_steps in decay_steps_list:  # EN: Sweep decay schedule parameter.
                t0 = perf_counter()  # EN: Start timer.
                W = block_oja_pca(  # EN: Run Oja to learn the subspace.
                    Xc=Xc,  # EN: Provide data.
                    k=k,  # EN: Provide k.
                    batch_size=batch_size,  # EN: Provide batch size.
                    n_epochs=n_epochs,  # EN: Provide epoch count.
                    eta0=float(eta0),  # EN: Provide eta0.
                    decay_steps=float(decay_steps),  # EN: Provide decay steps.
                    rng=rng,  # EN: Provide RNG.
                )  # EN: End Oja call.
                seconds = perf_counter() - t0  # EN: Stop timer.

                cos = principal_angle_cosines(U_ref=V_ref_k, U_hat=W)  # EN: Compute subspace alignment.
                evr = explained_variance_ratio(C=C, W=W)  # EN: Compute EVR.

                # EN: Cost proxy: epochs = full passes; steps = mini-batch updates (each does 2 matmuls with Xb).  # EN: Explain proxy.
                cost_hint = f"epochs={n_epochs}, steps={total_steps}, batch={batch_size}"  # EN: Human-readable cost hint.

                results.append(  # EN: Store run result.
                    RunResult(  # EN: Construct record.
                        method="Oja",  # EN: Method label.
                        k=k,  # EN: Store k.
                        config=f"eta0={eta0:.2g}, decay={decay_steps:.0f}",  # EN: Store hyperparameters.
                        min_cos=float(np.min(cos)),  # EN: Store worst principal-angle cosine.
                        mean_cos=float(np.mean(cos)),  # EN: Store mean principal-angle cosine.
                        evr=float(evr),  # EN: Store EVR.
                        seconds=float(seconds),  # EN: Store time.
                        cost_hint=cost_hint,  # EN: Store cost hint.
                    )  # EN: End record.
                )  # EN: End append.

    return results  # EN: Return all results.


def print_results_table(results: list[RunResult]) -> None:  # EN: Print a compact table of run results.
    if not results:  # EN: Handle empty results.
        print("(no results)")  # EN: Print message.
        return  # EN: Exit early.

    # EN: Sort by k then by method then by EVR (descending) for readability.  # EN: Explain sorting.
    results_sorted = sorted(results, key=lambda r: (r.k, r.method, -r.evr))  # EN: Sort results.

    header = "method | k | config                | min_cos | mean_cos | EVR     | seconds | cost"  # EN: Build header string.
    print(header)  # EN: Print header.
    print("-" * len(header))  # EN: Print divider line.
    for r in results_sorted:  # EN: Print each result row.
        print(  # EN: Print formatted row.
            f"{r.method:6} | "  # EN: Method column.
            f"{r.k:2d} | "  # EN: k column.
            f"{r.config:21} | "  # EN: Config column.
            f"{r.min_cos:7.4f} | "  # EN: min cosine.
            f"{r.mean_cos:8.4f} | "  # EN: mean cosine.
            f"{r.evr:7.4f} | "  # EN: EVR.
            f"{r.seconds:7.3f} | "  # EN: seconds.
            f"{r.cost_hint}"  # EN: cost hint.
        )  # EN: End print.


def main() -> None:  # EN: Run the benchmark on one synthetic dataset and print sweep results.
    rng = np.random.default_rng(SEED)  # EN: Create deterministic RNG.

    # EN: Build one dataset and reuse it for all sweeps (fair comparison).  # EN: Explain fairness goal.
    n_samples = 20000  # EN: Number of samples.
    d = 200  # EN: Feature dimension.
    spectrum = "slow"  # EN: Choose slow spectrum so hyperparameter effects are visible.
    noise_std = 0.20  # EN: Noise level.

    print_separator("Dataset")  # EN: Print dataset header.
    Xc = build_synthetic_data(n_samples=n_samples, d=d, spectrum=spectrum, noise_std=noise_std, rng=rng)  # EN: Generate centered data.
    C, evals, V_ref = reference_pca_from_covariance(Xc)  # EN: Compute reference covariance PCA.
    print(f"Xc shape: {Xc.shape}, spectrum={spectrum}, noise_std={noise_std}")  # EN: Print dataset summary.
    print("Top-10 eigenvalues(C):", evals[:10])  # EN: Show leading eigenvalues for intuition.

    # EN: Hyperparameter grids (keep small and readable; expand as needed).  # EN: Explain sweep design.
    k_list = [5, 10, 20]  # EN: Compare a few k values.

    p_list = [5, 10]  # EN: Oversampling choices for randomized SVD.
    q_list = [0, 1, 2]  # EN: Power iteration counts for randomized SVD.

    batch_size = 256  # EN: Mini-batch size for Oja.
    n_epochs = 2  # EN: Number of full passes over the data for Oja.
    eta0_list = [0.5, 1.0, 2.0]  # EN: Initial learning rates to compare.
    decay_steps_list = [200.0, 1000.0]  # EN: Decay timescales to compare.

    print_separator("Sweep: Randomized SVD (k,p,q)")  # EN: Announce randomized sweep.
    res_rand = run_randomized_svd_sweep(Xc=Xc, C=C, V_ref=V_ref, k_list=k_list, p_list=p_list, q_list=q_list, rng=rng)  # EN: Run sweep.
    print_results_table(res_rand)  # EN: Print table.

    print_separator("Sweep: Oja (k, learning rate schedule)")  # EN: Announce Oja sweep.
    res_oja = run_oja_sweep(  # EN: Run Oja sweep.
        Xc=Xc,  # EN: Data.
        C=C,  # EN: Covariance.
        V_ref=V_ref,  # EN: Reference eigenvectors.
        k_list=k_list,  # EN: k list.
        batch_size=batch_size,  # EN: Batch size.
        n_epochs=n_epochs,  # EN: Epoch count.
        eta0_list=eta0_list,  # EN: Learning rates.
        decay_steps_list=decay_steps_list,  # EN: Decay list.
        rng=rng,  # EN: RNG.
    )  # EN: End sweep call.
    print_results_table(res_oja)  # EN: Print table.

    print_separator("Notes")  # EN: Print notes section.
    print("- Randomized SVD is typically strong with small q (1–2) when spectrum is slow.")  # EN: Provide interpretation hint.
    print("- Oja can be competitive but depends strongly on learning-rate schedule and eigengap.")  # EN: Provide interpretation hint.
    print("- Use principal-angle cosines and EVR to judge subspace quality, not raw vector entries.")  # EN: Provide evaluation guidance.

    print_separator("Done")  # EN: End-of-script marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main.

