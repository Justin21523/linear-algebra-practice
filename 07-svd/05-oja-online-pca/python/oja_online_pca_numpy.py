"""  # EN: Start module docstring.
Oja's online PCA (NumPy): learn top-k principal components from streaming / mini-batch data.  # EN: Summarize what this script demonstrates.

Why "online PCA":  # EN: Explain the motivation.
  - When data is huge or arrives as a stream, full SVD/PCA is expensive or impossible to store.  # EN: Connect to large-scale settings.
  - Oja's rule updates the principal directions incrementally using small batches.  # EN: Explain incremental learning.

We work with centered samples x_t in R^d and covariance C = E[x x^T].  # EN: Define the covariance setting.
For top-k PCA we want the k-dimensional subspace maximizing variance, i.e. the span of top-k eigenvectors of C.  # EN: State the PCA goal.

Block Oja (subspace Oja) update (one mini-batch):  # EN: Provide the core update.
  - Estimate C_batch from a batch X (b×d) as (X^T X)/b.  # EN: Explain the batch covariance estimate.
  - Compute C_batch W and project out the component that breaks orthonormality.  # EN: Explain the Stiefel-manifold correction.
  - Re-orthonormalize W with QR (teaching-friendly and stable).  # EN: Explain the orthonormalization step.

We verify with an exact reference computed from the sample covariance eigen-decomposition:  # EN: Explain the evaluation approach.
  - Principal angle cosines between learned subspace and reference top-k subspace  # EN: Metric 1.
  - Explained variance ratio captured by learned subspace  # EN: Metric 2.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references in type annotations.

from dataclasses import dataclass  # EN: Use dataclass for structured result storage.

import numpy as np  # EN: Import NumPy for linear algebra routines.


EPS = 1e-12  # EN: Small epsilon to avoid division-by-zero.
SEED = 0  # EN: RNG seed for reproducible demos.
PRINT_PRECISION = 6  # EN: Console float precision for readable output.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing.


@dataclass(frozen=True)  # EN: Make the results immutable for safer usage.
class OjaHistory:  # EN: Store progress metrics recorded during training.
    steps: np.ndarray  # EN: Global step indices where we evaluated.
    min_cos: np.ndarray  # EN: Minimum principal-angle cosine at each evaluation (worst alignment).
    mean_cos: np.ndarray  # EN: Mean principal-angle cosine at each evaluation.
    evr: np.ndarray  # EN: Explained variance ratio at each evaluation.


def print_separator(title: str) -> None:  # EN: Print a section separator for console readability.
    print()  # EN: Add whitespace before the section.
    print("=" * 78)  # EN: Print a horizontal divider.
    print(title)  # EN: Print the section title.
    print("=" * 78)  # EN: Print a closing divider.


def fro_norm(A: np.ndarray) -> float:  # EN: Compute Frobenius norm of a matrix.
    return float(np.linalg.norm(A, ord="fro"))  # EN: Delegate to NumPy.


def l2_norm(x: np.ndarray) -> float:  # EN: Compute Euclidean norm (2-norm) of a vector.
    return float(np.linalg.norm(x, ord=2))  # EN: Delegate to NumPy.


def orthonormalize_columns(W: np.ndarray) -> np.ndarray:  # EN: Orthonormalize columns of W using thin QR.
    Q, _ = np.linalg.qr(W, mode="reduced")  # EN: Reduced QR yields orthonormal Q with same column span.
    return Q  # EN: Return orthonormal basis for the span of W.


def principal_angle_cosines(U_ref: np.ndarray, U_hat: np.ndarray) -> np.ndarray:  # EN: Compute cosines of principal angles between subspaces.
    M = U_ref.T @ U_hat  # EN: Cross-Gram matrix between two orthonormal bases (k×k).
    s = np.linalg.svd(M, compute_uv=False)  # EN: Singular values equal cosines of principal angles.
    s = np.clip(s, 0.0, 1.0)  # EN: Clip to [0,1] to avoid small numeric issues.
    return s  # EN: Return cosines (sorted descending by SVD).


def explained_variance_ratio(C: np.ndarray, W: np.ndarray) -> float:  # EN: Compute EVR captured by subspace span(W).
    num = float(np.trace(W.T @ C @ W))  # EN: Captured variance equals trace of projected covariance.
    den = float(np.trace(C))  # EN: Total variance equals trace of covariance.
    return num / max(den, EPS)  # EN: Return EVR with safety floor.


def build_synthetic_data_from_covariance(  # EN: Generate samples with a controlled covariance spectrum.
    n_samples: int,  # EN: Number of samples to generate.
    eigenvalues: np.ndarray,  # EN: Desired covariance eigenvalues (d,).
    noise_std: float,  # EN: Isotropic noise standard deviation added to samples.
    rng: np.random.Generator,  # EN: RNG for reproducibility.
) -> tuple[np.ndarray, np.ndarray]:  # EN: Return (X, V_true) where Cov(X)≈V diag(eigs) V^T.
    if eigenvalues.ndim != 1:  # EN: Validate eigenvalues shape.
        raise ValueError("eigenvalues must be a 1D array")  # EN: Reject invalid input.
    d = int(eigenvalues.size)  # EN: Feature dimension.
    if n_samples <= 0:  # EN: Validate sample count.
        raise ValueError("n_samples must be positive")  # EN: Reject invalid input.
    if np.any(eigenvalues < 0.0):  # EN: Validate PSD spectrum.
        raise ValueError("eigenvalues must be non-negative")  # EN: Reject invalid spectrum.

    # EN: Build a random orthonormal eigenvector matrix V_true using QR.  # EN: Explain construction.
    V_true, _ = np.linalg.qr(rng.standard_normal((d, d)))  # EN: Orthonormal basis in R^d.

    # EN: Generate Z ~ N(0,I) and transform it to have the desired covariance.  # EN: Explain sampling.
    Z = rng.standard_normal((n_samples, d))  # EN: Base isotropic Gaussian samples.
    X = (Z * np.sqrt(eigenvalues)) @ V_true.T  # EN: Apply scaling by sqrt(eigs) and rotate by V_true.

    # EN: Add isotropic noise so the problem is more realistic (not exactly low-rank).  # EN: Explain noise.
    X = X + noise_std * rng.standard_normal((n_samples, d))  # EN: Add i.i.d. Gaussian noise.
    return X.astype(float), V_true  # EN: Return data and the true eigenvector basis.


def reference_topk_from_covariance(Xc: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:  # EN: Compute reference top-k PCA from sample covariance.
    n, d = Xc.shape  # EN: Extract dimensions.
    if k <= 0 or k > d:  # EN: Validate k.
        raise ValueError("k must be in [1, d]")  # EN: Reject invalid k.
    C = (Xc.T @ Xc) / max(n - 1, 1)  # EN: Sample covariance (d×d).
    evals, evecs = np.linalg.eigh(C)  # EN: Eigen-decomposition of symmetric covariance.
    order = np.argsort(evals)[::-1]  # EN: Sort eigenvalues descending.
    evals = evals[order]  # EN: Reorder eigenvalues.
    evecs = evecs[:, order]  # EN: Reorder eigenvectors accordingly.
    V_k = evecs[:, :k]  # EN: Extract top-k eigenvectors (principal components).
    V_k = orthonormalize_columns(V_k)  # EN: Re-orthonormalize for numerical cleanliness.
    return C, evals, V_k  # EN: Return covariance, eigenvalues, and top-k eigenvectors.


def learning_rate(step: int, eta0: float, decay_steps: float) -> float:  # EN: Compute a simple decaying learning rate schedule.
    if decay_steps <= 0.0:  # EN: Validate decay steps.
        return eta0  # EN: If decay is disabled, return constant eta0.
    return eta0 / (1.0 + (step / decay_steps))  # EN: Use a 1/(1+t/T) style decay.


def block_oja_pca(  # EN: Learn top-k principal subspace with a mini-batch block Oja update.
    Xc: np.ndarray,  # EN: Centered data matrix (n×d); we iterate over rows as a stream.
    k: int,  # EN: Target number of principal components.
    batch_size: int,  # EN: Mini-batch size for each update.
    n_epochs: int,  # EN: Number of passes over the dataset (stream reuse for demo).
    eta0: float,  # EN: Initial learning rate.
    decay_steps: float,  # EN: Learning-rate decay timescale in "steps".
    eval_every: int,  # EN: Evaluate metrics every this many updates.
    V_ref_k: np.ndarray,  # EN: Reference top-k eigenvectors for evaluation (d×k).
    C_ref: np.ndarray,  # EN: Reference covariance matrix for explained variance computation (d×d).
    rng: np.random.Generator,  # EN: RNG for shuffling and initialization.
) -> tuple[np.ndarray, OjaHistory]:  # EN: Return learned W (d×k) and training history.
    n, d = Xc.shape  # EN: Extract dataset shape.
    if k <= 0 or k > d:  # EN: Validate k.
        raise ValueError("k must be in [1, d]")  # EN: Reject invalid k.
    if batch_size <= 0:  # EN: Validate batch size.
        raise ValueError("batch_size must be positive")  # EN: Reject invalid batch size.
    if n_epochs <= 0:  # EN: Validate epoch count.
        raise ValueError("n_epochs must be positive")  # EN: Reject invalid epoch count.

    # EN: Initialize W with random columns and orthonormalize (start on the Stiefel manifold).  # EN: Explain initialization.
    W = orthonormalize_columns(rng.standard_normal((d, k)).astype(float))  # EN: Random orthonormal basis for k-dim subspace.

    # EN: Prepare history buffers.  # EN: Explain what we record.
    steps: list[int] = []  # EN: Global steps at which we evaluate.
    min_cos: list[float] = []  # EN: Worst principal-angle cosine at each evaluation.
    mean_cos: list[float] = []  # EN: Mean principal-angle cosine at each evaluation.
    evr: list[float] = []  # EN: Explained variance ratio at each evaluation.

    global_step = 0  # EN: Count update steps across all epochs.

    for _ in range(n_epochs):  # EN: Loop over epochs (multiple passes for a clearer demo).
        perm = rng.permutation(n)  # EN: Shuffle sample order to reduce cyclic artifacts.
        for start in range(0, n, batch_size):  # EN: Iterate over mini-batches.
            idx = perm[start : start + batch_size]  # EN: Slice indices for this batch.
            Xb = Xc[idx, :]  # EN: Extract batch matrix (b×d).
            b = int(Xb.shape[0])  # EN: Actual batch size (last batch may be smaller).

            eta = learning_rate(step=global_step, eta0=eta0, decay_steps=decay_steps)  # EN: Compute learning rate for this step.

            # EN: Compute C_batch W without forming C_batch explicitly: (X^T X)W / b = X^T (XW) / b.  # EN: Explain matmul order.
            XW = Xb @ W  # EN: Project batch onto current subspace (b×k).
            Cw = (Xb.T @ XW) / max(b, 1)  # EN: Compute approximate covariance-times-W (d×k).

            # EN: Stiefel correction: remove component along W so update stays tangent to orthonormal constraints.  # EN: Explain correction term.
            G = Cw - W @ (W.T @ Cw)  # EN: Tangent-space gradient for maximizing variance with orthonormal columns.

            W = W + eta * G  # EN: Apply the Oja-style gradient step.
            W = orthonormalize_columns(W)  # EN: Re-orthonormalize for numerical stability and clean evaluation.

            global_step += 1  # EN: Advance global step counter.

            if eval_every > 0 and (global_step % eval_every == 0):  # EN: Periodically evaluate progress.
                cos = principal_angle_cosines(U_ref=V_ref_k, U_hat=W)  # EN: Compute subspace alignment to reference.
                steps.append(global_step)  # EN: Record step index.
                min_cos.append(float(np.min(cos)))  # EN: Record worst alignment cosine.
                mean_cos.append(float(np.mean(cos)))  # EN: Record average alignment cosine.
                evr.append(explained_variance_ratio(C=C_ref, W=W))  # EN: Record explained variance ratio.

    hist = OjaHistory(  # EN: Package recorded histories into a dataclass.
        steps=np.array(steps, dtype=int),  # EN: Convert step list to array.
        min_cos=np.array(min_cos, dtype=float),  # EN: Convert min_cos list to array.
        mean_cos=np.array(mean_cos, dtype=float),  # EN: Convert mean_cos list to array.
        evr=np.array(evr, dtype=float),  # EN: Convert evr list to array.
    )  # EN: End OjaHistory construction.
    return W, hist  # EN: Return learned basis and history.


def run_case(name: str, eigenvalues: np.ndarray, k: int) -> None:  # EN: Run one synthetic regime and show Oja convergence diagnostics.
    rng = np.random.default_rng(SEED)  # EN: Use deterministic RNG per case for comparable behavior.
    n_samples = 8000  # EN: Number of streaming samples (moderate for demo).
    noise_std = 0.20  # EN: Noise level to make the task non-trivial.
    batch_size = 128  # EN: Mini-batch size for each update.
    n_epochs = 3  # EN: Number of passes over the dataset (for clearer convergence plots in text form).
    eval_every = 25  # EN: Evaluate every N updates (keeps output compact).

    X, _ = build_synthetic_data_from_covariance(  # EN: Generate synthetic data with a controlled spectrum.
        n_samples=n_samples,  # EN: Provide sample count.
        eigenvalues=eigenvalues,  # EN: Provide target covariance spectrum.
        noise_std=noise_std,  # EN: Provide noise level.
        rng=rng,  # EN: Provide RNG.
    )  # EN: End data generation.

    Xc = X - np.mean(X, axis=0, keepdims=True)  # EN: Center data (important for PCA).
    C_ref, evals_ref, V_ref_k = reference_topk_from_covariance(Xc=Xc, k=k)  # EN: Compute reference top-k PCA.

    print_separator(f"Case: {name}")  # EN: Print case header.
    print(f"n_samples={n_samples}, d={Xc.shape[1]}, k={k}, noise_std={noise_std}")  # EN: Print setup summary.
    print("Top-10 eigenvalues (reference):", evals_ref[:10])  # EN: Show leading eigenvalues for intuition.
    print(f"Reference EVR(top-{k}) = {explained_variance_ratio(C=C_ref, W=V_ref_k):.6f}")  # EN: Print reference EVR for top-k.

    # EN: Run Oja with a reasonable decaying learning rate.  # EN: Explain why we pick a decaying schedule.
    eta0 = 1.0  # EN: Initial learning rate (tuned for this synthetic scale).
    decay_steps = 200.0  # EN: Decay timescale; larger -> slower decay.

    W, hist = block_oja_pca(  # EN: Train block Oja to learn the top-k subspace.
        Xc=Xc,  # EN: Provide centered data.
        k=k,  # EN: Provide target rank.
        batch_size=batch_size,  # EN: Provide batch size.
        n_epochs=n_epochs,  # EN: Provide epoch count.
        eta0=eta0,  # EN: Provide initial learning rate.
        decay_steps=decay_steps,  # EN: Provide decay schedule.
        eval_every=eval_every,  # EN: Provide evaluation frequency.
        V_ref_k=V_ref_k,  # EN: Provide reference top-k eigenvectors.
        C_ref=C_ref,  # EN: Provide covariance for EVR computation.
        rng=rng,  # EN: Provide RNG.
    )  # EN: End training call.

    # EN: Print a compact progress table.  # EN: Explain output formatting.
    if hist.steps.size > 0:  # EN: Only print if we actually recorded evaluations.
        print("\nstep | min_cos | mean_cos | EVR")  # EN: Print table header.
        for s, a, b, e in zip(hist.steps, hist.min_cos, hist.mean_cos, hist.evr):  # EN: Loop over history entries.
            print(f"{int(s):4d} | {a:7.4f} | {b:8.4f} | {e:7.4f}")  # EN: Print one row.

    cos_final = principal_angle_cosines(U_ref=V_ref_k, U_hat=W)  # EN: Compute final alignment.
    print("\nFinal principal-angle cosines:", cos_final)  # EN: Print final principal-angle cosines.
    print(f"Final EVR(top-{k}) = {explained_variance_ratio(C=C_ref, W=W):.6f}")  # EN: Print final EVR.
    print(f"Subspace distance (Fro) = {fro_norm(V_ref_k @ V_ref_k.T - W @ W.T):.3e}")  # EN: Print projection-matrix distance.


def main() -> None:  # EN: Run two regimes to show the effect of eigengap on Oja convergence.
    d = 80  # EN: Feature dimension.
    k = 5  # EN: Target number of principal components to learn.

    # EN: "Easy" regime: larger eigengap between λ_k and λ_{k+1} -> faster convergence.  # EN: Explain regime choice.
    eigenvalues_easy = np.concatenate([np.array([10.0, 4.0, 2.0, 1.0, 0.6]), 0.3 * np.ones(d - k)])  # EN: Build a spectrum with a clear gap.

    # EN: "Hard" regime: small eigengap -> slower / noisier convergence.  # EN: Explain why this is harder.
    eigenvalues_hard = np.concatenate([np.array([10.0, 9.5, 9.0, 8.5, 8.0]), 7.5 * np.ones(d - k)])  # EN: Build a spectrum with a small gap.

    run_case(name="Easy eigengap (faster convergence)", eigenvalues=eigenvalues_easy, k=k)  # EN: Run easy case.
    run_case(name="Hard eigengap (slower convergence)", eigenvalues=eigenvalues_hard, k=k)  # EN: Run hard case.

    print_separator("Done")  # EN: Print end marker.


if __name__ == "__main__":  # EN: Standard entrypoint guard.
    main()  # EN: Execute main.

