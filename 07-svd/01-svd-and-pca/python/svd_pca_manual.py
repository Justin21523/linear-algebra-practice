"""  # EN: Start module docstring.
SVD + PCA (Manual-ish): build a thin SVD via eigen-decomposition (eigh).  # EN: Explain the script purpose in English.

This file avoids calling np.linalg.svd() on purpose, so you can see the relationship:  # EN: Explain the constraint.
  A^T A v = (sigma^2) v, and then u = (A v) / sigma.  # EN: Summarize the core math identity.

It also demonstrates PCA via SVD on a centered data matrix X.  # EN: Mention the ML application.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references for type hints.

import math  # EN: Import math utilities for printing or sanity checks.
from dataclasses import dataclass  # EN: Use dataclass for small structured results.

import numpy as np  # EN: Import NumPy for array/matrix computations.


EPS = 1e-12  # EN: Numerical tolerance used to treat singular values as zero.
PRINT_PRECISION = 6  # EN: Default numeric printing precision for readability.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Configure NumPy printing for nicer console output.


@dataclass(frozen=True)  # EN: Create an immutable container for a thin SVD result.
class ThinSVD:  # EN: Store U, s, Vt for a thin SVD A ≈ U diag(s) Vt.
    U: np.ndarray  # EN: Left singular vectors (m×r) with orthonormal columns.
    s: np.ndarray  # EN: Singular values (r,) in descending order.
    Vt: np.ndarray  # EN: Right singular vectors transposed (r×n).


def print_separator(title: str) -> None:  # EN: Print a section separator for console readability.
    print()  # EN: Add a blank line before a section.
    print("=" * 70)  # EN: Print a horizontal separator line.
    print(title)  # EN: Print the section title.
    print("=" * 70)  # EN: Print another horizontal separator line.


def is_orthonormal_columns(Q: np.ndarray, tol: float = 1e-10) -> bool:  # EN: Check whether Q has orthonormal columns.
    if Q.ndim != 2:  # EN: Ensure we are working with a 2D matrix.
        raise ValueError("Q must be a 2D matrix")  # EN: Fail fast if the input shape is invalid.
    k = Q.shape[1]  # EN: Number of columns to check.
    gram = Q.T @ Q  # EN: Compute the Gram matrix of columns; should be close to I.
    return np.allclose(gram, np.eye(k), atol=tol, rtol=0.0)  # EN: Return True if the Gram matrix is near identity.


def frobenius_norm(A: np.ndarray) -> float:  # EN: Compute the Frobenius norm ‖A‖_F.
    return float(np.linalg.norm(A, ord="fro"))  # EN: Use NumPy's norm implementation with 'fro' order.


def thin_svd_via_eigh(A: np.ndarray, eps: float = EPS) -> ThinSVD:  # EN: Build a thin SVD using eigen-decomposition of A^T A.
    if A.ndim != 2:  # EN: Ensure A is a matrix.
        raise ValueError("A must be a 2D matrix")  # EN: Guard against invalid input.

    m, n = A.shape  # EN: Extract matrix dimensions.
    AtA = A.T @ A  # EN: Form the symmetric positive semidefinite matrix A^T A (n×n).

    # EN: eigh() is specialized for symmetric matrices and is numerically stable for AtA.
    eigenvalues, V = np.linalg.eigh(AtA)  # EN: Compute eigenvalues/eigenvectors of A^T A.

    # EN: Sort eigenpairs by descending eigenvalue so singular values are descending.
    order = np.argsort(eigenvalues)[::-1]  # EN: Indices that would sort eigenvalues descending.
    eigenvalues_sorted = eigenvalues[order]  # EN: Sort eigenvalues accordingly.
    V_sorted = V[:, order]  # EN: Reorder eigenvectors to match sorted eigenvalues.

    # EN: Singular values are sqrt of eigenvalues (clipped to avoid tiny negative due to floating error).
    singular_values = np.sqrt(np.clip(eigenvalues_sorted, 0.0, None))  # EN: Convert eigenvalues to singular values.

    # EN: Keep only non-zero (above eps) singular values for a thin SVD.
    keep = singular_values > eps  # EN: Boolean mask selecting non-zero singular values.
    s = singular_values[keep]  # EN: Keep the non-zero singular values (r,).
    V_r = V_sorted[:, keep]  # EN: Keep corresponding right singular vectors (n×r).

    if s.size == 0:  # EN: Handle the all-zero matrix case.
        U_empty = np.zeros((m, 0), dtype=float)  # EN: Provide an empty U with 0 columns.
        Vt_empty = np.zeros((0, n), dtype=float)  # EN: Provide an empty V^T with 0 rows.
        return ThinSVD(U=U_empty, s=s, Vt=Vt_empty)  # EN: Return a valid thin SVD container.

    # EN: Build U via u_i = (A v_i) / sigma_i (broadcast division over columns).
    U = A @ V_r  # EN: Compute A times each right singular vector to get unnormalized left vectors.
    U = U / s  # EN: Normalize each column by its singular value to get left singular vectors.

    # EN: Thin SVD uses Vt with shape (r×n).
    Vt = V_r.T  # EN: Transpose V to get V^T.

    return ThinSVD(U=U, s=s, Vt=Vt)  # EN: Return the computed thin SVD.


def reconstruct_from_thin_svd(svd: ThinSVD) -> np.ndarray:  # EN: Reconstruct A from a thin SVD representation.
    Sigma = np.diag(svd.s)  # EN: Build the r×r diagonal matrix of singular values.
    return svd.U @ Sigma @ svd.Vt  # EN: Reconstruct A ≈ U Σ V^T.


def best_rank_k_approximation(svd: ThinSVD, k: int) -> np.ndarray:  # EN: Compute the best rank-k approximation using top-k singular values.
    if k < 0:  # EN: Validate k is non-negative.
        raise ValueError("k must be non-negative")  # EN: Raise an error for invalid rank.
    r = svd.s.size  # EN: Rank of the thin SVD (number of kept singular values).
    k_eff = min(k, r)  # EN: Effective k cannot exceed r.
    if k_eff == 0:  # EN: Rank-0 approximation is the zero matrix.
        m = svd.U.shape[0]  # EN: Number of rows in the original matrix.
        n = svd.Vt.shape[1]  # EN: Number of columns in the original matrix.
        return np.zeros((m, n), dtype=float)  # EN: Return a zero matrix with the correct shape.

    U_k = svd.U[:, :k_eff]  # EN: Take the first k columns of U.
    s_k = svd.s[:k_eff]  # EN: Take the first k singular values.
    Vt_k = svd.Vt[:k_eff, :]  # EN: Take the first k rows of V^T.
    return U_k @ np.diag(s_k) @ Vt_k  # EN: Return A_k = U_k Σ_k V_k^T.


def pca_via_thin_svd(X: np.ndarray, k: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # EN: Perform PCA via thin SVD on centered data.
    if X.ndim != 2:  # EN: Ensure X is a 2D data matrix.
        raise ValueError("X must be a 2D matrix (n_samples×n_features)")  # EN: Enforce expected PCA input shape.

    n_samples, n_features = X.shape  # EN: Extract dataset dimensions.
    if n_samples < 2:  # EN: PCA needs at least 2 samples for variance estimates.
        raise ValueError("Need at least 2 samples for PCA")  # EN: Fail early for degenerate input.
    if k <= 0 or k > n_features:  # EN: Validate requested component count.
        raise ValueError("k must be in [1, n_features]")  # EN: Reject invalid k.

    mean = X.mean(axis=0)  # EN: Compute feature-wise mean for centering.
    Xc = X - mean  # EN: Center the data (important for PCA).

    # EN: Compute thin SVD of the centered data matrix.
    svd = thin_svd_via_eigh(Xc)  # EN: Use our manual-ish thin SVD implementation.

    # EN: Right singular vectors correspond to principal directions (components).
    V = svd.Vt.T  # EN: Convert V^T to V to get principal directions as columns.
    components = V[:, :k]  # EN: Select the first k principal directions.

    # EN: Project centered data to k-dimensional latent space.
    Z = Xc @ components  # EN: Compute PCA scores (n_samples×k).

    # EN: Reconstruct data back to the original feature space.
    X_hat = Z @ components.T + mean  # EN: Undo the projection and add the mean back.

    # EN: Explained variance is (singular_value^2) / (n_samples-1).
    eigenvalues = (svd.s**2) / (n_samples - 1)  # EN: Convert singular values to covariance eigenvalues.
    explained_variance_ratio = eigenvalues / eigenvalues.sum()  # EN: Normalize to get variance ratio per component.

    return Z, X_hat, eigenvalues, explained_variance_ratio  # EN: Return PCA outputs for analysis.


def main() -> None:  # EN: Run SVD and PCA demos with verification checks.
    print_separator("SVD Demo (Manual-ish via eigh): A = U Σ V^T")  # EN: Introduce the SVD demo.

    # EN: Use a small, readable example matrix for demonstration.
    A = np.array(  # EN: Define a demo matrix A.
        [  # EN: Start matrix literal rows.
            [3.0, 1.0],  # EN: Row 1.
            [2.0, 2.0],  # EN: Row 2.
            [0.0, 2.0],  # EN: Row 3.
        ],  # EN: End matrix literal rows.
        dtype=float,  # EN: Ensure float computations.
    )  # EN: Finish defining A.

    print("A =\n", A)  # EN: Print A for reference.

    svd = thin_svd_via_eigh(A)  # EN: Compute thin SVD using eigen-decomposition.

    print("\nU (thin) =\n", svd.U)  # EN: Print left singular vectors.
    print("\ns (singular values) =\n", svd.s)  # EN: Print singular values.
    print("\nV^T (thin) =\n", svd.Vt)  # EN: Print right singular vectors transposed.

    print_separator("Verify orthonormality and reconstruction")  # EN: Introduce verification checks.

    print("U has orthonormal columns? ->", is_orthonormal_columns(svd.U))  # EN: Check U^T U ≈ I.
    print("V has orthonormal columns? ->", is_orthonormal_columns(svd.Vt.T))  # EN: Check V^T V ≈ I.

    A_hat = reconstruct_from_thin_svd(svd)  # EN: Reconstruct A from U, s, V^T.
    err = frobenius_norm(A - A_hat)  # EN: Compute reconstruction error norm.
    print("\nA_hat =\n", A_hat)  # EN: Show reconstructed matrix.
    print(f"\nReconstruction error ||A - A_hat||_F = {err:.3e}")  # EN: Print reconstruction error.

    print_separator("Best rank-k approximation (k=1)")  # EN: Introduce low-rank approximation.
    A_1 = best_rank_k_approximation(svd, k=1)  # EN: Build rank-1 approximation using top singular triplet.
    err_1 = frobenius_norm(A - A_1)  # EN: Compute rank-1 approximation error.
    print("A_1 (rank-1) =\n", A_1)  # EN: Print rank-1 approximation.
    print(f"\n||A - A_1||_F = {err_1:.3e}")  # EN: Print approximation error.

    print_separator("PCA via SVD (on a small 2D toy dataset)")  # EN: Introduce the PCA demo.

    # EN: Classic 2D toy dataset used in many PCA tutorials.
    X = np.array(  # EN: Define sample data matrix X (n_samples×n_features).
        [  # EN: Start listing samples.
            [2.5, 2.4],  # EN: Sample 1.
            [0.5, 0.7],  # EN: Sample 2.
            [2.2, 2.9],  # EN: Sample 3.
            [1.9, 2.2],  # EN: Sample 4.
            [3.1, 3.0],  # EN: Sample 5.
            [2.3, 2.7],  # EN: Sample 6.
            [2.0, 1.6],  # EN: Sample 7.
            [1.0, 1.1],  # EN: Sample 8.
            [1.5, 1.6],  # EN: Sample 9.
            [1.1, 0.9],  # EN: Sample 10.
        ],  # EN: End listing samples.
        dtype=float,  # EN: Ensure float computations.
    )  # EN: Finish defining X.

    Z, X_hat, eigenvalues, evr = pca_via_thin_svd(X, k=1)  # EN: Compute 1D PCA projection and reconstruction.

    print("X (original) =\n", X)  # EN: Print original dataset.
    print("\nZ (1D projected) =\n", Z)  # EN: Print latent 1D coordinates.
    print("\nX_hat (reconstructed from 1D) =\n", X_hat)  # EN: Print reconstructed dataset.

    recon_err = frobenius_norm(X - X_hat)  # EN: Compute reconstruction error for PCA reconstruction.
    print(f"\nPCA reconstruction error ||X - X_hat||_F = {recon_err:.3e}")  # EN: Print reconstruction error.

    print("\nEigenvalues of covariance (via SVD) =\n", eigenvalues)  # EN: Print covariance eigenvalues.
    print("\nExplained variance ratio =\n", evr)  # EN: Print explained variance ratio.

    # EN: Sanity check: explained variance ratio should sum to ~1.
    print("\nSum(explained_variance_ratio) =", float(evr.sum()))  # EN: Print the sum of EVR.

    print_separator("Done")  # EN: Mark the end of the script.


if __name__ == "__main__":  # EN: Standard Python entrypoint guard.
    main()  # EN: Execute the demo when run as a script.

