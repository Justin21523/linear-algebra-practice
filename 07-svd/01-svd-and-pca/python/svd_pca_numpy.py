"""  # EN: Start module docstring.
SVD + PCA (NumPy): use np.linalg.svd() directly and verify properties.  # EN: Explain the script purpose.

This file is the "library-backed" counterpart to svd_pca_manual.py.  # EN: Describe relationship to manual file.
It demonstrates:  # EN: Introduce list of demonstrations.
  - SVD reconstruction A ≈ U Σ V^T  # EN: Mention reconstruction.
  - Low-rank approximation via top-k singular values  # EN: Mention low-rank approximation.
  - PCA via SVD on centered data, with explained variance ratio  # EN: Mention PCA results.
"""  # EN: End module docstring.

from __future__ import annotations  # EN: Enable forward references for type hints.

import numpy as np  # EN: Import NumPy for numerical linear algebra.


EPS = 1e-12  # EN: Numerical tolerance for comparisons.
PRINT_PRECISION = 6  # EN: Default printing precision.

np.set_printoptions(precision=PRINT_PRECISION, suppress=True)  # EN: Make NumPy arrays easier to read in console.


def print_separator(title: str) -> None:  # EN: Print a section separator to structure output.
    print()  # EN: Add whitespace before the separator.
    print("=" * 70)  # EN: Print a horizontal rule line.
    print(title)  # EN: Print section title.
    print("=" * 70)  # EN: Print another horizontal rule.


def frobenius_norm(A: np.ndarray) -> float:  # EN: Compute Frobenius norm of a matrix.
    return float(np.linalg.norm(A, ord="fro"))  # EN: Delegate to NumPy with Frobenius order.


def build_sigma(s: np.ndarray, m: int, n: int) -> np.ndarray:  # EN: Build a full (m×n) Sigma matrix from singular values.
    r = s.size  # EN: r = min(m, n) for full SVD output.
    Sigma = np.zeros((m, n), dtype=float)  # EN: Initialize Sigma as a zero matrix.
    Sigma[:r, :r] = np.diag(s)  # EN: Place singular values on the diagonal.
    return Sigma  # EN: Return the constructed Sigma matrix.


def best_rank_k_from_full_svd(U: np.ndarray, s: np.ndarray, Vt: np.ndarray, k: int) -> np.ndarray:  # EN: Compute rank-k approximation from full SVD.
    if k < 0:  # EN: Validate k.
        raise ValueError("k must be non-negative")  # EN: Reject invalid k.
    r = s.size  # EN: Number of available singular values.
    k_eff = min(k, r)  # EN: Clamp k to the available rank.
    if k_eff == 0:  # EN: Rank-0 approximation is zeros.
        return np.zeros((U.shape[0], Vt.shape[1]), dtype=float)  # EN: Return zeros with correct shape.
    return U[:, :k_eff] @ np.diag(s[:k_eff]) @ Vt[:k_eff, :]  # EN: Compute A_k = U_k Σ_k V_k^T.


def pca_via_svd(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:  # EN: Perform PCA using SVD on centered data.
    if X.ndim != 2:  # EN: Ensure X is a 2D matrix.
        raise ValueError("X must be a 2D matrix (n_samples×n_features)")  # EN: Guard invalid input.
    n_samples, n_features = X.shape  # EN: Extract data shape.
    if n_samples < 2:  # EN: Need at least 2 samples.
        raise ValueError("Need at least 2 samples for PCA")  # EN: Fail for degenerate case.

    mean = X.mean(axis=0)  # EN: Compute feature-wise mean.
    Xc = X - mean  # EN: Center data before SVD/PCA.

    # EN: Full SVD: Xc = U Σ V^T, where columns of V are principal directions.
    U, s, Vt = np.linalg.svd(Xc, full_matrices=False)  # EN: Compute economy SVD (thin).

    # EN: Covariance eigenvalues are (s^2)/(n_samples-1).
    eigenvalues = (s**2) / (n_samples - 1)  # EN: Convert singular values to covariance eigenvalues.
    explained_variance_ratio = eigenvalues / eigenvalues.sum()  # EN: Normalize to ratios.

    components = Vt.T  # EN: Principal directions as columns (n_features×n_features).
    scores = Xc @ components  # EN: Project to PCA coordinates (n_samples×n_features).

    return mean, Xc, components, eigenvalues, explained_variance_ratio  # EN: Return PCA artifacts.


def main() -> None:  # EN: Run SVD and PCA demos using NumPy SVD.
    print_separator("SVD Demo (NumPy): A = U Σ V^T")  # EN: Announce SVD demo.

    A = np.array(  # EN: Define the same demo matrix as in manual version for comparison.
        [  # EN: Start listing rows.
            [3.0, 1.0],  # EN: Row 1.
            [2.0, 2.0],  # EN: Row 2.
            [0.0, 2.0],  # EN: Row 3.
        ],  # EN: End listing rows.
        dtype=float,  # EN: Ensure float dtype.
    )  # EN: Finish defining A.

    print("A =\n", A)  # EN: Print A.

    U, s, Vt = np.linalg.svd(A, full_matrices=True)  # EN: Compute full SVD of A.
    Sigma = build_sigma(s, m=A.shape[0], n=A.shape[1])  # EN: Convert singular values into Sigma matrix.

    A_hat = U @ Sigma @ Vt  # EN: Reconstruct A from SVD.
    err = frobenius_norm(A - A_hat)  # EN: Compute reconstruction error.

    print("\nU =\n", U)  # EN: Print U.
    print("\ns =\n", s)  # EN: Print singular values.
    print("\nV^T =\n", Vt)  # EN: Print V^T.
    print("\nA_hat =\n", A_hat)  # EN: Print reconstructed A.
    print(f"\nReconstruction error ||A - A_hat||_F = {err:.3e}")  # EN: Print reconstruction error.

    print_separator("Low-rank approximation (k=1)")  # EN: Announce rank-k approximation demo.
    A_1 = best_rank_k_from_full_svd(U, s, Vt, k=1)  # EN: Build best rank-1 approximation from SVD.
    err_1 = frobenius_norm(A - A_1)  # EN: Compute approximation error.
    print("A_1 (rank-1) =\n", A_1)  # EN: Print rank-1 approximation.
    print(f"\n||A - A_1||_F = {err_1:.3e}")  # EN: Print approximation error.

    print_separator("PCA via SVD (centered data)")  # EN: Announce PCA demo.

    X = np.array(  # EN: Define the toy dataset.
        [  # EN: Samples.
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
        ],  # EN: End samples.
        dtype=float,  # EN: Ensure float dtype.
    )  # EN: Finish defining X.

    mean, Xc, components, eigenvalues, evr = pca_via_svd(X)  # EN: Run PCA and get key artifacts.

    print("mean =\n", mean)  # EN: Print mean vector.
    print("\nFirst principal direction (v1) =\n", components[:, 0])  # EN: Print first principal direction.
    print("\nEigenvalues (covariance) =\n", eigenvalues)  # EN: Print covariance eigenvalues.
    print("\nExplained variance ratio =\n", evr)  # EN: Print explained variance ratio.

    # EN: Demonstrate 1D projection and reconstruction using the first component.
    k = 1  # EN: Use one principal component.
    W = components[:, :k]  # EN: Projection matrix (n_features×k).
    Z = Xc @ W  # EN: Project centered data to 1D.
    X_hat = Z @ W.T + mean  # EN: Reconstruct to original space.
    recon_err = frobenius_norm(X - X_hat)  # EN: Compute reconstruction error.

    print("\nZ (1D scores) =\n", Z)  # EN: Print projected scores.
    print("\nX_hat (reconstructed) =\n", X_hat)  # EN: Print reconstructed data.
    print(f"\nPCA reconstruction error ||X - X_hat||_F = {recon_err:.3e}")  # EN: Print reconstruction error.

    print_separator("Done")  # EN: End of demo.


if __name__ == "__main__":  # EN: Python entrypoint.
    main()  # EN: Run main when executed as a script.

