import numpy as np
from scipy.sparse import coo_matrix
from sklearn.neighbors import NearestNeighbors
from sklearn.cluster import KMeans


def EDMD_matrix(
    psi_X: np.ndarray,
    psi_Y: np.ndarray,
    W: np.ndarray = None,
    return_full: bool = True
) -> tuple:
    """
    Compute the EDMD Koopman matrix K from basis matrices psi_X and psi_Y,
    optionally using a quadrature / weight matrix.

    The EDMD matrices are defined as:
        G = psi_X^T W psi_X
        A = psi_X^T W psi_Y
        L = psi_Y^T W psi_Y
    The Koopman operator is then computed via
        K = G^{pseudoinv} A

    Args:
        psi_X (np.ndarray): Basis function evaluations at time t, shape (M, N),
                            where M = number of data points, N = number of basis functions.
        psi_Y (np.ndarray): Basis function evaluations at time t+1, shape (M, N).
        W (np.ndarray, optional): Quadrature / weight matrix, shape (M, M).
                                  If None, uniform weights W = I/M are used.
                                  Can be diagonal (1D) or full (2D) matrix.
        return_full (bool, optional): If True, return (E, V, G, A, L) with eigendecomposition.
                                      If False, return just K. Default True.
    Returns:
        If return_full=True:
            tuple: (E, V, G, A, L) where
                E = eigenvalues of K
                V = eigenvectors of K
                G, A, L = Gram and cross-covariance matrices
        If return_full=False:
            np.ndarray: EDMD Koopman matrix K, shape (N, N)
    """
    M, N = psi_X.shape

    # Default uniform weighting
    if W is None:
        W = np.eye(M) / M
    else:
        # Convert 1D weight array to diagonal matrix
        if W.ndim == 1:
            W = np.diag(W)
        # Check shape
        if W.shape != (M, M):
            raise ValueError(f"Quadrature weight matrix W must have shape ({M}, {M})")

    # Compute Gram and cross-covariance matrices
    G = psi_X.conj().T @ W @ psi_X
    A = psi_X.conj().T @ W @ psi_Y
    L = psi_Y.conj().T @ W @ psi_Y

    # Calculate K
    K = np.linalg.solve(G, A)

    if return_full:
        E, V = np.linalg.eig(K)
        return E, V, G, A, L
    else:
        return K


def MDMD_matrix(X, Y, N, X_kmeans=None, seed: int = 0):
    """ 
    Calculate the Koopman matrix using MultDMD algorithm.

    Args: 
        X : ndarray (M, d)
            Input states (M samples, d-dimensional) at time t.
        Y : ndarray (M, d)
            Output states (M samples, d-dimensional) at time t+1 .
        N : int
            Number of clusters / basis functions.
        X_kmeans : ndarray, optional
            Dataset used to compute K-means centroids. If None, X is used.
        seed : int, optional
            Random seed for KMeans initialization. Defaults to 0.

    Returns:
        K_sparse : scipy.sparse matrix
            Sparse Koopman operator approximation.
        W : ndarray (N, N)
            Transition count matrix.
        xi : ndarray (M,)
            Cluster index assigned to each sample in X.
        centroids : ndarray (N, d)
            Cluster centroids defining the basis regions.
        E : ndarray
            Eigenvalues of K.
        V : ndarray
            Eigenvectors of K (canonicalized for reproducibility).
    """

    
    #If kmeans isn't fed in set the kmeans to the X data 
    if X_kmeans is None:
        X_kmeans = X

    #Construct centroids using kmeans
    kmeans = KMeans(n_clusters=N, random_state=seed, init='k-means++').fit(X_kmeans)
    centroids = kmeans.cluster_centers_
    nbrs = NearestNeighbors(n_neighbors=1).fit(centroids)

    M = X.shape[0]
    weight = np.ones(M) / M

    #Create W_ij matrix
    xi = nbrs.kneighbors(X, return_distance=False).flatten()
    xj = nbrs.kneighbors(Y, return_distance=False).flatten()

    pairs, H = np.unique(np.column_stack((xi, xj)), axis=0, return_inverse=True)
    ID1 = pairs[:, 0]
    ID2 = pairs[:, 1]
    T = np.bincount(H, weights=weight)

    #Spare coordinate matrix 
    W_sparse = coo_matrix((T, (ID1, ID2)), shape=(max(ID1) + 1, max(ID2) + 1))
    W = W_sparse.toarray()

    G = np.sum(W, axis=1)

    #Martix of minimisation problem
    #Filter out rows where G is zero to avoid division by zero
    non_zero_G_indices = np.where(G != 0)[0]
    G_filtered = G[non_zero_G_indices]
    W_filtered = W[non_zero_G_indices, :]

    min_j_g = (G_filtered[:, np.newaxis] - 2 * W_filtered) / G_filtered[:, np.newaxis]

    N_min = min_j_g.shape[0]
    I = np.argmin(min_j_g, axis=1)
    bi = np.arange(N_min)
    v = np.ones_like(I)

    # Create the K matrix with shape (N, N)
    K_sparse = coo_matrix((v, (non_zero_G_indices[bi], I)), shape=(N, N))

    # Compute eigendecomposition with canonical phase
    K = np.array(K_sparse.todense())
    E, V = np.linalg.eig(K)
    V = np.asarray(V)
    idx = np.argmax(np.abs(V), axis=0)
    phase = V[idx, np.arange(V.shape[1])]
    V /= phase / np.abs(phase)

    return K_sparse, W, xi, centroids, E, V


def calculate_residual(lam: complex, g: np.ndarray, G: np.ndarray, A: np.ndarray, L: np.ndarray) -> float:
    """
    Compute the residual for a single candidate eigenpair (lam, g).

    For a basis of N functions and M data points, the reisdual is computed as:

        r = sqrt( Re( g^T L g - lam g A.T g - conj(lam) g^T A g + |lam|^2 g^T G g)
                / Re(g^T G g) )

    Where:
        - G = (psi_X^T W psi_X)     : Gram matrix of basis functions at time t (quadrature weighted)
        - A = (psi_X^T W psi_Y)     : Cross-covariance matrix (quadrature weighted) 
        - L = (psi_Y^T W psi_Y)     : Gram-like matrix at next timestep (quadrature weighted)
        - g : candidate eigenvector shape (N,)
        - lam : candidate eigenvalue

    Args:
        lam (complex): Candidate eigenvalue lam.
        g (np.ndarray): Candidate eigenvector, shape (N,). 
        G (np.ndarray): Gram matrix of basis functions at time t, shape (N, N).
        A (np.ndarray): Cross-covariance matrix, shape (N, N).
        L (np.ndarray): Gram matrix at next timestep, shape (N, N).

    Returns:
        float: Residual of the eigenpair. Guaranteed non-negative. 
            Returns 0 if the denominator is zero or the computed ratio is negative.
    """
    
    # Compute numerator components
    term_L = np.vdot(g, L @ g)
    term_A_T = lam * np.vdot(g, A.T @ g)
    term_A = np.conj(lam) * np.vdot(g, A @ g)
    term_G = (np.abs(lam) ** 2) * np.vdot(g, G @ g)

    # Numerator
    numerator = np.real(term_L - term_A_T - term_A + term_G)

    # Denominator
    denominator = np.real(np.vdot(g, G @ g))

    # Safe division
    ratio = numerator / denominator if denominator != 0 else 0.0

    # Residual is sqrt of positive ratio, else 0
    return np.sqrt(ratio) if ratio > 0 else 0.0


def calculate_autocorr(snapshot_matrix: np.ndarray, num_lags: int = 750) -> np.ndarray:
    """
    Compute the autocorrelation of a time series dataset.

    Parameters
    ----------
    snapshot_matrix : np.ndarray
        2D array of shape (M, N) where M is the number of time snapshots 
        and N is the number of features (e.g., flattened vorticity field).
    num_lags : int, optional
        Number of time lags to compute the autocorrelation for. Default is 750.

    Returns
    -------
    np.ndarray
        2D array of shape (num_lags, N) containing the log of the absolute 
        value of the autocorrelation for each feature at each lag.

    Raises
    ------
    ValueError
        If `num_lags` is greater than or equal to the number of snapshots in the matrix.
    """

    M, N = snapshot_matrix.shape

    if num_lags >= M:
        raise ValueError(f"num_lags ({num_lags}) must be smaller than the number of snapshots ({M}).")

    autocorr_matrix = np.zeros((num_lags, N))

    for lag in range(1, num_lags):
        # Slice the matrix to align current and lagged data
        current_data = snapshot_matrix[:-lag, :]
        lagged_data = snapshot_matrix[lag:, :]

        # Compute the mean correlation for the current lag
        mean_corr = np.mean(current_data * lagged_data, axis=0)

        # Store the log of absolute value, adding a small epsilon to avoid log(0)
        autocorr_matrix[lag, :] = np.log(np.abs(mean_corr) + 1e-12)

    return autocorr_matrix