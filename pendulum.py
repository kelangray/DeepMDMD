"""
Nonlinear Pendulum
======================================================

Reproduces the nonlinear-pendulum Koopman experiments,
comparing DeepMDMD, MDMD and EDMD.

Sections
--------
1. System definition & trajectory generation
2. Cluster visualisation (trajectories · DeepMDMD · MDMD)
3. DeepMDMD training & embedding  (full model for eigenfunctions / spectra)
4. Basis construction, EDMD & eigendecomposition
5. Spectra & singular-value plot (saved)
6. Eigenfunction plot coloured on phase-plane trajectories (saved)

Dependencies
------------
    numpy, scipy, matplotlib, torch
    src.utils           (MDMD_matrix, calculate_residual)
    src                 (DeepMDMD)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import torch
from pathlib import Path
from scipy.integrate import solve_ivp
from matplotlib.collections import LineCollection
from sklearn.neighbors import NearestNeighbors

from src.deepmdmd import DeepMDMD
from src.utils import MDMD_matrix, EDMD_matrix, calculate_residual
 

# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# System
N_IC     = 20    # grid points per axis for initial conditions
IC_RANGE = 0.6   # initial conditions drawn from [-IC_RANGE, IC_RANGE]²

# Integration
T_TOTAL = 10.0
DT      = 0.1

# Cluster plot  (Section 2)
N_CLUSTERS_SMALL = 100                        # basis functions for the cluster figure
DIMS_SMALL       = [2, 128, 64, 10]          # encoder for the cluster-plot model
TRAJ_EXAMPLES    = [25, 50, 100, 150, 250, 399]   # trajectory indices to draw

# Full model  (Sections 3-7)
N_CLUSTERS = 1000
DIMS       = [2, 128, 64, 10]

# Eigenfunction mode selection  (Section 7)
SELECTED_MODES_DEEP = [2, 13, 24, 35, 46, 56]
SELECTED_MODES_MDMD = [0,  5, 10, 15, 19, 22]
N_MODES_TO_PLOT     = 6

# Eigenvalue / eigenvector filtering  (Sections 6 & 7)
MIN_SUPPORT  = 50     # minimum active clusters for an eigenvalue to be kept
SUPPORT_TOL  = 1e-12  # threshold for counting a cluster entry as "active"
LAMBDA_TOL   = 1e-8   # tolerance for discarding λ≈0 or λ≈1

# Figure output
FIGS_DIR = Path("results")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

# Font sizes — phase-plane figures (Sections 2, 7)
TITLE_FS  = 20
LABEL_FS  = 16
TICK_FS   = 14
SPINE_W   = 1.2

# Font sizes — spectra figure (Section 6)
SPECTRA_TITLE_FS  = 24
SPECTRA_LABEL_FS  = 20
SPECTRA_TICK_FS   = 18
SPECTRA_LEGEND_FS = 18


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------
def _style_phase_ax(ax, title=None, xlabel="x1", ylabel="x2",
                    xlim=(-1, 1), ylim=(-1.5, 1.5)):
    """Apply common phase-plane axis styling."""
    if title:
        ax.set_title(title, fontsize=TITLE_FS)
    ax.set_xlabel(xlabel, fontsize=LABEL_FS)
    ax.set_ylabel(ylabel, fontsize=LABEL_FS, rotation=360)
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_box_aspect(1)
    ax.yaxis.set_major_locator(ax.xaxis.get_major_locator())
    ax.tick_params(axis="both", labelsize=TICK_FS, width=SPINE_W)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_W)


def _style_spectra_ax(ax):
    """Apply spine / tick styling used in the spectra figure."""
    ax.tick_params(axis="both", labelsize=SPECTRA_TICK_FS, width=SPINE_W)
    for spine in ax.spines.values():
        spine.set_linewidth(SPINE_W)


def _colored_segments(ax, trajs, cluster_ids, n_clusters, cmap="plasma"):
    """Overlay cluster-coloured trajectory segments on *ax*."""
    n_segs = trajs.shape[1] - 1
    norm   = plt.Normalize(vmin=0, vmax=n_clusters - 1)
    for i, traj in enumerate(trajs):
        pts  = traj[:, :2]
        segs = np.stack([pts[:-1], pts[1:]], axis=1)
        lc   = LineCollection(segs, cmap=cmap, norm=norm, linewidth=1.0, alpha=0.95)
        lc.set_array(cluster_ids[i * n_segs: (i + 1) * n_segs])
        ax.add_collection(lc)


def _support_mask(E, V, min_support=MIN_SUPPORT, tol=SUPPORT_TOL):
    """Boolean mask: nonzero eigenvalue with sufficient eigenvector support."""
    V_arr   = np.asarray(V)
    support = np.sum(np.abs(V_arr) > tol, axis=0)
    return (E != 0) & (support > min_support), V_arr


def _sparse_safe_log(V, epsilon=1e-12):
    """Entrywise log of real eigenvector entries.
    Turns Hadamard products into sums: log(v_i * v_j) = log(v_i) + log(v_j).
    Computes log(max{|Re(v)|, epsilon}) for all entries, ensuring no -inf or NaN.
    """
    phi = np.asarray(V)
    return np.log(np.maximum(np.abs(np.real(phi)), epsilon))


# ---------------------------------------------------------------------------
# 1. System definition & trajectory generation
# ---------------------------------------------------------------------------
def pendulum(t, x):
    """Nonlinear pendulum RHS: dx1/dt = x2, dx2/dt = -sin(3*x1)"""
    x1, x2 = x
    return [x2, -np.sin(3 * x1)]


def make_initial_conditions(n_ic=N_IC, ic_range=IC_RANGE):
    """Return an array of grid initial conditions."""
    grid   = np.linspace(-ic_range, ic_range, n_ic)
    gx, gy = np.meshgrid(grid, grid)
    return np.column_stack((gx.flatten(), gy.flatten()))


def integrate_pendulum(x0_all):
    """Integrate the pendulum from every IC; return (trajs, X, Y)."""
    t_eval = np.linspace(0, T_TOTAL, int(T_TOTAL / DT))
    trajs  = []
    for x0_i in x0_all:
        sol = solve_ivp(pendulum, [0, T_TOTAL], x0_i, t_eval=t_eval, method="RK45")
        trajs.append(sol.y.T)
    trajs = np.array(trajs)   # (n_traj, N_steps, 2)
    X = trajs[:, :-1, :].reshape(-1, trajs.shape[2])
    Y = trajs[:, 1:,  :].reshape(-1, trajs.shape[2])
    print(f"  Trajectories: {trajs.shape[0]}   X shape: {X.shape}")
    return trajs, X, Y


# ---------------------------------------------------------------------------
# 2. Cluster visualisation  (trajectories · DeepMDMD · MDMD)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 3 & 4. Eigendecomposition
# ---------------------------------------------------------------------------
def prepare_spectrum(K_sparse, W,
                     support_threshold=MIN_SUPPORT,
                     tol=LAMBDA_TOL):
    """Eigendecompose K, Gram-normalise eigenvectors, filter, and sort by angle.

    Returns
    -------
    eigvals : filtered & sorted eigenvalues
    eigvecs : corresponding (Gram-normalised) eigenvectors
    theta   : corresponding angles in (-pi, pi]
    """
    K                = np.asarray(K_sparse.todense())
    eigvals, eigvecs = np.linalg.eig(K)

    # Gram normalisation
    G         = np.diag(np.sum(W, axis=1))
    gram_norm = np.real(np.einsum("ij,ij->j", np.conj(eigvecs), G @ eigvecs))
    valid     = gram_norm > 1e-12
    eigvecs[:, valid] /= np.sqrt(gram_norm[valid])

    # Sort by angle
    theta = np.arctan2(np.imag(eigvals), np.real(eigvals))
    order = np.argsort(theta)
    eigvals, eigvecs, theta = eigvals[order], eigvecs[:, order], theta[order]

    # Filter: positive angle, not lambda=1, nonzero, broad cluster support
    support = np.sum(np.abs(np.real(eigvecs)) > 1e-12, axis=0)
    mask = (
        (theta > 0.0)
        & (np.abs(eigvals - 1.0) > tol)
        & (np.abs(eigvals) > tol)
        & (support > support_threshold)
    )
    return eigvals[mask], eigvecs[:, mask], theta[mask]


# ---------------------------------------------------------------------------
# 5. Mode selection
# ---------------------------------------------------------------------------
def select_modes(n_available, requested):
    """Return valid integer indices into the filtered eigenvalue array."""
    if requested is not None:
        indices = np.asarray(requested, dtype=int)
        indices = indices[indices < n_available]
    else:
        n_ref   = min(N_MODES_TO_PLOT, n_available)
        indices = np.linspace(0, n_available - 1, n_ref, dtype=int)
    return indices[:N_MODES_TO_PLOT]


# ---------------------------------------------------------------------------
# 6. Spectra & singular-value plot (saved)
# ---------------------------------------------------------------------------
def plot_spectra(E_deepmdmd, V_deepmdmd, W_deepmdmd,
                 E_PCA,      V_PCA,      W_PCA,
                 E_edmd,     V_edmd,
                 G_edmd,     A_edmd,     L_edmd):
    """Two-row figure.

    Top row    : eigenvalue scatter on unit circle, coloured by residual
                 (DeepMDMD | MDMD | EDMD).
    Bottom row : singular-value decay of log-eigenvector matrices,
                 one panel per method (DeepMDMD | MDMD | EDMD).
    """
    # ── masks & residuals ────────────────────────────────────────────────────
    deep_mask, V_deep_arr = _support_mask(E_deepmdmd, V_deepmdmd)
    mdmd_mask, V_mdmd_arr = _support_mask(E_PCA,      V_PCA)
    edmd_support = np.sum(np.abs(np.asarray(V_edmd)) > SUPPORT_TOL, axis=0)
    edmd_mask    = np.isfinite(E_edmd) & (E_edmd != 0) & (edmd_support > MIN_SUPPORT)
    V_edmd_arr   = np.asarray(V_edmd)

    E_deep_plot = E_deepmdmd[deep_mask];  V_deep_plot = V_deep_arr[:, deep_mask]
    E_mdmd_plot = E_PCA[mdmd_mask];       V_mdmd_plot = V_mdmd_arr[:, mdmd_mask]
    E_edmd_plot = E_edmd[edmd_mask];      V_edmd_plot = V_edmd_arr[:, edmd_mask]

    print(
        f"  Spectra counts -> DeepMDMD: {E_deep_plot.size}/{E_deepmdmd.size}, "
        f"MDMD: {E_mdmd_plot.size}/{E_PCA.size}, EDMD: {E_edmd_plot.size}/{E_edmd.size}"
    )

    G_deep = np.diag(np.sum(W_deepmdmd, axis=1))
    L_deep = np.diag(np.sum(W_deepmdmd, axis=0))
    res_deep = np.array([
        calculate_residual(E_deep_plot[i], np.asarray(V_deep_plot[:, i]),
                           G_deep, W_deepmdmd, L_deep)
        for i in range(E_deep_plot.shape[0])
    ])

    G_mdmd = np.diag(np.sum(W_PCA, axis=1))
    L_mdmd = np.diag(np.sum(W_PCA, axis=0))
    res_mdmd = np.array([
        calculate_residual(E_mdmd_plot[i], np.asarray(V_mdmd_plot[:, i]),
                           G_mdmd, W_PCA, L_mdmd)
        for i in range(E_mdmd_plot.shape[0])
    ])

    res_edmd = np.array([
        calculate_residual(E_edmd_plot[i], np.asarray(V_edmd_plot[:, i]),
                           G_edmd, A_edmd, L_edmd)
        for i in range(E_edmd_plot.shape[0])
    ])

    non_empty = [r for r in [res_deep, res_mdmd, res_edmd] if r.size > 0]
    if not non_empty:
        raise ValueError("No eigenvalues available to plot.")
    res_max = max(r.max() for r in non_empty)
    res_min = min(r.min() for r in non_empty)

    # ── singular values ──────────────────────────────────────────────────────
    sv_deep = np.linalg.svd(_sparse_safe_log(V_deep_plot), compute_uv=False)
    sv_mdmd = np.linalg.svd(_sparse_safe_log(V_mdmd_plot), compute_uv=False)
    sv_edmd = np.linalg.svd(_sparse_safe_log(V_edmd_plot), compute_uv=False)

    sv_all  = np.concatenate([sv_deep, sv_mdmd, sv_edmd])
    sv_ymax = sv_all.max() * 2
    sv_ymin = max(sv_all[sv_all > 0].min() * 0.5, 1e-17)

    # ── layout ───────────────────────────────────────────────────────────────
    TOP_LEFT, TOP_RIGHT  = 0.06, 0.88
    TOP_TOP,  TOP_BOTTOM = 0.96, 0.26
    BOT_TOP,  BOT_BOTTOM = 0.30, 0.06
    CBAR_W,   CBAR_PAD   = 0.015, 0.02

    fig = plt.figure(figsize=(18, 11))
    gs_top = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                               left=TOP_LEFT, right=TOP_RIGHT,
                               bottom=TOP_BOTTOM, top=TOP_TOP)
    gs_bot = gridspec.GridSpec(1, 3, figure=fig, wspace=0.30,
                               left=TOP_LEFT, right=TOP_RIGHT,
                               bottom=BOT_BOTTOM, top=BOT_TOP)

    ax_deep    = fig.add_subplot(gs_top[0, 0])
    ax_mdmd    = fig.add_subplot(gs_top[0, 1])
    ax_edmd    = fig.add_subplot(gs_top[0, 2])
    ax_sv_deep = fig.add_subplot(gs_bot[0, 0])
    ax_sv_mdmd = fig.add_subplot(gs_bot[0, 1])
    ax_sv_edmd = fig.add_subplot(gs_bot[0, 2])

    # ── top row: eigenvalue scatter ──────────────────────────────────────────
    theta_circ  = np.linspace(0, 2 * np.pi, 500)
    unit_circle = np.exp(1j * theta_circ)

    plot_configs = [
        (ax_deep, E_deep_plot, res_deep, "DeepMDMD"),
        (ax_mdmd, E_mdmd_plot, res_mdmd, "MDMD"),
        (ax_edmd, E_edmd_plot, res_edmd, "EDMD"),
    ]

    scs = []
    for ax, E, res, title in plot_configs:
        if E.size > 0:
            sc = ax.scatter(E.real, E.imag, c=res, cmap="Spectral", s=30,
                            vmin=res_min, vmax=res_max)
            scs.append(sc)
        ax.plot(unit_circle.real, unit_circle.imag, "k-", linewidth=0.8)
        label = title if E.size > 0 else f"{title} (no eigenvalues)"
        ax.set_title(label,              fontsize=SPECTRA_TITLE_FS)
        ax.set_xlabel(r"Re($\lambda$)",  fontsize=SPECTRA_LABEL_FS)
        ax.set_ylabel(r"Im($\lambda$)",  fontsize=SPECTRA_LABEL_FS)
        ax.set_xticks([-1, -0.5, 0, 0.5, 1])
        ax.set_yticks([-1, -0.5, 0, 0.5, 1])
        ax.set_aspect("equal")
        _style_spectra_ax(ax)

    # Colorbar anchored to the height of the middle axes panel
    if scs:
        fig.canvas.draw()
        pos_mid = ax_mdmd.get_position()
        cbar_ax = fig.add_axes(
            [TOP_RIGHT + CBAR_PAD, pos_mid.y0, CBAR_W, pos_mid.y1 - pos_mid.y0]
        )
        cbar = fig.colorbar(scs[0], cax=cbar_ax)
        cbar.set_label("Residual", fontsize=SPECTRA_LABEL_FS)
        cbar.ax.tick_params(labelsize=SPECTRA_TICK_FS, width=SPINE_W)

    # ── bottom row: singular-value decay (three separate panels) ─────────────
    cmap = plt.cm.plasma
    sv_configs = [
        (ax_sv_deep, sv_deep, "DeepMDMD", cmap(0.1),  "^"),
        (ax_sv_mdmd, sv_mdmd, "MDMD",     cmap(0.5),  "s"),
        (ax_sv_edmd, sv_edmd, "EDMD",     cmap(0.65), "o"),
    ]
    for ax, sv, title, color, marker in sv_configs:
        ax.scatter(np.arange(len(sv)), sv, color=color, marker=marker, s=30)
        ax.set_yscale("log")
        ax.set_ylim(bottom=sv_ymin, top=sv_ymax)
        if title == "DeepMDMD":
            ax.set_xlim(left=0, right=500)
        elif title == "MDMD":
            ax.set_xlim(left=0, right=500)
        else:  # EDMD
            ax.set_xlim(left=0, right=1000)
        ax.set_aspect("auto")
        ax.set_xlabel("Index of singular value",  fontsize=SPECTRA_LABEL_FS)
        ax.set_ylabel("Singular value",           fontsize=SPECTRA_LABEL_FS)
        _style_spectra_ax(ax)

    out = FIGS_DIR / "pendulum_spectra.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Figure saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# 7. Eigenfunction plot coloured on phase-plane trajectories (saved)
# ---------------------------------------------------------------------------
def plot_eigenfunctions(trajs,
                        eigvecs_deep, theta_deep, xi_deep, modes_deep,
                        eigvecs_mdmd,  theta_mdmd,  xi_mdmd,  modes_mdmd):
    """
    Two-method, 2-row x 3-column panel.

    Top two rows    -> DeepMDMD eigenfunctions
    Bottom two rows -> MDMD eigenfunctions
    """
    n_plot     = min(len(modes_deep), len(modes_mdmd), N_MODES_TO_PLOT)
    modes_deep = modes_deep[:n_plot]
    modes_mdmd = modes_mdmd[:n_plot]
    n_segs     = trajs.shape[1] - 1
    ncols      = 3

    # Pre-compute eigenfunction values along all trajectories
    phi_deep_all = [np.real(np.asarray(eigvecs_deep[:, m]).ravel()[xi_deep]) for m in modes_deep]
    phi_mdmd_all = [np.real(np.asarray(eigvecs_mdmd[:, m]).ravel()[xi_mdmd]) for m in modes_mdmd]

    vmin_deep, vmax_deep = (np.min(np.concatenate(phi_deep_all)),
                            np.max(np.concatenate(phi_deep_all)))
    vmin_mdmd, vmax_mdmd = (np.min(np.concatenate(phi_mdmd_all)),
                            np.max(np.concatenate(phi_mdmd_all)))

    # 7-row grid: [plot, gap, plot, gap_BETWEEN_METHODS, plot, gap, plot]
    fig = plt.figure(figsize=(4.8 * ncols, 4.2 * 4.2))
    gs  = fig.add_gridspec(7, ncols,
                           height_ratios=[1.0, 0.3, 1.0, 0.55, 1.0, 0.3, 1.0])

    deep_axes = np.array([fig.add_subplot(gs[r, c]) for r in [0, 2] for c in range(ncols)])
    mdmd_axes = np.array([fig.add_subplot(gs[r, c]) for r in [4, 6] for c in range(ncols)])

    for r in [1, 3, 5]:
        for c in range(ncols):
            fig.add_subplot(gs[r, c]).axis("off")

    fig.subplots_adjust(wspace=0.28, hspace=0.0,
                        left=0.10, right=0.96, top=0.93, bottom=0.06)

    def _fill_axes(ax_list, phi_all, theta_arr, mode_indices, vmin, vmax):
        norm = plt.Normalize(vmin=vmin, vmax=vmax)
        for k, ax in enumerate(ax_list):
            if k >= n_plot:
                ax.axis("off")
                continue
            phi = phi_all[k]
            for i, traj in enumerate(trajs):
                phi_i = phi[i * n_segs: (i + 1) * n_segs]
                pts   = traj[:, :2]
                segs  = np.stack([pts[:-1], pts[1:]], axis=1)
                lc    = LineCollection(segs, cmap="Spectral", norm=norm,
                                       linewidth=1.0, alpha=0.9)
                lc.set_array(phi_i)
                ax.add_collection(lc)
            _style_phase_ax(
                ax, title=rf"$\lambda=\exp({theta_arr[mode_indices[k]]:.2f}i)$"
            )

    _fill_axes(deep_axes, phi_deep_all, theta_deep, modes_deep, vmin_deep, vmax_deep)
    _fill_axes(mdmd_axes, phi_mdmd_all, theta_mdmd, modes_mdmd, vmin_mdmd, vmax_mdmd)

    fig.text(0.04, 0.74,  "DeepMDMD", rotation=90, va="center", ha="center", fontsize=22)
    fig.text(0.04, 0.258, "MDMD",     rotation=90, va="center", ha="center", fontsize=22)

    out = FIGS_DIR / "pendulum_eigenfunctions.pdf"
    fig.savefig(out, bbox_inches="tight", pad_inches=0.5)
    print(f"  Figure saved: {out}")
    plt.show()


# ---------------------------------------------------------------------------
# Main execution
# ---------------------------------------------------------------------------

if __name__ == "__main__": 
    # 1. Trajectories
    print("Generating trajectories ...")
    x0_all      = make_initial_conditions()
    trajs, X, Y = integrate_pendulum(x0_all)

    # 2. Cluster visualisation
    print(f"\nPlotting cluster figure ...")
    print(f"  Building small bases (N={N_CLUSTERS_SMALL}) ...")
    # Train lightweight DeepMDMD for cluster visualization
    model_small, _ = DeepMDMD(
        X, Y,
        cluster_num=N_CLUSTERS_SMALL,
        dims=DIMS_SMALL,
        auto_epochs=50,
        Koopman_epochs=50,
        activation_fn="tanh",
        gamma=0,
        alpha=1,
        Koopman_interval=20,
        SEED=SEED,
    )
    model_small.eval()
    with torch.no_grad():
        X_emb_small = model_small.encoder(torch.tensor(X, dtype=torch.float32)).numpy()
        Y_emb_small = model_small.encoder(torch.tensor(Y, dtype=torch.float32)).numpy()

    _, _, xi_deep, _, _, _ = MDMD_matrix(X_emb_small, Y_emb_small, N=N_CLUSTERS_SMALL, seed=0)
    _, _, xi_mdmd, _, _, _ = MDMD_matrix(X,     Y,     N=N_CLUSTERS_SMALL, seed=0)

    fig, axes = plt.subplots(1, 3, figsize=(15, 5), constrained_layout=False)
    fig.subplots_adjust(wspace=0.35)

    # Left: example trajectories
    for i in TRAJ_EXAMPLES:
        pts = trajs[i, :, :]
        axes[0].plot(pts[:, 0], pts[:, 1],
                    color=plt.cm.plasma(0.1), linewidth=0.9, alpha=0.9)
    _style_phase_ax(axes[0], title="Trajectories")

    # Middle: DeepMDMD cluster assignments
    _colored_segments(axes[1], trajs, xi_deep, N_CLUSTERS_SMALL)
    _style_phase_ax(axes[1], title="DeepMDMD")

    # Right: MDMD cluster assignments
    _colored_segments(axes[2], trajs, xi_mdmd, N_CLUSTERS_SMALL)
    _style_phase_ax(axes[2], title="MDMD")

    out = FIGS_DIR / "pendulum_clusters.pdf"
    fig.savefig(out, bbox_inches="tight")
    print(f"  Figure saved: {out}")
    plt.show()

    # 3. DeepMDMD (full model)
    print("\nTraining DeepMDMD (full) ...")
    deepmdmd_model, _ = DeepMDMD(
        X, Y,
        cluster_num=N_CLUSTERS,
        dims=DIMS,
        auto_epochs=20,
        Koopman_epochs=20,
        activation_fn="tanh",
        gamma=0,
        alpha=1,
        Koopman_interval=20,
        SEED=SEED,
    )
    deepmdmd_model.eval()
    with torch.no_grad():
        X_emb = deepmdmd_model.encoder(torch.tensor(X, dtype=torch.float32)).numpy()
        Y_emb = deepmdmd_model.encoder(torch.tensor(Y, dtype=torch.float32)).numpy()

    # 4. Basis construction, EDMD & eigendecomposition
    print("\nBuilding bases ...")
    K_deep, W_deep, xi_deep, centroids_deep = MDMD_matrix(X_emb, Y_emb, N=N_CLUSTERS, seed=0)[:4]
    K_mdmd, W_mdmd, xi_mdmd, _ = MDMD_matrix(X,     Y,     N=N_CLUSTERS, seed=0)[:4]

    print("\nComputing EDMD ...")
    # Match notebook EDMD setup: build one-hot dictionary from DeepMDMD clusters.
    nbrs = NearestNeighbors(n_neighbors=1).fit(centroids_deep)
    yi_deep = nbrs.kneighbors(Y_emb, return_distance=False).flatten()

    psi_X = np.zeros((X.shape[0], N_CLUSTERS))
    psi_X[np.arange(X.shape[0]), xi_deep] = 1

    psi_Y = np.zeros((X.shape[0], N_CLUSTERS))
    psi_Y[np.arange(X.shape[0]), yi_deep] = 1

    E_edmd, V_edmd, G_edmd, A_edmd, L_edmd = EDMD_matrix(psi_X, psi_Y, W=None, return_full=True)
    print(f"  EDMD eigenvalues shape: {E_edmd.shape} (one-hot basis)")

    print("\nDecomposing MDMD / DeepMDMD spectra ...")
    eigvals_deep, eigvecs_deep, theta_deep = prepare_spectrum(K_deep, W_deep)
    eigvals_mdmd, eigvecs_mdmd, theta_mdmd = prepare_spectrum(K_mdmd, W_mdmd)

    # Raw eigendecompositions for the spectra / residual plot
    # Single eig call per matrix — results reused by plot_spectra
    E_deepmdmd, V_deepmdmd = np.linalg.eig(np.asarray(K_deep.todense()))
    E_PCA,      V_PCA      = np.linalg.eig(np.asarray(K_mdmd.todense()))

    # 5. Mode selection
    modes_deep = select_modes(len(eigvals_deep), SELECTED_MODES_DEEP)
    modes_mdmd = select_modes(len(eigvals_mdmd), SELECTED_MODES_MDMD)

    # 6. Spectra & singular-value plot
    print("\nPlotting spectra figure ...")
    plot_spectra(
        E_deepmdmd, V_deepmdmd, W_deep,
        E_PCA,      V_PCA,      W_mdmd,
        E_edmd,     V_edmd,
        G_edmd,     A_edmd,     L_edmd,
    )

    # 7. Eigenfunction plot
    print("\nPlotting eigenfunction figure ...")
    plot_eigenfunctions(
        trajs,
        eigvecs_deep, theta_deep, xi_deep, modes_deep,
        eigvecs_mdmd,  theta_mdmd,  xi_mdmd,  modes_mdmd)