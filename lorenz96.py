"""
Lorenz-96
======================================================

Reproduces the Lorenz-96 experiments.

Sections
--------
1. System definition & trajectory generation
2. DeepMDMD training & embedding (per forcing value)
3. Koopman eigendecomposition
4. Koopman eigenfunction plot coloured on latent trajectory (saved)

Dependencies
------------
    numpy, scipy, matplotlib, torch
    src.utils            (MDMD_matrix)
    src                  (DeepMDMD)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import torch
from pathlib import Path
from scipy.integrate import solve_ivp
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from src import utils
from src.deepmdmd import DeepMDMD


# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# System
N_DIM    = 9        # Lorenz-96 state dimension
F_VALUES = [2.0, 3.5, 4.2]

# Integration
T_SPINUP  = 100
T_TOTAL   = 1000
DT        = 0.1
N_DISCARD = 1000    # snapshots discarded after spin-up

# Model
N_CLUSTERS = 1000
DIMS       = [N_DIM, 128, 64, 3]   # encoder architecture; last entry = latent dim

# Eigenfunction plot
ROW_IDS_BY_F = {
    2.0: [4,  8, 12],
    3.5: [6, 12, 18],
    4.2: [8, 10, 12],
}
ROW_LABELS = {
    2.0: "Periodic",
    3.5: "Quasi-periodic",
    4.2: "Chaotic",
}

# Figure output
FIGS_DIR = Path("results")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

TITLE_FS = 26
LABEL_FS = 18
TICK_FS  = 16


# ---------------------------------------------------------------------------
# 1. System definition & trajectory generation
# ---------------------------------------------------------------------------
def lorenz96(t, x, F, N):
    """Lorenz-96 RHS: dx_i/dt = (x_{i+1} - x_{i-2}) * x_{i-1} - x_i + F"""
    d = np.zeros(N)
    for i in range(N):
        d[i] = (x[(i + 1) % N] - x[(i - 2) % N]) * x[(i - 1) % N] - x[i] + F
    return d


def integrate_lorenz96(F, N, seed=SEED):
    """Spin up, then integrate, returning one-step snapshot pairs (X, Y)."""
    rng    = np.random.default_rng(seed)
    x_init = rng.standard_normal(N)

    sol_spinup = solve_ivp(
        lorenz96, [0, T_SPINUP], x_init,
        args=(F, N), method="RK45", dense_output=True,
    )
    x_warm = sol_spinup.y[:, -1]

    t_eval = np.arange(0, T_TOTAL, DT)
    sol    = solve_ivp(
        lorenz96, [0, T_TOTAL], x_warm,
        args=(F, N), t_eval=t_eval, method="RK45", rtol=1e-10, atol=1e-12,
    )
    traj = sol.y.T[N_DISCARD:]
    print(f"  F={F}  X shape: {traj[:-1].shape}")
    return traj[:-1], traj[1:]


# ---------------------------------------------------------------------------
# 2 & 3. DeepMDMD training, embedding & Koopman eigendecomposition
# ---------------------------------------------------------------------------
def train_embed_decompose(X, Y):
    """Train DeepMDMD, embed data, compute Koopman eigendecomposition.

    Returns a dict with the keys needed for plotting.
    """
    model, _ = DeepMDMD(
        X, Y,
        cluster_num=N_CLUSTERS,
        dims=DIMS,
        auto_epochs=250,
        Koopman_epochs=100,
        activation_fn="tanh",
        gamma=0.2,
        alpha=1,
        Koopman_interval=20,
        SEED=SEED,
    )
    model.eval()
    with torch.no_grad():
        X_emb = model.encoder(torch.tensor(X, dtype=torch.float32)).numpy()
        Y_emb = model.encoder(torch.tensor(Y, dtype=torch.float32)).numpy()

    K_sparse, _, xi, _, E, V = utils.MDMD_matrix(X_emb, Y_emb, N=N_CLUSTERS)

    # Keep only non-zero eigenvalues; sort by angle in [0, pi]
    nz        = E != 0
    E_nz      = E[nz]
    V_nz      = np.asarray(V[:, nz])
    angles    = np.mod(np.angle(E_nz), 2 * np.pi)
    mask      = (angles >= 0.0) & (angles <= np.pi)
    angles_masked = angles[mask]
    real_masked   = np.real(E_nz[mask])

    sort_idx = np.lexsort((real_masked, angles_masked))  # stable by construction

    return dict(
        X_emb=X_emb,
        xi=xi,
        E_sorted=E_nz[mask][sort_idx],
        V_sorted=V_nz[:, mask][:, sort_idx],
    )


results = {}
for F in F_VALUES:
    print(f"\nTraining / eigendecomposition for F = {F}")
    X, Y       = integrate_lorenz96(F, N_DIM, seed=SEED)
    results[F] = train_embed_decompose(X, Y)


# ---------------------------------------------------------------------------
# 4. Koopman eigenfunction plot coloured on latent trajectory  (saved)
# ---------------------------------------------------------------------------
fig, axes = plt.subplots(
    3, 3, figsize=(18, 16),
    subplot_kw={"projection": "3d"},
    constrained_layout=True,
)

for row_idx, F in enumerate(F_VALUES):
    ax_row = axes[row_idx]
    data   = results[F]
    ids    = [i for i in ROW_IDS_BY_F[F] if i < data["E_sorted"].shape[0]]

    if not ids:
        for ax in ax_row:
            ax.axis("off")
        continue

    Es     = data["E_sorted"][ids]

    Gs     = data["V_sorted"][:, ids]
    points = data["X_emb"][:, :3]
    segments = np.stack([points[:-1], points[1:]], axis=1)
    phis     = [np.real(np.asarray(Gs[:, col]).ravel()[data["xi"]]) for col in range(Gs.shape[1])]
    row_vmin = min(phi.min() for phi in phis)
    row_vmax = max(phi.max() for phi in phis)

    for col_idx, ax in enumerate(ax_row):
        if col_idx >= len(ids):
            ax.axis("off")
            continue

        if row_idx == 0:
            ax.set_zticks([5.6, 5.4, 5.2, 5.0])
            ax.set_xticks([-2.75, -2.25, -1.75, -1.25])
            ax.set_yticks([4.5, 4.0, 3.5, 3.0])
        if row_idx == 1:
            ax.set_yticks([2.0, 1.0, 0.0, -1.0])

        theta = np.angle(Es[col_idx])
        phi   = phis[col_idx]

        line = Line3DCollection(segments, cmap="Spectral", linewidth=2.0, alpha=0.9)
        line.set_array(phi[:-1])
        line.set_clim(row_vmin, row_vmax)
        ax.add_collection3d(line)

        ax.text2D(0.5, 1, rf"$\lambda = \exp({theta:.2f}i)$",
                  fontsize=23, ha="center", va="top", transform=ax.transAxes)
        ax.set_xlabel("z1", fontsize=LABEL_FS, labelpad=12)
        ax.set_ylabel("z2", fontsize=LABEL_FS, labelpad=12)
        ax.set_zlabel("z3", fontsize=LABEL_FS, labelpad=12)
        ax.tick_params(axis="both", labelsize=TICK_FS)
        ax.view_init(elev=20, azim=115)
        ax.auto_scale_xyz(points[:, 0], points[:, 1], points[:, 2])

    ax_row[0].text2D(
        -0.12, 0.5, ROW_LABELS[F],
        transform=ax_row[0].transAxes,
        fontsize=28, va="center", rotation=90,
    )

fig.savefig(FIGS_DIR / "lorenz96_eigenfunctions.pdf", bbox_inches="tight", pad_inches=0.5)
plt.show()