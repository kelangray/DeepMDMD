"""
Cavity Flow
======================================================

Reproduces the cavity flow experiments.

Sections
--------
1. Data loading & preprocessing
2. PCA-based MDMD Koopman matrix
3. DeepMDMD Koopman matrix (learned latent space)
4. Koopman mode decomposition & trajectory prediction
5. Autocorrelation comparison + vorticity snapshot (saved)

Dependencies
------------
    numpy, matplotlib, torch, scikit-learn, mat73
    src.utils            (MDMD_matrix)
    src                  (DeepMDMD)
"""

# ---------------------------------------------------------------------------
# Imports
# ---------------------------------------------------------------------------
import numpy as np
import matplotlib.pyplot as plt
import mat73
import torch
from pathlib import Path
from matplotlib.ticker import FixedLocator
from matplotlib.cm import ScalarMappable
from matplotlib.gridspec import GridSpec
from sklearn.decomposition import PCA

from src import utils
from src.deepmdmd import DeepMDMD


# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
SEED = 42
np.random.seed(SEED)

# Cavity grid dimension (fixed by dataset)
D = 65

# Snapshot counts
N = 500          # number of basis functions / clusters
M = N            # training snapshots
T = 1000         # test / prediction horizon

# Noise level
NOISE_LEVEL = 0.4

# Model architecture
DIMS = [D * D, 256, 128, 64, 3]

# Autocorrelation lags
AUTOCORR_LAGS = 750
AUTOCORR_TOL  = 0.25   # threshold for selecting spatially active vorticity pixels

# Figure output
FIGS_DIR = Path("results")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

FSIZE_TITLE = 30
FSIZE_LABEL = 24
FSIZE_TICK  = 20


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------
data      = mat73.loadmat("cavity20kdata.mat")
grid      = data["Grid"]
vorticity = data["VORT"] - np.mean(data["VORT"], axis=1, keepdims=True)

# Add noise
rng             = np.random.randn(vorticity.shape[0], vorticity.shape[1])
vorticity_noisy = vorticity + NOISE_LEVEL * np.std(vorticity) * rng

# Training snapshots
X_noisy = vorticity_noisy[:, 0:M].T
Y_noisy = vorticity_noisy[:, 1:M + 1].T

# Shared normalisation (fit on noisy X)
mu    = np.mean(X_noisy)
sigma = np.std(X_noisy) + 1e-8

X = (X_noisy - mu) / sigma
Y = (Y_noisy - mu) / sigma

# Test data
X_test = (vorticity[:, M + 1:M + T + 1].T - mu) / sigma

x_grid, y_grid = np.meshgrid(grid, grid)


# ---------------------------------------------------------------------------
# 2. PCA-based MDMD Koopman matrix
# ---------------------------------------------------------------------------
pca = PCA(n_components=3, svd_solver="full")
X_embeddings_pca = pca.fit_transform(X)
Y_embeddings_pca = pca.transform(Y)

K_sparse_pca, _, xi_pca, _, _E_pca, _V_pca = utils.MDMD_matrix(X_embeddings_pca, Y_embeddings_pca, N=N)
K_pca = np.array(K_sparse_pca.todense())
E_pca, V_pca = np.linalg.eig(K_pca)


# ---------------------------------------------------------------------------
# 3. DeepMDMD Koopman matrix (learned latent space)
# ---------------------------------------------------------------------------
deepmdmd_model, _ = DeepMDMD(
    X, Y,
    cluster_num=N,
    dims=DIMS,
    auto_epochs=500,
    Koopman_epochs=250,
    SEED=SEED,
    activation_fn="tanh",
    gamma=0.25,
    alpha=1,
    Koopman_interval=20,
    pretrain_learning_rate=1e-3,
    finetune_learning_rate=1e-4,
    include_loss=True,
    dropout_rate=0.1,
)

deepmdmd_model.eval()
with torch.no_grad():
    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)
    X_embeddings_deepmdmd = deepmdmd_model.encoder(X_tensor).numpy()
    Y_embeddings_deepmdmd = deepmdmd_model.encoder(Y_tensor).numpy()

K_sparse_deep, _, xi_deep, _, _E_deep, _V_deep = utils.MDMD_matrix(
    X_embeddings_deepmdmd, Y_embeddings_deepmdmd, N=N
)
K_deepmdmd = np.array(K_sparse_deep.todense())
E_deepmdmd, V_deepmdmd = np.linalg.eig(K_deepmdmd)


# ---------------------------------------------------------------------------
# 4. Koopman mode decomposition & trajectory prediction
# ---------------------------------------------------------------------------
psi_X_pca = np.zeros((M, N));  psi_X_pca[np.arange(M), xi_pca]   = 1
psi_X_deep = np.zeros((M, N)); psi_X_deep[np.arange(M), xi_deep]  = 1

modes_pca      = np.linalg.lstsq(psi_X_pca  @ V_pca,      X,                     rcond=None)[0]
modes_deepmdmd = np.linalg.lstsq(psi_X_deep @ V_deepmdmd,  X_embeddings_deepmdmd, rcond=None)[0]

# Predict T steps forward from the last training snapshot
X_pred_deepmdmd = np.zeros((T, D * D))
X_pred_pca      = np.zeros((T, D * D))

for t in range(1, T + 1):
    z_deep = psi_X_deep[-1, :] @ V_deepmdmd @ np.diag(np.power(E_deepmdmd, t)) @ modes_deepmdmd
    with torch.no_grad():
        X_pred_deepmdmd[t - 1] = deepmdmd_model.decoder(
            torch.tensor(np.real(z_deep), dtype=torch.float32)
        ).numpy()

    X_pred_pca[t - 1] = np.real(psi_X_pca[-1, :] @ V_pca @ np.diag(np.power(E_pca, t)) @ modes_pca)


# ---------------------------------------------------------------------------
# 5. Autocorrelation comparison + vorticity snapshot  (single saved figure)
# ---------------------------------------------------------------------------


# Select spatially active vorticity pixels
X_test_avg  = np.mean(np.abs(X_test), axis=0)
idcs        = np.where(X_test_avg > AUTOCORR_TOL)[0]
idcs_sorted = idcs[np.flip(np.argsort(X_test_avg[idcs]))]

true_autocorr          = utils.calculate_autocorr(
    X_test[:, idcs_sorted], num_lags=AUTOCORR_LAGS
)
deepmdmd_pred_autocorr = utils.calculate_autocorr(
    X_pred_deepmdmd[:, idcs_sorted], num_lags=AUTOCORR_LAGS
)
pca_pred_autocorr      = utils.calculate_autocorr(
    X_pred_pca[:, idcs_sorted], num_lags=AUTOCORR_LAGS
)

# ── Configuration ───────────────────────────────────────────────────────────
t_snapshot = 750
ticks_xy   = [-1, -0.5, 0, 0.5, 1]
 
vort_methods = [
    (X_test,          "Exact"),
    (X_pred_deepmdmd, "DeepMDMD"),
    (X_pred_pca,      "MDMD"),
]
autocorr_methods = [
    (true_autocorr,          "Exact"),
    (deepmdmd_pred_autocorr, "DeepMDMD"),
    (pca_pred_autocorr,      "MDMD"),
]
 
# Shared vmin/vmax for vorticity row
all_vals = np.concatenate([arr[t_snapshot, :] for arr, _ in vort_methods], axis=0)
vmin_f   = np.percentile(all_vals, 2)
vmax_f   = np.percentile(all_vals, 98)
 
# GridSpec with constrained_layout
fig = plt.figure(figsize=(20, 10), layout="constrained")
gs  = GridSpec(
    nrows=2, ncols=4,
    figure=fig,
    width_ratios=[1, 1, 1, 0.07],
    height_ratios=[1.2, 1],
)
 
# ── Row 0 — Vorticity snapshots ─────────────────────────────────────────────
vort_axes = [fig.add_subplot(gs[0, c]) for c in range(3)]
cbar_ax_f = fig.add_subplot(gs[0, 3])
 
for c, (ax, (arr, name)) in enumerate(zip(vort_axes, vort_methods)):
    field = arr[t_snapshot, :].reshape(D, D, order="F")
    field = np.clip(field, vmin_f, vmax_f)
    ax.contourf(x_grid, y_grid, field, levels=50,
                cmap="Spectral", vmin=vmin_f, vmax=vmax_f)
    ax.set_title(name,    fontsize=FSIZE_TITLE, pad=8)
    ax.set_xlabel(r"$x$", fontsize=FSIZE_LABEL)
    ax.set_ylabel(r"$y$", fontsize=FSIZE_LABEL, rotation=0, labelpad=14)
    ax.set_xlim([-1, 1]); ax.set_ylim([-1, 1])
    ax.set_aspect(0.85)
    ax.xaxis.set_major_locator(FixedLocator(ticks_xy))
    ax.yaxis.set_major_locator(FixedLocator(ticks_xy))
    ax.set_xticklabels([str(v) for v in ticks_xy], fontsize=FSIZE_TICK)
    ax.set_yticklabels([str(v) for v in ticks_xy], fontsize=FSIZE_TICK)
    ax.tick_params(axis='both', labelsize=FSIZE_TICK)
 
cbar_f = fig.colorbar(
    ScalarMappable(norm=plt.Normalize(vmin=vmin_f, vmax=vmax_f), cmap="Spectral"),
    cax=cbar_ax_f,
)
cbar_f.set_label("Vorticity", fontsize=FSIZE_LABEL)
cbar_f.ax.tick_params(labelsize=FSIZE_TICK)
 
# ── Row 1 — Autocorrelation ─────────────────────────────────────────────────
autocorr_axes = [fig.add_subplot(gs[1, c]) for c in range(3)]
cbar_ax_a     = fig.add_subplot(gs[1, 3])
 
ims = []
for c, (ax, (Z, name)) in enumerate(zip(autocorr_axes, autocorr_methods)):
    im = ax.imshow(Z, cmap="Spectral", vmin=-6, vmax=-1,
                   aspect="auto", origin="lower")
    ims.append(im)
    ax.set_title(name,               fontsize=FSIZE_TITLE, pad=8)
    ax.set_xlabel("Vorticity field", fontsize=FSIZE_LABEL)
    ax.tick_params(axis='both',      labelsize=FSIZE_TICK)
 
    yticks = [y for y in [0, 250, 500, 750] if y <= Z.shape[0]]
    ax.set_yticks(yticks)
 
    ax.set_ylabel(r"Lag $\tau$", fontsize=FSIZE_LABEL)
    ax.set_yticklabels([str(y) for y in yticks], fontsize=FSIZE_TICK)
 
cbar_a = fig.colorbar(ims[0], cax=cbar_ax_a)
cbar_a.set_label(r"$|\mathcal{A}_{\mathbf{x}}(\tau)|$", fontsize=FSIZE_LABEL)
log_ticks = [-6, -5, -4, -3, -2, -1]
cbar_a.set_ticks(log_ticks)
cbar_a.set_ticklabels([r"$10^{%d}$" % t for t in log_ticks], fontsize=FSIZE_TICK)
cbar_a.ax.tick_params(labelsize=FSIZE_TICK)
 
fig.canvas.draw()  
SHIFT = 0.08    
 
for cax in [cbar_ax_f, cbar_ax_a]:
    pos = cax.get_position()
    cax.set_position([pos.x0 + SHIFT, pos.y0, pos.width * 0.8, pos.height])
 
fig.savefig(FIGS_DIR / "combined_snapshot_autocorr.pdf", dpi=300, bbox_inches="tight")
plt.show()