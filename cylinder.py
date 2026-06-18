"""
Cylinder Flow
======================================================

Reproduces the cylinder flow experiments.

Sections
--------
1. Data loading & preprocessing
2. PCA-based MDMD Koopman matrix
3. DeepMDMD Koopman matrix (learned latent space)
4. Koopman mode decomposition & trajectory prediction
5. Animation: latent trajectory + forecast (saved)
6. Noise sweep: latent trajectories + relative L2 error (saved)
7. Vorticity snapshot under 40% noise (saved)

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
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as mcolors
import mat73
import torch
from pathlib import Path
from matplotlib.animation import FuncAnimation, PillowWriter
from matplotlib.gridspec import GridSpec
from matplotlib.ticker import MaxNLocator
from sklearn.decomposition import PCA

from src import utils
from src.deepmdmd import DeepMDMD


# ---------------------------------------------------------------------------
# Global settings
# ---------------------------------------------------------------------------
SEED = 1
np.random.seed(SEED)

# Snapshot counts
M = 80           # training snapshots
N = M            # number of basis functions / clusters
T_ANIM = 150     # frames for animation

# Noise sweep
NOISE_LEVELS  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]
LATENT_LEVELS = {0.0, 0.2, 0.4}   # noise levels for which latent trajectories are stored
STORE_PREDS_AT = 0.4               # noise level for which full predictions are stored

# Model architecture  (D set after data load)
LATENT_DIM = 3
HIDDEN_DIMS = [256, 128, 64]

# Figure output
FIGS_DIR = Path("results")
FIGS_DIR.mkdir(parents=True, exist_ok=True)

FSIZE_TITLE = 14
FSIZE_LABEL = 12
FSIZE_TICK  = 10


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def build_field(indices, values, clip_sigma=1.0):
    """Place `values` at `indices` in a flat 800×200 field, clip, reshape."""
    field_flat = np.zeros(800 * 200)
    field_flat[indices] = values
    v = field_flat[indices]
    lo = np.mean(np.real(v)) - clip_sigma * np.std(np.real(v))
    hi = np.mean(np.real(v)) + clip_sigma * np.std(np.real(v))
    field_flat[indices] = np.clip(v, lo, hi)
    return field_flat.reshape([800, 200], order="F")


def cylinder_mask(indices):
    """Boolean mask (True = outside cylinder support, i.e. to be hidden)."""
    mask_flat = np.ones(800 * 200, dtype=bool)
    mask_flat[indices] = False
    return mask_flat.reshape([800, 200], order="F")


# ---------------------------------------------------------------------------
# 1. Data loading & preprocessing
# ---------------------------------------------------------------------------
vort_data = mat73.loadmat("cylinderdata.mat")
II        = vort_data["II"].astype(int) - 1   # 0-indexed support indices
vorticity = vort_data["VORT"]
x_grid    = vort_data["Xgrid"]
y_grid    = vort_data["Ygrid"]

vorticity_std = float(np.std(vorticity))

# Initial noise level (0 = clean baseline for model training)
NOISE_LEVEL  = 0.0
rng          = np.random.randn(vorticity.shape[0], vorticity.shape[1])
vorticity_noisy = vorticity + NOISE_LEVEL * vorticity_std * rng

# Training snapshots
X_noisy = vorticity_noisy[:, 0:M].T
Y_noisy = vorticity_noisy[:, 1:M + 1].T

# Normalisation (fit on noisy X)
mu    = np.mean(X_noisy)
sigma = np.std(X_noisy)
X = (X_noisy - mu) / sigma
Y = (Y_noisy - mu) / sigma

# Test data (clean vorticity, same normalisation)
X_test = (vorticity[:, M + 1:].T - mu) / sigma

# Derived dimension
D = X.shape[1]
DIMS = [D] + HIDDEN_DIMS + [LATENT_DIM]

mask_2d = cylinder_mask(II)
cmap_flow = plt.get_cmap("Spectral").copy()
cmap_flow.set_bad(color="white")


# ---------------------------------------------------------------------------
# 2. PCA-based MDMD Koopman matrix
# ---------------------------------------------------------------------------
pca = PCA(n_components=LATENT_DIM, svd_solver="full")
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
    activation_fn="relu",
    gamma=0.25,
    alpha=1,
    Koopman_interval=20,
    pretrain_learning_rate=1e-3,
    finetune_learning_rate=1e-4,
    include_loss=True,
    SEED=SEED,
)

deepmdmd_model.eval()
with torch.no_grad():
    X_embeddings_deepmdmd = deepmdmd_model.encoder(torch.tensor(X, dtype=torch.float32)).numpy()
    Y_embeddings_deepmdmd = deepmdmd_model.encoder(torch.tensor(Y, dtype=torch.float32)).numpy()

K_sparse_deep, _, xi_deep, _, _E_deep, _V_deep = utils.MDMD_matrix(
    X_embeddings_deepmdmd, Y_embeddings_deepmdmd, N=N
)
K_deepmdmd = np.array(K_sparse_deep.todense())
E_deepmdmd, V_deepmdmd = np.linalg.eig(K_deepmdmd)


# ---------------------------------------------------------------------------
# 4. Koopman mode decomposition & trajectory prediction
# ---------------------------------------------------------------------------
psi_X_pca  = np.zeros((M, N)); psi_X_pca[np.arange(M),  xi_pca]  = 1
psi_X_deep = np.zeros((M, N)); psi_X_deep[np.arange(M), xi_deep] = 1

modes_pca      = np.linalg.lstsq(psi_X_pca  @ V_pca,      X,                     rcond=None)[0]
modes_deepmdmd = np.linalg.lstsq(psi_X_deep @ V_deepmdmd,  X_embeddings_deepmdmd, rcond=None)[0]

T = X_test.shape[0]
X_pred_deepmdmd = np.zeros((T, D))
X_pred_pca      = np.zeros((T, D))
Z_pred_deepmdmd = np.zeros((T, LATENT_DIM))

for t in range(1, T + 1):
    z_deep = psi_X_deep[-1, :] @ V_deepmdmd @ np.diag(np.power(E_deepmdmd, t)) @ modes_deepmdmd
    Z_pred_deepmdmd[t - 1] = np.real(z_deep)
    with torch.no_grad():
        X_pred_deepmdmd[t - 1] = deepmdmd_model.decoder(
            torch.tensor(np.real(z_deep), dtype=torch.float32)
        ).numpy()
    X_pred_pca[t - 1] = np.real(psi_X_pca[-1, :] @ V_pca @ np.diag(np.power(E_pca, t)) @ modes_pca)


# ---------------------------------------------------------------------------
# 5. Animation: latent trajectory + forecast  (saved)
# ---------------------------------------------------------------------------
n_frames = T_ANIM

# Pre-build masked state fields
true_frames  = np.zeros((n_frames, 800, 200))
state_frames = np.zeros((n_frames, 800, 200))

for t in range(n_frames):
    u_true = np.zeros(800 * 200); u_true[II]  = X_test[t, :]
    u_pred = np.zeros(800 * 200); u_pred[II]  = X_pred_deepmdmd[t, :]
    true_frames[t]  = u_true.reshape([800, 200], order="F")
    state_frames[t] = u_pred.reshape([800, 200], order="F")

flat_valid = np.concatenate([true_frames[:, ~mask_2d], state_frames[:, ~mask_2d]], axis=0)
vmin_anim  = np.percentile(flat_valid, 2)
vmax_anim  = np.percentile(flat_valid, 98)

norm_anim    = mcolors.Normalize(vmin=vmin_anim, vmax=vmax_anim)
mappable_anim = cm.ScalarMappable(norm=norm_anim, cmap=cmap_flow)
mappable_anim.set_array([])

fig_anim = plt.figure(figsize=(17, 6))
gs_anim  = fig_anim.add_gridspec(2, 2, width_ratios=[1.0, 2.0], wspace=0.15, hspace=0.22)
ax_latent_anim = fig_anim.add_subplot(gs_anim[:, 0], projection="3d")
ax_true_anim   = fig_anim.add_subplot(gs_anim[0, 1])
ax_state_anim  = fig_anim.add_subplot(gs_anim[1, 1])

x_min, x_max = float(x_grid.min()), float(x_grid.max())
y_min, y_max = float(y_grid.min()), float(y_grid.max())

ax_latent_anim.set_title("Latent Trajectory")
ax_latent_anim.set_xlabel("z1"); ax_latent_anim.set_ylabel("z2"); ax_latent_anim.set_zlabel("z3")
ax_latent_anim.set_xlim(Z_pred_deepmdmd[:n_frames, 0].min(), Z_pred_deepmdmd[:n_frames, 0].max())
ax_latent_anim.set_ylim(Z_pred_deepmdmd[:n_frames, 1].min(), Z_pred_deepmdmd[:n_frames, 1].max())
ax_latent_anim.set_zlim(Z_pred_deepmdmd[:n_frames, 2].min(), Z_pred_deepmdmd[:n_frames, 2].max())
ax_latent_anim.xaxis.set_major_locator(MaxNLocator(nbins=5))
ax_latent_anim.yaxis.set_major_locator(MaxNLocator(nbins=5))
ax_latent_anim.zaxis.set_major_locator(MaxNLocator(nbins=5))
ax_latent_anim.tick_params(axis="both", which="major", labelsize=9, pad=2)
ax_latent_anim.zaxis.set_tick_params(labelsize=9, pad=2)

line_anim, = ax_latent_anim.plot(
    [], [], [],
    color=plt.cm.plasma(0.1),
    lw=2
)

point_anim, = ax_latent_anim.plot(
    [], [], [],
    linestyle="None",
    marker="o",
    color=plt.cm.plasma(0.5),
    ms=4
)


def _init_panel(ax, frames, t=0):
    field = np.clip(frames[t], vmin_anim, vmax_anim)
    field = np.ma.array(field, mask=mask_2d)
    ax.contourf(x_grid, y_grid, field, levels=50, cmap=cmap_flow,
                vmin=vmin_anim, vmax=vmax_anim)
    ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max])
    ax.set_aspect("auto")


_init_panel(ax_true_anim,  true_frames);  ax_true_anim.set_title("Exact")
_init_panel(ax_state_anim, state_frames); ax_state_anim.set_title("Forecast")

cbar_ax_anim = fig_anim.add_axes([0.94, 0.12, 0.015, 0.76])
fig_anim.colorbar(mappable_anim, cax=cbar_ax_anim, orientation="vertical")


def _update(t):
    line_anim.set_data(Z_pred_deepmdmd[:t + 1, 0], Z_pred_deepmdmd[:t + 1, 1])
    line_anim.set_3d_properties(Z_pred_deepmdmd[:t + 1, 2])
    point_anim.set_data([Z_pred_deepmdmd[t, 0]], [Z_pred_deepmdmd[t, 1]])
    point_anim.set_3d_properties([Z_pred_deepmdmd[t, 2]])

    for ax, frames, title in [
        (ax_true_anim,  true_frames,  "Exact"),
        (ax_state_anim, state_frames, "Forecast"),
    ]:
        ax.clear()
        field = np.clip(frames[t], vmin_anim, vmax_anim)
        field = np.ma.array(field, mask=mask_2d)
        ax.contourf(x_grid, y_grid, field, levels=50, cmap=cmap_flow,
                    vmin=vmin_anim, vmax=vmax_anim)
        ax.set_xlim([x_min, x_max]); ax.set_ylim([y_min, y_max])
        ax.set_aspect("auto"); ax.set_title(title)

    return [line_anim, point_anim]


anim = FuncAnimation(fig_anim, _update, frames=range(n_frames), blit=False, interval=50)
anim.save(FIGS_DIR / "cylinder_latent_forecast.gif", writer=PillowWriter(fps=20))
plt.show()


# ---------------------------------------------------------------------------
# 6. Noise sweep: latent trajectories + relative L2 error  (saved)
# ---------------------------------------------------------------------------
rel_l2_mdmd     = []
rel_l2_deepmdmd = []
latent_results  = {}   # {noise_level: Z_pred array}
sweep_results   = {}   # {noise_level: {'X_pred_deepmdmd': ..., 'X_pred_pca': ...}}

for nn in NOISE_LEVELS:
    print(f"\nProcessing noise level: {nn:.1%}")

    # Rebuild noisy training pairs for this noise level
    rng_n           = np.random.randn(vorticity.shape[0], vorticity.shape[1])
    vorticity_noisy_n = vorticity + nn * vorticity_std * rng_n

    X_noisy_n = vorticity_noisy_n[:, 0:M].T
    Y_noisy_n = vorticity_noisy_n[:, 1:M + 1].T

    mu_n    = np.mean(X_noisy_n)
    sigma_n = np.std(X_noisy_n)
    X_n = ((X_noisy_n - mu_n) / sigma_n).astype(np.float32)
    Y_n = ((Y_noisy_n - mu_n) / sigma_n).astype(np.float32)

    # PCA Koopman (reuse fitted pca)
    X_emb_pca_n = pca.transform(X_n)
    Y_emb_pca_n = pca.transform(Y_n)
    K_sparse_n, _, xi_pca_n, _, _E_n, _V_n = utils.MDMD_matrix(X_emb_pca_n, Y_emb_pca_n, N=N)
    K_pca_n = np.array(K_sparse_n.todense())
    E_pca_n, V_pca_n = np.linalg.eig(K_pca_n)

    # DeepMDMD Koopman (reuse trained model)
    deepmdmd_model.eval()
    with torch.no_grad():
        X_emb_deep_n = deepmdmd_model.encoder(torch.tensor(X_n, dtype=torch.float32)).cpu().numpy().astype(np.float32)
        Y_emb_deep_n = deepmdmd_model.encoder(torch.tensor(Y_n, dtype=torch.float32)).cpu().numpy().astype(np.float32)

    K_sparse_deep_n, _, xi_deep_n, _, _E_deep_n, _V_deep_n = utils.MDMD_matrix(X_emb_deep_n, Y_emb_deep_n, N=N)
    K_deep_n = np.array(K_sparse_deep_n.todense())
    E_deep_n, V_deep_n = np.linalg.eig(K_deep_n)

    # Indicator bases & modes
    psi_X_pca_n  = np.zeros((M, N), dtype=np.float32); psi_X_pca_n[np.arange(M),  xi_pca_n]  = 1
    psi_X_deep_n = np.zeros((M, N), dtype=np.float32); psi_X_deep_n[np.arange(M), xi_deep_n] = 1

    modes_pca_n  = np.linalg.lstsq(psi_X_pca_n  @ V_pca_n,  X_n,           rcond=None)[0]
    modes_deep_n = np.linalg.lstsq(psi_X_deep_n @ V_deep_n, X_emb_deep_n,  rcond=None)[0]

    T_test = X_test.shape[0]
    keep_latent = round(nn, 1) in LATENT_LEVELS
    keep_preds  = round(nn, 1) == STORE_PREDS_AT

    if keep_latent:
        Z_latent_n = np.zeros((T_test, LATENT_DIM), dtype=np.float32)
    if keep_preds:
        X_pred_deep_store = np.zeros((T_test, D), dtype=np.float32)
        X_pred_pca_store  = np.zeros((T_test, D), dtype=np.float32)

    mdmd_sum = 0.0
    deep_sum = 0.0

    for t in range(1, T_test + 1):
        z_deep_n = psi_X_deep_n[-1, :] @ V_deep_n @ np.diag(np.power(E_deep_n, t)) @ modes_deep_n
        z_real_n = np.real(z_deep_n).astype(np.float32)

        if keep_latent:
            Z_latent_n[t - 1] = z_real_n

        with torch.no_grad():
            x_deep_n = deepmdmd_model.decoder(
                torch.tensor(z_real_n, dtype=torch.float32)
            ).cpu().numpy().astype(np.float32)

        x_pca_n = np.real(
            psi_X_pca_n[-1, :] @ V_pca_n @ np.diag(np.power(E_pca_n, t)) @ modes_pca_n
        ).astype(np.float32)

        x_ref   = X_test[t - 1]
        ref_norm = max(np.linalg.norm(x_ref), 1e-12)

        mdmd_sum += np.linalg.norm(x_pca_n  - x_ref) / ref_norm
        deep_sum += np.linalg.norm(x_deep_n - x_ref) / ref_norm

        if keep_preds:
            X_pred_deep_store[t - 1] = x_deep_n
            X_pred_pca_store[t - 1]  = x_pca_n

    rel_l2_mdmd.append(mdmd_sum / T_test)
    rel_l2_deepmdmd.append(deep_sum / T_test)

    if keep_latent:
        latent_results[round(nn, 1)] = Z_latent_n
    if keep_preds:
        sweep_results[round(nn, 1)] = {
            "X_pred_deepmdmd": X_pred_deep_store,
            "X_pred_pca":      X_pred_pca_store,
        }

    print(
        f"Completed {nn:.1%} | relL2 MDMD={rel_l2_mdmd[-1]:.4e}, "
        f"DeepMDMD={rel_l2_deepmdmd[-1]:.4e}"
    )

# ── Plot: latent trajectories + L2 error curve ──────────────────────────────
noise_percent   = [int(n * 100) for n in NOISE_LEVELS]
latent_levels_pct = [0, 20, 40]

latent_trajs  = []
latent_labels = []
for pct in latent_levels_pct:
    key = round(pct / 100.0, 1)
    if key in latent_results:
        latent_trajs.append(latent_results[key])
        latent_labels.append(f"{pct}% noise")

fig_sweep = plt.figure(figsize=(14, 4))
gs_sweep  = fig_sweep.add_gridspec(1, 2, width_ratios=[1.8, 1])
ax_lat    = fig_sweep.add_subplot(gs_sweep[0, 0], projection="3d")
ax_err    = fig_sweep.add_subplot(gs_sweep[0, 1])

cmap_sweep   = plt.cm.Spectral
latent_colors = cmap_sweep(np.linspace(0.7, 1.0, max(len(latent_trajs), 1)))
markers = ["o", "^", "s"]

for i, (Z_lat, label) in enumerate(zip(latent_trajs, latent_labels)):
    ax_lat.plot3D(
        Z_lat[:, 0], Z_lat[:, 1], Z_lat[:, 2],
        color=latent_colors[i], linewidth=1.5, alpha=0.85,
        label=label, marker=markers[i], markersize=5,
    )

ax_lat.set_xlabel(r"$z_1$", fontsize=FSIZE_LABEL, labelpad=5)
ax_lat.set_ylabel(r"$z_2$", fontsize=FSIZE_LABEL, labelpad=5)
ax_lat.set_zlabel(r"$z_3$", fontsize=FSIZE_LABEL, labelpad=5, rotation=0)
ax_lat.tick_params(axis="both", which="both", labelsize=FSIZE_TICK)
ax_lat.set_title("Latent space trajectories", fontsize=FSIZE_TITLE, y=1.03)
ax_lat.set_zticks([-2, 2, 6])
ax_lat.set_yticks([0, 5, 10])
ax_lat.set_xticks([-2, -6, -10])
ax_lat.set_box_aspect([1.5, 1.5, 1])
ax_lat.set_anchor("C")
ax_lat.legend(fontsize=FSIZE_LABEL, loc="upper right",
              bbox_to_anchor=(1.5, 0.3), frameon=False)

err_colors = cmap_sweep(np.linspace(0, 1, 2))
ax_err.plot(noise_percent, rel_l2_mdmd,     linewidth=2, c=err_colors[0], label="MDMD")
ax_err.plot(noise_percent, rel_l2_deepmdmd, linewidth=2, c=err_colors[1], label="DeepMDMD")
ax_err.grid(True, which="both", linewidth=0.6, alpha=0.7)
ax_err.set_title("Forecast error vs. noise", fontsize=FSIZE_TITLE, y=1.03)
ax_err.set_xlabel("Noise level (%)", fontsize=FSIZE_LABEL)
ax_err.set_ylabel(r"Relative $L^2$ error", fontsize=FSIZE_LABEL)
ax_err.tick_params(axis="both", which="both", labelsize=FSIZE_TICK)
ax_err.legend(frameon=False, fontsize=FSIZE_LABEL)
ax_err.set_box_aspect(0.6)
ax_err.set_xlim(0, 100)

pos = ax_lat.get_position()
ax_lat.set_position([pos.x0, pos.y0 - 1, pos.width, pos.height])
fig_sweep.canvas.draw() 

pos_lat = ax_lat.get_position()
pos_err = ax_err.get_position()

vertical_offset = -0.05   # ← tune this (positive = up, negative = down)

top_lat = pos_lat.y0 + pos_lat.height
new_height = pos_err.height
new_y0 = top_lat - new_height + vertical_offset

ax_err.set_position([pos_err.x0, new_y0, pos_err.width, new_height])
fig_sweep.savefig(FIGS_DIR / "cylinder_noise_sweep.pdf", bbox_inches="tight")
plt.show()


# ---------------------------------------------------------------------------
# 7. Vorticity snapshot under 40% noise  (saved)
# ---------------------------------------------------------------------------
nn_plot = STORE_PREDS_AT
K_snap  = 0   # first prediction step

X_pred_pca_plot      = sweep_results[nn_plot]["X_pred_pca"]
X_pred_deepmdmd_plot = sweep_results[nn_plot]["X_pred_deepmdmd"]

C_true     = build_field(II, X_test[K_snap, :])
C_pred_pca = build_field(II, X_pred_pca_plot[K_snap, :])
C_pred_deep = build_field(II, X_pred_deepmdmd_plot[K_snap, :])

C_true_masked      = np.ma.array(C_true,      mask=mask_2d)
C_pred_pca_masked  = np.ma.array(C_pred_pca,  mask=mask_2d)
C_pred_deep_masked = np.ma.array(C_pred_deep, mask=mask_2d)

cmap_snap = mpl.colors.ListedColormap(plt.get_cmap("Spectral")(np.linspace(0, 1, 256)))
cmap_snap.set_bad("white")

vmin_snap = min(C_true.min(), C_pred_pca.min(), C_pred_deep.min())
vmax_snap = max(C_true.max(), C_pred_pca.max(), C_pred_deep.max())

imshow_kw = dict(
    origin="lower",
    extent=[x_grid.min(), x_grid.max(), y_grid.min(), y_grid.max()],
    cmap=cmap_snap,
    interpolation="bilinear",
    aspect="auto",
    vmin=vmin_snap,
    vmax=vmax_snap,
)

fig_snap, axs = plt.subplots(3, 1, figsize=(8, 9), constrained_layout=True)

axs[0].imshow(C_true_masked.T,      **imshow_kw); axs[0].set_title("Exact",                              fontsize=FSIZE_TITLE)
axs[1].imshow(C_pred_deep_masked.T, **imshow_kw); axs[1].set_title(f"DeepMDMD ({int(nn_plot*100)}% noise)", fontsize=FSIZE_TITLE)
im_snap = axs[2].imshow(C_pred_pca_masked.T,  **imshow_kw); axs[2].set_title(f"MDMD ({int(nn_plot*100)}% noise)",     fontsize=FSIZE_TITLE)

for ax in axs:
    ax.set_xlim([-2, 14]); ax.set_ylim([-2.3, 2.3])
    ax.set_xlabel("x", fontsize=FSIZE_LABEL)
    ax.set_ylabel("y", fontsize=FSIZE_LABEL, rotation=360)
    ax.tick_params(axis="both", which="major", labelsize=FSIZE_TICK)

cbar_snap = fig_snap.colorbar(im_snap, ax=axs, orientation="horizontal",
                               shrink=0.9, pad=0.01, aspect=40)
cbar_snap.set_label("Vorticity", fontsize=FSIZE_LABEL)
cbar_snap.ax.tick_params(labelsize=FSIZE_TICK)

fig_snap.savefig(FIGS_DIR / "cylinder_snapshot_40pct_noise.pdf", bbox_inches="tight")
plt.show()