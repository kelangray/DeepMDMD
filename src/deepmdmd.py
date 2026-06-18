"""
deep_mdmd.py
============
Deep Multiplicative Dynamic Mode Decomposition (DeepMDMD)

Combines a deep autoencoder with soft clustering and a
Koopman-operator loss so that cluster dynamics respect
a learned Koopman matrix K built from the embedded data.

Architecture overview
---------------------
Pretraining phase:
    Autoencoder (encoder + decoder) trained with MSE reconstruction loss.

Fine-tuning phase:
    DeepMDMDModel wraps the pretrained encoder/decoder and adds a
    SoftAssignment (Student-t kernel).  The model is jointly trained
    with:
    - Reconstruction loss
    - Squared Koopman residual loss between predicted cluster
      transitions and next-step soft assignments, weighted by gamma
"""

from typing import Optional, Tuple
import os

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.cluster import KMeans
from torch.nn import Parameter
from torch.utils.data import DataLoader, TensorDataset
import matplotlib.pyplot as plt
from matplotlib.ticker import LogLocator, LogFormatterMathtext

from . import utils

# ---------------------------------------------------------------------------
# Reproducibility helpers
# ---------------------------------------------------------------------------

def set_random_seed(seed: int, deterministic: bool = True) -> torch.device:
    """Seed all relevant RNGs and return the best available device.

    Args:
        seed: Integer seed for NumPy, PyTorch CPU, and CUDA.
        deterministic: If True, forces cuDNN into deterministic mode (may
            reduce throughput but guarantees reproducibility).

    Returns:
        torch.device: 'cuda' if a GPU is available, otherwise 'cpu'.
    """
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.use_deterministic_algorithms(True)
    
        torch.set_num_threads(1) 
        os.environ["OMP_NUM_THREADS"] = "1"
        os.environ["MKL_NUM_THREADS"] = "1"
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"  # needed for CUDA determinism
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    return device

# ---------------------------------------------------------------------------
# Activation helper
# ---------------------------------------------------------------------------

_ACTIVATIONS = {
    "relu": nn.ReLU,
    "tanh": nn.Tanh,
    "elu": nn.ELU,
    "sigmoid": nn.Sigmoid,
    "leaky_relu": nn.LeakyReLU,
}


def _get_activation(activation_fn: str) -> nn.Module:
    """Instantiate an activation module by name.

    Args:
        activation_fn: Case-insensitive name of the activation function.

    Returns:
        Instantiated nn.Module for the requested activation.

    Raises:
        ValueError: If the name is not recognised.
    """
    key = activation_fn.lower()
    if key not in _ACTIVATIONS:
        raise ValueError(
            f"Unknown activation '{activation_fn}'. "
            f"Available: {list(_ACTIVATIONS.keys())}"
        )
    # Instantiate fresh each call so layers don't share the same Module object.
    return _ACTIVATIONS[key]()

# ---------------------------------------------------------------------------
# Autoencoder
# ---------------------------------------------------------------------------

class Autoencoder(nn.Module):
    """Symmetric fully-connected autoencoder.

    The encoder maps input_dim → dims[1] → … → latent_dim (dims[-1]).
    The decoder is the mirror image.  Activations are inserted between
    every pair of layers except the final layer of each sub-network,
    preserving an unbounded latent space and reconstruction range.

    Dropout layers can optionally be inserted after hidden layers to reduce
    overfitting on noisy data by preventing co-adaptation of features.

    Args:
        dims: List of layer widths including input and latent dimensions,
            e.g. [256, 128, 64, 3].
        activation_fn: Name of the non-linearity to use (see _ACTIVATIONS).
        dropout_rate: Dropout probability for hidden layers (default 0 = no dropout).
            Recommended: 0.3-0.5 for noisy data. Dropout is applied after each
            hidden layer activation, but not after the latent layer or output layer.
    """

    def __init__(self, dims: list, activation_fn: str = "tanh", dropout_rate: float = 0.0):
        super().__init__()
        self.dims = dims
        self.activation_fn = activation_fn
        self.dropout_rate = dropout_rate

        # --- Encoder ---
        enc_layers: list[nn.Module] = []
        for i in range(len(dims) - 1):
            enc_layers.append(nn.Linear(dims[i], dims[i + 1]))
            # No activation or dropout after the final (latent) layer.
            if i < len(dims) - 2:
                enc_layers.append(_get_activation(activation_fn))
                if dropout_rate > 0:
                    enc_layers.append(nn.Dropout(dropout_rate))
        self.encoder = nn.Sequential(*enc_layers)

        # --- Decoder (mirror of encoder) ---
        dec_layers: list[nn.Module] = []
        for i in range(len(dims) - 1, 0, -1):
            dec_layers.append(nn.Linear(dims[i], dims[i - 1]))
            # No activation or dropout after the final (output) layer.
            if i > 1:
                dec_layers.append(_get_activation(activation_fn))
                if dropout_rate > 0:
                    dec_layers.append(nn.Dropout(dropout_rate))
        self.decoder = nn.Sequential(*dec_layers)

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Map input to latent space."""
        return self.encoder(x)

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Map latent code back to input space."""
        return self.decoder(z)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full encode-decode pass; returns reconstruction."""
        return self.decoder(self.encoder(x))

# ---------------------------------------------------------------------------
# Soft cluster assignment (Student-t kernel)
# ---------------------------------------------------------------------------

class SoftAssignment(nn.Module):
    """Differentiable soft cluster assignment using a Student-t kernel.

    The assignment probability q_{ij} measures how likely datapoint i belongs
    to cluster j.

    The cluster centres are learnable parameters updated during the
    fine-tuning phase.

    Args:
        cluster_number: Number of clusters N
        embedding_dimension: Dimensionality of the latent space.
        cluster_centers: Initial centre coordinates, shape (M, embedding_dim).
    """

    def __init__(
        self,
        cluster_number: int,
        embedding_dimension: int,
        cluster_centers: torch.Tensor,
    ) -> None:
        super().__init__()
        self.cluster_number = cluster_number
        self.embedding_dimension = embedding_dimension
        # Learnable cluster centres.
        self.cluster_centers = Parameter(cluster_centers)

    def forward(self, x: torch.Tensor, alpha: float = 1) -> torch.Tensor:
        """Compute soft assignments for a batch of embeddings.

        Args:
            x: Latent embeddings, shape (N, embedding_dim).
            alpha: Degrees of freedom for the Student-t kernel (default 1.0
                collapses to Cauchy; larger values approximate a Gaussian).

        Returns:
            Soft assignment matrix Q of shape (M, N).
        """        
        # Squared distances: (N, K)
        norm_sq = torch.sum((x.unsqueeze(1) - self.cluster_centers) ** 2, dim=2)
        numerator = (1.0 + norm_sq / alpha).pow(-(alpha + 1) / 2.0)
        return numerator / (numerator.sum(dim=1, keepdim=True))

# ---------------------------------------------------------------------------
# Full DeepMDMD model
# ---------------------------------------------------------------------------

class DeepMDMDModel(nn.Module):
    """Combines the autoencoder with soft cluster assignment.

    During the fine-tuning phase the encoder, decoder, and cluster centres
    are all updated jointly.

    Args:
        encoder: Pretrained encoder sub-network.
        decoder: Pretrained decoder sub-network.
        cluster_number: Number of clusters N.
        embed_dim: Latent-space dimensionality.
        init_centers: K-means initialised cluster centres, shape (N, embed_dim).
    """

    def __init__(
        self,
        encoder: nn.Module,
        decoder: nn.Module,
        cluster_number: int,
        embed_dim: int,
        init_centers: torch.Tensor,
    ):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.assignment = SoftAssignment(cluster_number, embed_dim, init_centers)

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Full forward pass returning soft assignments and reconstruction.

        Returns:
            q: Soft assignment matrix, shape (M, M).
            x_recon: Reconstructed input, shape (M, input_dim).
        """
        z = self.encoder(x)
        q = self.assignment(z)
        x_recon = self.decoder(z)
        return q, x_recon

    def ae_forward(self, x: torch.Tensor) -> torch.Tensor:
        """AE-only forward pass (encode → decode, no cluster assignment).

        Exposes the same interface as Autoencoder.forward so that
        reconstruction loss computation can be reused in both the pretraining
        and fine-tuning phases without any wrapper.

        Args:
            x: Input tensor.

        Returns:
            Reconstruction tensor.
        """
        return self.decoder(self.encoder(x))


# ---------------------------------------------------------------------------
# Shared loss computation
# ---------------------------------------------------------------------------

def compute_reconstruction_loss(
    ae: nn.Module,
    batch_x: torch.Tensor,
    criterion: nn.Module,
) -> torch.Tensor:
    """Reconstruction loss only: MSE(x_recon, x)."""
    x_recon = ae(batch_x)
    recon_loss = criterion(x_recon, batch_x)
    return recon_loss


# ---------------------------------------------------------------------------
# Embedding extraction helper
# ---------------------------------------------------------------------------

def compute_embeddings(
    model: DeepMDMDModel,
    loader: DataLoader,
    device: torch.device,
) -> torch.Tensor:
    """Extract encoder embeddings for all samples in a DataLoader.

    Args:
        model: DeepMDMDModel (only the encoder is used).
        loader: DataLoader yielding (x, _) batches.
        device: Target device.

    Returns:
        Embeddings tensor of shape (N, embed_dim) on CPU.
    """
    embeddings = []
    model.eval()
    with torch.no_grad():
        for batch_x, _ in loader:
            embeddings.append(model.encoder(batch_x.to(device)).cpu())
    return torch.cat(embeddings, dim=0)


# ---------------------------------------------------------------------------
# Pretraining
# ---------------------------------------------------------------------------

def pretrain_autoencoder(
    ae: Autoencoder,
    X: torch.Tensor,
    Y: torch.Tensor,
    device: torch.device,
    num_epochs: int = 100,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    seed: int = 1,
) -> None:
    """Pretrain the autoencoder with reconstruction loss.

    X and Y are paired consecutive snapshots of timeseries at time t and t+1, respectively. 

    The DataLoaders for X and Y use the same seed so that their shuffle
    order is synchronised; paired (x_t, x_{t+1}) examples therefore
    remain aligned across batches.

    Args:
        ae: Autoencoder to train (mutated in-place).
        X: State snapshots at time t, shape (M, input_dim).
        Y: State snapshots at time t+1, shape (M, input_dim).
        device: Training device.
        num_epochs: Number of passes over the dataset.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay (L2 regularization).
        batch_size: Mini-batch size.
        seed: RNG seed for DataLoader shuffling.
    """
    optimizer = torch.optim.Adam(
        ae.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    criterion = nn.MSELoss()

    # Both loaders share the same seed so pairs stay aligned after shuffle.
    def _make_loader(tensor: torch.Tensor) -> DataLoader:
        dataset = TensorDataset(tensor, tensor)  # (x, x) — label unused
        return DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(seed),
        )

    X_loader = _make_loader(X)
    Y_loader = _make_loader(Y)

    ae.to(device)
    for epoch in range(num_epochs):
        ae.train()
        total_recon, n_batches = 0.0, 0

        for (batch_x, _), (batch_y, _) in zip(X_loader, Y_loader):
            batch_x, batch_y = batch_x.to(device), batch_y.to(device)

            optimizer.zero_grad()
            recon_loss = compute_reconstruction_loss(ae, batch_x, criterion)
            recon_loss.backward()
            optimizer.step()

            total_recon += recon_loss.item()
            n_batches += 1

        if (epoch + 1) % 50 == 0:
            print(
                f"  AE Epoch {epoch+1}/{num_epochs} | "
                f"Recon={total_recon/n_batches:.6f}"
            )


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------

def _train_deepmdmd(
    deepmdmd: DeepMDMDModel,
    X_loader: DataLoader,
    Y_loader: DataLoader,
    cluster_num: int,
    device: torch.device,
    num_epochs: int = 50,
    update_interval: int = 10,
    learning_rate: float = 1e-3,
    weight_decay: float = 0.0,
    gamma: float = 0.5,
    alpha: float = 1.0,
) -> dict:
    """Fine-tune the DeepMDMD model.

    Every `update_interval` epochs the Koopman matrix K is
    recomputed from the current embeddings via utils.MDMD_matrix.  The joint
    loss is:

        L = gamma * recon
            + MSE(q_X @ K @ G_invsqrt, q_Y @ G_invsqrt)

    Args:
        deepmdmd: Model to fine-tune (mutated in-place).
        X_loader: DataLoader for states at time t.
        Y_loader: DataLoader for states at time t+1.
        cluster_num: Number of clusters (passed to utils.MDMD_matrix).
        device: Training device.
        num_epochs: Total fine-tuning epochs.
        update_interval: How often to recompute the Koopman matrix K.
        learning_rate: Adam learning rate.
        weight_decay: Adam weight decay (L2 regularization).
        gamma: Weight of the base loss (reconstruction) term.
        alpha: Degrees of freedom for the Student-t soft-assignment kernel.

    Returns:
        Loss history dictionary with keys 'epochs', 'recon', 'koopman', 'total'.
    """
    deepmdmd = deepmdmd.to(device)
    optimizer = torch.optim.Adam(
        deepmdmd.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    reconstruction_criterion = nn.MSELoss()

    # K and W are recomputed lazily; initialise to None so the first pass
    # always triggers a refresh.
    K: Optional[torch.Tensor] = None
    W: Optional[torch.Tensor] = None

    # Initialize loss history tracking
    loss_history = {
        'epochs': [],
        'recon': [],
        'koopman': [],
        'total': [],
        'lambda': gamma,
    }

    for epoch in range(num_epochs):
        # --- Refresh Koopman matrix K every `update_interval` epochs ---
        if epoch % update_interval == 0:
            X_emb = compute_embeddings(deepmdmd, X_loader, device).numpy()
            Y_emb = compute_embeddings(deepmdmd, Y_loader, device).numpy()

            K_sparse, W_np, _xi, _centroids, _E, _V = utils.MDMD_matrix(
                X_emb, Y_emb, N=cluster_num
            )
            K_np = np.asarray(K_sparse.todense(), dtype=np.float32)
            K = torch.as_tensor(K_np, dtype=torch.float32, device=device)
            W = torch.as_tensor(W_np, dtype=torch.float32, device=device)
            G_invsqrt = torch.diag(1 / torch.sqrt(torch.sum(W, axis=1)))
            
            K.requires_grad_(False)
            W.requires_grad_(False)
            G_invsqrt.requires_grad_(False)


        deepmdmd.train()

        # --- Batched optimization (AE + Koopman residual per batch) ---
        total_loss = 0.0
        total_recon = 0.0
        total_koopman_residual = 0.0
        n_batches = 0
        for (batch_x, _), (batch_y, _) in zip(X_loader, Y_loader):
            batch_x = batch_x.to(device)
            batch_y = batch_y.to(device)

            # Reconstruction on the current batch.
            recon_loss = compute_reconstruction_loss(
                deepmdmd.ae_forward,
                batch_x,
                reconstruction_criterion,
            )

            q_X, _ = deepmdmd(batch_x)
            q_Y, _ = deepmdmd(batch_y)    

            koopman_residual_loss = F.mse_loss(
                q_X @ K @ G_invsqrt,
                q_Y @ G_invsqrt,
            )

            loss = gamma * recon_loss + koopman_residual_loss

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            total_recon += recon_loss.item()
            total_koopman_residual += koopman_residual_loss.item()
            n_batches += 1

        if (epoch + 1) % update_interval == 0:
            print(
                f"  DeepMDMD Epoch {epoch+1}/{num_epochs} | "
                f"Loss={total_loss/max(n_batches, 1):.6f} | "
                f"Recon={total_recon/max(n_batches, 1):.6f} | "
                f"KoopmanResidual={total_koopman_residual/max(n_batches, 1):.6f}"
            )

        # Record loss history for this epoch (aggregate across batches)
        avg_recon = total_recon / max(n_batches, 1)
        avg_koopman = total_koopman_residual / max(n_batches, 1)
        avg_total = total_loss / max(n_batches, 1)
        
        loss_history['epochs'].append(epoch + 1)
        loss_history['recon'].append(avg_recon)
        loss_history['koopman'].append(avg_koopman)
        loss_history['total'].append(avg_total)
    
    return loss_history


# ---------------------------------------------------------------------------
# Loss visualization
# ---------------------------------------------------------------------------

def plot_loss_history(loss_history: dict, output_dir: str = "results") -> None:
    """Plot loss history from fine-tuning phase.
    Produces a figure with three subplots showing Koopman loss,
    reconstruction loss, and total loss over epochs.
    Args:
        loss_history: Dictionary returned by _train_deepmdmd with keys
            'epochs', 'recon', 'koopman', 'total'.
        output_dir: Directory to save the plot. Created if it doesn't exist.
    """
    os.makedirs(output_dir, exist_ok=True)
    epochs = loss_history['epochs']
    recon_loss = loss_history['recon']
    koopman_loss = loss_history['koopman']
    total_loss = loss_history['total']
    cmap = plt.cm.plasma
    koopman_color, recon_color, total_color = cmap(0.1), cmap(0.5), cmap(0.65)
    fig, axes = plt.subplots(1, 3, figsize=(19, 4.2), constrained_layout=True)
    # Koopman loss
    axes[0].semilogy(epochs, koopman_loss, color=koopman_color, linewidth=4)
    axes[0].set_ylim(top=10 ** np.ceil(np.log10(max(koopman_loss))))
    axes[0].set_xlim(0, 250)
    axes[0].set_xlabel('Training iteration', fontsize=22)
    axes[0].set_title('Clustering Loss', fontsize=26)
    axes[0].grid(True, alpha=0.3)
    axes[0].yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    axes[0].yaxis.set_major_formatter(LogFormatterMathtext())
    axes[0].yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=20))
    axes[0].tick_params(axis='both', which='both', labelsize=18)
    # Reconstruction loss
    axes[1].semilogy(epochs, recon_loss, color=recon_color, linewidth=4)
    axes[1].set_ylim(top=10 ** np.ceil(np.log10(max(recon_loss))))
    axes[1].set_xlim(0, 250)
    axes[1].set_xlabel('Training iteration', fontsize=22)
    axes[1].set_title('Reconstruction Loss', fontsize=26)
    axes[1].grid(True, alpha=0.3)
    axes[1].yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    axes[1].yaxis.set_major_formatter(LogFormatterMathtext())
    axes[1].yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=20))
    axes[1].tick_params(axis='both', which='both', labelsize=18)
    # Total loss
    axes[2].semilogy(epochs, total_loss, color=total_color, linewidth=4)
    axes[2].set_ylim(top=10 ** np.ceil(np.log10(max(total_loss))))
    axes[2].set_xlim(0, 250)
    axes[2].set_xlabel('Training iteration', fontsize=22)
    axes[2].set_title('Total loss', fontsize=26)
    axes[2].grid(True, alpha=0.3)
    axes[2].yaxis.set_major_locator(LogLocator(base=10, numticks=6))
    axes[2].yaxis.set_major_formatter(LogFormatterMathtext())
    axes[2].yaxis.set_minor_locator(LogLocator(base=10, subs='auto', numticks=20))
    axes[2].tick_params(axis='both', which='both', labelsize=18)
    plt.savefig(os.path.join(output_dir, 'loss_history.pdf'), bbox_inches='tight')
    print(f"\nLoss history plot saved to {os.path.join(output_dir, 'loss_history.pdf')}")
    plt.close()


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def DeepMDMD(
    X: np.ndarray,
    Y: np.ndarray,
    cluster_num: int,
    dims: list,
    X_kmeans: Optional[np.ndarray] = None,
    auto_epochs: int = 100,
    Koopman_epochs: int = 200,
    Koopman_interval: int = 100,
    SEED: int = 1,
    device: Optional[torch.device] = None,
    activation_fn: str = "tanh",
    learning_rate: float = 1e-3,
    pretrain_learning_rate: Optional[float] = None,
    finetune_learning_rate: Optional[float] = None,
    weight_decay: float = 0.0,
    batch_size: int = 256,
    gamma: float = 0.5,
    alpha: float = 1.0,
    dropout_rate: float = 0.0,
    include_loss: bool = False,
) -> Tuple["DeepMDMDModel", Autoencoder]:
    """Train a DeepMDMD model.

    Two-phase training:
             1. Pretrain an autoencoder on (X, Y) pairs with reconstruction
                loss.
             2. Fine-tune a DeepMDMDModel jointly using AE losses and Koopman
                 residual MSE loss.

    Example usage:

        deepmdmd_model, ae_pretrained = DeepMDMD(
            X, Y,
            cluster_num=N,
            dims=dims,
            auto_epochs=500,
            Koopman_epochs=100,
            activation_fn='relu',
            gamma=0.5,
            dropout_rate=0.3,  # For improved noise robustness
        )

    Args:
        X: State snapshots at time t, shape (N_samples, input_dim).
        Y: State snapshots at time t+1, shape (N_samples, input_dim).
        cluster_num: Number of Koopman clusters / basis functions.
        dims: Autoencoder layer widths including input and latent,
            e.g. [input_dim, 256, 64, latent_dim].
        X_kmeans: Optional subset of X used for K-means initialisation.
            Defaults to all of X when None.
        auto_epochs: Number of autoencoder pretraining epochs.
        Koopman_epochs: Number of DeepMDMD fine-tuning epochs.
        Koopman_interval: How often (in epochs) to recompute the Koopman
            transition kernel K during fine-tuning.
        SEED: Global RNG seed for reproducibility.
        device: Target device.  Auto-detected when None.
        activation_fn: Non-linearity for the autoencoder (see _ACTIVATIONS).
        learning_rate: Shared Adam learning rate for both training phases.
            Used as a fallback when phase-specific learning rates are not
            provided.
        pretrain_learning_rate: Adam learning rate for autoencoder
            pretraining phase. If None, defaults to learning_rate.
        finetune_learning_rate: Adam learning rate for DeepMDMD fine-tuning
            phase. If None, defaults to learning_rate.
        weight_decay: Adam weight decay (L2 regularization) for both phases.
        batch_size: Mini-batch size for the pretraining phase.
        gamma: Weight of the Koopman residual term in the fine-tuning joint
            objective. Range [0, 1].
        alpha: Degrees of freedom for the Student-t soft-assignment kernel.
        dropout_rate: Fraction of hidden units to drop during training (default 0).
            Recommended: 0.1-0.5 for noisy data to reduce overfitting. Dropout is
            applied after each hidden layer activation but not at the latent or
            output layers. Improves generalization by preventing co-adaptation
            of features.
        include_loss: If True, generate and save a loss history plot to
            results/loss_history.pdf.

    Returns:
        deepmdmd_model: Fully trained DeepMDMDModel.
        ae_pretrained: Snapshot of the autoencoder after pretraining,
            before any fine-tuning (useful for ablation studies).
    """
    if device is None:
        device = set_random_seed(SEED)
    else:
        set_random_seed(SEED)

    X_tensor = torch.tensor(X, dtype=torch.float32)
    Y_tensor = torch.tensor(Y, dtype=torch.float32)

    # Preserve backwards compatibility while allowing phase-specific rates.
    pretrain_lr = (
        learning_rate if pretrain_learning_rate is None else pretrain_learning_rate
    )
    finetune_lr = (
        learning_rate if finetune_learning_rate is None else finetune_learning_rate
    )

    # ------------------------------------------------------------------
    # Phase 1 – Autoencoder pretraining
    # ------------------------------------------------------------------
    print("Phase 1: Pretraining autoencoder...")
    ae = Autoencoder(dims, activation_fn=activation_fn, dropout_rate=dropout_rate)
    pretrain_autoencoder(
        ae, X_tensor, Y_tensor, device,
        num_epochs=auto_epochs,
        learning_rate=pretrain_lr,
        weight_decay=weight_decay,
        batch_size=batch_size,
        seed=SEED,
    )

    # Preserve a snapshot of the pretrained weights for later inspection.
    ae_pretrained = Autoencoder(dims, activation_fn=activation_fn, dropout_rate=dropout_rate)
    ae_pretrained.load_state_dict(ae.state_dict())
    ae_pretrained.to(device)
    ae_pretrained.eval()

    # K-means initialisation of cluster centres
    kmeans_data = (
        torch.tensor(X_kmeans, dtype=torch.float32)
        if X_kmeans is not None
        else X_tensor
    )

    ae.eval()
    with torch.no_grad():
        embeddings_np = ae.encode(kmeans_data.to(device)).cpu().numpy()

    kmeans = KMeans(n_clusters=cluster_num, random_state=SEED, init='k-means++', n_init=1)
    kmeans.fit(embeddings_np)
    init_centers = torch.tensor(kmeans.cluster_centers_, dtype=torch.float32)

    # ------------------------------------------------------------------
    # Phase 2 – Build DeepMDMDModel and fine-tune
    # ------------------------------------------------------------------
    # Fine-tuning loaders with synchronized shuffling so X and Y stay paired while still getting shuffled mini-batches.
    def _synced_loader(tensor: torch.Tensor) -> DataLoader:
        return DataLoader(
            TensorDataset(tensor, tensor),
            batch_size=batch_size,
            shuffle=True,
            generator=torch.Generator().manual_seed(SEED),
        )

    X_dec_loader = _synced_loader(X_tensor)
    Y_dec_loader = _synced_loader(Y_tensor)

    embed_dim = dims[-1]
    print("Phase 2: Building and fine-tuning DeepMDMD model...")
    deepmdmd_model = DeepMDMDModel(
        ae.encoder, ae.decoder, cluster_num, embed_dim, init_centers
    )
    loss_history = _train_deepmdmd(
        deepmdmd_model,
        X_dec_loader,
        Y_dec_loader,
        cluster_num,
        device,
        num_epochs=Koopman_epochs,
        update_interval=Koopman_interval,
        learning_rate=finetune_lr,
        weight_decay=weight_decay,
        gamma=gamma,
        alpha=alpha,
    )

    # Optionally generate and save loss history plot
    if include_loss:
        plot_loss_history(loss_history, output_dir="results")

    return deepmdmd_model, ae_pretrained