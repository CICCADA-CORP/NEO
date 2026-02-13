"""
Fine-tuning pipeline for stem-aware DAC encoding.

Adapts the pre-trained DAC model to better handle isolated stems (vocals,
drums, bass, melody) by fine-tuning on multi-track datasets like MUSDB18-HQ.

The core insight is that a vanilla DAC model, trained on full mixes, wastes
codebook capacity on frequency bands that are empty in isolated stems.  By
fine-tuning with a stem-conditioned loss, we teach the encoder to allocate
codes more efficiently per stem type, improving quality at the same bitrate.

Architecture overview::

    ┌─────────────────────────────────────────────────────┐
    │  MUSDB18-HQ Dataset                                 │
    │  ├── Track 1/  vocals.wav, drums.wav, bass.wav, …  │
    │  ├── Track 2/  …                                    │
    │  └── …                                              │
    └────────────┬────────────────────────────────────────┘
                 │  StemDataset (random crop + augment)
                 ▼
    ┌─────────────────────────────────────────────────────┐
    │  DAC Encoder  →  RVQ Quantiser  →  DAC Decoder      │
    │       ↓                                  ↓          │
    │   latent z                         reconstructed x̂  │
    └────────────┬───────────────┬────────────────────────┘
                 │               │
                 ▼               ▼
    ┌──────────────┐   ┌────────────────────────────┐
    │ Commitment   │   │  Multi-scale spectral loss  │
    │ + codebook   │   │  + time-domain L1 loss      │
    │ loss (VQ)    │   │  + stem-weighted PESQ proxy │
    └──────┬───────┘   └──────────┬─────────────────┘
           └──────────┬───────────┘
                      ▼
                Total loss  →  AdamW optimiser

Usage:
    python -m neo_neural.train_stems --dataset ./data/musdb18hq --epochs 50
    python -m neo_neural.train_stems --dataset ./data/musdb18hq --epochs 100 \\
        --batch-size 8 --lr 1e-4 --stem-type vocals
"""

from __future__ import annotations

import argparse
import logging
from dataclasses import dataclass, field
from pathlib import Path

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------


@dataclass
class TrainConfig:
    """All hyper-parameters and paths for a training run.

    Attributes:
        dataset_path: Root of the MUSDB18-HQ dataset (train / test splits
            detected automatically).
        output_dir: Where to save checkpoints, logs, and exported models.
        stem_type: Which stem type to fine-tune for.  ``"all"`` trains a
            single model across all stem types (stem label passed as
            conditioning).
        epochs: Total number of training epochs.
        batch_size: Mini-batch size (per GPU).
        learning_rate: Peak learning rate for AdamW.
        weight_decay: AdamW weight decay coefficient.
        warmup_steps: Number of linear-warmup steps before cosine decay.
        crop_length_sec: Length (in seconds) of random crops from each track.
        sample_rate: Target sample rate (must match the DAC model).
        n_codebooks: Number of RVQ codebooks in the DAC model.
        commitment_weight: Weight of the VQ commitment loss term.
        spectral_weight: Weight of the multi-scale spectral loss term.
        time_weight: Weight of the time-domain L1 loss term.
        checkpoint_every: Save a checkpoint every N epochs.
        seed: Random seed for reproducibility.
        num_workers: DataLoader worker count.
        device: Torch device string.
        model_type: DAC model variant (``"44khz"``, ``"24khz"``, ``"16khz"``).
    """

    dataset_path: Path = Path("./data/musdb18hq")
    output_dir: Path = Path("./checkpoints")
    stem_type: str = "all"
    epochs: int = 50
    batch_size: int = 4
    learning_rate: float = 3e-4
    weight_decay: float = 1e-2
    warmup_steps: int = 500
    crop_length_sec: float = 2.0
    sample_rate: int = 44_100
    n_codebooks: int = 9
    commitment_weight: float = 0.25
    spectral_weight: float = 1.0
    time_weight: float = 0.1
    checkpoint_every: int = 5
    seed: int = 42
    num_workers: int = 4
    device: str = "cuda"
    model_type: str = "44khz"

    # Stem-specific loss scaling.  Vocals get higher weight because
    # perceptual quality of voice is critical.
    stem_loss_weights: dict[str, float] = field(
        default_factory=lambda: {
            "vocals": 1.5,
            "drums": 1.0,
            "bass": 1.0,
            "other": 0.8,
        }
    )


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------
#
# Implementation plan for StemDataset(torch.utils.data.Dataset):
#
# 1. __init__(root: Path, split: str, stem_type: str, crop_samples: int,
#             sample_rate: int)
#    - Scan *root*/{split}/ for track directories.
#    - Each track dir contains: vocals.wav, drums.wav, bass.wav, other.wav,
#      mixture.wav.
#    - Build an index list of (track_path, stem_name) tuples.
#    - If stem_type != "all", filter to that single stem.
#
# 2. __getitem__(idx) -> dict[str, Tensor]
#    - Load the stem WAV at the native rate.
#    - Random-crop to crop_samples length (pad with zeros if shorter).
#    - Normalise to [-1, 1] (peak normalisation).
#    - Apply optional augmentations:
#        a. Random gain ±6 dB.
#        b. Random polarity flip (50 %).
#    - Return {"audio": Tensor[1, crop_samples],
#              "stem_label": int,  # 0=vocals, 1=drums, 2=bass, 3=other
#              "track_name": str}
#
# 3. __len__() -> int
#    - Return length of the index.


STEM_LABELS: dict[str, int] = {
    "vocals": 0,
    "drums": 1,
    "bass": 2,
    "other": 3,
}
"""Mapping from stem name to integer label."""


def _build_dataset(
    config: TrainConfig,
    split: str,
) -> "torch.utils.data.Dataset":
    """Build a ``StemDataset`` for the given split.

    Args:
        config: Training configuration.
        split: One of ``"train"`` or ``"test"``.

    Returns:
        A ``torch.utils.data.Dataset`` yielding stem crops.

    Raises:
        ImportError: If PyTorch or torchaudio are not installed.
        FileNotFoundError: If the dataset split directory does not exist.
    """
    try:
        import torch
        import torchaudio
    except ImportError as exc:
        raise ImportError(
            "Training requires PyTorch and torchaudio. Install with: pip install torch torchaudio"
        ) from exc

    import random

    crop_samples = int(config.crop_length_sec * config.sample_rate)

    class StemDataset(torch.utils.data.Dataset):
        """Dataset of stem audio crops from a MUSDB18-HQ-style directory.

        Each item is a random crop from a single stem WAV file, with
        augmentations applied (random gain, polarity flip).
        """

        def __init__(
            self,
            root: Path,
            split: str,
            stem_type: str,
            crop_samples: int,
            sample_rate: int,
        ) -> None:
            self.crop_samples = crop_samples
            self.sample_rate = sample_rate

            split_dir = root / split
            if not split_dir.is_dir():
                raise FileNotFoundError(f"Dataset split directory not found: {split_dir}")

            stem_names = [stem_type] if stem_type != "all" else list(STEM_LABELS.keys())

            self.index: list[tuple[Path, str]] = []
            for track_dir in sorted(split_dir.iterdir()):
                if not track_dir.is_dir():
                    continue
                for stem_name in stem_names:
                    stem_path = track_dir / f"{stem_name}.wav"
                    if stem_path.exists():
                        self.index.append((track_dir, stem_name))

            logger.info(
                "StemDataset[%s]: %d items from %s",
                split,
                len(self.index),
                split_dir,
            )

        def __len__(self) -> int:
            return len(self.index)

        def __getitem__(self, idx: int) -> dict:
            track_dir, stem_name = self.index[idx]
            stem_path = track_dir / f"{stem_name}.wav"

            # Load audio
            waveform, sr = torchaudio.load(stem_path)  # [C, N]

            # Resample if necessary
            if sr != self.sample_rate:
                waveform = torchaudio.functional.resample(waveform, sr, self.sample_rate)

            # Convert to mono by averaging channels
            if waveform.shape[0] > 1:
                waveform = waveform.mean(dim=0, keepdim=True)  # [1, N]

            n_samples = waveform.shape[-1]

            # Random crop or zero-pad
            if n_samples >= self.crop_samples:
                start = random.randint(0, n_samples - self.crop_samples)
                waveform = waveform[:, start : start + self.crop_samples]
            else:
                # Pad with zeros on the right
                pad_len = self.crop_samples - n_samples
                waveform = torch.nn.functional.pad(waveform, (0, pad_len))

            # Peak normalise to [-1, 1]
            peak = waveform.abs().max()
            if peak > 0:
                waveform = waveform / peak

            # Augmentation: random gain ±6 dB
            gain_db = random.uniform(-6.0, 6.0)
            gain_linear = 10.0 ** (gain_db / 20.0)
            waveform = waveform * gain_linear

            # Augmentation: random polarity flip (50%)
            if random.random() < 0.5:
                waveform = -waveform

            # Clamp to prevent clipping beyond [-1, 1]
            waveform = waveform.clamp(-1.0, 1.0)

            return {
                "audio": waveform,  # [1, T]
                "stem_label": STEM_LABELS[stem_name],
                "track_name": track_dir.name,
            }

    return StemDataset(
        root=config.dataset_path,
        split=split,
        stem_type=config.stem_type,
        crop_samples=crop_samples,
        sample_rate=config.sample_rate,
    )


# ---------------------------------------------------------------------------
# Loss functions
# ---------------------------------------------------------------------------
#
# Implementation plan for the composite loss:
#
# 1. MultiScaleSpectralLoss
#    - Compute STFT at multiple window sizes (256, 512, 1024, 2048).
#    - For each scale: L1(|STFT(x)| - |STFT(x̂)|) + L1(log|STFT(x)| - log|STFT(x̂)|)
#    - Average across scales.
#
# 2. TimeDomainLoss
#    - Simple L1 loss between original and reconstructed waveforms.
#
# 3. CommitmentLoss
#    - Already computed inside DAC's quantiser — extract from model outputs.
#
# 4. StemWeightedLoss (combines the above)
#    - total = (spectral_weight * spectral_loss
#               + time_weight * time_loss
#               + commitment_weight * commitment_loss)
#    - Multiply by per-stem scaling factor from config.stem_loss_weights.


def _compute_loss(
    original: "torch.Tensor",
    reconstructed: "torch.Tensor",
    commitment_loss: "torch.Tensor",
    stem_label: "torch.Tensor",
    config: TrainConfig,
) -> "torch.Tensor":
    """Compute the composite training loss for one mini-batch.

    Combines multi-scale spectral loss, time-domain L1 loss, and VQ
    commitment loss, weighted by stem type.

    Args:
        original: Ground-truth waveform tensor ``[B, 1, T]``.
        reconstructed: Reconstructed waveform tensor ``[B, 1, T]``.
        commitment_loss: Scalar commitment loss from the RVQ quantiser.
        stem_label: Integer stem labels ``[B]`` for per-stem weighting.
        config: Training config with loss weights.

    Returns:
        Total scalar loss tensor.

    Raises:
        ImportError: If PyTorch is not installed.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Training requires PyTorch. Install with: pip install torch") from exc

    # Flatten to [B, T] for STFT computation
    orig_flat = original.squeeze(1)  # [B, T]
    recon_flat = reconstructed.squeeze(1)  # [B, T]

    # --- Multi-scale spectral loss ---
    stft_windows = [256, 512, 1024, 2048]
    spectral_loss = torch.tensor(0.0, device=original.device)
    eps = 1e-7  # small constant for log stability

    for win_size in stft_windows:
        hop_length = win_size // 4

        orig_stft = torch.stft(
            orig_flat,
            n_fft=win_size,
            hop_length=hop_length,
            win_length=win_size,
            window=torch.hann_window(win_size, device=original.device),
            return_complex=True,
        )
        recon_stft = torch.stft(
            recon_flat,
            n_fft=win_size,
            hop_length=hop_length,
            win_length=win_size,
            window=torch.hann_window(win_size, device=original.device),
            return_complex=True,
        )

        orig_mag = orig_stft.abs()
        recon_mag = recon_stft.abs()

        # L1 between magnitudes
        mag_loss = torch.nn.functional.l1_loss(recon_mag, orig_mag)

        # L1 between log magnitudes
        log_mag_loss = torch.nn.functional.l1_loss(
            torch.log(recon_mag + eps),
            torch.log(orig_mag + eps),
        )

        spectral_loss = spectral_loss + (mag_loss + log_mag_loss)

    spectral_loss = spectral_loss / len(stft_windows)

    # --- Time-domain L1 loss ---
    time_loss = torch.nn.functional.l1_loss(reconstructed, original)

    # --- Weighted combination ---
    total = (
        config.spectral_weight * spectral_loss
        + config.time_weight * time_loss
        + config.commitment_weight * commitment_loss
    )

    # --- Per-stem weighting ---
    label_to_stem = {v: k for k, v in STEM_LABELS.items()}
    stem_weights = torch.tensor(
        [
            config.stem_loss_weights.get(label_to_stem.get(lab.item(), "other"), 1.0)
            for lab in stem_label
        ],
        device=original.device,
        dtype=original.dtype,
    )
    # Average stem weight across the batch
    stem_weight = stem_weights.mean()

    return total * stem_weight


# ---------------------------------------------------------------------------
# Validation helper
# ---------------------------------------------------------------------------


def _validate(
    model: object,
    val_loader: object,
    config: TrainConfig,
) -> float:
    """Run one validation pass and return the average loss.

    Args:
        model: The DAC model (in eval mode internally).
        val_loader: Validation DataLoader.
        config: Training configuration.

    Returns:
        Average validation loss as a Python float.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Training requires PyTorch. Install with: pip install torch") from exc

    model.eval()  # type: ignore[union-attr]
    total_loss = 0.0
    n_batches = 0

    with torch.no_grad():
        for batch in val_loader:  # type: ignore[union-attr]
            audio = batch["audio"].to(config.device)
            stem_label = batch["stem_label"].to(config.device)

            z = model.encoder(audio)  # type: ignore[union-attr]
            z_q, codes, commitment, codebook_loss, _ = model.quantizer(z)  # type: ignore[union-attr]
            reconstructed = model.decoder(z_q)  # type: ignore[union-attr]

            loss = _compute_loss(audio, reconstructed, commitment, stem_label, config)
            total_loss += loss.item()
            n_batches += 1

    return total_loss / max(n_batches, 1)


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------


def train(config: TrainConfig) -> None:
    """Execute the full fine-tuning pipeline.

    High-level steps:

    1. **Setup**: seed RNGs, create output dirs, initialise logging.
    2. **Load model**: ``dac.DAC.load(...)`` → move to device → set to train
       mode.  Freeze the quantiser codebook embeddings for the first
       ``warmup_steps`` to prevent codebook collapse.
    3. **Load data**: Instantiate train and validation ``StemDataset`` with
       ``DataLoader`` (pin_memory, drop_last, shuffle).
    4. **Optimiser**: ``AdamW(lr, weight_decay)`` with linear warmup +
       cosine annealing schedule.
    5. **Training loop** (per epoch):
       a. For each batch:
          - Forward pass through encoder → quantiser → decoder.
          - Compute composite loss via ``_compute_loss()``.
          - Backward + gradient clipping (max_norm=1.0) + optimiser step.
          - Log metrics (loss, learning rate, codebook utilisation).
       b. After ``checkpoint_every`` epochs:
          - Run validation (same forward pass, no grad).
          - Save checkpoint: ``{output_dir}/epoch_{n}.pt``
          - Log validation metrics.
    6. **Final export**: Save best model and optionally trigger ONNX export.

    Args:
        config: Full training configuration.

    Raises:
        ImportError: If PyTorch or dac are not installed.
    """
    try:
        import torch
    except ImportError as exc:
        raise ImportError("Training requires PyTorch. Install with: pip install torch") from exc

    logger.info("=== NEO Stem-Aware DAC Fine-Tuning ===")
    logger.info("Config: %s", config)

    # ------------------------------------------------------------------
    # Step 1 — Setup
    # ------------------------------------------------------------------
    torch.manual_seed(config.seed)
    config.output_dir.mkdir(parents=True, exist_ok=True)

    # ------------------------------------------------------------------
    # Step 2 — Load pre-trained DAC
    # ------------------------------------------------------------------
    import dac
    import dac.utils

    model_path = dac.utils.download(model_type=config.model_type)
    model = dac.DAC.load(model_path)
    model.to(config.device)
    model.train()

    # Freeze codebook embeddings during warmup to prevent collapse.
    for param in model.quantizer.codebook.parameters():
        param.requires_grad = False

    # ------------------------------------------------------------------
    # Step 3 — Datasets & DataLoaders
    # ------------------------------------------------------------------
    from torch.utils.data import DataLoader

    train_ds = _build_dataset(config, split="train")
    val_ds = _build_dataset(config, split="test")

    train_loader = DataLoader(
        train_ds,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.num_workers,
        pin_memory=True,
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=config.batch_size,
        shuffle=False,
        num_workers=config.num_workers,
        pin_memory=True,
    )

    # ------------------------------------------------------------------
    # Step 4 — Optimiser & scheduler
    # ------------------------------------------------------------------
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=config.learning_rate,
        weight_decay=config.weight_decay,
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=config.epochs * len(train_loader),
    )

    # ------------------------------------------------------------------
    # Step 5 — Training loop
    # ------------------------------------------------------------------
    global_step = 0
    best_val_loss = float("inf")

    for epoch in range(1, config.epochs + 1):
        model.train()
        epoch_loss = 0.0

        for batch in train_loader:
            audio = batch["audio"].to(config.device)  # [B, 1, T]
            stem_label = batch["stem_label"].to(config.device)  # [B]

            # Forward
            z = model.encoder(audio)
            z_q, codes, commitment, codebook_loss, _ = model.quantizer(z)
            reconstructed = model.decoder(z_q)

            # Unfreeze codebooks after warmup
            if global_step == config.warmup_steps:
                for p in model.quantizer.codebook.parameters():
                    p.requires_grad = True
                logger.info(
                    "Unfreezing codebook embeddings at step %d",
                    global_step,
                )

            # Loss
            loss = _compute_loss(audio, reconstructed, commitment, stem_label, config)

            # Backward
            optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
            scheduler.step()

            epoch_loss += loss.item()
            global_step += 1

        avg_loss = epoch_loss / len(train_loader)
        logger.info(
            "Epoch %d/%d — train loss: %.5f",
            epoch,
            config.epochs,
            avg_loss,
        )

        # Validation & checkpointing
        if epoch % config.checkpoint_every == 0:
            val_loss = _validate(model, val_loader, config)
            logger.info("  val loss: %.5f", val_loss)

            ckpt_path = config.output_dir / f"epoch_{epoch}.pt"
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "val_loss": val_loss,
                },
                ckpt_path,
            )
            logger.info("  checkpoint → %s", ckpt_path)

            if val_loss < best_val_loss:
                best_val_loss = val_loss
                best_path = config.output_dir / "best_model.pt"
                torch.save(model.state_dict(), best_path)
                logger.info("  ★ new best model → %s", best_path)

    # ------------------------------------------------------------------
    # Step 6 — Final export
    # ------------------------------------------------------------------
    logger.info("Training complete.  Best val loss: %.5f", best_val_loss)
    logger.info("Run `python -m neo_neural.export_onnx` to export to ONNX.")


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry-point for stem-aware DAC fine-tuning.

    Parses command-line arguments, builds a :class:`TrainConfig`, and
    invokes :func:`train`.
    """
    parser = argparse.ArgumentParser(
        description="Fine-tune DAC for stem-aware encoding",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m neo_neural.train_stems --dataset ./data/musdb18hq --epochs 50\n"
            "  python -m neo_neural.train_stems --dataset ./data/musdb18hq "
            "--stem-type vocals --lr 1e-4\n"
        ),
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        required=True,
        help="Path to MUSDB18-HQ dataset root",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./checkpoints"),
        help="Output directory for checkpoints (default: ./checkpoints/)",
    )
    parser.add_argument(
        "--stem-type",
        type=str,
        default="all",
        choices=["all", "vocals", "drums", "bass", "other"],
        help="Stem type to fine-tune for (default: all)",
    )
    parser.add_argument(
        "--model-type",
        type=str,
        default="44khz",
        choices=["16khz", "24khz", "44khz"],
        help="DAC model variant (default: 44khz)",
    )
    parser.add_argument("--epochs", type=int, default=50, help="Training epochs (default: 50)")
    parser.add_argument("--batch-size", type=int, default=4, help="Batch size (default: 4)")
    parser.add_argument("--lr", type=float, default=3e-4, help="Learning rate (default: 3e-4)")
    parser.add_argument("--seed", type=int, default=42, help="Random seed (default: 42)")
    parser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Torch device (default: cuda)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable debug-level logging",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
    )

    config = TrainConfig(
        dataset_path=args.dataset,
        output_dir=args.output,
        stem_type=args.stem_type,
        model_type=args.model_type,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        seed=args.seed,
        device=args.device,
    )

    train(config)


if __name__ == "__main__":
    main()
