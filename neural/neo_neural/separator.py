"""
Source separation using Demucs v4.

Splits a stereo mix into stems (vocals, drums, bass, other) for encoding
into the NEO format.  Uses Meta's Hybrid Transformer Demucs (``htdemucs``)
as the default backbone.

The module can be used programmatically::

    from neo_neural.separator import load_separator, separate, save_stems

    model = load_separator("htdemucs_ft")
    stems = separate(Path("mix.wav"), model)
    save_stems(stems, Path("./stems"), sample_rate=44100)

Or from the command line::

    python -m neo_neural.separator input.wav --output ./stems/
    python -m neo_neural.separator input.wav --output ./stems/ --model htdemucs_ft
"""

from __future__ import annotations

import argparse
import logging
import time
from pathlib import Path

import numpy as np
import soundfile as sf
import torch
import torchaudio

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

#: Stem names produced by Demucs (htdemucs / htdemucs_ft).
STEM_NAMES: tuple[str, ...] = ("drums", "bass", "other", "vocals")

#: Default Demucs model variant — fine-tuned hybrid transformer.
DEFAULT_MODEL: str = "htdemucs_ft"

#: Number of audio channels expected by Demucs (stereo).
DEMUCS_CHANNELS: int = 2

#: Default overlap between segments during inference (seconds).
_DEFAULT_OVERLAP: float = 0.25

# Maximum waveform length (in samples) fed to Demucs in one chunk.
# Prevents OOM on long files — set to ~10 minutes at 44.1 kHz stereo.
_MAX_CHUNK_SAMPLES: int = 44_100 * 60 * 10


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------


def load_separator(
    model_name: str = DEFAULT_MODEL,
    device: str | None = None,
) -> torch.nn.Module:
    """Load a pretrained Demucs source-separation model.

    The model weights are downloaded automatically on first use and cached
    by ``torch.hub``.

    Args:
        model_name: Name of the Demucs model to load.  Common options are
            ``"htdemucs"`` (fast) and ``"htdemucs_ft"`` (fine-tuned, better
            quality but slower).
        device: Torch device string (e.g. ``"cpu"``, ``"cuda"``).  When
            ``None`` the function picks CUDA if available, otherwise CPU.

    Returns:
        The loaded Demucs model in eval mode, moved to *device*.

    Raises:
        ImportError: If the ``demucs`` package is not installed.
        RuntimeError: If model download or loading fails.
    """
    try:
        from demucs.pretrained import get_model
    except ImportError as exc:
        raise ImportError(
            "The 'demucs' package is required for source separation.  "
            "Install it with:  pip install 'neo-neural[separation]'"
        ) from exc

    if device is None:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Loading Demucs model '%s' on %s …", model_name, device)
    t0 = time.perf_counter()

    model = get_model(model_name)
    model.to(device)
    model.eval()

    elapsed = time.perf_counter() - t0
    logger.info("Demucs '%s' loaded in %.2f s.", model_name, elapsed)
    return model


def separate(
    audio_path: Path,
    model: torch.nn.Module,
    *,
    device: str | None = None,
    shifts: int = 1,
    overlap: float = _DEFAULT_OVERLAP,
) -> dict[str, np.ndarray]:
    """Run source separation on an audio file.

    Reads the audio from *audio_path*, resamples to the model's native rate
    if necessary, runs Demucs inference, and returns each stem as a NumPy
    array.

    Args:
        audio_path: Path to the input audio file (WAV, FLAC, MP3, etc.).
        model: A loaded Demucs model (from :func:`load_separator`).
        device: Torch device override.  Defaults to the model's current
            device.
        shifts: Number of random shifts for test-time augmentation.
            Higher values improve quality but increase compute linearly.
        overlap: Fractional overlap between chunks (0.0 – 1.0).

    Returns:
        Dictionary mapping stem names (``"vocals"``, ``"drums"``, ``"bass"``,
        ``"other"``) to NumPy float32 arrays of shape
        ``[channels, samples]``.

    Raises:
        FileNotFoundError: If *audio_path* does not exist.
        RuntimeError: If inference fails (OOM, unsupported format, etc.).
    """
    from demucs.apply import apply_model

    if not audio_path.exists():
        raise FileNotFoundError(f"Audio file not found: {audio_path}")

    if device is None:
        device = str(next(model.parameters()).device)

    # Load audio -----------------------------------------------------------
    logger.info("Loading audio from %s …", audio_path)
    waveform, sr = torchaudio.load(str(audio_path))

    # Ensure stereo.
    if waveform.shape[0] == 1:
        waveform = waveform.repeat(DEMUCS_CHANNELS, 1)
    elif waveform.shape[0] > DEMUCS_CHANNELS:
        waveform = waveform[:DEMUCS_CHANNELS, :]

    # Resample to model's expected sample rate if needed.
    model_sr: int = model.samplerate  # type: ignore[attr-defined]
    if sr != model_sr:
        logger.info("Resampling %d Hz → %d Hz …", sr, model_sr)
        resampler = torchaudio.transforms.Resample(orig_freq=sr, new_freq=model_sr)
        waveform = resampler(waveform)

    # Add batch dimension: [channels, samples] → [1, channels, samples].
    waveform = waveform.unsqueeze(0).to(device)

    # Normalise to prevent clipping during inference.
    ref = waveform.mean(0)
    waveform_mean = ref.mean()
    waveform_std = ref.std()
    waveform = (waveform - waveform_mean) / (waveform_std + 1e-8)

    # Separate --------------------------------------------------------------
    logger.info(
        "Running separation (shifts=%d, overlap=%.2f) …",
        shifts,
        overlap,
    )
    t0 = time.perf_counter()

    with torch.no_grad():
        estimates = apply_model(
            model,
            waveform,
            shifts=shifts,
            overlap=overlap,
        )
    # estimates shape: [batch, sources, channels, samples]

    elapsed = time.perf_counter() - t0
    logger.info("Separation complete in %.2f s.", elapsed)

    # De-normalise.
    estimates = estimates * (waveform_std + 1e-8) + waveform_mean

    # Build output dict ----------------------------------------------------
    source_names: list[str] = model.sources  # type: ignore[attr-defined]
    stems: dict[str, np.ndarray] = {}

    for idx, name in enumerate(source_names):
        stem_tensor = estimates[0, idx]  # [channels, samples]
        stems[name] = stem_tensor.cpu().numpy().astype(np.float32)

    logger.info("Stems extracted: %s", ", ".join(stems))
    return stems


def save_stems(
    stems: dict[str, np.ndarray],
    output_dir: Path,
    sample_rate: int,
    *,
    subtype: str = "PCM_24",
) -> list[Path]:
    """Save separated stems as individual WAV files.

    Args:
        stems: Dictionary mapping stem names to NumPy arrays of shape
            ``[channels, samples]``.
        output_dir: Directory to write the WAV files into.  Created if it
            does not exist.
        sample_rate: Sample rate for the output WAV files.
        subtype: Sound-file PCM subtype (default ``"PCM_24"`` for 24-bit).

    Returns:
        List of ``Path`` objects for the written WAV files.

    Raises:
        ValueError: If a stem array has an unexpected shape.
        OSError: If writing a file fails.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []

    for name, audio in stems.items():
        if audio.ndim != 2:
            raise ValueError(
                f"Expected stem '{name}' to have 2 dimensions [channels, samples], "
                f"got shape {audio.shape}"
            )

        out_path = output_dir / f"{name}.wav"

        # soundfile expects [samples, channels].
        sf.write(
            str(out_path),
            audio.T,
            samplerate=sample_rate,
            subtype=subtype,
        )

        logger.info(
            "Saved stem '%s' → %s  (%.1f s, %d ch)",
            name,
            out_path,
            audio.shape[1] / sample_rate,
            audio.shape[0],
        )
        written.append(out_path)

    return written


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry-point for source separation.

    Parses arguments, loads Demucs, separates the input file, and saves
    each stem as a WAV file in the output directory.
    """
    parser = argparse.ArgumentParser(
        description="Separate an audio file into stems using Demucs v4",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m neo_neural.separator song.wav --output ./stems/\n"
            "  python -m neo_neural.separator song.wav -o stems/ --model htdemucs\n"
        ),
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Path to the input audio file (WAV, FLAC, MP3, …)",
    )
    parser.add_argument(
        "--output",
        "-o",
        type=Path,
        default=Path("./stems"),
        help="Output directory for stem WAV files (default: ./stems/)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_MODEL,
        help=f"Demucs model name (default: {DEFAULT_MODEL})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Torch device (default: auto-detect CUDA/CPU)",
    )
    parser.add_argument(
        "--shifts",
        type=int,
        default=1,
        help="Number of random shifts for TTA (default: 1)",
    )
    parser.add_argument(
        "--overlap",
        type=float,
        default=_DEFAULT_OVERLAP,
        help=f"Overlap between inference chunks (default: {_DEFAULT_OVERLAP})",
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

    if not args.input.exists():
        logger.error("Input file does not exist: %s", args.input)
        raise SystemExit(1)

    logger.info("=== NEO Source Separator ===")
    logger.info("Input : %s", args.input.resolve())
    logger.info("Output: %s", args.output.resolve())
    logger.info("Model : %s", args.model)

    # Load model.
    model = load_separator(args.model, device=args.device)

    # Separate.
    stems = separate(
        args.input,
        model,
        device=args.device,
        shifts=args.shifts,
        overlap=args.overlap,
    )

    # Determine output sample rate from the model.
    model_sr: int = model.samplerate  # type: ignore[attr-defined]

    # Save stems.
    written = save_stems(stems, args.output, sample_rate=model_sr)

    logger.info("Done — %d stems written to %s", len(written), args.output.resolve())
    for p in written:
        logger.info("  • %s", p.name)


if __name__ == "__main__":
    main()
