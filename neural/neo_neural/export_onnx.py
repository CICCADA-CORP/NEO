"""
ONNX Export Pipeline for DAC models.

Exports the DAC encoder and decoder as separate ONNX models
for inference in Rust via the ``ort`` crate.

The encoder converts raw audio waveforms into quantized RVQ codes,
and the decoder reconstructs audio from those codes.  Splitting them
into two ONNX graphs lets the Rust side run encode / decode independently
and keeps each model small enough for efficient loading.

Usage:
    python -m neo_neural.export_onnx --model 44khz --output ./models/
    python -m neo_neural.export_onnx --model 24khz --output ./models/ --validate
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

import dac
import dac.utils
import numpy as np
import onnx
import onnxruntime as ort
import torch

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MODEL_SAMPLE_RATES: dict[str, int] = {
    "16khz": 16_000,
    "24khz": 24_000,
    "44khz": 44_100,
}

DEFAULT_OPSET_VERSION = 17

# Duration (in seconds) of the dummy waveform used for tracing.
_TRACE_DURATION_SEC = 1.0

# Tolerance for numerical validation between PyTorch and ONNX outputs.
_VALIDATION_ATOL = 1e-4
_VALIDATION_RTOL = 1e-3


# ---------------------------------------------------------------------------
# Encoder / Decoder wrapper modules
# ---------------------------------------------------------------------------


class _EncoderWrapper(torch.nn.Module):
    """Thin wrapper that isolates the DAC encoder + quantiser path.

    ``dac.DAC`` bundles encoder, quantiser and decoder in one module.
    For ONNX export we need a clean ``forward()`` that maps
    ``audio_data → codes``.
    """

    def __init__(self, model: dac.DAC) -> None:
        super().__init__()
        self.encoder = model.encoder
        self.quantizer = model.quantizer

    def forward(self, audio_data: torch.Tensor) -> torch.Tensor:
        """Encode raw audio into RVQ codes.

        Args:
            audio_data: Waveform tensor of shape ``[batch, channels, samples]``.

        Returns:
            Integer code tensor of shape ``[batch, n_codebooks, frames]``.
        """
        z = self.encoder(audio_data)
        _, codes, _, _, _ = self.quantizer(z)
        return codes


class _DecoderWrapper(torch.nn.Module):
    """Thin wrapper that isolates the DAC quantiser-decode + decoder path.

    Maps ``codes → audio_out`` so the Rust side can reconstruct audio
    from integer RVQ codes alone.
    """

    def __init__(self, model: dac.DAC) -> None:
        super().__init__()
        self.quantizer = model.quantizer
        self.decoder = model.decoder

    def forward(self, codes: torch.Tensor) -> torch.Tensor:
        """Decode RVQ codes back into a waveform.

        Args:
            codes: Integer tensor of shape ``[batch, n_codebooks, frames]``.

        Returns:
            Reconstructed waveform of shape ``[batch, channels, samples]``.
        """
        z = self.quantizer.from_codes(codes)[0]
        audio_out = self.decoder(z)
        return audio_out


# ---------------------------------------------------------------------------
# Core functions
# ---------------------------------------------------------------------------


def load_dac_model(model_type: str, device: str = "cpu") -> dac.DAC:
    """Download (if needed) and load a pretrained DAC model.

    Args:
        model_type: One of ``"16khz"``, ``"24khz"``, ``"44khz"``.
        device: Torch device string (``"cpu"``, ``"cuda"``, etc.).

    Returns:
        The loaded ``dac.DAC`` model in eval mode on *device*.

    Raises:
        ValueError: If *model_type* is not a recognised DAC variant.
    """
    if model_type not in MODEL_SAMPLE_RATES:
        raise ValueError(
            f"Unknown model type '{model_type}'. Choose from: {', '.join(MODEL_SAMPLE_RATES)}"
        )

    logger.info("Downloading / locating DAC '%s' checkpoint …", model_type)
    model_path = dac.utils.download(model_type=model_type)
    logger.info("Loading DAC model from %s", model_path)

    model = dac.DAC.load(model_path)
    model.to(device)
    model.eval()

    logger.info(
        "DAC '%s' loaded — sample rate=%d Hz, device=%s",
        model_type,
        MODEL_SAMPLE_RATES[model_type],
        device,
    )
    return model


def export_encoder(
    model: dac.DAC,
    output_path: Path,
    sample_rate: int,
    *,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> Path:
    """Export the DAC encoder (+ quantiser) to an ONNX file.

    Args:
        model: A loaded ``dac.DAC`` model (eval mode, CPU recommended).
        output_path: Directory where the ONNX file will be written.
        sample_rate: Native sample rate of *model* (used for dummy input length).
        opset_version: ONNX opset version (default 17).

    Returns:
        The ``Path`` of the written ``.onnx`` file.

    Raises:
        RuntimeError: If ``torch.onnx.export`` fails.
        onnx.checker.ValidationError: If the exported graph is malformed.
    """
    encoder = _EncoderWrapper(model)
    encoder.eval()

    n_samples = int(sample_rate * _TRACE_DURATION_SEC)
    dummy_input = torch.randn(1, 1, n_samples)

    onnx_path = output_path / f"dac_encoder_{sample_rate}hz.onnx"

    logger.info("Tracing encoder with dummy input shape %s …", list(dummy_input.shape))
    t0 = time.perf_counter()

    torch.onnx.export(
        encoder,
        (dummy_input,),
        str(onnx_path),
        opset_version=opset_version,
        input_names=["audio_data"],
        output_names=["codes"],
        dynamic_axes={
            "audio_data": {0: "batch", 2: "samples"},
            "codes": {0: "batch", 2: "frames"},
        },
    )

    elapsed = time.perf_counter() - t0
    logger.info("Encoder exported in %.2f s → %s", elapsed, onnx_path)

    # Validate the ONNX graph structure.
    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX checker passed for encoder model.")

    return onnx_path


def export_decoder(
    model: dac.DAC,
    output_path: Path,
    sample_rate: int,
    *,
    opset_version: int = DEFAULT_OPSET_VERSION,
) -> Path:
    """Export the DAC decoder to an ONNX file.

    The decoder accepts integer RVQ codes and produces a reconstructed
    waveform.

    Args:
        model: A loaded ``dac.DAC`` model (eval mode, CPU recommended).
        output_path: Directory where the ONNX file will be written.
        sample_rate: Native sample rate of *model* (used for dummy input sizing).
        opset_version: ONNX opset version (default 17).

    Returns:
        The ``Path`` of the written ``.onnx`` file.

    Raises:
        RuntimeError: If ``torch.onnx.export`` fails.
        onnx.checker.ValidationError: If the exported graph is malformed.
    """
    decoder = _DecoderWrapper(model)
    decoder.eval()

    # Obtain a realistic codes tensor by running the encoder once.
    n_samples = int(sample_rate * _TRACE_DURATION_SEC)
    with torch.no_grad():
        dummy_audio = torch.randn(1, 1, n_samples)
        z = model.encoder(dummy_audio)
        _, codes, _, _, _ = model.quantizer(z)

    onnx_path = output_path / f"dac_decoder_{sample_rate}hz.onnx"

    logger.info(
        "Tracing decoder with codes shape %s …",
        list(codes.shape),
    )
    t0 = time.perf_counter()

    torch.onnx.export(
        decoder,
        (codes,),
        str(onnx_path),
        opset_version=opset_version,
        input_names=["codes"],
        output_names=["audio_out"],
        dynamic_axes={
            "codes": {0: "batch", 2: "frames"},
            "audio_out": {0: "batch", 2: "samples"},
        },
    )

    elapsed = time.perf_counter() - t0
    logger.info("Decoder exported in %.2f s → %s", elapsed, onnx_path)

    onnx_model = onnx.load(str(onnx_path))
    onnx.checker.check_model(onnx_model)
    logger.info("ONNX checker passed for decoder model.")

    return onnx_path


def validate_export(
    model: dac.DAC,
    encoder_path: Path,
    decoder_path: Path,
    sample_rate: int,
) -> bool:
    """Compare PyTorch and ONNX Runtime outputs for numerical equivalence.

    Runs a short random waveform through both the original PyTorch model and
    the exported ONNX encoder + decoder, then checks that:

    1. The encoder codes match exactly (integer comparison).
    2. The reconstructed audio matches within a tight tolerance.

    Args:
        model: The original ``dac.DAC`` model used during export.
        encoder_path: Path to the exported encoder ``.onnx`` file.
        decoder_path: Path to the exported decoder ``.onnx`` file.
        sample_rate: Sample rate that matches the model variant.

    Returns:
        ``True`` if validation passes, ``False`` otherwise.

    Raises:
        FileNotFoundError: If either ONNX file does not exist.
    """
    if not encoder_path.exists():
        raise FileNotFoundError(f"Encoder ONNX not found: {encoder_path}")
    if not decoder_path.exists():
        raise FileNotFoundError(f"Decoder ONNX not found: {decoder_path}")

    n_samples = int(sample_rate * _TRACE_DURATION_SEC)
    dummy_audio = torch.randn(1, 1, n_samples)

    # --- PyTorch reference ---------------------------------------------------
    with torch.no_grad():
        z = model.encoder(dummy_audio)
        _, pt_codes, _, _, _ = model.quantizer(z)
        z_hat = model.quantizer.from_codes(pt_codes)[0]
        pt_audio_out = model.decoder(z_hat)

    pt_codes_np: np.ndarray = pt_codes.cpu().numpy()
    pt_audio_np: np.ndarray = pt_audio_out.cpu().numpy()

    # --- ONNX Runtime inference ----------------------------------------------
    so = ort.SessionOptions()
    so.log_severity_level = 3  # suppress verbose logs

    enc_session = ort.InferenceSession(str(encoder_path), sess_options=so)
    dec_session = ort.InferenceSession(str(decoder_path), sess_options=so)

    (onnx_codes,) = enc_session.run(
        ["codes"],
        {"audio_data": dummy_audio.cpu().numpy()},
    )
    (onnx_audio_out,) = dec_session.run(
        ["audio_out"],
        {"codes": onnx_codes},
    )

    # --- Compare -------------------------------------------------------------
    codes_match = np.array_equal(pt_codes_np, onnx_codes)
    audio_close = np.allclose(
        pt_audio_np,
        onnx_audio_out,
        atol=_VALIDATION_ATOL,
        rtol=_VALIDATION_RTOL,
    )

    if codes_match:
        logger.info("✓ Encoder codes match exactly between PyTorch and ONNX.")
    else:
        mismatches = int(np.sum(pt_codes_np != onnx_codes))
        total = int(pt_codes_np.size)
        logger.warning("✗ Encoder code mismatch: %d / %d elements differ.", mismatches, total)

    if audio_close:
        max_diff = float(np.max(np.abs(pt_audio_np - onnx_audio_out)))
        logger.info(
            "✓ Decoded audio matches (max abs diff = %.6f, atol=%.4f).",
            max_diff,
            _VALIDATION_ATOL,
        )
    else:
        max_diff = float(np.max(np.abs(pt_audio_np - onnx_audio_out)))
        logger.warning(
            "✗ Decoded audio mismatch — max abs diff = %.6f (atol=%.4f).",
            max_diff,
            _VALIDATION_ATOL,
        )

    passed = codes_match and audio_close
    if passed:
        logger.info("Validation PASSED ✓")
    else:
        logger.warning("Validation FAILED ✗")
    return passed


# ---------------------------------------------------------------------------
# CLI entry-point
# ---------------------------------------------------------------------------


def main() -> None:
    """CLI entry-point for exporting DAC models to ONNX.

    Parses command-line arguments, loads the requested DAC variant,
    exports encoder and decoder, and optionally validates the export.
    """
    parser = argparse.ArgumentParser(
        description="Export DAC encoder/decoder to ONNX for Rust inference",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "Examples:\n"
            "  python -m neo_neural.export_onnx --model 44khz --output ./models/\n"
            "  python -m neo_neural.export_onnx --model 24khz --output ./models/ --validate\n"
        ),
    )
    parser.add_argument(
        "--model",
        choices=sorted(MODEL_SAMPLE_RATES),
        default="44khz",
        help="DAC model variant to export (default: 44khz)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("./models"),
        help="Directory for output ONNX files (default: ./models/)",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run numerical validation after export",
    )
    parser.add_argument(
        "--opset-version",
        type=int,
        default=DEFAULT_OPSET_VERSION,
        help=f"ONNX opset version (default: {DEFAULT_OPSET_VERSION})",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device for model loading (default: cpu)",
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

    sample_rate = MODEL_SAMPLE_RATES[args.model]
    args.output.mkdir(parents=True, exist_ok=True)

    logger.info("=== NEO ONNX Export Pipeline ===")
    logger.info("Model: DAC %s (%d Hz)", args.model, sample_rate)
    logger.info("Output directory: %s", args.output.resolve())
    logger.info("Opset version: %d", args.opset_version)

    # Load the DAC model.
    model = load_dac_model(args.model, device=args.device)

    # Move to CPU for ONNX tracing (avoids device-specific ops in graph).
    model.cpu()
    model.eval()

    # Export encoder and decoder.
    encoder_path = export_encoder(model, args.output, sample_rate, opset_version=args.opset_version)
    decoder_path = export_decoder(model, args.output, sample_rate, opset_version=args.opset_version)

    logger.info("Export complete:")
    logger.info("  Encoder → %s", encoder_path)
    logger.info("  Decoder → %s", decoder_path)

    # Optionally validate.
    if args.validate:
        logger.info("Running numerical validation …")
        passed = validate_export(model, encoder_path, decoder_path, sample_rate)
        if not passed:
            logger.error("Validation failed — exported models may be inaccurate.")
            sys.exit(1)

    logger.info("Done.")


if __name__ == "__main__":
    main()
