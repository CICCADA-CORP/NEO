"""
Forked Descript Audio Codec (DAC) — adapted for NEO stem-aware encoding.

Based on: https://github.com/descriptinc/descript-audio-codec (MIT License)

This package is a thin adaptation layer on top of the upstream
``descript-audio-codec`` library.  Rather than vendoring the entire DAC
source tree, we re-export key classes and provide NEO-specific wrappers
that add stem-aware encoding behaviour.

Planned Modifications
---------------------

1. **Stem-conditioned encoder** — Inject a stem-type embedding into the
   encoder bottleneck so the model can allocate codebook capacity
   differently for vocals vs. drums vs. bass vs. other.

2. **Adaptive RVQ depth** — Allow the quantiser to use fewer codebooks
   for "easy" stems (e.g. bass is spectrally simple) and more for
   "hard" stems (e.g. vocals), reducing bitrate without quality loss.

3. **ONNX-friendly forward path** — Minor graph surgery to ensure all
   operations trace cleanly through ``torch.onnx.export()`` (replace
   in-place ops, avoid data-dependent control flow).

4. **Residual coding utilities** — Helpers that compute ``mix − Σ(stems)``
   residual signals for lossless reconstruction when the NEO container
   stores both the stem codes and a lightweight residual layer.

Current Status
--------------

The upstream ``dac`` package (``descript-audio-codec >= 1.0.0``) is used
directly.  The modifications listed above will land incrementally during
Phase 2 of the NEO project.  For now this module simply re-exports the
public API so that the rest of ``neo_neural`` can import from a single
namespace::

    from neo_neural.dac_fork import DAC, DACFile

Re-exports
----------

.. autosummary::
    DAC
    DACFile
"""

from __future__ import annotations

__all__ = [
    "DAC",
    "DACFile",
    "load_model",
]

# ---------------------------------------------------------------------------
# Re-exports from upstream descript-audio-codec
# ---------------------------------------------------------------------------

try:
    from dac import DAC, DACFile
    from dac.utils import download as _download
except ImportError:
    # Allow the package to be imported even when dac is not installed
    # (e.g. for documentation builds or lightweight dependency checks).
    DAC = None  # type: ignore[assignment,misc]
    DACFile = None  # type: ignore[assignment,misc]
    _download = None  # type: ignore[assignment]


def load_model(
    model_type: str = "44khz",
    device: str = "cpu",
) -> "DAC":
    """Convenience loader for a pretrained DAC model.

    Downloads the checkpoint on first call (cached by ``dac.utils``) and
    returns the model in eval mode on the requested device.

    Args:
        model_type: One of ``"16khz"``, ``"24khz"``, ``"44khz"``.
        device: Torch device string (``"cpu"``, ``"cuda"``, ``"mps"``, …).

    Returns:
        A ``dac.DAC`` model instance ready for inference.

    Raises:
        ImportError: If the ``descript-audio-codec`` package is not installed.
        ValueError: If *model_type* is not recognised by the upstream library.
    """
    if DAC is None or _download is None:
        raise ImportError(
            "The 'descript-audio-codec' package is required but not installed. "
            "Install it with:  pip install descript-audio-codec>=1.0.0"
        )

    import torch

    model_path = _download(model_type=model_type)
    model = DAC.load(model_path)
    model.to(torch.device(device))
    model.eval()
    return model
