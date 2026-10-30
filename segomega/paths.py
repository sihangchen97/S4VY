"""Shared path configuration.

vggt_omega is a pip-installed dependency (see pyproject.toml), so it is imported
directly wherever needed; no sys.path surgery is required anymore. Only the
checkpoint location remains env-driven.
"""
from __future__ import annotations

import os

# Default checkpoint for the 1B / 512px variant.
VGGT_OMEGA_CKPT = os.environ.get(
    "VGGT_OMEGA_CKPT",
    "/PATH/TO/checkpoints/VGGT-Omega/vggt_omega_1b_512.pt",
)
