from __future__ import annotations

import torch
from loguru import logger


class DeviceResolver:
    """Selects the best available PyTorch execution device."""

    def resolve(self, verbose: bool = True) -> torch.device:
        """Return CUDA when available, otherwise CPU.

        Args:
            verbose: Whether to log the selected device.

        Returns:
            Resolved torch device.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        if verbose:
            logger.info("Using device: {}", device)
        return device
