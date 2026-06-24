from __future__ import annotations

import os
import random

import numpy as np
import torch


class ReproducibilityManager:
    """Configures deterministic seeds across Python, NumPy, and PyTorch."""

    def set_seed(self, seed: int) -> None:
        """Set global random seeds for reproducible experiments.

        Args:
            seed: Seed value applied to all supported RNG backends.
        """
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed(seed)
            torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        os.environ["PYTHONHASHSEED"] = str(seed)
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.use_deterministic_algorithms(True, warn_only=True)
