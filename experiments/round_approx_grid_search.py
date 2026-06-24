from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Iterable, List

import matplotlib.pyplot as plt
import pandas as pd
import torch
from loguru import logger

from softstairs_qat.core.quantization_params import QuantizationParamsCalculator
from softstairs_qat.core.soft_stairs import SoftStairs


@dataclass
class RoundApproxConfig:
    """Configuration for the round-approximation grid search."""

    r_values: List[float] | None = None
    n_bits: int = 32
    output_weights_path: str = "tinyllm_weights.pt"
    results_csv: str = "round_approx_results.csv"

    def __post_init__(self) -> None:
        if self.r_values is None:
            self.r_values = [0.9, 0.95, 0.98, 0.99, 0.995, 0.999, 0.9995, 0.9999]


class WeightExtractor:
    """Extracts or synthesizes model weights for quantization analysis."""

    def load_or_create(self, model_name: str = "arnir0/TinyLLM") -> Dict[str, torch.Tensor]:
        try:
            from transformers import AutoModel
            model = AutoModel.from_pretrained(model_name)
            weights = {
                name: param.detach().cpu().clone()
                for name, param in model.named_parameters()
            }
            logger.info("Loaded {} tensors from {}", len(weights), model_name)
            return weights
        except Exception as exc:
            logger.warning("Could not load {}: {}", model_name, exc)
            logger.warning("Creating synthetic weights for demonstration.")
            return {
                "0.weight": torch.randn(256, 128) * 0.08,
                "0.bias": torch.randn(256) * 0.01,
                "2.weight": torch.randn(128, 256) * 0.08,
                "2.bias": torch.randn(128) * 0.01,
            }

    def save(self, weights: Dict[str, torch.Tensor], path: str) -> None:
        torch.save(weights, path)
        logger.info("Weights saved to {}", path)

    def load(self, path: str) -> Dict[str, torch.Tensor]:
        weights = torch.load(path, weights_only=True)
        logger.info("Weights loaded from {}", path)
        return weights


class SoftStairsGridEvaluator:
    """Evaluates SoftStairs quantization quality across r values."""

    def __init__(self, n_bits: int = 32) -> None:
        self.n_bits = n_bits
        self._params_calculator = QuantizationParamsCalculator()

    def quantize_tensor(self, tensor: torch.Tensor, r: float) -> torch.Tensor:
        params = self._params_calculator.compute(tensor, self.n_bits, symmetric=False)
        normalized = tensor / params.scale + params.zero_point
        soft_stairs = SoftStairs(r=r, modified=False)
        rounded = soft_stairs.forward(normalized).clamp(params.q_min, params.q_max)
        return (rounded - params.zero_point) * params.scale

    def compute_error_metrics(
        self,
        original: torch.Tensor,
        quantized: torch.Tensor,
    ) -> Dict[str, float]:
        diff = original - quantized
        mse = torch.mean(diff ** 2).item()
        mae = torch.mean(torch.abs(diff)).item()
        relative = (
            torch.norm(diff) / torch.norm(original)
        ).item() if torch.norm(original) > 0 else 0.0
        return {"mse": mse, "mae": mae, "relative_l2": relative}

    def evaluate_r_grid(
        self,
        weights: Dict[str, torch.Tensor],
        r_values: Iterable[float],
    ) -> pd.DataFrame:
        records: List[Dict[str, float | str]] = []
        for r_value in r_values:
            logger.info("Testing r = {}", r_value)
            for name, tensor in weights.items():
                if tensor.ndim < 1:
                    continue
                flat_tensor = tensor.float()
                quantized = self.quantize_tensor(flat_tensor, r_value)
                metrics = self.compute_error_metrics(flat_tensor, quantized)
                records.append({
                    "layer": name,
                    "r": r_value,
                    "shape_rows": flat_tensor.shape[0] if flat_tensor.ndim > 0 else 1,
                    "shape_cols": flat_tensor.numel() // max(flat_tensor.shape[0], 1),
                    **metrics,
                })
                logger.info(
                    "  {}: mse={:.6e}, relative_l2={:.6e}",
                    name,
                    metrics["mse"],
                    metrics["relative_l2"],
                )
        return pd.DataFrame(records)


class RoundApproxExperiment:
    """Coordinates weight extraction and SoftStairs grid evaluation."""

    def __init__(self, config: RoundApproxConfig | None = None) -> None:
        self.config = config or RoundApproxConfig()
        self._extractor = WeightExtractor()
        self._evaluator = SoftStairsGridEvaluator(n_bits=self.config.n_bits)

    def run(self) -> pd.DataFrame:
        weights = self._extractor.load_or_create()
        self._extractor.save(weights, self.config.output_weights_path)
        weights = self._extractor.load(self.config.output_weights_path)

        frame = self._evaluator.evaluate_r_grid(weights, self.config.r_values)
        frame.to_csv(self.config.results_csv, index=False)
        self._plot_r_dependency(frame)
        logger.info("Saved results to {}", self.config.results_csv)
        return frame

    def _plot_r_dependency(self, frame: pd.DataFrame) -> None:
        summary = frame.groupby("r")[["mse", "mae", "relative_l2"]].mean().reset_index()
        fig, axes = plt.subplots(1, 3, figsize=(15, 4))
        for axis, metric in zip(axes, ["mse", "mae", "relative_l2"]):
            axis.plot(summary["r"], summary[metric], marker="o")
            axis.set_xlabel("r")
            axis.set_ylabel(metric)
            axis.set_title(f"Mean {metric} vs r")
            axis.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig("round_approx_r_dependency.png", dpi=200, bbox_inches="tight")
        plt.close(fig)
        logger.info("Saved plot to round_approx_r_dependency.png")


def main() -> None:
    logger.info("=" * 80)
    logger.info("QUANTIZATION REPRESENTATION EXPERIMENT - GRID SEARCH")
    logger.info("=" * 80)
    RoundApproxExperiment().run()


if __name__ == "__main__":
    from softstairs_qat.utils import configure_logging

    paths = configure_logging(name="round_approx_grid_search")
    logger.info("Log file: {}", paths.log_file)
    main()
