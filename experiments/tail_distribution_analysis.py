from __future__ import annotations

from dataclasses import dataclass
from math import erfc
from typing import Dict, List

import numpy as np
import pandas as pd
from loguru import logger

from softstairs_qat.core.variance_controller import VarianceController


@dataclass(frozen=True)
class DtypeConfig:
    """Quantization dtype configuration."""

    name: str
    bitwidth: int
    ceiling: int
    sigma_a_max: float


@dataclass
class TailExperimentConfig:
    """Configuration for tail distribution analysis."""

    rows: int = 1024
    cols: int = 1024
    rank: int = 32
    sigma_w: float = 1.0
    num_experiments: int = 10
    mu_w_options: List[int] | None = None
    sigma_a_percentages: List[float] | None = None

    def __post_init__(self) -> None:
        if self.mu_w_options is None:
            self.mu_w_options = [-6, -4, -2, 0, 2, 4, 10]
        if self.sigma_a_percentages is None:
            self.sigma_a_percentages = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]


class MatrixGenerator:
    """Generates random weight and low-rank correction matrices."""

    def __init__(self, seed: int = 42) -> None:
        self._rng = np.random.default_rng(seed)

    def generate_weight(self, rows: int, cols: int, mean: float, sigma: float) -> np.ndarray:
        return self._rng.normal(mean, sigma, (rows, cols))

    def generate_low_rank_product(
        self,
        rows: int,
        cols: int,
        rank: int,
        sigma_a: float,
        sigma_b: float,
    ) -> np.ndarray:
        matrix_a = self._rng.normal(0.0, sigma_a, (rows, rank))
        matrix_b = self._rng.normal(0.0, sigma_b, (rank, cols))
        return matrix_a @ matrix_b


class TailStatisticsCollector:
    """Collects tail statistics for combined weight tensors."""

    QUANTILES = [0.5, 0.9, 0.95, 0.99, 0.995, 0.999, 0.9995, 0.9999]

    def count_integer_values(self, values: np.ndarray, ceiling: int) -> int:
        rounded = np.round(values).astype(int)
        unique_ints = np.unique(rounded)
        unique_ints = unique_ints[(unique_ints >= -ceiling) & (unique_ints <= ceiling)]
        return len(unique_ints)

    def analyze(
        self,
        values: np.ndarray,
        dtype_config: DtypeConfig,
        mu_w: float,
        sigma_a: float,
        sigma_b: float,
        sigma_w: float,
    ) -> Dict[str, float | int]:
        abs_values = np.abs(values)
        quantile_values = np.percentile(abs_values, [q * 100 for q in self.QUANTILES])
        sigma_emp = float(np.std(values))
        max_abs = float(np.max(abs_values))
        ceiling = dtype_config.ceiling

        if sigma_emp > 0 and ceiling > 0:
            prob_exceed_normal = erfc(ceiling / sigma_emp / np.sqrt(2)) * 100
        else:
            prob_exceed_normal = 0.0

        prob_exceed_emp = float(np.sum(abs_values > ceiling) / len(values) * 100)
        values_in_range = values[np.abs(values) <= ceiling]
        max_fit = float(np.max(values_in_range)) if len(values_in_range) else 0.0

        return {
            "type": dtype_config.name,
            "mu_W": mu_w,
            "sigma_W": sigma_w,
            "sigma_A": sigma_a,
            "sigma_B": sigma_b,
            "T": ceiling,
            "n_values": len(values),
            "sigma": sigma_emp,
            "max_val_abs": max_abs,
            "max_fit": max_fit,
            "prob_exceed_normal": prob_exceed_normal,
            "prob_exceed_emp": prob_exceed_emp,
            "n_integers": self.count_integer_values(values, ceiling),
            **{f"q_{quantile}": value for quantile, value in zip(self.QUANTILES, quantile_values)},
        }


class TailDistributionExperiment:
    """Runs the W + AB tail distribution grid experiment."""

    DTYPE_CONFIGS = {
        "int4": DtypeConfig("int4", 4, 7, 4 / 6),
        "int8": DtypeConfig("int8", 8, 127, 8 / 6),
        "int16": DtypeConfig("int16", 16, 32767, 16 / 6),
        "int32": DtypeConfig("int32", 32, 2147483647, 32 / 6),
    }

    def __init__(self, config: TailExperimentConfig | None = None) -> None:
        self.config = config or TailExperimentConfig()
        self._generator = MatrixGenerator()
        self._collector = TailStatisticsCollector()
        self._variance_controller = VarianceController()

    def run(self) -> pd.DataFrame:
        records = []
        rows, cols, rank = self.config.rows, self.config.cols, self.config.rank
        num_elements = rows * cols
        dynamic_coeff = np.sqrt(2 * np.log(num_elements))

        for dtype_name, dtype_config in self.DTYPE_CONFIGS.items():
            logger.info("Type: {} (T={})", dtype_name, dtype_config.ceiling)
            for mu_w in self.config.mu_w_options:
                weight = self._generator.generate_weight(
                    rows, cols, mu_w, self.config.sigma_w,
                )
                l_bound = dtype_config.ceiling - np.max(weight)

                for percentage in self.config.sigma_a_percentages:
                    sigma_a = percentage * dtype_config.sigma_a_max
                    sigma_b = self._variance_controller.compute_sigma_b_max(
                        l_bound, sigma_a, rank, rows, cols,
                    )

                    combined_samples = []
                    for _ in range(self.config.num_experiments):
                        correction = self._generator.generate_low_rank_product(
                            rows, cols, rank, sigma_a, sigma_b,
                        )
                        combined_samples.append((weight + correction).reshape(-1))

                    values = np.concatenate(combined_samples)
                    stats = self._collector.analyze(
                        values,
                        dtype_config,
                        mu_w,
                        sigma_a,
                        sigma_b,
                        self.config.sigma_w,
                    )
                    records.append(stats)
                    logger.info(
                        "  mu_W={:>3} pct={:>4.0%} max|S|={:.2e} exceed={:.1f}%",
                        mu_w,
                        percentage,
                        stats["max_val_abs"],
                        stats["prob_exceed_emp"],
                    )

        frame = pd.DataFrame(records)
        frame.to_csv("tail_distribution_results.csv", index=False)
        logger.info("Saved {} rows to tail_distribution_results.csv", len(frame))
        logger.info("Dynamic coefficient sqrt(2 ln N) = {:.4f}", dynamic_coeff)
        return frame


def main() -> None:
    logger.info("=" * 80)
    logger.info("Experiment: tail distribution analysis for W + AB")
    logger.info("=" * 80)
    TailDistributionExperiment().run()


if __name__ == "__main__":
    from softstairs_qat.utils import configure_logging

    paths = configure_logging(name="tail_distribution_analysis")
    logger.info("Log file: {}", paths.log_file)
    main()
