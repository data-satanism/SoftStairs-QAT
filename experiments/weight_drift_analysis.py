from __future__ import annotations

import sys
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Dict, List

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset
from torchvision import datasets, transforms

from softstairs_qat.core.quantization_params import QuantizationParamsCalculator
from softstairs_qat.core.soft_stairs import SoftStairs
from softstairs_qat.utils import ReproducibilityManager


@dataclass
class DriftExperimentConfig:
    """Configuration for the weight drift experiment."""

    dataset: str = "MNIST"
    n_bits: int = 32
    n_epochs: int = 30
    learning_rates: List[float] | None = None
    master_seed: int = 42

    def __post_init__(self) -> None:
        if self.learning_rates is None:
            self.learning_rates = [0.0005, 0.001, 0.005, 0.01]


class QuantizationStrategy(ABC):
    """Base quantization strategy used during training."""

    @abstractmethod
    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        """Quantize a weight tensor for the forward pass."""


class RoundSTEStrategy(QuantizationStrategy):
    """Round-with-STE baseline quantizer."""

    def __init__(self, n_bits: int = 32) -> None:
        self._calculator = QuantizationParamsCalculator()
        self.n_bits = n_bits

    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        params = self._calculator.compute(weight, self.n_bits, symmetric=False)
        scaled = weight / params.scale + params.zero_point
        quantized = torch.round(scaled).clamp(params.q_min, params.q_max)
        return (quantized - params.zero_point) * params.scale


class SoftStairsSTEStrategy(QuantizationStrategy):
    """SoftStairs STE quantizer with optional modified variant."""

    def __init__(self, r: float = 0.99, modified: bool = False, n_bits: int = 32) -> None:
        self._calculator = QuantizationParamsCalculator()
        self._soft_stairs = SoftStairs(r=r, modified=modified)
        self.n_bits = n_bits

    def quantize(self, weight: torch.Tensor) -> torch.Tensor:
        params = self._calculator.compute(weight, self.n_bits, symmetric=False)
        scaled = weight / params.scale + params.zero_point
        soft_rounded = self._soft_stairs.forward(scaled)
        return (soft_rounded - params.zero_point) * params.scale


class MLPClassifier(nn.Module):
    """Simple MLP used for drift experiments."""

    def __init__(
        self,
        input_dim: int = 784,
        hidden_dims: tuple[int, ...] = (512, 256, 128),
        output_dim: int = 10,
        dropout: float = 0.3,
    ) -> None:
        super().__init__()
        layers: List[nn.Module] = []
        prev_dim = input_dim
        for hidden_dim in hidden_dims:
            layers.extend([
                nn.Linear(prev_dim, hidden_dim),
                nn.BatchNorm1d(hidden_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
            ])
            prev_dim = hidden_dim
        layers.append(nn.Linear(prev_dim, output_dim))
        self.model = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if len(x.shape) > 2:
            x = x.view(x.size(0), -1)
        return self.model(x)


class DatasetFactory:
    """Creates dataloaders for supported drift experiment datasets."""

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed

    def create(self, dataset_name: str, batch_size: int = 128):
        if dataset_name == "MNIST":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,)),
            ])
            train_dataset = datasets.MNIST("./data", train=True, download=True, transform=transform)
            test_dataset = datasets.MNIST("./data", train=False, download=True, transform=transform)
            input_dim, output_dim = 784, 10
        elif dataset_name == "CIFAR10":
            transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
            ])
            train_dataset = datasets.CIFAR10(
                "./data", train=True, download=True, transform=transform)
            test_dataset = datasets.CIFAR10(
                "./data", train=False, download=True, transform=transform)
            input_dim, output_dim = 3072, 10
        else:
            data = load_digits()
            x_train, x_test, y_train, y_test = train_test_split(
                data.data,
                data.target,
                test_size=0.2,
                random_state=self.seed,
                stratify=data.target,
            )
            scaler = StandardScaler()
            x_train = torch.tensor(scaler.fit_transform(x_train), dtype=torch.float32)
            x_test = torch.tensor(scaler.transform(x_test), dtype=torch.float32)
            y_train = torch.tensor(y_train, dtype=torch.long)
            y_test = torch.tensor(y_test, dtype=torch.long)
            train_dataset = TensorDataset(x_train, y_train)
            test_dataset = TensorDataset(x_test, y_test)
            input_dim, output_dim = x_train.shape[1], 10

        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
        test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
        return train_loader, test_loader, input_dim, output_dim


class DriftExperimentRunner:
    """Trains an MLP while tracking weight drift statistics."""

    def __init__(self, config: DriftExperimentConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._dataset_factory = DatasetFactory(seed=config.master_seed)

    def _compute_weight_stats(self, model: nn.Module) -> tuple[float, float]:
        weights = []
        for name, param in model.named_parameters():
            if "weight" in name:
                weights.append(param.data.flatten())
        if not weights:
            return 0.0, 0.0
        stacked = torch.cat(weights)
        return stacked.mean().item(), stacked.abs().sum().item()

    def train_with_strategy(
        self,
        strategy: QuantizationStrategy,
        learning_rate: float,
        verbose: bool = False,
    ) -> Dict[str, List[float]]:
        train_loader, test_loader, input_dim, output_dim = self._dataset_factory.create(
            self.config.dataset,
        )
        model = MLPClassifier(input_dim=input_dim, output_dim=output_dim).to(self.device)
        optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=1e-5)
        criterion = nn.CrossEntropyLoss()

        stats: Dict[str, List[float]] = {
            "epoch": [],
            "step": [],
            "mean_weight": [],
            "l1_norm": [],
            "loss": [],
            "accuracy": [],
        }
        step_counter = 0

        for epoch in range(1, self.config.n_epochs + 1):
            model.train()
            for data, target in train_loader:
                data, target = data.to(self.device), target.to(self.device)
                originals = {}
                for name, param in model.named_parameters():
                    if param.requires_grad and "weight" in name:
                        originals[name] = param.data.clone()
                        param.data = strategy.quantize(param.data)

                optimizer.zero_grad()
                loss = criterion(model(data), target)
                loss.backward()

                mean_weight, l1_norm = self._compute_weight_stats(model)
                stats["epoch"].append(epoch)
                stats["step"].append(step_counter)
                stats["mean_weight"].append(mean_weight)
                stats["l1_norm"].append(l1_norm)
                stats["loss"].append(loss.item())

                for name, param in model.named_parameters():
                    if name in originals:
                        param.data = originals[name]
                optimizer.step()
                step_counter += 1

            accuracy = self._evaluate(model, test_loader, strategy)
            for _ in range(len(train_loader)):
                stats["accuracy"].append(accuracy)

            if verbose and (epoch % 5 == 0 or epoch == self.config.n_epochs):
                logger.info(
                    "Epoch {:3d} | Loss: {:.4f} | Acc: {:.4f}",
                    epoch,
                    np.mean(stats["loss"][-len(train_loader):]),
                    accuracy,
                )

        return stats

    def _evaluate(
        self,
        model: nn.Module,
        test_loader: DataLoader,
        strategy: QuantizationStrategy,
    ) -> float:
        model.eval()
        correct = 0
        total = 0
        originals = {}
        for name, param in model.named_parameters():
            if param.requires_grad and "weight" in name:
                originals[name] = param.data.clone()
                param.data = strategy.quantize(param.data)

        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(self.device), target.to(self.device)
                pred = model(data).argmax(dim=1)
                correct += (pred == target).sum().item()
                total += target.size(0)

        for name, param in model.named_parameters():
            if name in originals:
                param.data = originals[name]
        return correct / total

    def run_all(self) -> Dict[float, Dict[str, Dict[str, List[float]]]]:
        results: Dict[float, Dict[str, Dict[str, List[float]]]] = {}
        strategies = {
            "Round": RoundSTEStrategy(self.config.n_bits),
            "SoftStairs": SoftStairsSTEStrategy(r=0.99, modified=False, n_bits=self.config.n_bits),
            "SoftStairs_Modified": SoftStairsSTEStrategy(
                r=0.99, modified=True, n_bits=self.config.n_bits,
            ),
        }

        for learning_rate in self.config.learning_rates:
            logger.info("Training for learning rate = {}", learning_rate)
            results[learning_rate] = {}
            for name, strategy in strategies.items():
                logger.info("  Strategy: {}", name)
                results[learning_rate][name] = self.train_with_strategy(
                    strategy, learning_rate, verbose=True,
                )
        return results


class ResultsExporter:
    """Persists experiment statistics to CSV files."""

    def save_first_steps(
        self,
        results: Dict[float, Dict[str, Dict[str, List[float]]]],
        learning_rates: List[float],
        n_steps: int,
        filename: str,
    ) -> pd.DataFrame:
        max_steps = n_steps
        for learning_rate in learning_rates:
            for strategy in results[learning_rate]:
                max_steps = min(max_steps, len(results[learning_rate][strategy]["loss"]))

        data = {"step": list(range(max_steps))}
        for learning_rate in learning_rates:
            for strategy in results[learning_rate]:
                column = f"LR_{learning_rate}_{strategy}"
                data[column] = results[learning_rate][strategy]["loss"][:max_steps]
        frame = pd.DataFrame(data)
        frame.to_csv(filename, index=False)
        return frame

    def save_full_statistics(
        self,
        results: Dict[float, Dict[str, Dict[str, List[float]]]],
        filename: str,
    ) -> pd.DataFrame:
        records = []
        for learning_rate, strategies in results.items():
            for strategy_name, stats in strategies.items():
                for index in range(len(stats["step"])):
                    records.append({
                        "learning_rate": learning_rate,
                        "strategy": strategy_name,
                        "epoch": stats["epoch"][index],
                        "step": stats["step"][index],
                        "mean_weight": stats["mean_weight"][index],
                        "l1_norm": stats["l1_norm"][index],
                        "loss": stats["loss"][index],
                        "accuracy": stats["accuracy"][index],
                    })
        frame = pd.DataFrame(records)
        frame.to_csv(filename, index=False)
        return frame


class DriftPlotter:
    """Plots summary charts for drift experiments."""

    def plot_summary(
        self,
        results: Dict[float, Dict[str, Dict[str, List[float]]]],
        dataset_name: str,
        save_prefix: str = "drift_analysis",
    ) -> None:
        learning_rates = sorted(results.keys())
        strategies = ["Round", "SoftStairs", "SoftStairs_Modified"]
        colors = {"Round": "red", "SoftStairs": "blue", "SoftStairs_Modified": "green"}

        fig, axes = plt.subplots(2, 2, figsize=(14, 10))
        metrics = [
            ("accuracy", "Final Accuracy"),
            ("l1_norm", "Final L1 Norm"),
            ("mean_weight", "Final Mean Weight"),
            ("loss", "Final Loss"),
        ]

        for axis, (metric, title) in zip(axes.flatten(), metrics):
            for strategy in strategies:
                values = [results[lr][strategy][metric][-1] for lr in learning_rates]
                axis.plot(
                    learning_rates,
                    values,
                    marker="o",
                    label=strategy,
                    color=colors[strategy])
            axis.set_xscale("log")
            axis.set_xlabel("Learning Rate")
            axis.set_ylabel(title)
            axis.set_title(f"{title} ({dataset_name})")
            axis.grid(True, alpha=0.3)
            axis.legend()

        plt.tight_layout()
        plt.savefig(f"{save_prefix}_summary_{dataset_name}.png", dpi=200, bbox_inches="tight")
        plt.close(fig)


def main() -> None:
    config = DriftExperimentConfig(n_epochs=5)
    ReproducibilityManager().set_seed(config.master_seed)

    runner = DriftExperimentRunner(config)
    results = runner.run_all()

    exporter = ResultsExporter()
    exporter.save_first_steps(
        results,
        config.learning_rates,
        n_steps=100,
        filename=f"first_100_steps_loss_{config.dataset}.csv",
    )
    exporter.save_full_statistics(results, filename=f"full_statistics_{config.dataset}.csv")

    DriftPlotter().plot_summary(results, config.dataset)
    logger.info("Weight drift experiment completed.")


if __name__ == "__main__":
    from softstairs_qat.utils import configure_logging

    paths = configure_logging(name="weight_drift_analysis")
    logger.info("Log file: {}", paths.log_file)
    try:
        main()
    except Exception:
        logger.exception("Weight drift experiment failed")
        sys.exit(1)
