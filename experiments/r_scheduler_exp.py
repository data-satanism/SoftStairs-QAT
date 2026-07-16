#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Set, Tuple

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
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, TensorDataset

import sys
from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parent.parent
SRC_DIR = ROOT_DIR / "src"
sys.path.insert(0, str(SRC_DIR))

from softstairs_qat import QuantizationConfig
from softstairs_qat.core.quantizer import SoftStairsQuantizer
from softstairs_qat.utils import ReproducibilityManager, DeviceResolver, configure_logging

warnings.filterwarnings("ignore")


@dataclass
class ExperimentSettings:
    """Hyperparameters for the scheduler ablation experiment."""

    batch_size: int = 64
    test_size: float = 0.2
    hidden_size: int = 64
    n_epochs: int = 30
    learning_rate: float = 0.001
    n_bits: int = 4
    symmetric: bool = False
    modified: bool = False
    is_lora: bool = False

    r_start: float = 0.5
    r_end: float = 0.9999
    r_tau: float = 8.0
    r_step: int = 100
    strategies: List[str] = field(
        default_factory=lambda: ["linear", "exp", "step", "cos"]
    )

    excluded_modules: Set[str] = field(default_factory=set)

    seed: int = 42
    warmup_epochs: int = 0


class Perceptron(nn.Module):
    def __init__(self, input_size: int, hidden_size: int, num_classes: int):
        super().__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, num_classes)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x


class DigitsDataModule:
    def __init__(
        self,
        batch_size: int = 64,
        test_size: float = 0.2,
        seed: int = 42,
    ):
        self.batch_size = batch_size
        self.test_size = test_size
        self.seed = seed

    def dataloaders(self) -> Tuple[DataLoader, DataLoader, int, int]:
        digits = load_digits()
        X, y = digits.data, digits.target

        scaler = StandardScaler()
        X = scaler.fit_transform(X)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, random_state=self.seed, stratify=y
        )

        X_train = torch.FloatTensor(X_train)
        y_train = torch.LongTensor(y_train)
        X_test = torch.FloatTensor(X_test)
        y_test = torch.LongTensor(y_test)

        train_loader = DataLoader(
            TensorDataset(X_train, y_train),
            batch_size=self.batch_size,
            shuffle=True,
            num_workers=0,
        )
        test_loader = DataLoader(
            TensorDataset(X_test, y_test),
            batch_size=self.batch_size,
            shuffle=False,
            num_workers=0,
        )

        return train_loader, test_loader, X_train.shape[1], len(np.unique(y))


class PerceptronFactory:
    def create(self, input_size: int, hidden_size: int, num_classes: int) -> Perceptron:
        return Perceptron(input_size, hidden_size, num_classes)


class MetricsEvaluator:
    def evaluate(
        self,
        model: nn.Module,
        dataloader: DataLoader,
        device: torch.device,
    ) -> Tuple[float, float]:
        model.eval()
        predictions: List[int] = []
        labels: List[int] = []

        with torch.no_grad():
            for inputs, targets in dataloader:
                inputs = inputs.to(device)
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                preds = logits.argmax(dim=1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(targets.numpy())

        return (
            accuracy_score(labels, predictions),
            f1_score(labels, predictions, average="weighted"),
        )


class SchedulerAblationRunner:
    def __init__(
        self,
        settings: ExperimentSettings,
        device_resolver: DeviceResolver | None = None,
        metrics_evaluator: MetricsEvaluator | None = None,
    ) -> None:
        self.settings = settings
        self.device_resolver = device_resolver or DeviceResolver()
        self.metrics_evaluator = metrics_evaluator or MetricsEvaluator()
        self._results: List[Dict[str, Any]] = []

    def _create_base_model(self, input_size: int, num_classes: int) -> Perceptron:
        return PerceptronFactory().create(
            input_size, self.settings.hidden_size, num_classes
        )

    def _get_base_model_state(self, model: nn.Module) -> Dict[str, torch.Tensor]:
        return model.state_dict()

    def _load_base_model_state(
        self,
        model: nn.Module,
        state_dict: Dict[str, torch.Tensor],
    ) -> None:
        model.load_state_dict(state_dict)

    def _create_quantization_config(self, strategy: str) -> QuantizationConfig:
        return QuantizationConfig(
            r=self.settings.r_start,
            n_bits=self.settings.n_bits,
            symmetric=self.settings.symmetric,
            modified=self.settings.modified,
            is_lora=self.settings.is_lora,
            r_scheduler_strategy=strategy,
            r_start=self.settings.r_start,
            r_end=self.settings.r_end,
            r_tau=self.settings.r_tau,
            r_step=self.settings.r_step,
        )

    def _dequant_finalized_model(
        self,
        int_model: nn.Module,
        quantizer: SoftStairsQuantizer,
        input_size: int,
        num_classes: int,
        device: torch.device,
    ) -> Perceptron:
        """
        After finalize(), weights are int in code space and scale hooks are gone.
        Build a float clone with physical weights: W = (W_q - zp) * scale
        for a correct post-finalize evaluation.
        """
        float_model = self._create_base_model(input_size, num_classes)
        with torch.no_grad():
            for name, module in int_model.named_modules():
                if name not in quantizer._scales:
                    continue
                if not hasattr(module, "weight"):
                    continue
                scale = quantizer._scales[name]
                zp = quantizer._zero_points[name]
                w_int = module.weight.data.float()
                w_phys = (w_int - zp) * scale
                
                target = float_model.get_submodule(name)
                target.weight.data.copy_(w_phys.to(dtype=target.weight.dtype))
                
                if module.bias is not None and target.bias is not None:
                    target.bias.data.copy_(module.bias.data.float().to(dtype=target.bias.dtype))
        
        return float_model.to(device)

    def _train_softstairs(
        self,
        strategy: str,
        train_loader: DataLoader,
        test_loader: DataLoader,
        input_size: int,
        num_classes: int,
        base_state: Dict[str, torch.Tensor],
        device: torch.device,
    ) -> Dict[str, Any]:
        """Train with SoftStairs quantization (non-LoRA)."""
        model = self._create_base_model(input_size, num_classes)
        self._load_base_model_state(model, base_state)
        model.to(device)

        total_steps = len(train_loader) * self.settings.n_epochs
        config = self._create_quantization_config(strategy)

        quantizer = SoftStairsQuantizer(
            model=model,
            config=config,
            total_steps=total_steps,
            excluded_modules=set(self.settings.excluded_modules),
        )

        optimizer = optim.SGD(
            model.parameters(),
            lr=self.settings.learning_rate,
        )
        criterion = nn.CrossEntropyLoss()

        train_losses: List[float] = []
        train_accs: List[float] = []
        train_f1s: List[float] = []
        test_accs: List[float] = []
        test_f1s: List[float] = []
        r_values: List[float] = []

        best_test_acc = 0.0
        best_test_f1 = 0.0
        best_epoch = 0

        logger.info(f"Training with SoftStairs strategy: {strategy}")
        logger.info(f"Total steps: {total_steps}")
        logger.info(f"Excluded modules: {self.settings.excluded_modules or '{}'}")
        logger.info(f"Quantized layers: {sorted(quantizer._scales.keys())}")

        for epoch in range(self.settings.n_epochs):
            current_r = quantizer.get_current_r()
            r_values.append(current_r)

            model.train()
            epoch_loss = 0.0
            


            for inputs, labels in train_loader:
                inputs, labels = inputs.to(device), labels.to(device)

                optimizer.zero_grad()
                outputs = model(inputs)
                logits = outputs.logits if hasattr(outputs, "logits") else outputs
                loss = criterion(logits, labels)
                if epoch == 0: 
                    logger.info(
                    f"Startng Loss={loss:.4f}, "

                )
                loss.backward()
                optimizer.step()
                quantizer.step()

                epoch_loss += loss.item()

            train_acc, train_f1 = self.metrics_evaluator.evaluate(
                model, train_loader, device
            )
            test_acc, test_f1 = self.metrics_evaluator.evaluate(
                model, test_loader, device
            )

            avg_loss = epoch_loss / len(train_loader)
            train_losses.append(avg_loss)
            train_accs.append(train_acc)
            train_f1s.append(train_f1)
            test_accs.append(test_acc)
            test_f1s.append(test_f1)

            if test_acc > best_test_acc:
                best_test_acc = test_acc
                best_test_f1 = test_f1
                best_epoch = epoch + 1

            logger.info(
                f"  Epoch {epoch+1:2d}/{self.settings.n_epochs}: "
                f"Loss={avg_loss:.4f}, "
                f"Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}, "
                f"Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}, "
                f"r={current_r:.6f}"
            )

        logger.info(f"  Best Test Acc: {best_test_acc:.4f} (epoch {best_epoch})")
        logger.info(f"  Best Test F1:  {best_test_f1:.4f} (epoch {best_epoch})")

        logger.info("  Finalizing model...")
        model = quantizer.finalize()

        eval_model = self._dequant_finalized_model(
            model, quantizer, input_size, num_classes, device
        )
        final_test_acc, final_test_f1 = self.metrics_evaluator.evaluate(
            eval_model, test_loader, device
        )
        final_train_acc, final_train_f1 = self.metrics_evaluator.evaluate(
            eval_model, train_loader, device
        )

        return {
            "method": f"softstairs_{strategy}",
            "best_test_acc": best_test_acc,
            "best_test_f1": best_test_f1,
            "best_epoch": best_epoch,
            "final_test_acc": final_test_acc,
            "final_test_f1": final_test_f1,
            "final_train_acc": final_train_acc,
            "final_train_f1": final_train_f1,
            "final_loss": train_losses[-1],
            "train_losses": train_losses,
            "train_accs": train_accs,
            "train_f1s": train_f1s,
            "test_accs": test_accs,
            "test_f1s": test_f1s,
            "r_values": r_values,
        }

    # def _train_torch_qat(
    #     self,
    #     train_loader: DataLoader,
    #     test_loader: DataLoader,
    #     input_size: int,
    #     num_classes: int,
    #     base_state: Dict[str, torch.Tensor],
    #     device: torch.device,
    # ) -> Dict[str, Any]:
    #     """Train with PyTorch QAT (baseline)."""
    #     model = self._create_base_model(input_size, num_classes)
    #     self._load_base_model_state(model, base_state)
    #     model.to(device)

    #     model.qconfig = get_default_qat_qconfig("fbgemm")
    #     model = prepare_qat(model)
    #     model.to(device)

    #     optimizer = optim.Adam(
    #         model.parameters(),
    #         lr=self.settings.learning_rate,
    #     )
    #     criterion = nn.CrossEntropyLoss()

    #     train_losses: List[float] = []
    #     train_accs: List[float] = []
    #     train_f1s: List[float] = []
    #     test_accs: List[float] = []
    #     test_f1s: List[float] = []

    #     best_test_acc = 0.0
    #     best_test_f1 = 0.0
    #     best_epoch = 0

    #     logger.info("Training with PyTorch QAT (baseline)")

    #     for epoch in range(self.settings.n_epochs):
    #         model.train()
    #         epoch_loss = 0.0

    #         for inputs, labels in train_loader:
    #             inputs, labels = inputs.to(device), labels.to(device)

    #             optimizer.zero_grad()
    #             outputs = model(inputs)
    #             logits = outputs.logits if hasattr(outputs, "logits") else outputs
    #             loss = criterion(logits, labels)
    #             loss.backward()
    #             optimizer.step()

    #             epoch_loss += loss.item()

    #         train_acc, train_f1 = self.metrics_evaluator.evaluate(
    #             model, train_loader, device
    #         )
    #         test_acc, test_f1 = self.metrics_evaluator.evaluate(
    #             model, test_loader, device
    #         )

    #         avg_loss = epoch_loss / len(train_loader)
    #         train_losses.append(avg_loss)
    #         train_accs.append(train_acc)
    #         train_f1s.append(train_f1)
    #         test_accs.append(test_acc)
    #         test_f1s.append(test_f1)

    #         if test_acc > best_test_acc:
    #             best_test_acc = test_acc
    #             best_test_f1 = test_f1
    #             best_epoch = epoch + 1

    #         logger.info(
    #             f"  Epoch {epoch+1:2d}/{self.settings.n_epochs}: "
    #             f"Loss={avg_loss:.4f}, "
    #             f"Train Acc={train_acc:.4f}, Train F1={train_f1:.4f}, "
    #             f"Test Acc={test_acc:.4f}, Test F1={test_f1:.4f}"
    #         )

    #     logger.info(f"  Best Test Acc: {best_test_acc:.4f} (epoch {best_epoch})")
    #     logger.info(f"  Best Test F1:  {best_test_f1:.4f} (epoch {best_epoch})")

    #     logger.info("  Converting to quantized model...")
    #     model.eval()
    #     model = convert(model.cpu(), inplace=False).to(device)

    #     final_test_acc, final_test_f1 = self.metrics_evaluator.evaluate(
    #         model, test_loader, device
    #     )
    #     final_train_acc, final_train_f1 = self.metrics_evaluator.evaluate(
    #         model, train_loader, device
    #     )

    #     return {
    #         "method": "torch_qat",
    #         "best_test_acc": best_test_acc,
    #         "best_test_f1": best_test_f1,
    #         "best_epoch": best_epoch,
    #         "final_test_acc": final_test_acc,
    #         "final_test_f1": final_test_f1,
    #         "final_train_acc": final_train_acc,
    #         "final_train_f1": final_train_f1,
    #         "final_loss": train_losses[-1],
    #         "train_losses": train_losses,
    #         "train_accs": train_accs,
    #         "train_f1s": train_f1s,
    #         "test_accs": test_accs,
    #         "test_f1s": test_f1s,
    #         "r_values": None,
    #     }

    def run_ablation(self) -> List[Dict[str, Any]]:
        device = self.device_resolver.resolve()
        logger.info("=" * 80)
        logger.info("QAT ABLATION STUDY: Perceptron + Digits")
        logger.info("=" * 80)
        logger.info(f"SoftStairs strategies: {self.settings.strategies}")
        logger.info("Baseline: PyTorch QAT")
        logger.info(f"r range: {self.settings.r_start} -> {self.settings.r_end}")
        logger.info(f"Epochs: {self.settings.n_epochs}")
        logger.info(f"Hidden size: {self.settings.hidden_size}")
        logger.info(f"Batch size: {self.settings.batch_size}")
        logger.info(f"Learning rate: {self.settings.learning_rate}")
        logger.info(f"n_bits: {self.settings.n_bits}")
        logger.info(f"excluded_modules: {self.settings.excluded_modules or '{}'}")
        logger.info(f"Device: {device}")
        logger.info("=" * 80)

        logger.info("\nLoading Digits dataset...")
        data_module = DigitsDataModule(
            batch_size=self.settings.batch_size,
            seed=self.settings.seed,
        )
        train_loader, test_loader, input_size, num_classes = data_module.dataloaders()

        logger.info(f"Input size: {input_size}, Classes: {num_classes}")
        logger.info(
            f"Train: {len(train_loader.dataset)}, Test: {len(test_loader.dataset)}"
        )

        logger.info("\nCreating base model...")
        base_model = self._create_base_model(input_size, num_classes)
        base_state = self._get_base_model_state(base_model)

        self._results = []

        logger.info(f"\n{'='*60}")
        logger.info("BASELINE: PyTorch QAT")
        logger.info("=" * 60)

        # self._results.append(
        #     self._train_torch_qat(
        #         train_loader=train_loader,
        #         test_loader=test_loader,
        #         input_size=input_size,
        #         num_classes=num_classes,
        #         base_state=base_state,
        #         device=device,
        #     )
        # )

        for strategy in self.settings.strategies:
            logger.info(f"\n{'='*60}")
            logger.info(f"SoftStairs with strategy: {strategy}")
            logger.info("=" * 60)

            self._results.append(
                self._train_softstairs(
                    strategy=strategy,
                    train_loader=train_loader,
                    test_loader=test_loader,
                    input_size=input_size,
                    num_classes=num_classes,
                    base_state=base_state,
                    device=device,
                )
            )

        return self._results

    @property
    def results(self) -> List[Dict[str, Any]]:
        return self._results


def plot_results(
    results: List[Dict[str, Any]],
    settings: ExperimentSettings,
    show_plot: bool = False,
) -> pd.DataFrame:
    """Visualize ablation study results."""
    baseline = next((r for r in results if r["method"] == "torch_qat"), None)
    softstairs_results = [r for r in results if r["method"] != "torch_qat"]

    colors = {
        "linear": "blue",
        "exp": "orange",
        "step": "green",
        "cos": "red",
        "torch_qat": "black",
    }
    markers = {
        "linear": "o",
        "exp": "s",
        "step": "^",
        "cos": "D",
        "torch_qat": "x",
    }

    fig, axes = plt.subplots(2, 3, figsize=(18, 10))

    ax1 = axes[0, 0]
    if baseline:
        epochs = range(1, len(baseline["test_accs"]) + 1)
        ax1.plot(epochs, baseline["test_accs"], "k-", linewidth=2, label="PyTorch QAT")
    for res in softstairs_results:
        strat = res["method"].replace("softstairs_", "")
        epochs = range(1, len(res["test_accs"]) + 1)
        ax1.plot(
            epochs,
            res["test_accs"],
            marker=markers.get(strat, ""),
            color=colors.get(strat, "gray"),
            linestyle="-",
            label=f"SoftStairs ({strat})",
            linewidth=2,
            markersize=4,
            markevery=5,
        )
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Test Accuracy")
    ax1.set_title("Test Accuracy")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    ax2 = axes[0, 1]
    if baseline:
        epochs = range(1, len(baseline["test_f1s"]) + 1)
        ax2.plot(epochs, baseline["test_f1s"], "k-", linewidth=2, label="PyTorch QAT")
    for res in softstairs_results:
        strat = res["method"].replace("softstairs_", "")
        epochs = range(1, len(res["test_f1s"]) + 1)
        ax2.plot(
            epochs,
            res["test_f1s"],
            marker=markers.get(strat, ""),
            color=colors.get(strat, "gray"),
            linestyle="-",
            label=f"SoftStairs ({strat})",
            linewidth=2,
            markersize=4,
            markevery=5,
        )
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Test F1 Score")
    ax2.set_title("Test F1 Score")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    ax3 = axes[0, 2]
    if baseline:
        epochs = range(1, len(baseline["train_accs"]) + 1)
        ax3.plot(epochs, baseline["train_accs"], "k-", linewidth=2, label="PyTorch QAT")
    for res in softstairs_results:
        strat = res["method"].replace("softstairs_", "")
        epochs = range(1, len(res["train_accs"]) + 1)
        ax3.plot(
            epochs,
            res["train_accs"],
            marker=markers.get(strat, ""),
            color=colors.get(strat, "gray"),
            linestyle="--",
            label=f"SoftStairs ({strat})",
            linewidth=2,
            markersize=4,
            markevery=5,
        )
    ax3.set_xlabel("Epoch")
    ax3.set_ylabel("Train Accuracy")
    ax3.set_title("Train Accuracy")
    ax3.legend()
    ax3.grid(True, alpha=0.3)

    ax4 = axes[1, 0]
    for res in softstairs_results:
        strat = res["method"].replace("softstairs_", "")
        if res.get("r_values"):
            r_values = res["r_values"]
            steps_per_epoch = max(len(r_values) // settings.n_epochs, 1)
            epoch_r = []
            for i in range(settings.n_epochs):
                idx = min((i + 1) * steps_per_epoch - 1, len(r_values) - 1)
                epoch_r.append(r_values[idx])
            epochs = range(1, len(epoch_r) + 1)
            ax4.plot(
                epochs,
                epoch_r,
                color=colors.get(strat, "gray"),
                linestyle="--",
                marker="o",
                markevery=5,
                markersize=4,
                label=f"r ({strat})",
                linewidth=2,
            )
    ax4.set_xlabel("Epoch")
    ax4.set_ylabel("r")
    ax4.set_title("r Dynamics (end of epoch)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    ax5 = axes[1, 1]
    results_bar = []
    labels_bar = []
    if baseline:
        results_bar.append(baseline["best_test_acc"])
        labels_bar.append("PyTorch QAT")
    for res in softstairs_results:
        results_bar.append(res["best_test_acc"])
        labels_bar.append(res["method"].replace("softstairs_", ""))
    bar_colors = ["black"] + [
        colors.get(r["method"].replace("softstairs_", ""), "blue")
        for r in softstairs_results
    ]
    bars = ax5.bar(labels_bar, results_bar, color=bar_colors)
    ax5.set_ylabel("Best Test Accuracy")
    ax5.set_title("Best Test Accuracy")
    ax5.grid(True, alpha=0.3, axis="y")
    for bar, value in zip(bars, results_bar):
        ax5.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    ax6 = axes[1, 2]
    results_bar_f1 = []
    labels_bar_f1 = []
    if baseline:
        results_bar_f1.append(baseline["best_test_f1"])
        labels_bar_f1.append("PyTorch QAT")
    for res in softstairs_results:
        results_bar_f1.append(res["best_test_f1"])
        labels_bar_f1.append(res["method"].replace("softstairs_", ""))
    bar_colors = ["black"] + [
        colors.get(r["method"].replace("softstairs_", ""), "blue")
        for r in softstairs_results
    ]
    bars = ax6.bar(labels_bar_f1, results_bar_f1, color=bar_colors)
    ax6.set_ylabel("Best Test F1")
    ax6.set_title("Best Test F1 Score")
    ax6.grid(True, alpha=0.3, axis="y")
    for bar, value in zip(bars, results_bar_f1):
        ax6.text(
            bar.get_x() + bar.get_width() / 2.0,
            bar.get_height() + 0.005,
            f"{value:.4f}",
            ha="center",
            va="bottom",
            fontsize=9,
        )

    plt.tight_layout()
    plt.savefig("qat_ablation_results.png", dpi=150)
    if show_plot:
        plt.show()
    else:
        plt.close()

    logger.info("\n" + "=" * 80)
    logger.info("QAT ABLATION STUDY RESULTS")
    logger.info("=" * 80)

    df = pd.DataFrame(results)
    logger.info("\n" + df[["method", "best_test_acc", "best_test_f1", "best_epoch"]].to_string(index=False))

    if baseline is not None:
        baseline_acc = baseline["best_test_acc"]
        baseline_f1 = baseline["best_test_f1"]
        logger.info(
            f"\nGain over PyTorch QAT (Acc={baseline_acc:.4f}, F1={baseline_f1:.4f}):"
        )
        for res in softstairs_results:
            strat = res["method"].replace("softstairs_", "")
            gain_acc = res["best_test_acc"] - baseline_acc
            gain_f1 = res["best_test_f1"] - baseline_f1
            logger.info(
                f"  {strat}: Acc {gain_acc:+.4f} (best: {res['best_test_acc']:.4f}), "
                f"F1 {gain_f1:+.4f} (best: {res['best_test_f1']:.4f})"
            )

    return df


def main() -> None:
    """Run the QAT ablation experiment."""
    ReproducibilityManager().set_seed(42)

    settings = ExperimentSettings(
        n_epochs=30,
        batch_size=64,
        hidden_size=64,
        learning_rate=1e-3,
        n_bits=16,
        symmetric=False,
        modified=False,
        r_start=0.5,
        r_end=0.9999,
        r_tau=8.0,
        r_step=100,
        strategies=["linear", "exp", "step", "cos"],
        excluded_modules=set(),
        is_lora=False,
        seed=42,
        warmup_epochs=0,
    )

    logger.info("\n" + "=" * 80)
    logger.info("QAT ABLATION STUDY: Perceptron + SoftStairs vs PyTorch QAT")
    logger.info("=" * 80)
    logger.info(f"Model: Perceptron with {settings.hidden_size} neurons")
    logger.info("Dataset: Digits (8x8 images, 10 classes)")
    logger.info("Baseline: PyTorch QAT")
    logger.info(f"SoftStairs strategies: {settings.strategies}")
    logger.info(f"r: {settings.r_start} -> {settings.r_end}, tau={settings.r_tau}")
    logger.info(f"Epochs: {settings.n_epochs}")
    logger.info(f"n_bits: {settings.n_bits}")
    logger.info(f"excluded_modules: {settings.excluded_modules or '{}'}")
    logger.info("=" * 80)

    runner = SchedulerAblationRunner(settings)
    results = runner.run_ablation()

    df = pd.DataFrame(results)
    df.to_csv("qat_ablation_results.csv", index=False)
    logger.info("\n Results saved to qat_ablation_results.csv")

    plot_results(results, settings, show_plot=False)
    logger.info("\n Plots saved to qat_ablation_results.png")

    logger.info("\n" + "=" * 80)
    logger.info("QAT ABLATION STUDY COMPLETED")
    logger.info("=" * 80)


if __name__ == "__main__":
    paths = configure_logging(name="qat_ablation")
    logger.info(f"Log file: {paths.log_file}")
    main()