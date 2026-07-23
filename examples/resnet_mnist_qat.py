from __future__ import annotations

from dataclasses import dataclass
from typing import Tuple

import torch
import torch.nn as nn
import torch.optim as optim
from loguru import logger
from sklearn.metrics import accuracy_score, f1_score
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import AutoConfig, AutoModelForImageClassification

from softstairs_qat import QuantizationConfig, wrap_model_for_quantization
from softstairs_qat.utils import DeviceResolver, ReproducibilityManager


@dataclass
class ExperimentSettings:
    """Hyperparameters for the MNIST QAT demonstration."""

    batch_size: int = 64
    data_fraction: float = 0.3
    n_epochs: int = 3
    learning_rate: float = 1e-4
    rank: int = 4
    r: float = 0.99
    n_bits: int = 32
    seed: int = 42


class MnistResNetDataModule:
    """Loads MNIST with transforms suitable for a pretrained ResNet-18."""

    def __init__(self, batch_size: int = 64, data_fraction: float = 1.0) -> None:
        """Create a data module.

        Args:
            batch_size: Mini-batch size for loaders.
            data_fraction: Fraction of train/test data to keep.
        """
        self.batch_size = batch_size
        self.data_fraction = data_fraction

    def _build_transform(self):
        """Return torchvision transforms for ResNet-style inputs."""
        return transforms.Compose([
            transforms.Resize(224),
            transforms.Grayscale(num_output_channels=3),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225],
            ),
        ])

    def dataloaders(self) -> Tuple[DataLoader, DataLoader]:
        """Build train and test dataloaders.

        Returns:
            Tuple of (train_loader, test_loader).
        """
        transform = self._build_transform()
        full_train = datasets.MNIST(
            root="./data", train=True, download=True, transform=transform,
        )
        full_test = datasets.MNIST(
            root="./data", train=False, download=True, transform=transform,
        )

        if self.data_fraction < 1.0:
            train_size = int(len(full_train) * self.data_fraction)
            test_size = int(len(full_test) * self.data_fraction)
            train_indices = torch.randperm(len(full_train))[:train_size].tolist()
            test_indices = torch.randperm(len(full_test))[:test_size].tolist()
            train_dataset = Subset(full_train, train_indices)
            test_dataset = Subset(full_test, test_indices)
        else:
            train_dataset = full_train
            test_dataset = full_test

        train_loader = DataLoader(
            train_dataset, batch_size=self.batch_size, shuffle=True,
        )
        test_loader = DataLoader(
            test_dataset, batch_size=self.batch_size, shuffle=False,
        )
        return train_loader, test_loader


class ResNet18ClassifierFactory:
    """Builds a HuggingFace ResNet-18 image classifier."""

    def create(self, num_classes: int = 10) -> nn.Module:
        """Instantiate ResNet-18 with a custom classification head.

        Args:
            num_classes: Number of output labels.

        Returns:
            Image classification model.
        """
        config = AutoConfig.from_pretrained("microsoft/resnet-18")
        config.num_labels = num_classes
        return AutoModelForImageClassification.from_pretrained(
            "microsoft/resnet-18",
            config=config,
            ignore_mismatched_sizes=True,
        )


class MetricsEvaluator:
    """Computes classification metrics on a dataloader."""

    def evaluate(self, model: nn.Module, dataloader: DataLoader, device: torch.device):
        """Compute accuracy and weighted F1.

        Args:
            model: Model to evaluate.
            dataloader: Evaluation dataloader.
            device: Execution device.

        Returns:
            Tuple of (accuracy, f1_score).
        """
        model.eval()
        predictions = []
        labels = []
        with torch.no_grad():
            for images, targets in dataloader:
                images = images.to(device)
                outputs = model(images)
                preds = outputs.logits.argmax(dim=1)
                predictions.extend(preds.cpu().numpy())
                labels.extend(targets.numpy())
        accuracy = accuracy_score(labels, predictions)
        f1 = f1_score(labels, predictions, average="weighted")
        return accuracy, f1


class QATExperimentRunner:
    """Runs baseline or quantized training loops."""

    def __init__(
        self,
        settings: ExperimentSettings,
        device_resolver: DeviceResolver | None = None,
        metrics_evaluator: MetricsEvaluator | None = None,
    ) -> None:
        """Create an experiment runner.

        Args:
            settings: Experiment hyperparameters.
            device_resolver: Device selection helper.
            metrics_evaluator: Metric computation helper.
        """
        self.settings = settings
        self.device_resolver = device_resolver or DeviceResolver()
        self.metrics_evaluator = metrics_evaluator or MetricsEvaluator()

    def run(self, use_quantization: bool = True) -> Tuple[float, float, int]:
        """Train and evaluate a ResNet-18 model.

        Args:
            use_quantization: Whether to wrap the model with SoftStairs QAT.

        Returns:
            Tuple of (accuracy, f1, soft_stairs_calls).
        """
        device = self.device_resolver.resolve()
        logger.info("Data fraction: {}%", self.settings.data_fraction * 100)

        data_module = MnistResNetDataModule(
            batch_size=self.settings.batch_size,
            data_fraction=self.settings.data_fraction,
        )
        train_loader, test_loader = data_module.dataloaders()

        model_factory = ResNet18ClassifierFactory()
        model = model_factory.create(num_classes=10).to(device)

        if use_quantization:
            config = QuantizationConfig(
                rank=self.settings.rank,
                r=self.settings.r,
                n_bits=self.settings.n_bits,
                target_modules=(nn.Linear,),
            )
            wrapped_model = wrap_model_for_quantization(model, config=config)
            wrapped_model.to(device)
            optimizer = optim.Adam(wrapped_model.parameters(), lr=self.settings.learning_rate)
            logger.info("Using quantization wrapper (Linear layers only)")
        else:
            wrapped_model = model
            optimizer = optim.Adam(model.parameters(), lr=self.settings.learning_rate)
            logger.info("No quantization (baseline)")

        criterion = nn.CrossEntropyLoss()

        logger.info("Starting training...")
        for epoch in range(self.settings.n_epochs):
            wrapped_model.train()
            epoch_loss = 0.0
            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = wrapped_model(images)
                loss = criterion(outputs.logits, labels)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()
            avg_loss = epoch_loss / len(train_loader)
            logger.info("Epoch {}/{}, Loss: {:.4f}", epoch + 1, self.settings.n_epochs, avg_loss)

        accuracy, f1 = self.metrics_evaluator.evaluate(
            wrapped_model, test_loader, device,
        )
        logger.info("Test Accuracy: {:.4f}, F1: {:.4f}", accuracy, f1)


        return accuracy, f1


def main() -> None:
    ReproducibilityManager().set_seed(42)

    settings = ExperimentSettings()
    runner = QATExperimentRunner(settings)

    logger.info("=" * 50)
    logger.info("Experiment: ResNet-18 + MNIST")
    logger.info("=" * 50)

    logger.info("--- Baseline (no quantization) ---")
    acc_baseline, f1_baseline, _ = runner.run(use_quantization=False)

    logger.info("--- With quantization (SoftStairs) ---")
    acc_quant, f1_quant, _ = runner.run(use_quantization=True)

    logger.info("=" * 50)
    logger.info("RESULTS COMPARISON:")
    logger.info("Baseline: Accuracy={:.4f}, F1={:.4f}", acc_baseline, f1_baseline)
    logger.info("Quantized: Accuracy={:.4f}, F1={:.4f}", acc_quant, f1_quant)


if __name__ == "__main__":
    from softstairs_qat.utils import configure_logging

    paths = configure_logging(name="resnet_mnist_qat")
    logger.info("Log file: {}", paths.log_file)
    main()
