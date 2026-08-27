# experiments/ebm/ebm_model.py
from __future__ import annotations

import csv
import os
import random
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Set, Tuple

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
from torch.utils.data import DataLoader
from torchvision import transforms
from torchvision.datasets import MNIST

import lightning as pl
from lightning.pytorch.callbacks import Callback

ROOT = Path(__file__).resolve().parents[2]
DATASET_PATH = os.environ.get("PATH_DATASETS", str(ROOT / "data"))
OUTPUT_ROOT = ROOT / "experiments" / "ebm" / "outputs"


class CNNModel(nn.Module):
    def __init__(self, hidden_features: int = 32, out_dim: int = 1, **kwargs):
        super().__init__()
        c_hid1 = hidden_features // 2
        c_hid2 = hidden_features
        c_hid3 = hidden_features * 2
        self.cnn_layers = nn.Sequential(
            nn.Conv2d(1, c_hid1, kernel_size=5, stride=2, padding=4),
            nn.SiLU(),
            nn.Conv2d(c_hid1, c_hid2, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c_hid2, c_hid3, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(c_hid3, c_hid3, kernel_size=3, stride=2, padding=1),
            nn.SiLU(),
            nn.Flatten(),
            nn.Linear(c_hid3 * 4, c_hid3),
            nn.SiLU(),
            nn.Linear(c_hid3, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn_layers(x).squeeze(dim=-1)


class Sampler:
    def __init__(self, model: nn.Module, img_shape, sample_size: int, max_len: int = 8192):
        self.model = model
        self.img_shape = img_shape
        self.sample_size = sample_size
        self.max_len = max_len
        self.examples = [(torch.rand((1,) + img_shape) * 2 - 1) for _ in range(sample_size)]

    def sample_new_exmps(self, steps: int = 60, step_size: float = 10.0) -> torch.Tensor:
        device = next(self.model.parameters()).device
        n_new = np.random.binomial(self.sample_size, 0.05)
        rand_imgs = torch.rand((n_new,) + self.img_shape) * 2 - 1
        old_imgs = torch.cat(random.choices(self.examples, k=self.sample_size - n_new), dim=0)
        inp_imgs = torch.cat([rand_imgs, old_imgs], dim=0).detach().to(device)
        inp_imgs = self.generate_samples(self.model, inp_imgs, steps=steps, step_size=step_size)
        self.examples = list(inp_imgs.cpu().chunk(self.sample_size, dim=0)) + self.examples
        self.examples = self.examples[: self.max_len]
        return inp_imgs

    @staticmethod
    def generate_samples(
        model: nn.Module,
        inp_imgs: torch.Tensor,
        steps: int = 60,
        step_size: float = 10.0,
    ) -> torch.Tensor:
        is_training = model.training
        model.eval()
        for p in model.parameters():
            p.requires_grad = False

        inp_imgs = inp_imgs.detach().requires_grad_(True)
        had_grad = torch.is_grad_enabled()
        torch.set_grad_enabled(True)
        noise = torch.randn_like(inp_imgs)

        for _ in range(steps):
            noise.normal_(0, 0.005)
            inp_imgs.data.add_(noise.data).clamp_(-1.0, 1.0)
            out_imgs = -model(inp_imgs)
            out_imgs.sum().backward()
            inp_imgs.grad.data.clamp_(-0.03, 0.03)
            inp_imgs.data.add_(-step_size * inp_imgs.grad.data)
            inp_imgs.grad.detach_()
            inp_imgs.grad.zero_()
            inp_imgs.data.clamp_(-1.0, 1.0)

        for p in model.parameters():
            p.requires_grad = True
        model.train(is_training)
        torch.set_grad_enabled(had_grad)
        return inp_imgs


def get_mnist_dataloaders(batch_size: int = 128, num_workers: int = 0) -> Tuple[DataLoader, DataLoader]:
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.5,), (0.5,)),
    ])
    train_set = MNIST(root=DATASET_PATH, train=True, transform=transform, download=True)
    val_set = MNIST(root=DATASET_PATH, train=False, transform=transform, download=True)
    train_loader = data.DataLoader(
        train_set, batch_size=batch_size, shuffle=True, drop_last=True,
        num_workers=num_workers, pin_memory=torch.cuda.is_available(),
    )
    val_loader = data.DataLoader(
        val_set, batch_size=256, shuffle=False, num_workers=num_workers,
    )
    return train_loader, val_loader


def grad_to_2d(grad: torch.Tensor) -> np.ndarray:
    arr = grad.abs().cpu().numpy()
    if arr.ndim == 4:
        return arr.mean(axis=(0, 1))
    if arr.ndim == 2:
        return arr
    return arr.reshape(arr.shape[0], -1)


def save_gradient_heatmaps(grads: Dict[str, torch.Tensor], step: int, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    np.savez(out_dir / f"grads_step{step}.npz", **{k: v.cpu().numpy() for k, v in grads.items()})
    for name, grad in grads.items():
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(grad_to_2d(grad), aspect="auto", cmap="viridis")
        ax.set_title(f"{name} | step={step}")
        plt.colorbar(im)
        safe = name.replace(".", "_")
        fig.savefig(out_dir / f"heatmap_{safe}_step{step}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


def compute_snapshot_steps(n_epochs: int, steps_per_epoch: int) -> Set[int]:
    total = n_epochs * steps_per_epoch
    return {0, total // 2, max(total - 1, 0)}

def classification_metrics(tp: int, fp: int, fn: int, tn: int) -> dict:
    precision = tp / (tp + fp + 1e-8)
    recall = tp / (tp + fn + 1e-8)
    f1 = 2 * precision * recall / (precision + recall + 1e-8)
    accuracy = (tp + tn) / (tp + fp + fn + tn + 1e-8)
    return {
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "accuracy": float(accuracy),
    }


@dataclass
class TrainHistory:
    epoch: int
    train_loss: float
    val_cdiv: float
    avg_real: float
    avg_fake: float
    val_precision: float
    val_recall: float
    val_f1: float
    val_accuracy: float
    t: Optional[float] = None

METRICS_CSV_COLUMNS = [
    "epoch",
    "train_loss",
    "val_cdiv",
    "avg_real",
    "avg_fake",
    "val_precision",
    "val_recall",
    "val_f1",
    "val_accuracy",
    "t",
]

def compute_ebm_loss(
    cnn: nn.Module,
    sampler: Sampler,
    real_imgs: torch.Tensor,
    alpha: float = 0.1,
) -> Tuple[torch.Tensor, dict]:
    small_noise = torch.randn_like(real_imgs) * 0.005
    real_imgs = real_imgs.add(small_noise).clamp(-1.0, 1.0)
    fake_imgs = sampler.sample_new_exmps(steps=60, step_size=10.0)
    inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
    real_out, fake_out = cnn(inp_imgs).chunk(2, dim=0)
    reg_loss = alpha * (real_out ** 2 + fake_out ** 2).mean()
    cdiv_loss = fake_out.mean() - real_out.mean()
    loss = reg_loss + cdiv_loss
    return loss, {
        "loss": loss.item(),
        "reg_loss": reg_loss.item(),
        "cdiv_loss": cdiv_loss.item(),
        "avg_real": real_out.mean().item(),
        "avg_fake": fake_out.mean().item(),
    }

def append_metrics_row(out_dir: Path, row: dict, reset: bool = False) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "metrics.csv"
    if reset and csv_path.exists():
        csv_path.unlink()
    write_header = not csv_path.exists()
    row_dict = {k: row.get(k) for k in METRICS_CSV_COLUMNS}
    with csv_path.open("a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
        if write_header:
            writer.writeheader()
        writer.writerow(row_dict)

def _metric_float(value) -> Optional[float]:
    if value is None:
        return None
    return float(value.item() if hasattr(value, "item") else value)


class MetricsCsvCallback(Callback):
    def __init__(self, out_dir: Path, reset: bool = True, max_epochs: Optional[int] = None):
        self.out_dir = Path(out_dir)
        self.reset = reset
        self.max_epochs = max_epochs
        self._initialized = False
    def on_train_epoch_end(self, trainer: pl.Trainer, pl_module: "DeepEnergyModel") -> None:
        m = trainer.callback_metrics
        row = {
            "epoch": trainer.current_epoch,
            "train_loss": _metric_float(m.get("train_loss")),
            "val_cdiv": _metric_float(m.get("val_cdiv")),
            "avg_real": _metric_float(m.get("avg_real")),
            "avg_fake": _metric_float(m.get("avg_fake")),
            "val_precision": _metric_float(m.get("val_precision")),
            "val_recall": _metric_float(m.get("val_recall")),
            "val_f1": _metric_float(m.get("val_f1")),
            "val_accuracy": _metric_float(m.get("val_accuracy")),
            "t": _metric_float(m.get("t")) if pl_module.quantizer else None,
        }
        append_metrics_row(
            self.out_dir, row, reset=(self.reset and not self._initialized)
        )
        if self.max_epochs is not None and row["train_loss"] is not None:
            t_str = ""
            if pl_module.quantizer is not None:
                t_str = f", t={pl_module.quantizer.get_current_t():.4f}"
            epoch = trainer.current_epoch + 1
            print(
                f"[{pl_module.hparams.run_name}] epoch {epoch}/{self.max_epochs} "
                f"loss={row['train_loss']:.4f} "
                f"val_cdiv={row['val_cdiv']:.4f} "
                f"P={row['val_precision']:.3f} "
                f"R={row['val_recall']:.3f}{t_str}"
            )
        self._initialized = True

        
class DeepEnergyModel(pl.LightningModule):
    """Lightning EBM (Tutorial 7). QAT: передайте qconfig=."""
    def __init__(
        self,
        img_shape: Tuple[int, ...] = (1, 28, 28),
        batch_size: int = 128,
        lr: float = 1e-4,
        weight_decay: float = 0.0,
        alpha: float = 0.1,
        beta1: float = 0.0,
        run_name: str = "run",
        snapshot_steps: Optional[Set[int]] = None,
        quantizer=None,
        qconfig=None,
        detection_threshold: float = 0.0,
        log_t_every_n_steps: int = 50,
    ):
        super().__init__()
        self.save_hyperparameters(ignore=["quantizer", "qconfig"])
        self.cnn = CNNModel()
        self.sampler: Optional[Sampler] = None
        self.quantizer = quantizer
        if qconfig is not None:
            from softstairs_qat import SoftStairsQuantizer
            self.quantizer = SoftStairsQuantizer(
                self.cnn, qconfig, excluded_modules=set()
            )
        self.snapshot_steps = snapshot_steps or set()
        self.detection_threshold = detection_threshold
        self.log_t_every_n_steps = log_t_every_n_steps
        self.out_dir = OUTPUT_ROOT / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.grad_dir = self.out_dir / "gradients"
        self._t_schedule_path = self.out_dir / "t_schedule.csv"
        self._t_schedule_initialized = False
        self._val_cdivs: list[float] = []
        self._val_reals: list[float] = []
        self._val_fakes: list[float] = []
        self._val_tp = self._val_fp = self._val_fn = self._val_tn = 0

    def setup(self, stage: Optional[str] = None) -> None:
        self.sampler = Sampler(
            self.cnn,
            img_shape=self.hparams.img_shape,
            sample_size=self.hparams.batch_size,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.cnn(x)
    
    def training_step(self, batch, batch_idx):
        real_imgs, _ = batch
        loss, metrics = compute_ebm_loss(
            self.cnn, self.sampler, real_imgs, alpha=self.hparams.alpha
        )
        self.log("train_loss", metrics["loss"], on_step=False, on_epoch=True, prog_bar=True)
        self.log("train_cdiv", metrics["cdiv_loss"], on_epoch=True)
        return loss
    
    def validation_step(self, batch, batch_idx):
        real_imgs, _ = batch
        fake_imgs = torch.rand_like(real_imgs) * 2 - 1
        real_out, fake_out = self(torch.cat([real_imgs, fake_imgs])).chunk(2)
        self._val_cdivs.append((fake_out.mean() - real_out.mean()).item())
        self._val_reals.append(real_out.mean().item())
        self._val_fakes.append(fake_out.mean().item())
        th = self.detection_threshold
        self._val_tp += int((real_out > th).sum())
        self._val_fn += int((real_out <= th).sum())
        self._val_fp += int((fake_out > th).sum())
        self._val_tn += int((fake_out <= th).sum())

    def on_validation_epoch_end(self) -> None:
        if not self._val_cdivs:
            return
        det = classification_metrics(self._val_tp, self._val_fp, self._val_fn, self._val_tn)
        self.log("val_cdiv", float(np.mean(self._val_cdivs)), prog_bar=True)
        self.log("avg_real", float(np.mean(self._val_reals)))
        self.log("avg_fake", float(np.mean(self._val_fakes)))
        self.log("val_precision", det["precision"])
        self.log("val_recall", det["recall"])
        self.log("val_f1", det["f1"])
        self.log("val_accuracy", det["accuracy"])
        if self.quantizer is not None:
            self.log("t", float(self.quantizer.get_current_t()))
        self._val_cdivs.clear()
        self._val_reals.clear()
        self._val_fakes.clear()
        self._val_tp = self._val_fp = self._val_fn = self._val_tn = 0

    def on_train_epoch_end(self) -> None:
        torch.save(self.cnn.state_dict(), self.out_dir / "last.pt")

    def configure_optimizers(self):
        opt = torch.optim.Adam(
            self.cnn.parameters(),
            lr=self.hparams.lr,
            betas=(self.hparams.beta1, 0.999),
            weight_decay=self.hparams.weight_decay,
        )
        sched = torch.optim.lr_scheduler.StepLR(opt, step_size=1, gamma=0.97)
        return {"optimizer": opt, "lr_scheduler": {"scheduler": sched, "interval": "epoch"}}
    
    def optimizer_step(self, epoch, batch_idx, optimizer, optimizer_closure):
        optimizer_closure()
        torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=0.1)
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if self.quantizer is not None:
            self.quantizer.step()
            if self.global_step % self.log_t_every_n_steps == 0:
                self._log_t_schedule()
        if self.global_step in self.snapshot_steps:
            grads = {
                n: p.grad.detach().clone()
                for n, p in self.cnn.named_parameters()
                if p.grad is not None
            }
            save_gradient_heatmaps(grads, self.global_step, self.grad_dir)

    def _log_t_schedule(self) -> None:
        write_header = not self._t_schedule_initialized
        with self._t_schedule_path.open("a", newline="") as f:
            w = csv.DictWriter(f, fieldnames=["step", "t", "epoch"])
            if write_header:
                w.writeheader()
                self._t_schedule_initialized = True
            w.writerow({
                "step": self.global_step,
                "t": self.quantizer.get_current_t(),
                "epoch": self.current_epoch,
            })


def fit_ebm_model(
    model: DeepEnergyModel,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
) -> pl.Trainer:
    trainer = pl.Trainer(
        max_epochs=max_epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=1,
        callbacks=[MetricsCsvCallback(model.out_dir, reset=True)],
        logger=False,
        enable_checkpointing=False,
        enable_model_summary=False,
    )
    trainer.fit(model, train_loader, val_loader)
    return trainer


# class EBMTrainer:
#     def __init__(
#         self,
#         img_shape: Tuple[int, ...],
#         batch_size: int,
#         lr: float = 1e-4,
#         weight_decay: float = 0.0,
#         alpha: float = 0.1,
#         beta1: float = 0.0,
#         run_name: str = "run",
#         snapshot_steps: Optional[Set[int]] = None,
#         device: Optional[torch.device] = None,
#         detection_threshold: float = 0.0,
#     ):
#         self.img_shape = img_shape
#         self.batch_size = batch_size
#         self.lr = lr
#         self.weight_decay = weight_decay
#         self.alpha = alpha
#         self.beta1 = beta1
#         self.run_name = run_name
#         self.snapshot_steps = snapshot_steps or set()
#         self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

#         self.cnn: Optional[CNNModel] = None
#         self.sampler: Optional[Sampler] = None
#         self.optimizer: Optional[optim.Optimizer] = None
#         self.scheduler: Optional[optim.lr_scheduler.StepLR] = None
#         self.quantizer = None
#         self.global_step = 0
#         self.history: list[TrainHistory] = []

#         self.out_dir = OUTPUT_ROOT / run_name
#         self.out_dir.mkdir(parents=True, exist_ok=True)
#         self.grad_dir = self.out_dir / "gradients"
#         self.detection_threshold = detection_threshold

#     def build_model(self) -> CNNModel:
#         self.cnn = CNNModel().to(self.device)
#         self.sampler = Sampler(self.cnn, img_shape=self.img_shape, sample_size=self.batch_size)
#         return self.cnn

#     def configure_optimizers(self) -> optim.Optimizer:
#         assert self.cnn is not None
#         optimizer = optim.Adam(
#             self.cnn.parameters(),
#             lr=self.lr,
#             betas=(self.beta1, 0.999),
#             weight_decay=self.weight_decay,
#         )
#         self.scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.97)
#         return optimizer

#     def compute_loss(self, real_imgs: torch.Tensor) -> Tuple[torch.Tensor, dict]:
#         return compute_ebm_loss(self.cnn, self.sampler, real_imgs, self.alpha)

#     @torch.no_grad()
#     def validate_detection(
#         self,
#         val_loader: DataLoader,
#         fake_mode: str = "uniform",
#         max_batches: Optional[int] = None,
#     ) -> dict:
#         """
#         Binary detection: real MNIST vs fake.
#         fake_mode: "uniform" (быстро) or "mcmc" (медленно, через sampler).
#         """
#         assert self.cnn is not None
#         if fake_mode == "mcmc":
#             assert self.sampler is not None

#         self.cnn.eval()
#         tp = fp = fn = tn = 0
#         threshold = self.detection_threshold

#         for batch_idx, (real_imgs, _) in enumerate(val_loader):
#             if max_batches is not None and batch_idx >= max_batches:
#                 break

#             real_imgs = real_imgs.to(self.device)

#             if fake_mode == "uniform":
#                 fake_imgs = torch.rand_like(real_imgs) * 2 - 1
#             elif fake_mode == "mcmc":
#                 fake_imgs = self.sampler.sample_new_exmps(steps=60, step_size=10.0)
#             else:
#                 raise ValueError(f"Unknown fake_mode: {fake_mode}")

#             scores = self.cnn(torch.cat([real_imgs, fake_imgs], dim=0))
#             real_scores, fake_scores = scores.chunk(2, dim=0)

#             tp += (real_scores > threshold).sum().item()
#             fn += (real_scores <= threshold).sum().item()
#             fp += (fake_scores > threshold).sum().item()
#             tn += (fake_scores <= threshold).sum().item()

#         self.cnn.train()
#         metrics = classification_metrics(tp, fp, fn, tn)

#         prefix = "val" if fake_mode == "uniform" else "val_mcmc"
#         return {
#             f"{prefix}_precision": metrics["precision"],
#             f"{prefix}_recall": metrics["recall"],
#             f"{prefix}_f1": metrics["f1"],
#             f"{prefix}_accuracy": metrics["accuracy"],
#         }

#     @torch.no_grad()
#     def validate(self, val_loader: DataLoader) -> dict:
#         assert self.cnn is not None
#         self.cnn.eval()
#         cdivs, reals, fakes = [], [], []

#         for real_imgs, _ in val_loader:
#             real_imgs = real_imgs.to(self.device)
#             fake_imgs = torch.rand_like(real_imgs) * 2 - 1
#             inp_imgs = torch.cat([real_imgs, fake_imgs], dim=0)
#             real_out, fake_out = self.cnn(inp_imgs).chunk(2, dim=0)
#             cdivs.append((fake_out.mean() - real_out.mean()).item())
#             reals.append(real_out.mean().item())
#             fakes.append(fake_out.mean().item())

#         self.cnn.train()

#         metrics = {
#             "val_cdiv": float(np.mean(cdivs)),
#             "val_real": float(np.mean(reals)),
#             "val_fake": float(np.mean(fakes)),
#         }
#         metrics.update(self.validate_detection(val_loader, fake_mode="uniform"))
#         return metrics

#     def optimizer_step(self) -> None:
#         assert self.cnn is not None and self.optimizer is not None

#         torch.nn.utils.clip_grad_norm_(self.cnn.parameters(), max_norm=0.1)
#         self.optimizer.step()
#         self.optimizer.zero_grad(set_to_none=True)

#         if self.quantizer is not None:
#             self.quantizer.step()

#         self._maybe_snapshot_gradients()

#     def _maybe_snapshot_gradients(self) -> None:
#         if self.global_step not in self.snapshot_steps or self.cnn is None:
#             return

#         grads = {}
#         for name, param in self.cnn.named_parameters():
#             if param.grad is not None:
#                 grads[name] = param.grad.detach().clone()

#         save_gradient_heatmaps(grads, self.global_step, self.grad_dir)

#     def train_one_epoch(self, train_loader: DataLoader, epoch: int) -> dict:
#         assert self.cnn is not None and self.optimizer is not None

#         self.cnn.train()
#         losses = []

#         for batch in train_loader:
#             real_imgs, _ = batch
#             real_imgs = real_imgs.to(self.device)

#             self.optimizer.zero_grad(set_to_none=True)
#             loss, metrics = self.compute_loss(real_imgs)
#             loss.backward()
#             self.optimizer_step()

#             losses.append(metrics["loss"])
#             self.global_step += 1

#         if self.scheduler is not None:
#             self.scheduler.step()

#         return {"train_loss": float(np.mean(losses))}

#     def save_history_row(self, epoch: int, train_metrics: dict, val_metrics: dict) -> None:
#         t = self.quantizer.get_current_t() if self.quantizer is not None else None
#         row = TrainHistory(
#             epoch=epoch,
#             train_loss=train_metrics["train_loss"],
#             val_cdiv=val_metrics["val_cdiv"],
#             avg_real=val_metrics["val_real"],
#             avg_fake=val_metrics["val_fake"],
#             val_precision=val_metrics["val_precision"],
#             val_recall=val_metrics["val_recall"],
#             val_f1=val_metrics["val_f1"],
#             val_accuracy=val_metrics["val_accuracy"],
#             t=t,
#         )
#         self.history.append(row)

#         csv_path = self.out_dir / "metrics.csv"
#         write_header = not csv_path.exists()
#         row_dict = {k: row.__dict__[k] for k in METRICS_CSV_COLUMNS}
#         with csv_path.open("a", newline="") as f:
#             writer = csv.DictWriter(f, fieldnames=METRICS_CSV_COLUMNS)
#             if write_header:
#                 writer.writeheader()
#             writer.writerow(row_dict)

#     def train(
#         self,
#         train_loader: DataLoader,
#         val_loader: DataLoader,
#         n_epochs: int,
#         eval_mcmc_detection_every: int = 5,
#         mcmc_detection_batches: int = 4,
#     ) -> None:
#         self.build_model()
#         self.optimizer = self.configure_optimizers()
#         csv_path = self.out_dir / "metrics.csv"
#         if csv_path.exists():
#             csv_path.unlink()
#         self.history = []

#         for epoch in range(n_epochs):
#             train_metrics = self.train_one_epoch(train_loader, epoch)
#             val_metrics = self.validate(val_loader)

#             if epoch % eval_mcmc_detection_every == 0 or epoch == n_epochs - 1:
#                 val_metrics.update(
#                     self.validate_detection(
#                         val_loader,
#                         fake_mode="mcmc",
#                         max_batches=mcmc_detection_batches,
#                     )
#                 )
#             else:
#                 val_metrics.update({
#                     "val_mcmc_precision": float("nan"),
#                     "val_mcmc_recall": float("nan"),
#                     "val_mcmc_f1": float("nan"),
#                     "val_mcmc_accuracy": float("nan"),
#                 })

#             self.save_history_row(epoch, train_metrics, val_metrics)

#             t_str = f", t={self.quantizer.get_current_t():.4f}" if self.quantizer else ""
#             print(
#                 f"[{self.run_name}] epoch {epoch + 1}/{n_epochs} "
#                 f"loss={train_metrics['train_loss']:.4f} "
#                 f"val_cdiv={val_metrics['val_cdiv']:.4f} "
#                 f"P={val_metrics['val_precision']:.3f} "
#                 f"R={val_metrics['val_recall']:.3f}{t_str}"
#             )

#         torch.save(self.cnn.state_dict(), self.out_dir / "last.pt")