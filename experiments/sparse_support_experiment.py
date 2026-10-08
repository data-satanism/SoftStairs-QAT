"""Sparse-support optimization experiment for SoftStairs-QAT.

The experiment tests whether sparse optimization supports derived from the
on-the-fly weight-type tracker (:mod:`softstairs_qat.analysis.weight_types`)
can replace dense QAT updates without sacrificing fake-quantized accuracy.

Protocol
--------
1. **Discovery phase** -- dense QAT for ``discovery_fraction`` of the total
   epochs while the :class:`WeightTypeTracker` observes the unmasked QAT
   gradients.  At the end the supports are frozen:
   ``core = time_active >= core_threshold`` and
   ``churning = churn_min_time_active <= time_active < core_threshold``.
   The model checkpoint, tracker state and masks are saved.
2. **Forked optimization** -- every mode reloads the *same* discovery
   checkpoint (identical weights, optimizer init, quantizer scales and t
   schedule, seeds and data ordering) and trains the remaining epochs.  The
   only difference between modes is which optimizer gradients survive:

   ================  =========================================================
   mode              support receiving optimizer updates
   ================  =========================================================
   dense             all quantized weights (forked dense baseline)
   dynamic_active    tracker active set recomputed every
                     ``support_update_interval`` steps
   persistent_core   discovery-time core (fixed)
   core_churning     core | churning (fixed); with ``churn_stride > 1`` the
                     churning-only weights receive updates only every
                     ``churn_stride``-th step
   random_core       fixed random support with |R| = |core|
   random_active     fixed random support with |R| = median dynamic-active
                     support size (falls back to the discovery-phase median)
   ================  =========================================================

   The tracker always sees the *unmasked* gradients, so the dynamic support
   keeps reflecting the underlying QAT pressure; masking happens after
   ``backward()`` and before ``optimizer.step()`` and is the only
   optimization modification.  Biases and excluded modules are never masked:
   sparse modes constrain the quantized-weight population only.

3. **Evaluation** -- fake-quantized accuracy is measured on the test split for
   every mode and summarized against the dense baseline.

Run from the repository root::

    python experiments/sparse_support_experiment.py --epochs 30 \
        --modes dense dynamic_active persistent_core core_churning random_core random_active

Every important parameter is exposed through the CLI; ``default_args()``
builds the same namespace programmatically (used by
``experiments/CV/dynamic_training.ipynb``).

Memory: the tracker holds roughly 130 bytes per tracked weight on the compute
device plus per-step temporaries (the quantile normalization sorts the flat
score vector).  On memory-limited GPUs combine a smaller ``--batch-size`` with
``--track-exclude`` (a regex; excluded quantized weights are neither tracked
nor masked and train densely in every mode).
"""

from __future__ import annotations

import argparse
import gc
import json
import re
import sys
import time
from collections import defaultdict
from pathlib import Path

import pandas as pd
import pytorch_lightning as pl
import torch
import torch.nn as nn
import torchvision.models as models
from pytorch_lightning.loggers import CSVLogger
from torchmetrics import Accuracy, MeanMetric, MetricCollection
from torchmetrics.classification import F1Score
from torchvision import transforms
from torchvision.datasets import STL10

_REPO_ROOT = str(Path(__file__).resolve().parents[1])
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)

from softstairs_qat import QuantizationConfig, SoftStairsQuantizer
from softstairs_qat.analysis import ModelWeightTypeTracker, WeightTypeConfig
from softstairs_qat.analysis.tracking import jaccard

MODES = ("dense", "dynamic_active", "persistent_core", "core_churning", "random_core", "random_active")

MODE_DESCRIPTIONS = {
    "dense": "forked dense QAT baseline",
    "dynamic_active": "tracker active set, recomputed periodically",
    "persistent_core": "discovery-time persistent core (time_active >= core-threshold)",
    "core_churning": "core | churning union (time_active >= churning-min)",
    "random_core": "random support with |R| = |core|",
    "random_active": "random support with |R| = median dynamic-active support size",
}

SUMMARY_COLUMNS = (
    "mode",
    "test_acc",
    "test_acc_quant",
    "best_val_acc_quant",
    "final_val_acc_quant",
    "support_fraction_mean",
    "support_size_final",
    "update_budget",
    "update_budget_fraction",
    "effective_updates",
    "jaccard_active_core_mean",
    "core_coverage_mean",
    "churning_coverage_mean",
    "support_change_rate_mean",
    "acc_drop_vs_dense",
    "relative_acc_drop_vs_dense",
)


# ----------------------------------------------------------------------
# quantizer / tracker helpers
# ----------------------------------------------------------------------
def quantized_weight_mapping(model: nn.Module, exclude_regex: str | None = None) -> dict[str, torch.Tensor]:
    """Collect the trainable quantized-weight leaves (``*_orig`` parameters).

    Args:
        model: Quantized model produced by ``SoftStairsQuantizer``.
        exclude_regex: Optional regex; matching parameter names are skipped
            (use it to shrink the tracker on memory-limited GPUs).

    Returns:
        Ordered mapping from parameter name to tensor.
    """
    pattern = re.compile(exclude_regex) if exclude_regex else None
    mapping = {
        name: parameter
        for name, parameter in model.named_parameters()
        if parameter.requires_grad
        and name.endswith(SoftStairsQuantizer._orig_suffix)
        and not (pattern is not None and pattern.search(name))
    }
    return mapping


def fast_forward_quantizer(quantizer: SoftStairsQuantizer | None, steps: int) -> None:
    """Advance the t schedule by ``steps`` epochs without touching weights.

    The scheduler is deterministic in ``current_step`` for every non-adaptive
    strategy, so a forked run reproduces the discovery run's temperature.

    Args:
        quantizer: Live quantizer, or ``None``.
        steps: Number of ``quantizer.step()`` calls to replay.
    """
    if quantizer is None:
        return
    for _ in range(int(steps)):
        quantizer.step()


def build_tracker_config(args: argparse.Namespace) -> WeightTypeConfig:
    """Build the tracker configuration from CLI arguments.

    Args:
        args: Parsed CLI namespace.

    Returns:
        Tracker configuration; discovery bookkeeping is handled by the
        experiment phases, so the tracker-internal discovery is disabled.
    """
    return WeightTypeConfig(
        active_threshold=args.active_threshold,
        reservoir_threshold=args.reservoir_threshold,
        normalization_quantile=args.normalization_quantile,
        decay_activity=args.decay_activity,
        decay_sensitivity=args.decay_sensitivity,
        decay_velocity=args.decay_velocity,
        persistent_active_threshold=args.core_threshold,
        churn_min_time_active=args.churning_min,
        churn_min_spells=args.churn_min_spells,
        discovery_fraction=None,
        total_steps=None,
    )


def _state_to_cpu(state: dict) -> dict:
    """Move every tensor of a tracker state dict to CPU for portable storage."""
    return {key: value.cpu() if torch.is_tensor(value) else value for key, value in state.items()}


def _free_gpu_memory() -> None:
    """Release finished phases' GPU memory.

    Lightning trainers/modules/callbacks form reference cycles, so a plain
    ``del`` does not free the previous phase's tracker tensors; the collector
    must run before the CUDA cache is emptied.
    """
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


# ----------------------------------------------------------------------
# Lightning callbacks
# ----------------------------------------------------------------------
class SSQStepCallback(pl.Callback):
    """Advance the SoftStairs t schedule once per epoch (resume-aware).

    Unlike the notebook ``SSQStep`` this callback keys the stepping on its own
    local epoch counter: the first epoch of a *fit call* never steps, because
    forked runs are fast-forwarded to the discovery temperature beforehand.
    """

    def __init__(self) -> None:
        self._local_epoch: int | None = None

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self._local_epoch = 0

    def on_train_epoch_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        quantizer = getattr(pl_module, "quantizer", None)
        if quantizer is None:
            return
        if self._local_epoch is None:
            self._local_epoch = 0
        if self._local_epoch > 0:
            quantizer.step()
        self._local_epoch += 1
        pl_module.log("quantizer_t", quantizer.t, on_epoch=True, on_step=False)
        pl_module.log("quantizer_tback", quantizer.current_backward_t, on_step=False)


class SparseSupportController(pl.Callback):
    """Track weight types and apply mode-specific support masking.

    On every optimizer step the callback feeds the *unmasked* QAT gradients
    into the :class:`ModelWeightTypeTracker` and then zeroes optimizer
    gradients outside the current support mask.  This is the only optimization
    modification; the tracker, metrics and masks are purely observational.

    For ``dynamic_active`` the support is the tracker's current active set,
    recomputed every ``support_update_interval`` steps (optionally padded to
    ``dynamic_min_fraction`` with the highest ``ema_activity`` weights).  All
    other modes take their support from the ``mask_fn`` supplied by the
    orchestrator.
    """

    def __init__(
        self,
        mode: str,
        *,
        tracker_config: WeightTypeConfig,
        mask_fn=None,
        tracker_state: dict | None = None,
        exclude_regex: str | None = None,
        support_update_interval: int = 1,
        dynamic_min_fraction: float = 0.0,
        log_interval: int = 50,
        prefix: str = "sp",
    ) -> None:
        if mode not in MODES and mode != "discovery":
            raise ValueError(f"unknown mode {mode!r}; expected one of {MODES + ('discovery',)}")
        self.mode = mode
        self.tracker_config = tracker_config
        self.mask_fn = mask_fn
        self.tracker_state = tracker_state
        self.exclude_regex = exclude_regex or None
        self.support_update_interval = max(1, int(support_update_interval))
        self.dynamic_min_fraction = float(dynamic_min_fraction)
        self.log_interval = max(1, int(log_interval))
        self.prefix = prefix

        self.tracker: ModelWeightTypeTracker | None = None
        self.params: dict[str, torch.Tensor] | None = None
        self.layout = None
        self.num_weights = 0
        self._step = 0
        self._frozen: dict[str, torch.Tensor] = {}
        self._support: torch.Tensor | None = None
        self.active_size_history: list[int] = []
        self._budget = 0
        self._effective = 0
        self._logged_steps = 0
        self._support_changes = 0
        self._sums: dict[str, float] = defaultdict(float)
        self.best: dict[str, float] = {}
        self.last: dict[str, float] = {}

    # ------------------------------------------------------------------
    # setup
    # ----------------------------------------------------------------------
    def set_frozen_masks(self, core: torch.Tensor, churning: torch.Tensor) -> None:
        """Attach the discovery-time core / churning masks (flat, any device)."""
        self._frozen = {"core": core, "churning": churning}

    def on_fit_start(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        self.params = quantized_weight_mapping(pl_module.model, self.exclude_regex)
        if not self.params:
            raise RuntimeError("no quantized weight leaves found; was the model wrapped by SoftStairsQuantizer?")
        self.tracker = ModelWeightTypeTracker(self.params, config=self.tracker_config)
        if self.tracker_state is not None:
            self.tracker.load_state_dict(self.tracker_state)
            sample_device = next(iter(self.params.values())).device
            if self.tracker.tracker.device != sample_device:
                self.tracker.tracker.to(sample_device)
        self.layout = self.tracker.layout
        self.num_weights = self.layout.num_parameters
        self._step = self.tracker.tracker.seen
        self._support = None

    # ------------------------------------------------------------------
    # support construction
    # ----------------------------------------------------------------------
    def _frozen_mask(self, name: str) -> torch.Tensor | None:
        """Frozen mask on the tracker device, or ``None``."""
        mask = self._frozen.get(name)
        if mask is None:
            return None
        device = self.tracker.tracker.device
        return mask if mask.device == device else mask.to(device)

    def _dynamic_support(self) -> torch.Tensor:
        """Current active support from the tracker, optionally size-floored."""
        inner = self.tracker.tracker
        active = inner.active_mask
        if self.dynamic_min_fraction > 0.0 and inner.ema_activity is not None:
            if float(active.float().mean()) < self.dynamic_min_fraction:
                target = max(int(active.sum()), int(round(self.dynamic_min_fraction * self.num_weights)))
                top = torch.topk(inner.ema_activity, target).indices
                padded = torch.zeros_like(active)
                padded[top] = True
                active = padded
        self.active_size_history.append(int(active.sum()))
        return active

    def _record_support_change(self, support: torch.Tensor) -> None:
        if self._support is not None and support is not self._support:
            self._sums["support_jaccard"] += float(jaccard(support, self._support))
            self._support_changes += 1

    # ------------------------------------------------------------------
    # optimizer-step hook: track with true grads, then mask
    # ----------------------------------------------------------------------
    @torch.no_grad()
    def on_before_optimizer_step(self, trainer: pl.Trainer, pl_module: pl.LightningModule, optimizer) -> None:
        if self.tracker is None or not self.params:
            return
        quantizer = getattr(pl_module, "quantizer", None)
        t = float(quantizer.t) if quantizer is not None else None
        grads: dict[str, torch.Tensor] = {}
        for name, parameter in self.params.items():
            grads[name] = parameter.grad if parameter.grad is not None else torch.zeros_like(parameter)
        self.tracker.update(gradients=grads, parameters=dict(self.params), t=t)
        self._step += 1
        step = self._step

        if self.mode == "dynamic_active":
            if self._support is None or step % self.support_update_interval == 0:
                support = self._dynamic_support()
                self._record_support_change(support)
                self._support = support
        elif self.mask_fn is not None:
            support = self.mask_fn(step)
            if support is not None:
                self._record_support_change(support)
                self._support = support

        self._apply_mask(grads)
        self._account(step, grads)
        if step % self.log_interval == 0:
            self._log(pl_module)

    def _apply_mask(self, grads: dict[str, torch.Tensor]) -> None:
        """Zero optimizer gradients outside the current support (in place)."""
        if self._support is None or self.mode in ("dense", "discovery"):
            return
        views = self.layout.unflatten_masks(self._support)
        for name, grad in grads.items():
            view = views.get(name)
            if view is not None:
                grad.masked_fill_(~view.to(grad.device), 0.0)

    def _account(self, step: int, grads: dict[str, torch.Tensor]) -> None:
        support = self._support
        size = self.num_weights if support is None else int(support.sum())
        self._budget += size
        if support is None:
            self._effective += sum(int(torch.count_nonzero(grad)) for grad in grads.values())
        else:
            views = self.layout.unflatten_masks(support)
            effective = 0
            for name, grad in grads.items():
                view = views.get(name)
                if view is not None:
                    effective += int(torch.count_nonzero(grad[view]))
            self._effective += effective
        self._sums["support_fraction"] += size / self.num_weights
        core = self._frozen_mask("core")
        if core is not None:
            active = self.tracker.tracker.active_mask
            self._sums["jaccard_active_core"] += float(jaccard(active, core))
            core_size = int(core.sum())
            if core_size:
                self._sums["core_coverage"] += float((active & core).sum()) / core_size
            churning = self._frozen_mask("churning")
            churning_size = int(churning.sum()) if churning is not None else 0
            if churning_size:
                self._sums["churning_coverage"] += float((active & churning).sum()) / churning_size
        self._logged_steps += 1

    def _log(self, pl_module: pl.LightningModule) -> None:
        steps = max(1, self._logged_steps)
        size = self.num_weights if self._support is None else int(self._support.sum())
        metrics = {
            "support_size": float(size),
            "support_fraction": size / self.num_weights,
            "update_budget": float(self._budget),
            "effective_updates": float(self._effective),
        }
        if self._sums["jaccard_active_core"]:
            metrics["jaccard_active_core"] = self._sums["jaccard_active_core"] / steps
        if self._sums["core_coverage"]:
            metrics["core_coverage"] = self._sums["core_coverage"] / steps
        if self._sums["churning_coverage"]:
            metrics["churning_coverage"] = self._sums["churning_coverage"] / steps
        if self._support_changes:
            metrics["support_change_rate"] = 1.0 - self._sums["support_jaccard"] / self._support_changes
        for key, value in metrics.items():
            pl_module.log(f"{self.prefix}/{key}", value, on_step=True, on_epoch=True)
        tracker_metrics = self.tracker.get_metrics()
        for key, value in tracker_metrics.items():
            if not isinstance(value, (int, float)) or isinstance(value, bool):
                continue
            if key in ("step", "tracking_steps") or key.endswith("_total") or key.endswith("_total_rate"):
                continue
            pl_module.log(f"wt/{key}", float(value), on_step=True, on_epoch=True)

    # ------------------------------------------------------------------
    # validation bookkeeping
    # ----------------------------------------------------------------------
    def on_validation_epoch_end(self, trainer: pl.Trainer, pl_module: pl.LightningModule) -> None:
        if getattr(trainer, "sanity_checking", False):
            return
        for key in ("mtr/acc", "mtr/acc_quant", "mtr/f1", "mtr/f1_quant", "loss/val", "loss/val_quant"):
            value = trainer.callback_metrics.get(key)
            if value is None:
                continue
            scalar = float(value)
            self.last[key] = scalar
            self.best[key] = max(self.best.get(key, float("-inf")), scalar)

    # ------------------------------------------------------------------
    # results
    # ----------------------------------------------------------------------
    def result(self) -> dict:
        """Aggregate support statistics of this callback over its fit."""
        if self.tracker is None:
            return {}
        steps = max(1, self._logged_steps)
        out = {
            "num_tracked_weights": self.num_weights,
            "tracking_steps": self._step,
            "support_fraction_mean": self._sums["support_fraction"] / steps,
            "update_budget": self._budget,
            "update_budget_fraction": self._budget / max(1, self.num_weights * max(1, self._step)),
            "effective_updates": self._effective,
        }
        if self._sums["jaccard_active_core"]:
            out["jaccard_active_core_mean"] = self._sums["jaccard_active_core"] / steps
        if self._sums["core_coverage"]:
            out["core_coverage_mean"] = self._sums["core_coverage"] / steps
        if self._sums["churning_coverage"]:
            out["churning_coverage_mean"] = self._sums["churning_coverage"] / steps
        if self._support_changes:
            out["support_change_rate_mean"] = 1.0 - self._sums["support_jaccard"] / self._support_changes
        if self.mode == "dynamic_active" and self.active_size_history:
            out["support_size_final"] = self.active_size_history[-1]
        return out


# ----------------------------------------------------------------------
# model / data / training (reused from experiments/CV/inception_stl10.ipynb)
# ----------------------------------------------------------------------
def get_inception(n_classes: int) -> nn.Module:
    """Inception v3 (ImageNet weights) with an STL-10 head, as in the notebook."""
    model = models.inception_v3(weights=models.Inception_V3_Weights.IMAGENET1K_V1)
    model.fc = nn.Linear(model.fc.in_features, 10)
    model.aux_logits = True
    if model.aux_logits:
        model.AuxLogits.fc = nn.Linear(model.AuxLogits.fc.in_features, n_classes)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    return model.to(device)


def build_transforms():
    """Notebook transforms for STL-10 resized to Inception's 299x299 input."""
    normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    test_transform = transforms.Compose([transforms.Resize((299, 299)), transforms.ToTensor(), normalize])
    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomResizedCrop((299, 299), scale=(0.8, 1.0), ratio=(0.9, 1.1)),
        transforms.ToTensor(),
        normalize,
    ])
    return train_transform, test_transform


def make_loaders(args: argparse.Namespace, seed: int):
    """Notebook data pipeline, re-seeded so every phase sees identical ordering."""
    pl.seed_everything(seed)
    train_transform, test_transform = build_transforms()
    train_dataset = STL10(root=args.data_root, split="train", transform=train_transform, download=False)
    val_dataset = STL10(root=args.data_root, split="train", transform=test_transform, download=False)
    n_train = int(args.max_train_samples) if args.max_train_samples else len(train_dataset)
    n_val = int(args.max_val_samples) if args.max_val_samples else len(val_dataset)
    train_set, _ = torch.utils.data.random_split(train_dataset, [n_train, len(train_dataset) - n_train])
    pl.seed_everything(seed)
    _, val_set = torch.utils.data.random_split(val_dataset, [len(val_dataset) - n_val, n_val])
    test_set = STL10(root=args.data_root, split="test", transform=test_transform, download=True)
    if args.max_test_samples:
        test_set = torch.utils.data.Subset(test_set, range(int(args.max_test_samples)))
    train_loader = torch.utils.data.DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        pin_memory=True,
        num_workers=args.num_workers,
    )
    val_loader = torch.utils.data.DataLoader(
        val_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False, drop_last=False, num_workers=args.num_workers
    )
    return train_loader, val_loader, test_loader


class STL10Module(pl.LightningModule):
    """STL-10 Inception module with SoftStairs QAT (notebook module, SoftStairs path)."""

    def __init__(
        self,
        model_name,
        model_hparams,
        optimizer_name,
        optimizer_hparams,
        fake_quant=False,
        quantization_hparams={},
    ):
        super().__init__()
        self.save_hyperparameters()
        n_classes = self.hparams.model_hparams["n_classes"]
        self.model = get_inception(n_classes)
        if hasattr(self.model, "aux_logits"):
            self.model.aux_logits = True
        self.fake_quant = fake_quant
        self._prepare_quantization()
        self.loss_module = nn.CrossEntropyLoss()
        self.train_metrics = MetricCollection({
            "mtr/acc": Accuracy(task="multiclass", num_classes=n_classes),
            "mtr/f1": F1Score(task="multiclass", num_classes=n_classes, average="macro"),
        })
        self.val_metrics = MetricCollection({
            "mtr/acc": Accuracy(task="multiclass", num_classes=n_classes),
            "mtr/f1": F1Score(task="multiclass", num_classes=n_classes, average="macro"),
        })
        if self.fake_quant:
            self.val_metrics_quant = MetricCollection({
                "mtr/acc_quant": Accuracy(task="multiclass", num_classes=n_classes),
                "mtr/f1_quant": F1Score(task="multiclass", num_classes=n_classes, average="macro"),
            })

    def _prepare_quantization(self):
        excluded_modules = {n for n, _ in self.model.named_modules() if "bn" in n or "aux" in n.lower()}
        qconfig = QuantizationConfig(**self.hparams.quantization_hparams)
        self.quantizer = SoftStairsQuantizer(self.model, qconfig, excluded_modules=excluded_modules)

    def forward(self, imgs):
        if self.training and hasattr(self.model, "aux_logits") and self.model.aux_logits:
            output, aux_output = self.model(imgs)
            return output, aux_output
        return self.model(imgs)

    def configure_optimizers(self):
        if self.hparams.optimizer_name == "Adam":
            optimizer = torch.optim.AdamW(self.parameters(), **self.hparams.optimizer_hparams)
        elif self.hparams.optimizer_name == "SGD":
            optimizer = torch.optim.SGD(self.parameters(), **self.hparams.optimizer_hparams)
        else:
            raise ValueError(f'Unknown optimizer: "{self.hparams.optimizer_name}"')
        return [optimizer], []

    def training_step(self, batch, batch_idx):
        imgs, labels = batch
        if hasattr(self.model, "aux_logits") and self.model.aux_logits:
            logits, aux_logits = self.model(imgs)
            loss = self.loss_module(logits, labels) + 0.4 * self.loss_module(aux_logits, labels)
        else:
            logits = self.model(imgs)
            loss = self.loss_module(logits, labels)
        preds = logits.argmax(-1)
        self.train_metrics.update(preds, labels)
        self.log_dict(self.train_metrics, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss/train", loss, on_step=True, on_epoch=True, prog_bar=False)
        return loss

    def on_train_epoch_start(self):
        if self.quantizer:
            self.log("epoch_start_quant_error", self.quantizer.estimate_current_quant_error(), on_epoch=True)

    def on_train_epoch_end(self):
        if self.quantizer:
            self.log("epoch_end_quant_error", self.quantizer.estimate_current_quant_error(), on_epoch=True)
        self.train_metrics.reset()

    def validation_step(self, batch, batch_idx):
        imgs, labels = batch
        if hasattr(self.model, "aux_logits"):
            self.model.aux_logits = False
        logits = self.model(imgs)
        loss = self.loss_module(logits, labels)
        self.val_metrics.update(logits.argmax(-1), labels)
        self.log_dict(self.val_metrics, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss/val", loss, on_epoch=True, on_step=True)
        if hasattr(self.model, "aux_logits") and self.training:
            self.model.aux_logits = True
        if not (self.fake_quant and self.quantizer):
            return loss
        state_dict = {key: value.clone() for key, value in self.model.state_dict().items()}
        self.quantizer.fake_quantize()
        logits = self.model(imgs)
        loss_quant = self.loss_module(logits, labels)
        self.val_metrics_quant.update(logits.argmax(-1), labels)
        self.log_dict(self.val_metrics_quant, on_step=True, on_epoch=True, prog_bar=True)
        self.log("loss/val_quant", loss_quant, on_epoch=True, on_step=True)
        self.model.load_state_dict(state_dict)
        del state_dict
        self.quantizer.activate_hooks()
        return loss

    def on_validation_epoch_end(self):
        self.val_metrics.reset()
        if self.fake_quant and hasattr(self, "val_metrics_quant"):
            self.val_metrics_quant.reset()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def build_module(args: argparse.Namespace, quantization_hparams: dict) -> STL10Module:
    """Fresh module with deterministic init (call after ``seed_everything``)."""
    return STL10Module(
        model_name="InceptionNet",
        model_hparams={"n_classes": 10},
        optimizer_name=args.optimizer,
        optimizer_hparams={"lr": args.lr, "weight_decay": args.weight_decay},
        fake_quant=True,
        quantization_hparams=quantization_hparams,
    )


def build_trainer(args: argparse.Namespace, *, max_epochs: int, logger, callbacks) -> pl.Trainer:
    """Single-device trainer with the experiment's logging/limit settings."""
    kwargs = dict(
        accelerator="auto",
        devices=1,
        max_epochs=max_epochs,
        logger=logger,
        callbacks=list(callbacks),
        default_root_dir=str(Path(args.results_dir) / "trainer"),
        enable_checkpointing=False,
        enable_progress_bar=bool(args.progress),
        log_every_n_steps=max(1, args.log_interval),
    )
    if args.limit_batches:
        kwargs["limit_train_batches"] = args.limit_batches
    if args.eval_batches:
        kwargs["limit_val_batches"] = args.eval_batches
    return pl.Trainer(**kwargs)


# ----------------------------------------------------------------------
# evaluation
# ----------------------------------------------------------------------
@torch.no_grad()
def evaluate(module: pl.LightningModule, loader, *, fake_quant: bool = True, max_batches: int | None = None) -> dict:
    """Evaluate a module; ``fake_quant`` additionally reports fake-quantized metrics.

    Mirrors the notebook validation path: clone the state, ``fake_quantize()``,
    evaluate, restore the state and re-activate the hooks.
    """
    quantizer = getattr(module, "quantizer", None)
    if quantizer is not None and quantizer._scales:
        module.to(next(iter(quantizer._scales.values())).device)
    device = next(module.parameters()).device
    n_classes = module.hparams.model_hparams["n_classes"]
    module.eval()
    if hasattr(module.model, "aux_logits"):
        module.model.aux_logits = False
    acc = Accuracy(task="multiclass", num_classes=n_classes).to(device)
    acc_quant = Accuracy(task="multiclass", num_classes=n_classes).to(device)
    loss_mean = MeanMetric(nan_strategy="ignore").to(device)
    loss_quant_mean = MeanMetric(nan_strategy="ignore").to(device)
    loss_fn = module.loss_module
    state = None
    if fake_quant and getattr(module, "quantizer", None) is not None:
        state = {key: value.clone() for key, value in module.model.state_dict().items()}
        module.quantizer.fake_quantize()
    for index, (imgs, labels) in enumerate(loader):
        if max_batches is not None and index >= max_batches:
            break
        imgs = imgs.to(device)
        labels = labels.to(device)
        logits = module.model(imgs)
        acc.update(logits.argmax(-1), labels)
        loss_mean.update(loss_fn(logits, labels))
        if state is not None:
            logits = module.model(imgs)
            acc_quant.update(logits.argmax(-1), labels)
            loss_quant_mean.update(loss_fn(logits, labels))
    if state is not None:
        module.model.load_state_dict(state)
        module.quantizer.activate_hooks()
        del state
    result = {"test_acc": float(acc.compute()), "test_loss": float(loss_mean.compute())}
    if fake_quant and getattr(module, "quantizer", None) is not None:
        result["test_acc_quant"] = float(acc_quant.compute())
        result["test_loss_quant"] = float(loss_quant_mean.compute())
    else:
        result["test_acc_quant"] = result["test_acc"]
        result["test_loss_quant"] = result["test_loss"]
    return result


# ----------------------------------------------------------------------
# support factories
# ----------------------------------------------------------------------
def _support_factory(
    args: argparse.Namespace,
    mode: str,
    core: torch.Tensor,
    churning: torch.Tensor,
    dynamic_sizes: list[int] | None,
):
    """Resolve the optimization-phase support for a non-dynamic mode.

    Args:
        args: Parsed CLI namespace.
        mode: One of ``persistent_core`` / ``core_churning`` / ``random_core`` /
            ``random_active`` (``dense`` and ``dynamic_active`` need no mask).
        core: Flat discovery-time core mask.
        churning: Flat discovery-time churning mask.
        dynamic_sizes: Per-recompute active-support sizes of the
            ``dynamic_active`` run (falls back to the discovery history).

    Returns:
        ``(mask_fn, meta)`` where ``mask_fn(step)`` returns a flat support mask
        (or ``None``) and ``meta`` describes the frozen support.
    """
    if mode in ("dense", "dynamic_active"):
        return None, {}
    if mode == "persistent_core":
        return lambda step: core, {"support_size": int(core.sum())}
    if mode == "core_churning":
        union = core | churning
        stride = max(1, int(args.churn_stride))

        def churn_mask(step: int) -> torch.Tensor:
            if stride > 1 and step % stride != 0:
                return core
            return union

        return churn_mask, {"support_size": int(union.sum())}
    if mode == "random_core":
        size = int(core.sum())
        seed = args.seed * 1009 + args.random_seed
    elif mode == "random_active":
        if dynamic_sizes:
            size = int(round(float(pd.Series(dynamic_sizes).median())))
        else:
            raise ValueError("random_active needs the dynamic_active run (run it first) to size the support")
        seed = args.seed * 1009 + args.random_seed + 1
    else:
        raise ValueError(f"unknown mode {mode!r}")
    generator = torch.Generator()
    generator.manual_seed(int(seed))
    mask = torch.zeros(core.numel(), dtype=torch.bool)
    indices = torch.randperm(core.numel(), generator=generator)[:size]
    mask[indices] = True
    print(f"[support] {mode}: frozen random support of {size} weights (seed {seed})")
    return (lambda step: mask), {"support_size": size, "random_seed": seed}


# ----------------------------------------------------------------------
# experiment orchestration
# ----------------------------------------------------------------------
def _quantization_hparams(args: argparse.Namespace) -> dict:
    return dict(
        t_scheduler_strategy=args.strategy,
        async_t_factor=args.async_factor,
        t_start=args.t_start,
        t_end=args.t_end,
        t_tau=args.t_tau,
        n_bits=args.n_bits,
        type=args.qtype,
        normalized=args.normalized,
        symmetric=args.symmetric,
        early_power=args.early_power,
        min_majorant=args.min_majorant,
        half_shift=args.half_shift,
        n_steps=args.epochs,
    )


def run_experiment(args: argparse.Namespace) -> pd.DataFrame:
    """Run the full discovery + forked-modes experiment and return the summary."""
    started = time.time()
    pl.seed_everything(args.seed, workers=True)
    out = Path(args.results_dir)
    discovery_dir = out / "discovery"
    discovery_dir.mkdir(parents=True, exist_ok=True)
    modes = tuple(args.modes)
    discovery_epochs = max(1, int(round(args.discovery_fraction * args.epochs)))
    if discovery_epochs >= args.epochs:
        raise ValueError("discovery_fraction*epochs must leave at least one optimization epoch")
    quantization_hparams = _quantization_hparams(args)
    tracker_config = build_tracker_config(args)
    print(f"[experiment] discovery: {discovery_epochs}/{args.epochs} epochs | modes: {modes}")
    print(f"[experiment] quantization: {quantization_hparams}")

    # ---------------- discovery phase (dense QAT + tracking) ----------------
    train_loader, val_loader, test_loader = make_loaders(args, args.seed)
    module = build_module(args, quantization_hparams)
    discovery_controller = SparseSupportController(
        "discovery",
        tracker_config=tracker_config,
        exclude_regex=args.track_exclude or None,
        log_interval=args.log_interval,
    )
    trainer = build_trainer(
        args,
        max_epochs=discovery_epochs,
        logger=CSVLogger(save_dir=str(discovery_dir), name="logs"),
        callbacks=[SSQStepCallback(), discovery_controller],
    )
    trainer.fit(module, train_loader, val_loader)

    tracker = discovery_controller.tracker
    time_active = tracker.time_active().detach().cpu()
    core = time_active >= float(args.core_threshold)
    churning = (time_active >= float(args.churning_min)) & ~core
    core_size, churning_size = int(core.sum()), int(churning.sum())
    print(
        f"[discovery] weights={tracker.num_parameters} steps={tracker.tracker.seen} "
        f"core={core_size} churning={churning_size}"
    )
    torch.save(
        {
            "core": core,
            "churning": churning,
            "time_active": time_active,
            "active_size_history": discovery_controller.active_size_history,
            "core_threshold": args.core_threshold,
            "churning_min": args.churning_min,
            "num_tracked_weights": tracker.num_parameters,
            "parameter_names": tuple(tracker.layout.names),
        },
        discovery_dir / "masks.pt",
    )
    torch.save(_state_to_cpu(tracker.state_dict()), discovery_dir / "tracker_state.pt")
    tracker.export_weight_statistics(path=str(discovery_dir / "weight_stats.pt"))
    discovery_ckpt = discovery_dir / "discovery.ckpt"
    trainer.save_checkpoint(str(discovery_ckpt))
    state = torch.load(discovery_ckpt, map_location="cpu", weights_only=False)["state_dict"]
    tracker_state = torch.load(discovery_dir / "tracker_state.pt", map_location="cpu", weights_only=False)
    discovery_sizes = list(discovery_controller.active_size_history)
    del trainer, module, discovery_controller, tracker
    _free_gpu_memory()

    # ---------------- forked optimization phases ----------------
    rows: list[dict] = []
    dynamic_sizes: list[int] | None = None
    for mode in modes:
        print(f"\n[mode] === {mode}: {MODE_DESCRIPTIONS[mode]} ===")
        pl.seed_everything(args.seed, workers=True)
        module = build_module(args, quantization_hparams)
        module.load_state_dict(state)
        fast_forward_quantizer(module.quantizer, discovery_epochs)
        mask_fn, mask_meta = _support_factory(args, mode, core, churning, dynamic_sizes)
        controller = SparseSupportController(
            mode,
            tracker_config=tracker_config,
            mask_fn=mask_fn,
            tracker_state=tracker_state,
            exclude_regex=args.track_exclude or None,
            support_update_interval=args.support_update_interval,
            dynamic_min_fraction=args.dynamic_min_fraction,
            log_interval=args.log_interval,
        )
        controller.set_frozen_masks(core, churning)
        train_loader, val_loader, test_loader = make_loaders(args, args.seed)
        trainer = build_trainer(
            args,
            max_epochs=args.epochs - discovery_epochs,
            logger=CSVLogger(save_dir=str(out), name=mode),
            callbacks=[SSQStepCallback(), controller],
        )
        trainer.fit(module, train_loader, val_loader)
        test_metrics = evaluate(module, test_loader, fake_quant=True, max_batches=args.eval_batches or None)
        row = {
            "mode": mode,
            "epochs_optimization": args.epochs - discovery_epochs,
            "support_size_final": mask_meta.get("support_size"),
            **controller.result(),
            **test_metrics,
            "best_val_acc": controller.best.get("mtr/acc", float("nan")),
            "best_val_acc_quant": controller.best.get("mtr/acc_quant", float("nan")),
            "final_val_acc": controller.last.get("mtr/acc", float("nan")),
            "final_val_acc_quant": controller.last.get("mtr/acc_quant", float("nan")),
            "log_dir": str(CSVLogger(save_dir=str(out), name=mode).log_dir),
            "core_size": core_size,
            "churning_size": churning_size,
        }
        rows.append(row)
        if mode == "dynamic_active":
            dynamic_sizes = list(controller.active_size_history)
        brief = {key: row[key] for key in ("test_acc_quant", "support_fraction_mean", "update_budget")}
        print(f"[mode] {mode}: {json.dumps(brief, default=str)}")
        del trainer, module, controller
        _free_gpu_memory()

    # ---------------- summary ----------------
    summary = pd.DataFrame(rows)
    if "dense" in set(summary["mode"]):
        dense_row = summary.loc[summary["mode"] == "dense"].iloc[0]
        summary["acc_drop_vs_dense"] = float(dense_row["test_acc_quant"]) - summary["test_acc_quant"]
        dense_acc = float(dense_row["test_acc_quant"])
        summary["relative_acc_drop_vs_dense"] = summary["acc_drop_vs_dense"] / dense_acc if dense_acc else float("nan")
    else:
        summary["acc_drop_vs_dense"] = float("nan")
        summary["relative_acc_drop_vs_dense"] = float("nan")
    columns = [column for column in SUMMARY_COLUMNS if column in summary.columns]
    extra = [column for column in summary.columns if column not in columns]
    summary = summary[columns + extra]
    summary.to_csv(out / "summary.csv", index=False)
    with open(out / "summary.json", "w") as handle:
        json.dump({"args": vars(args), "results": rows}, handle, indent=2, default=str)
    print(f"\n[summary] total time {time.time() - started:.0f}s, results under {out}")
    print(summary[[column for column in SUMMARY_COLUMNS if column in summary.columns]].to_string(index=False))
    return summary


# ----------------------------------------------------------------------
# CLI
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    """CLI parser exposing every experiment parameter."""
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    run = parser.add_argument_group("run")
    run.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES), help="modes to run, in order")
    run.add_argument("--epochs", type=int, default=30, help="total epochs (discovery + optimization)")
    run.add_argument(
        "--discovery-fraction", type=float, default=0.25, help="fraction of epochs spent in dense discovery"
    )
    run.add_argument("--results-dir", default="runs/sparse_support", help="output directory")
    run.add_argument("--seed", type=int, default=42, help="base seed for init, data ordering and masks")
    run.add_argument("--random-seed", type=int, default=7, help="extra seed mixed into the random supports")
    run.add_argument("--batch-size", type=int, default=64)
    run.add_argument("--num-workers", type=int, default=4)
    run.add_argument("--data-root", default="experiments/SimCLR/data", help="STL10 data directory")
    run.add_argument("--max-train-samples", type=int, default=4000, help="train split size (0 = full)")
    run.add_argument("--max-val-samples", type=int, default=1000, help="val split size (0 = full)")
    run.add_argument("--max-test-samples", type=int, default=0, help="test subset size (0 = full)")
    run.add_argument("--limit-batches", type=int, default=0, help="limit train batches per epoch (smoke tests)")
    run.add_argument("--eval-batches", type=int, default=0, help="limit val/test batches (smoke tests)")
    run.add_argument("--progress", action="store_true", help="show the progress bar")
    run.add_argument("--log-interval", type=int, default=50, help="steps between support-metric log emissions")
    optimizer = parser.add_argument_group("optimizer")
    optimizer.add_argument("--optimizer", default="Adam", choices=("Adam", "SGD"))
    optimizer.add_argument("--lr", type=float, default=1e-3)
    optimizer.add_argument("--weight-decay", type=float, default=0.0)
    quant = parser.add_argument_group("quantization")
    quant.add_argument("--n-bits", type=int, default=8)
    quant.add_argument("--strategy", default="cyclic", help="t_scheduler_strategy")
    quant.add_argument("--async-factor", type=float, default=1.0, help="async_t_factor")
    quant.add_argument("--t-start", type=float, default=0.9)
    quant.add_argument("--t-end", type=float, default=0.0005)
    quant.add_argument("--t-tau", type=float, default=8.0)
    quant.add_argument("--qtype", default="standard", choices=("naive", "standard", "shifted"))
    quant.add_argument("--normalized", action=argparse.BooleanOptionalAction, default=True)
    quant.add_argument("--symmetric", action=argparse.BooleanOptionalAction, default=False)
    quant.add_argument("--early-power", type=float, default=0.95)
    quant.add_argument("--min-majorant", type=float, default=0.0)
    quant.add_argument("--half-shift", action=argparse.BooleanOptionalAction, default=False)
    tracker = parser.add_argument_group("tracker")
    tracker.add_argument("--active-threshold", type=float, default=0.8)
    tracker.add_argument("--reservoir-threshold", type=float, default=0.3)
    tracker.add_argument("--normalization-quantile", type=float, default=0.9)
    tracker.add_argument("--decay-activity", type=float, default=0.9)
    tracker.add_argument("--decay-sensitivity", type=float, default=0.9)
    tracker.add_argument("--decay-velocity", type=float, default=0.9)
    tracker.add_argument("--core-threshold", type=float, default=0.90, help="persistent_active_threshold")
    tracker.add_argument("--churning-min", type=float, default=0.10, help="churn_min_time_active")
    tracker.add_argument("--churn-min-spells", type=int, default=2)
    tracker.add_argument("--track-exclude", default="", help="regex of parameter names to exclude from tracking")
    support = parser.add_argument_group("support scheduling")
    support.add_argument(
        "--support-update-interval", type=int, default=1, help="dynamic_active recompute interval (steps)"
    )
    support.add_argument(
        "--churn-stride", type=int, default=1, help="core_churning: update churning-only weights every k-th step"
    )
    support.add_argument(
        "--dynamic-min-fraction", type=float, default=0.0, help="floor for the dynamic support fraction"
    )
    return parser


def default_args(**overrides) -> argparse.Namespace:
    """Programmatic defaults for notebooks: build the CLI namespace and override fields."""
    parser = build_parser()
    args = parser.parse_args([])
    for key, value in overrides.items():
        if not hasattr(args, key):
            raise KeyError(f"unknown experiment argument {key!r}")
        setattr(args, key, value)
    return args


def main(argv=None) -> None:
    run_experiment(build_parser().parse_args(argv))


if __name__ == "__main__":
    main()
