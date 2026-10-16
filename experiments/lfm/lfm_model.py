# experiments/lfm/lfm_model.py
from __future__ import annotations

import csv
import math
import os
from pathlib import Path
from typing import Dict, Optional, Set, Tuple, Union

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
 
from dataclasses import dataclass, field

from datasets import load_dataset
from transformers import (
        AutoTokenizer,
        DataCollatorForLanguageModeling,
        Lfm2Config,
        Lfm2ForCausalLM,
    )

ROOT = Path(__file__).resolve().parents[2]
OUTPUT_ROOT = ROOT / "experiments" / "lfm" / "outputs"

DEFAULT_MODEL_ID = "LiquidAI/LFM2-350M"
DEFAULT_MAX_SEQ_LEN = 512



def make_mini_lfm2_config(
    hidden_size: int = 256,
    num_hidden_layers: int = 8,
    num_attention_heads: int = 8,
    num_key_value_heads: int = 4,
    intermediate_size: int = 1024,
    full_attn_idxs: Optional[list[int]] = None,
    max_position_embeddings: int = 2048,
    vocab_size: int = 65536,
) -> Lfm2Config:
    if full_attn_idxs is None:
        full_attn_idxs = [2, 5]
    return Lfm2Config(
        hidden_size=hidden_size,
        intermediate_size=intermediate_size,
        num_hidden_layers=num_hidden_layers,
        num_attention_heads=num_attention_heads,
        num_key_value_heads=num_key_value_heads,
        max_position_embeddings=max_position_embeddings,
        vocab_size=vocab_size,
        conv_L_cache=3,
        conv_bias=False,
        full_attn_idxs=full_attn_idxs,
        block_auto_adjust_ff_dim=False,
        tie_word_embeddings=True,
    )


def load_lfm2_causal_lm(
    model_id: str = DEFAULT_MODEL_ID,
    lfm2_config: Optional[Lfm2Config] = None,
    torch_dtype: torch.dtype = torch.float32,
) -> Lfm2ForCausalLM:
    if lfm2_config is not None:
        return Lfm2ForCausalLM.from_pretrained(
            model_id,
            config=lfm2_config,
            torch_dtype=torch_dtype,
            trust_remote_code=True,
        )
    return Lfm2ForCausalLM.from_pretrained(
        model_id,
        torch_dtype=torch_dtype,
        trust_remote_code=True,
    )


def load_tokenizer(model_id: str = DEFAULT_MODEL_ID):
    tokenizer = AutoTokenizer.from_pretrained(model_id, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    return tokenizer


def _tokenize_text_dataset(ds, tokenizer, max_length: int):
    def _fn(examples):
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=max_length,
            padding=False,
        )

    return ds.map(_fn, batched=True, remove_columns=ds.column_names)


def get_wikitext_dataloaders(
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 4,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    tokenizer = load_tokenizer(model_id)
    raw = load_dataset("wikitext", "wikitext-2-raw-v1")

    train_ds = raw["train"].filter(lambda x: len(x["text"].strip()) > 0)
    val_ds = raw["validation"].filter(lambda x: len(x["text"].strip()) > 0)

    if limit_train is not None:
        train_ds = train_ds.select(range(min(limit_train, len(train_ds))))
    if limit_val is not None:
        val_ds = val_ds.select(range(min(limit_val, len(val_ds))))

    train_ds = _tokenize_text_dataset(train_ds, tokenizer, max_length)
    val_ds = _tokenize_text_dataset(val_ds, tokenizer, max_length)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, tokenizer


def get_smoltalk_dataloaders(
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 4,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
    limit_train: Optional[int] = 2000,
    limit_val: Optional[int] = 200,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    tokenizer = load_tokenizer(model_id)
    raw = load_dataset("HuggingFaceTB/smoltalk", "all")

    train_raw = raw["train"]
    val_raw = raw["test"]

    if limit_train is not None:
        train_raw = train_raw.select(range(min(limit_train, len(train_raw))))
    if limit_val is not None:
        val_raw = val_raw.select(range(min(limit_val, len(val_raw))))

    def to_text(example):
        text = tokenizer.apply_chat_template(
            example["messages"],
            tokenize=False,
            add_generation_prompt=False,
        )
        return {"text": text}

    train_ds = train_raw.map(to_text, remove_columns=train_raw.column_names)
    val_ds = val_raw.map(to_text, remove_columns=val_raw.column_names)

    train_ds = _tokenize_text_dataset(train_ds, tokenizer, max_length)
    val_ds = _tokenize_text_dataset(val_ds, tokenizer, max_length)

    train_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])
    val_ds.set_format(type="torch", columns=["input_ids", "attention_mask"])

    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        drop_last=True,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collator,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )
    return train_loader, val_loader, tokenizer


def get_text_dataloaders(
    dataset: str = "wikitext",
    model_id: str = DEFAULT_MODEL_ID,
    batch_size: int = 4,
    max_length: int = DEFAULT_MAX_SEQ_LEN,
    limit_train: Optional[int] = None,
    limit_val: Optional[int] = None,
    num_workers: int = 0,
) -> Tuple[DataLoader, DataLoader, AutoTokenizer]:
    if dataset == "wikitext":
        return get_wikitext_dataloaders(
            model_id=model_id,
            batch_size=batch_size,
            max_length=max_length,
            limit_train=limit_train,
            limit_val=limit_val,
            num_workers=num_workers,
        )
    if dataset == "smoltalk":
        return get_smoltalk_dataloaders(
            model_id=model_id,
            batch_size=batch_size,
            max_length=max_length,
            limit_train=limit_train,
            limit_val=limit_val,
            num_workers=num_workers,
        )
    raise ValueError(f"Unknown dataset: {dataset}. Use 'wikitext' or 'smoltalk'.")



def compute_snapshot_steps(n_epochs: int, steps_per_epoch: int) -> Set[int]:
    total = n_epochs * steps_per_epoch
    return {0, total // 2, max(total - 1, 0)}


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
        if grad.ndim != 2:
            continue
        fig, ax = plt.subplots(figsize=(6, 4))
        im = ax.imshow(grad_to_2d(grad), aspect="auto", cmap="viridis")
        ax.set_title(f"{name} | step={step}")
        plt.colorbar(im)
        safe = name.replace(".", "_")
        fig.savefig(out_dir / f"heatmap_{safe}_step{step}.png", dpi=120, bbox_inches="tight")
        plt.close(fig)


METRICS_CSV_COLUMNS = [
    "step",
    "epoch",
    "t",
    "train_loss",
    "val_loss",
    "val_perplexity",
    "quant_error",
]


def compute_lfm_causal_loss(
    model: Lfm2ForCausalLM,
    batch: dict,
    device: torch.device,
) -> Tuple[torch.Tensor, dict]:
    batch = {k: v.to(device) for k, v in batch.items()}
    outputs = model(
        input_ids=batch["input_ids"],
        attention_mask=batch.get("attention_mask"),
        labels=batch["labels"],
    )
    loss = outputs.loss
    return loss, {"loss": float(loss.item())}


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


@dataclass
class TrainHistory:
    step: int
    epoch: int
    train_loss: float
    val_loss: Optional[float] = None
    val_perplexity: Optional[float] = None
    t: Optional[float] = None
    quant_error: Optional[float] = None


class LFMCausalTrainer:
    """LFM2 + SoftStairs / torchao baseline. Один metrics.csv, 1 строка / epoch."""

    def __init__(
        self,
        model_id: str = DEFAULT_MODEL_ID,
        lfm2_config: Optional[Lfm2Config] = None,
        lr: float = 2e-5,
        weight_decay: float = 0.0,
        beta1: float = 0.9,
        run_name: str = "run",
        snapshot_steps: Optional[Set[int]] = None,
        quantizer=None,
        qconfig=None,
        torch_qconfig=None,
        gradient_clip_val: float = 1.0,
        device: Optional[torch.device] = None,
    ):
        self.model_id = model_id
        self.lfm2_config = lfm2_config
        self.lr = lr
        self.weight_decay = weight_decay
        self.beta1 = beta1
        self.run_name = run_name
        self.snapshot_steps = snapshot_steps or set()
        self.gradient_clip_val = gradient_clip_val
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.lfm = load_lfm2_causal_lm(
            model_id=model_id,
            lfm2_config=lfm2_config,
            torch_dtype=torch.float32,
        ).to(self.device)

        if torch_qconfig is not None:
            from torchao.quantization import quantize_
            from torchao.quantization.qat import QATConfig
            quantize_(self.lfm, QATConfig(torch_qconfig, step="prepare"))

        self.quantizer = quantizer
        if qconfig is not None:
            from softstairs_qat import SoftStairsQuantizer
            self.quantizer = SoftStairsQuantizer(
                self.lfm,
                qconfig,
                excluded_modules=set(),
            )

        self.optimizer = torch.optim.AdamW(
            self.lfm.parameters(),
            lr=self.lr,
            betas=(self.beta1, 0.999),
            weight_decay=self.weight_decay,
        )
        self.scheduler: Optional[torch.optim.lr_scheduler.CosineAnnealingLR] = None

        self.out_dir = OUTPUT_ROOT / run_name
        self.out_dir.mkdir(parents=True, exist_ok=True)
        self.grad_dir = self.out_dir / "gradients"

        self.global_step = 0
        self.history: list[TrainHistory] = []
        self._metrics_csv_initialized = False

    @torch.no_grad()
    def validate(self, val_loader: DataLoader) -> tuple[float, float]:
        self.lfm.eval()
        losses: list[float] = []
        for batch in val_loader:
            _, metrics = compute_lfm_causal_loss(self.lfm, batch, self.device)
            losses.append(metrics["loss"])
        val_loss = float(np.mean(losses)) if losses else float("nan")
        val_ppl = float(math.exp(min(val_loss, 20.0)))
        return val_loss, val_ppl

    def _log_epoch_metrics(
        self,
        epoch: int,
        train_loss: float,
        val_loss: float,
        val_ppl: float,
        quant_error: Optional[float] = None,
    ) -> None:
        append_metrics_row(
            self.out_dir,
            {
                "step": self.global_step,
                "epoch": epoch,
                "t": self.quantizer.get_current_t() if self.quantizer else None,
                "train_loss": train_loss,
                "val_loss": val_loss,
                "val_perplexity": val_ppl,
                "quant_error": quant_error,
            },
            reset=(not self._metrics_csv_initialized),
        )

    def _maybe_save_grad_snapshots(self) -> None:
        if self.global_step not in self.snapshot_steps:
            return
        grads = {
            n: p.grad.detach().clone()
            for n, p in self.lfm.named_parameters()
            if p.grad is not None and p.grad.ndim == 2
        }
        if grads:
            save_gradient_heatmaps(grads, self.global_step, self.grad_dir)

    def fit(
        self,
        train_loader: DataLoader,
        val_loader: DataLoader,
        max_epochs: int,
    ) -> list[TrainHistory]:
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=max(max_epochs, 1)
        )

        for epoch in range(max_epochs):
            self.lfm.train()
            epoch_losses: list[float] = []

            for batch in train_loader:
                self.optimizer.zero_grad(set_to_none=True)
                loss, metrics = compute_lfm_causal_loss(self.lfm, batch, self.device)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.lfm.parameters(), max_norm=self.gradient_clip_val
                )
                self.optimizer.step()

                epoch_losses.append(metrics["loss"])

                if self.quantizer is not None:
                    self.quantizer.step()

                self._maybe_save_grad_snapshots()
                self.global_step += 1

            train_loss = float(np.mean(epoch_losses))
            val_loss, val_ppl = self.validate(val_loader)
            self.scheduler.step()
            quant_error = None
            if self.quantizer is not None:
                quant_error = float(
                    self.quantizer.estimate_current_quant_error(directed=False)
                )
            self._log_epoch_metrics(
                epoch, train_loss, val_loss, val_ppl, quant_error=quant_error
)

            t_str = ""
            qe_str = ""
            if self.quantizer is not None:
                t_str = f" t={self.quantizer.get_current_t():.4f}"
                qe_str = f" qerr={quant_error:.4f}"
            print(
                f"[{self.run_name}] epoch {epoch + 1}/{max_epochs} "
                f"loss={train_loss:.4f} val_loss={val_loss:.4f} ppl={val_ppl:.2f}{t_str}{qe_str}"
            )
            self.history.append(
                TrainHistory(
                    step=self.global_step,
                    epoch=epoch,
                    train_loss=train_loss,
                    val_loss=val_loss,
                    val_perplexity=val_ppl,
                    t=self.quantizer.get_current_t() if self.quantizer else None,
                    quant_error=quant_error,
                )
            )

        return self.history


def fit_lfm_model(
    trainer: LFMCausalTrainer,
    train_loader: DataLoader,
    val_loader: DataLoader,
    max_epochs: int,
) -> LFMCausalTrainer:
    trainer.fit(train_loader, val_loader, max_epochs=max_epochs)
    return trainer