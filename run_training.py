import argparse
import os 
import sys 

from ultralytics import YOLO

# your imports
from softstairs_qat import SoftStairsQuantizer, QuantizationConfig


DATASET = "HomeObjects-3K.yaml"
IMGSIZE = 640
BATCH_SIZE = 64


def get_qconfig(strategy, t_start, steps, n_bits, naive):
    return QuantizationConfig(
        n_bits=n_bits,
        normalized=True,
        t_scheduler_strategy=strategy,
        t_start=t_start,
        t_end=1e-4,
        n_steps=steps * BATCH_SIZE,
        naive=naive
    )


def run_experiment_nb(model_name, n_epochs, strategy, t_start, n_bits, naive):

    qconfig = get_qconfig(
        strategy,
        t_start,
        n_epochs,
        n_bits,
        naive
    )

    from ultralytics.models.yolo.detect import DetectionTrainer

    class SSQATTrainer(DetectionTrainer):
        def _build_train_pipeline(self):
            import math 
            from ultralytics.utils import LOCAL_RANK
            """Build dataloaders, optimizer, and scheduler for current batch size."""
            batch_size = self.batch_size // max(self.world_size, 1)
            self.train_loader = self.get_dataloader(
                self.data["train"], batch_size=batch_size, rank=LOCAL_RANK, mode="train"
            )
            final_batch_size = len(self.train_loader.sampler) % self.train_loader.batch_size or self.train_loader.batch_size
            if self.args.imgsz < 2 * self.stride and not self.train_loader.drop_last and final_batch_size == 1:
                raise ValueError(
                    f"final batch=1 training at imgsz={self.args.imgsz} gives BatchNorm a single value per channel; "
                    f"change batch or use imgsz >= {2 * self.stride}"
                )
            # Note: When training DOTA dataset, double batch size could get OOM on images with >2000 objects.
            self.test_loader = self.get_dataloader(
                self.data.get("val") or self.data.get("test"),
                batch_size=batch_size if self.args.task in {"obb", "semantic", "depth"} else batch_size * 2,
                rank=LOCAL_RANK,
                mode="val",
            )
            self.accumulate = max(round(self.args.nbs / self.batch_size), 1)  # accumulate loss before optimizing
            weight_decay = 0.
            iterations = math.ceil(len(self.train_loader.dataset) / max(self.batch_size, self.args.nbs)) * self.epochs
            self.optimizer = self.build_optimizer(
                model=self.model,
                name=self.args.optimizer,
                lr=self.args.lr0,
                momentum=self.args.momentum,
                decay=weight_decay,
                iterations=iterations,
            )
            self._setup_scheduler()

        def get_model(self, cfg=None, weights=None, verbose=True):
            model = super().get_model(cfg, weights, verbose)

            excluded_modules = {
                n for n, p in model.named_modules()
                if "bn" in n
            }

            self.quantizer = SoftStairsQuantizer(
                model,
                qconfig,
                excluded_modules=excluded_modules,
            )

            return model
        
        def optimizer_step(self):
            """
            Custom optimizer step for SSQAT training.
            
            Args:
                epoch: Current epoch number
                batch: Current batch index
                optimizer: The optimizer being used
                loss: The loss value from the current batch
            """
            # Call the parent optimizer_step first
            super().optimizer_step()
            if self.quantizer is not None:
                self.quantizer.step()
                
        def get_validator(self):
            """Return a DetectionValidator for YOLO model validation."""
            from ultralytics.models import yolo
            from copy import copy 
            import torch 

            import json
            import time
            from pathlib import Path

            import numpy as np
            import torch
            import torch.distributed as dist

            from ultralytics.cfg import get_cfg, get_save_dir
            from ultralytics.data.utils import check_cls_dataset, check_det_dataset, convert_ndjson_to_yolo_if_needed
            from ultralytics.nn.autobackend import AutoBackend
            from ultralytics.utils import LOCAL_RANK, LOGGER, RANK, TQDM, callbacks, colorstr, emojis
            from ultralytics.utils.checks import check_imgsz
            from ultralytics.utils.ops import Profile, linear_sum_assignment
            from ultralytics.utils.torch_utils import (
                attempt_compile,
                autocast,
                select_device,
                smart_inference_mode,
                torch_distributed_zero_first,
                unwrap_model,
            )
            

            class NormValidator(yolo.detect.DetectionValidator):
                @smart_inference_mode()
                def __call__(self, trainer=None, model=None):
                    """Execute validation process, running inference on dataloader and computing performance metrics.

                    Args:
                        trainer (object, optional): Trainer object that contains the model to validate.
                        model (nn.Module, optional): Model to validate if not using a trainer.

                    Returns:
                        (dict): Dictionary containing validation statistics.
                    """
                    self.training = 1 # trainer is not None
                    augment = self.args.augment and (not self.training)
                    if self.training:
                        self.device = trainer.device
                        self.data = trainer.data
                        # Keep training validation read-only: inputs may be fp16, but EMA/model weights stay fp32 under autocast.
                        self.args.quantize = 16 if (self.device.type != "cpu" and trainer.amp) else None
                        model = trainer.ema.ema or trainer.model
                        if trainer.args.compile and hasattr(model, "_orig_mod"):
                            model = model._orig_mod  # validate non-compiled original model to avoid issues
                        model = model.float()
                        self.loss = {k: torch.zeros_like(v) for k, v in trainer.loss_items.items()}
                        self.args.plots &= trainer.stopper.possible_stop or (trainer.epoch == trainer.epochs - 1)
                        model.eval()
                    
                    self.run_callbacks("on_val_start")
                    dt = (
                        Profile(device=self.device),
                        Profile(device=self.device),
                        Profile(device=self.device),
                        Profile(device=self.device),
                    )
                    bar = TQDM(self.dataloader, desc=self.get_desc(), total=len(self.dataloader))
                    self.init_metrics(unwrap_model(model))
                    self.jdict = []  # empty before each val
                    for batch_i, batch in enumerate(bar):
                        self.run_callbacks("on_val_batch_start")
                        self.batch_i = batch_i
                        # Preprocess
                        with dt[0]:
                            batch = self.preprocess(batch)

                        with autocast(self.training and self.args.quantize == 16, device=self.device.type):
                            # Inference
                            with dt[1]:
                                preds = model(batch["img"], augment=augment)

                            # Loss
                            with dt[2]:
                                if self.training:
                                    for k, v in model.loss(batch, preds)[1].items():
                                        self.loss[k] += v

                        # Postprocess
                        with dt[3]:
                            preds = self.postprocess(preds)

                        self.update_metrics(preds, batch)
                        if self.args.plots and batch_i < 3 and RANK in {-1, 0}:
                            self.plot_val_samples(batch, batch_i)
                            self.plot_predictions(batch, preds, batch_i)

                        self.run_callbacks("on_val_batch_end")

                    stats = {}
                    self.gather_stats()
                    if RANK in {-1, 0}:
                        stats = self.get_stats()
                        self.speed = dict(zip(self.speed.keys(), (x.t / len(self.dataloader.dataset) * 1e3 for x in dt)))
                        self.finalize_metrics()
                        self.print_results()
                        self.run_callbacks("on_val_end")

                    if self.training:
                        # Reduce loss across all GPUs
                        loss = {k: v.clone().detach() for k, v in self.loss.items()}
                        if trainer.world_size > 1:
                            for v in loss.values():
                                dist.reduce(v, dst=0, op=dist.ReduceOp.AVG)
                        if RANK > 0:
                            return
                        loss = {k: v.cpu() / len(self.dataloader) for k, v in loss.items()}
                        results = {**stats, **trainer.label_loss_items(loss, prefix="val")}
                        return {k: round(float(v), 5) for k, v in results.items()}  # return results as 5 decimal place floats
                    
            return NormValidator(
                            self.test_loader, save_dir=self.save_dir, args=copy(self.args), _callbacks=self.callbacks
                        )



    model = YOLO(model_name)
    # def on_train_epoch_end(self):
    #     self.quantizer.step()
    # model.add_callback('on_fit_epoch_end', on_train_epoch_end)

    model.train(
        data=DATASET,
        epochs=n_epochs,
        name=f"{strategy}-{t_start}-{qconfig.n_bits}b-{n_epochs}e",
        imgsz=IMGSIZE,
        trainer=SSQATTrainer,
        save_period=10, batch=BATCH_SIZE
    )
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--model", required=True)
    parser.add_argument("--epochs", type=int, required=True)
    parser.add_argument("--strategy", required=True)
    parser.add_argument("--t-start", type=float, required=True)
    parser.add_argument("--bits", type=int, default=8)
    parser.add_argument("--naive", type=bool, default=False)

    args = parser.parse_args()

    print(
        f"Running: strategy={args.strategy}, "
        f"t_start={args.t_start}, "
        f"bits={args.bits}, "
        f"epochs={args.epochs}"
        f'naive: {args.naive}'
    )

    run_experiment_nb(
        args.model,
        args.epochs,
        args.strategy,
        args.t_start,
        args.bits,
        args.naive
    )