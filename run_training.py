import argparse
import os 
import sys 

from ultralytics import YOLO

# your imports
from softstairs_qat import SoftStairsQuantizer, QuantizationConfig


DATASET = "HomeObjects-3K.yaml"
IMGSIZE = 640
BATCH_SIZE = 64


def get_qconfig(strategy, t_start, steps, n_bits=8):
    return QuantizationConfig(
        n_bits=n_bits,
        normalized=True,
        t_scheduler_strategy=strategy,
        t_start=t_start,
        t_end=1e-4,
        n_steps=steps * BATCH_SIZE,
    )


def run_experiment_nb(model_name, n_epochs, strategy, t_start, n_bits=8):

    qconfig = get_qconfig(
        strategy,
        t_start,
        n_epochs,
        n_bits,
    )

    from ultralytics.models.yolo.detect import DetectionTrainer

    class SSQATTrainer(DetectionTrainer):

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

    args = parser.parse_args()

    print(
        f"Running: strategy={args.strategy}, "
        f"t_start={args.t_start}, "
        f"bits={args.bits}, "
        f"epochs={args.epochs}"
    )

    run_experiment_nb(
        args.model,
        args.epochs,
        args.strategy,
        args.t_start,
        args.bits,
    )