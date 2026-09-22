import torch
from torchmetrics import Metric


class ClassificationMargin(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("margin_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits: torch.Tensor, target: torch.Tensor):
        true_logit = logits.gather(1, target.unsqueeze(1)).squeeze(1)
        masked = logits.clone()
        masked.scatter_(1, target.unsqueeze(1), float('-inf'))
        max_other = masked.max(dim=1).values
        margin = true_logit - max_other
        self.margin_sum += margin.sum()
        self.n += margin.numel()

    def compute(self):
        return self.margin_sum / self.n.clamp(min=1)


class LogitNormDiff(Metric):
    def __init__(self, dist_sync_on_step=False):
        super().__init__(dist_sync_on_step=dist_sync_on_step)
        self.add_state("norm_sum", default=torch.tensor(0.0), dist_reduce_fx="sum")
        self.add_state("n", default=torch.tensor(0), dist_reduce_fx="sum")

    def update(self, logits_fp: torch.Tensor, logits_quant: torch.Tensor):
        per_example = (logits_fp - logits_quant).norm(p=2, dim=1)
        self.norm_sum += per_example.sum()
        self.n += per_example.numel()

    def compute(self):
        return self.norm_sum / self.n.clamp(min=1)