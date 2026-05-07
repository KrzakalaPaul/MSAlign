from torch.optim.lr_scheduler import LambdaLR
import torch
import numpy as np

def optimizer_with_scheduler(self):
    optimizer = torch.optim.AdamW(self.parameters(), lr=self.hparams["lr"], weight_decay=self.hparams["weight_decay"])
    warmup_steps = self.hparams.n_warmup_steps
    total_steps = self.hparams.n_max_steps

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step) / float(max(1, warmup_steps))
        # cosine annealing after warmup
        progress = float(step - warmup_steps) / float(max(1, total_steps - warmup_steps))
        return 0.5 * (1.0 + np.cos(np.pi * progress))

    scheduler = {
        "scheduler": LambdaLR(optimizer, lr_lambda),
        "interval": "step",     # update per step
        "frequency": 1
    }
    return [optimizer], [scheduler]