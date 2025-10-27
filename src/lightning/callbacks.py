import torch
import pytorch_lightning as L


class KLAnnealingCallback(L.Callback):
    def __init__(self, total_anneal_steps: int, initial_beta: float = 0.0, final_beta: float = 1.0):
        super().__init__()
        self.total_anneal_steps = total_anneal_steps
        self.initial_beta = initial_beta
        self.final_beta = final_beta
        
    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
        current_step = trainer.global_step
        if current_step >= self.total_anneal_steps:
            beta = self.final_beta
        else:
            anneal_rate = current_step / self.total_anneal_steps
            beta = self.initial_beta + (self.final_beta - self.initial_beta) * anneal_rate
            
        pl_module.kl_beta = beta
