import torch 
import os 
import pandas as pd
from transformers import TrainerCallback
class SpaceSaverCallback(TrainerCallback):
    def on_save(self, args, state, control, **kwargs):
        if not state.best_model_checkpoint:
            return
        checkpoint_dir = state.best_model_checkpoint
        optim_file = os.path.join(checkpoint_dir, "optimizer.pt")
        sched_file = os.path.join(checkpoint_dir, "scheduler.pt")
        try:
            if os.path.exists(optim_file):
                os.remove(optim_file)
                print(f"SpaceSaver: Deleted {optim_file}")
            
            if os.path.exists(sched_file):
                os.remove(sched_file)
        except Exception as e:
            print(f"Could not clean up checkpoint: {e}")

import torch
import math
import numpy as np

def pearson_torch(preds, targets):
    vx = preds - torch.mean(preds)
    vy = targets - torch.mean(targets)
    numerator = torch.sum(vx * vy)
    denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
    return numerator / (denominator + 1e-8)

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)

    pred_v = logits[:, 0]
    pred_a = logits[:, 1]
    
    gold_v = labels[:, 0]
    gold_a = labels[:, 1]

    pred_v = torch.sigmoid(pred_v) * 8 + 1 # scale to [1, 9]
    pred_a =  torch.sigmoid(pred_a) * 8 + 1 # scale to [1, 9]
    
    pcc_v = pearson_torch(pred_v, gold_v)
    pcc_a = pearson_torch(pred_a, gold_a)

    sse_v = torch.sum((gold_v - pred_v) ** 2)
    sse_a = torch.sum((gold_a - pred_a) ** 2)
    
    total_sse = sse_v + sse_a
    n_samples = gold_v.shape[0]
    
    rmse_va = torch.sqrt(total_sse / n_samples)
    
    return {
        'PCC_V': pcc_v.item(),
        'PCC_A': pcc_a.item(),
        'RMSE_VA': rmse_va.item()
    }
def save_training_history(trainer, args):
    os.makedirs("logs", exist_ok=True)
    history = trainer.state.log_history
    df = pd.DataFrame(history)
    filename = f"logs/{args.output_dir}.csv"
    df.to_csv(filename, index=False)
