import torch
import numpy as np
import math
import os
import pandas as pd

def pearson_torch(preds, targets):
    vx = preds - torch.mean(preds)
    vy = targets - torch.mean(targets)
    numerator = torch.sum(vx * vy)
    denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
    return numerator / (denominator + 1e-8)

def bin_to_score_torch(bin_idxs):
    """
    Converts Bin Indices (0-31) back to Float Scores (1.0-9.0).
    Logic: (BinID * 0.25) + 1.0 + (0.25 / 2.0)
    """
    return (bin_idxs.float() * 0.25) + 1.0 + 0.125


def pearson_torch(preds, targets):
    vx = preds - torch.mean(preds)
    vy = targets - torch.mean(targets)
    numerator = torch.sum(vx * vy)
    denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
    return numerator / (denominator + 1e-8)

def bin_to_score_torch(bin_idxs):
    return (bin_idxs.float() * 0.25) + 1.0 + 0.125

def compute_metrics(eval_pred):
    logits, labels = eval_pred

    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)

    if logits.dim() == 2:
        logits_v = logits[:, 0:32]
        logits_a = logits[:, 32:64]
    else:
        logits_v = logits[:, 0, :]
        logits_a = logits[:, 1, :]

    bin_pred_v = torch.argmax(logits_v, dim=1)
    bin_pred_a = torch.argmax(logits_a, dim=1)

    pred_v = bin_to_score_torch(bin_pred_v)
    pred_a = bin_to_score_torch(bin_pred_a)

    gold_bin_v = labels[:, 0]
    gold_bin_a = labels[:, 1]
    
    gold_v = bin_to_score_torch(gold_bin_v)
    gold_a = bin_to_score_torch(gold_bin_a)

    min_len = min(len(pred_v), len(gold_v))
    pcc_v = pearson_torch(pred_v[:min_len], gold_v[:min_len])
    pcc_a = pearson_torch(pred_a[:min_len], gold_a[:min_len])

    sse_v = torch.sum((gold_v[:min_len] - pred_v[:min_len]) ** 2)
    sse_a = torch.sum((gold_a[:min_len] - pred_a[:min_len]) ** 2)
    
    rmse_va = torch.sqrt((sse_v + sse_a) / min_len)
    rmse_norm = rmse_va / math.sqrt(128)

    return {
        'PCC_V': pcc_v.item(),
        'PCC_A': pcc_a.item(),
        'RMSE_VA': rmse_norm.item()
    }

def save_training_history(trainer, args):
    os.makedirs("logs", exist_ok=True)
    history = trainer.state.log_history
    df = pd.DataFrame(history)
    
    clean_name = args.output_dir.replace("/", "_").replace(".", "")
    filename = f"logs/history_{clean_name}.csv"
    
    df.to_csv(filename, index=False)
    print(f"Training Logs saved to: {filename}")

def save_training_history(trainer, args):
    """
    Saves the training logs to a CSV file.
    It is now saved both in the global logs directory and the experiment info directory.
    """
    history = trainer.state.log_history
    df = pd.DataFrame(history)
    
    # Global log
    os.makedirs("logs", exist_ok=True)
    clean_name = args.output_dir.replace("/", "_").replace(".", "")
    global_filename = f"logs/history_{clean_name}.csv"
    df.to_csv(global_filename, index=False)
    
    # Experiment specific log
    exp_log_dir = os.path.join(f"./models/{args.output_dir}", "experiment_info")
    os.makedirs(exp_log_dir, exist_ok=True)
    exp_filename = os.path.join(exp_log_dir, "training_history.csv")
    df.to_csv(exp_filename, index=False)
    print(f"Training history saved to {exp_filename}")

def create_experiment_dir(output_dir, data, folder_name="experiment_info"):
    """
    Creates a directory inside output_dir and saves experiment data to a text file.
    """
    target_dir = os.path.join(output_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)
    
    file_path = os.path.join(target_dir, "experiment_data.txt")
    with open(file_path, "w") as f:
        if isinstance(data, dict):
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write(str(data))
    
    print(f"Experiment data saved to {file_path}")
    return target_dir
    print(f"Training Logs saved to: {filename}")