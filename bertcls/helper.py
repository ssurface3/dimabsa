import torch
import numpy as np
import math
import os
import pandas as pd

# --- 1. Math Helpers ---
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

# --- 2. Metric Function (Fixed for CLS) ---
def compute_metrics(eval_pred):
    logits, labels = eval_pred

    # 1. Convert to Tensor
    if not isinstance(logits, torch.Tensor):
        logits = torch.tensor(logits, dtype=torch.float32)
    if not isinstance(labels, torch.Tensor):
        labels = torch.tensor(labels, dtype=torch.float32)

    # 2. Safety Slicing (Handle both cases just in case)
    # If 2D [N, 64] (The Fix)
    if logits.dim() == 2:
        logits_v = logits[:, 0:32]  # First 32 cols
        logits_a = logits[:, 32:64] # Last 32 cols
    # If 3D [N, 2, 32] (Old way)
    elif logits.dim() == 3:
        logits_v = logits[:, 0, :]
        logits_a = logits[:, 1, :]
    
    # 3. Argmax (Find Bin ID)
    pred_bin_v = torch.argmax(logits_v, dim=1)
    pred_bin_a = torch.argmax(logits_a, dim=1)

    # 4. Convert Bins -> Scores
    pred_v = bin_to_score_torch(pred_bin_v)
    pred_a = bin_to_score_torch(pred_bin_a)

    # 5. Handle Labels
    gold_v = bin_to_score_torch(labels[:, 0])
    gold_a = bin_to_score_torch(labels[:, 1])
    
    # --- DEBUG CHECK ---
    # Un-comment if you still get errors to see exact mismatch
    # print(f"DEBUG SHAPES: Pred {pred_v.shape}, Gold {gold_v.shape}")

    # 6. Calculate Metrics (Safe Slicing)
    # Force them to be the same length to prevent crash
    min_len = min(len(pred_v), len(gold_v))
    pcc_v = pearson_torch(pred_v[:min_len], gold_v[:min_len])
    pcc_a = pearson_torch(pred_a[:min_len], gold_a[:min_len])

    # RMSE
    sse_v = torch.sum((gold_v[:min_len] - pred_v[:min_len]) ** 2)
    sse_a = torch.sum((gold_a[:min_len] - pred_a[:min_len]) ** 2)
    
    rmse_va = torch.sqrt((sse_v + sse_a) / min_len)
    rmse_norm = rmse_va / math.sqrt(128)

    return {
        'PCC_V': pcc_v.item(),
        'PCC_A': pcc_a.item(),
        'RMSE_VA': rmse_norm.item()
    }

# --- 3. Save History Function (THIS WAS MISSING) ---
def save_training_history(trainer, args):
    """
    Saves the training logs to a CSV file.
    """
    # Create logs directory
    os.makedirs("logs", exist_ok=True)
    
    # Extract logs
    history = trainer.state.log_history
    df = pd.DataFrame(history)
    
    # Save
    # Clean directory path to make a valid filename
    clean_name = args.output_dir.replace("/", "_").replace(".", "")
    filename = f"logs/history_{clean_name}.csv"
    
    df.to_csv(filename, index=False)
    print(f"✅ Training Logs saved to: {filename}")