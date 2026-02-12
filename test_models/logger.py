import os
import pandas as pd

def log_experiment(args, metrics, log_path="master_experiment_log.csv"):
    experiment_data = {
        "timestamp": pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
        "exp_id": getattr(args, "output_dir", "N/A"),
        "model_name": getattr(args, "model_name", "N/A"),
        "learning_rate": getattr(args, "lr", "N/A"),
        "batch_size": getattr(args, "batch_size", "N/A"),
        "epochs": getattr(args, "epochs", "N/A"),
        "max_len": getattr(args, "max_len", "N/A"),
        "grad_accum": getattr(args, "grad_accum", "N/A"),
        "pcc_v": metrics.get("avg_v"),
        "pcc_a": metrics.get("avg_a"),
        "rmse_va": metrics.get("avg_r")
    }

    df = pd.DataFrame([experiment_data])

    if not os.path.isfile(log_path):
        df.to_csv(log_path, index=False)
    else:
        df.to_csv(log_path, mode='a', header=False, index=False)
