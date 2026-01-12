import os
import sys

os.environ["CUDA_VISIBLE_DEVICES"] = "0" 
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WANDB_SILENT'] = 'true'

import argparse
import torch
import torch.nn as nn
import warnings
import pandas as pd
import numpy as np
from scipy.stats import pearsonr
from sklearn.model_selection import train_test_split
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding,
    logging
)

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

from dataloader import Dataloader

# --- 2. ADVANCED METRICS (PCC & RMSE) ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred


    pred_v = predictions[:, 0]
    pred_a = predictions[:, 1]
    label_v = labels[:, 0]
    label_a = labels[:, 1]

    # 1. Pearson Correlation (PCC)
    # Handle edge case where standard deviation is 0 (constant prediction)
    try:
        pcc_v, _ = pearsonr(label_v, pred_v)
        pcc_a, _ = pearsonr(label_a, pred_a)
    except:
        pcc_v, pcc_a = 0.0, 0.0

    # 2. RMSE (Root Mean Squared Error)
    # We calculate separately then average, or total. 
    # Based on your screenshot, RMSE_VA is usually the global RMSE.
    mse = ((predictions - labels) ** 2).mean()
    rmse_va = np.sqrt(mse)

    return {
        "rmse_va": rmse_va,
        "pcc_v": pcc_v,
        "pcc_a": pcc_a
    }

# --- 3. CUSTOM LOSS FUNCTION (CCC + MSE) ---
class CCCLoss(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x, y):
        # Concordance Correlation Coefficient
        vx = x - torch.mean(x)
        vy = y - torch.mean(y)
        rho = torch.sum(vx * vy) / (torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2)) + 1e-8)
        
        sigma_x = torch.var(x) + 1e-8
        sigma_y = torch.var(y) + 1e-8
        mu_x = torch.mean(x)
        mu_y = torch.mean(y)
        
        numerator = 2 * rho * torch.sqrt(sigma_x) * torch.sqrt(sigma_y)
        denominator = sigma_x + sigma_y + (mu_x - mu_y) ** 2
        
        ccc = numerator / denominator
        return 1.0 - ccc

class CustomTrainer(Trainer):
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        # Custom Loss Logic:
        # We combine MSE (Distance) with CCC (Correlation)
        # Loss = 0.5 * MSE + 0.5 * (1 - CCC)
        
        loss_fct_mse = nn.MSELoss()
        loss_fct_ccc = CCCLoss()
        
        # Calculate separately for Valence and Arousal
        mse_loss = loss_fct_mse(logits, labels)
        
        ccc_v = loss_fct_ccc(logits[:, 0], labels[:, 0])
        ccc_a = loss_fct_ccc(logits[:, 1], labels[:, 1])
        ccc_loss = (ccc_v + ccc_a) / 2
        
        # Weighted combination (Tune this if needed, 50/50 is standard)
        loss = 0.5 * mse_loss + 0.5 * ccc_loss

        return (loss, outputs) if return_outputs else loss

# --- 4. UTILS ---
def save_training_history(trainer, args):
    os.makedirs("logs", exist_ok=True)
    history = trainer.state.log_history
    df = pd.DataFrame(history)
    filename = f"logs/{args.output_dir}_metrics.csv"
    df.to_csv(filename, index=False)

# --- 5. MAIN ---
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="microsoft/deberta-v3-large")
    parser.add_argument("--data_path", type=str, default="data/train.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=False)
    args = parser.parse_args()

    print(f"--- Training: {args.model_name} ---")
    
    # Load Data
    temp_loader = Dataloader(args.data_path) 
    full_df = pd.DataFrame(temp_loader.data)
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)

    # Tokenizer
    # Always load tokenizer from the BASE model configuration on Hugging Face
    # to avoid the "DeBERTa V2/V3" local loading bug.
    tokenizer_source = args.model_name
    if "checkpoint" in args.model_name or "final" in args.model_name:
        # If loading a local path, try to guess the base tokenizer
        if "deberta" in args.model_name: tokenizer_source = "microsoft/deberta-v3-large"
        elif "roberta" in args.model_name: tokenizer_source = "roberta-large"
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"
    
    train_dataset = Dataloader(train_df, tokenizer)
    eval_dataset = Dataloader(val_df, tokenizer)
    
    # Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2, 
        problem_type="regression",
        pad_token_id=tokenizer.pad_token_id
    )
    
    model.config.pad_token_id = tokenizer.pad_token_id

    training_args = TrainingArguments(
        output_dir=f"./models/{args.output_dir}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="rmse_va", # Optimize for the competition metric!
        greater_is_better=False,         # Lower RMSE is better
        report_to="none", 
        fp16=torch.cuda.is_available()
    )

    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)

    # USE CUSTOM TRAINER
    trainer = CustomTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator 
    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    final_path = f"./models/{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    save_training_history(trainer, args)
    
    # Cleanup
    import shutil
    for item in os.listdir(f"./models/{args.output_dir}"):
        if item.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(f"./models/{args.output_dir}", item))

if __name__ == "__main__":
    main()