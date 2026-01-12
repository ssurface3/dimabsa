import os
# --- FORCE SINGLE GPU ---
os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ['WANDB_SILENT'] = 'true'

import argparse
import torch
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

# --- ROBUST METRICS ---
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    
    # Clip predictions to valid range (1-9) to avoid massive RMSE spikes
    predictions = np.clip(predictions, 1.0, 9.0)

    mse = ((predictions - labels) ** 2).mean()
    rmse_va = np.sqrt(mse)
    
    # Pearson Correlation
    try:
        pcc_v, _ = pearsonr(labels[:, 0], predictions[:, 0])
        pcc_a, _ = pearsonr(labels[:, 1], predictions[:, 1])
    except:
        pcc_v, pcc_a = 0.0, 0.0

    return {
        "rmse": rmse_va, # Renamed to standard 'rmse'
        "pcc_v": pcc_v,
        "pcc_a": pcc_a
    }

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, required=True)
    parser.add_argument("--data_path", type=str, default="data/eng_laptop_train_alltasks.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=4) # DeBERTa Large fits on 4
    parser.add_argument("--lr", type=float, default=1e-5)    # Safe LR
    parser.add_argument("--grad_accum", type=int, default=4) # Accumulate to 16
    args = parser.parse_args()

    print(f"--- STABLE TRAINING: {args.model_name} ---")
    
    # 1. Load Data
    # We verify data is actually loaded
    temp_loader = Dataloader(args.data_path) 
    print(f"DEBUG: Loaded {len(temp_loader.data)} rows. Example target: {temp_loader.data[0]['Target']}")
    
    if len(temp_loader.data) == 0:
        raise ValueError("CRITICAL: Dataset is empty! Check your path.")

    full_df = pd.DataFrame(temp_loader.data)
    train_df, val_df = train_test_split(full_df, test_size=0.15, random_state=42)

    # 2. Tokenizer (CRITICAL FIX)
    # If using DeBERTa, FORCE the tokenizer from the Hub to prevent local file errors
    tokenizer_source = args.model_name
    if "deberta" in args.model_name.lower():
        tokenizer_source = "microsoft/deberta-v3-large"
        print(f"⚠️  Forcing Tokenizer from Hub: {tokenizer_source}")
    
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_source)

    train_dataset = Dataloader(train_df, tokenizer)
    eval_dataset = Dataloader(val_df, tokenizer)
    
    # 3. Model
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2, 
        problem_type="regression"
    )
    
    # 4. Training Args (Standard & Safe)
    training_args = TrainingArguments(
        output_dir=f"./models/{args.output_dir}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size * 2,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        weight_decay=0.01,
        
        # Strategies
        eval_strategy="epoch",
        save_strategy="epoch",
        save_total_limit=1,
        load_best_model_at_end=True,
        metric_for_best_model="rmse", 
        greater_is_better=False,
        
        # Speed & Logging
        fp16=torch.cuda.is_available(),
        report_to="none",
        logging_steps=25 # Log often so you see if loss is dropping
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=DataCollatorWithPadding(tokenizer=tokenizer)
    )

    print("🚀 Starting Training...")
    trainer.train()

    # Save Final
    final_path = f"./models/{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    # Save Metrics Log
    history = pd.DataFrame(trainer.state.log_history)
    history.to_csv(f"logs/{args.output_dir}.csv", index=False)
    
    # Clean space
    import shutil
    for item in os.listdir(f"./models/{args.output_dir}"):
        if item.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(f"./models/{args.output_dir}", item))

if __name__ == "__main__":
    main()
