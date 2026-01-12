import os    
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Silence TensorFlow/CUDA
os.environ['WANDB_SILENT'] = 'true'       # Silence WandB

import argparse
import torch
import warnings
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
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
import shutil
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
def compute_metrics(eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    return {"mse": mse, "mae": mae}

def save_training_history(trainer, args):
    os.makedirs("logs", exist_ok=True)
    history = trainer.state.log_history
    df = pd.DataFrame(history)
    filename = f"logs/{args.output_dir}.csv"
    df.to_csv(filename, index=False)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", type=str, default="bert-base-uncased")
    parser.add_argument("--data_path", type=str, default="data/train.jsonl")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=2e-5)
    parser.add_argument("--grad_accum", type=int, default=1)
    parser.add_argument("--resume_from_checkpoint", type=str, default=None)
    args = parser.parse_args()
    if "mmbert" in args.model_name.lower() or "modernbert" in args.model_name.lower():
        print(f"Model '{args.model_name}' detected. Forcing CUDA_VISIBLE_DEVICES='0' to prevent Dynamo/FX errors.")
        os.environ["CUDA_VISIBLE_DEVICES"] = "0"

    print(f"--- Training: {args.model_name} ---")
    
    temp_loader = Dataloader(args.data_path) 
    full_df = pd.DataFrame(temp_loader.data)
    train_df, val_df = train_test_split(full_df, test_size=0.2, random_state=42)


    tokenizer = AutoTokenizer.from_pretrained(args.model_name)

    # only for one bert al ike  
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right" 
    
    
    train_dataset = Dataloader(train_df, tokenizer)
    eval_dataset = Dataloader(val_df, tokenizer)
    
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
        metric_for_best_model="mae",
        greater_is_better=False,
        report_to="none", 
        fp16=torch.cuda.is_available()
    )

    
    data_collator = DataCollatorWithPadding(tokenizer=tokenizer)
    space_saver = SpaceSaverCallback()
    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer,
        data_collator=data_collator ,        
        callbacks=[space_saver] 

    )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        trainer.train()

    final_path = f"./models/{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    save_training_history(trainer, args)
    
    for item in os.listdir(f"./models/{args.output_dir}"):
        if item.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(f"./models/{args.output_dir}", item))

if __name__ == "__main__":
    main()