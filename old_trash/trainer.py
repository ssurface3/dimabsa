import argparse
import os
import torch
from sklearn.metrics import mean_absolute_error, mean_squared_error
from transformers import (
    AutoTokenizer, 
    AutoModelForSequenceClassification, 
    Trainer, 
    TrainingArguments,
    DataCollatorWithPadding
)
from dataloader import Dataloader

path_to_data = "/Users/anatoliifrolov/Desktop/epstein files/extractor/dimabsa/eng_restaurant_train_alltasks.jsonl"
parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
parser.add_argument("--data_path", type=str, default="data/train.jsonl")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
args = parser.parse_args()

def compute_metrics(self, eval_pred):
    predictions, labels = eval_pred
    mse = mean_squared_error(labels, predictions)
    mae = mean_absolute_error(labels, predictions)
    return {"mse": mse, "mae": mae}


def main():
    print(f"Parameters chosen: {args.output_dir} ---")
    tokenizer = AutoTokenizer.from_pretrained(args.model_name) 
    dataloader = Dataloader(args.data_path)    
    train_data, val_data = dataloader.tt_split()
    
    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2, #val + arousal 
        problem_type="regression" 
    )

    training_args = TrainingArguments(
        output_dir=f"./models/{args.output_dir}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        learning_rate=args.lr,
        weight_decay=0.01,
        
        # Strategies
        evaluation_strategy="epoch",  
        save_strategy="epoch",        
        load_best_model_at_end=True,  
        metric_for_best_model="mae",  
        greater_is_better=False,      
        
        # System
        logging_dir='./logs',
        logging_steps=50,
        fp16=torch.cuda.is_available() 
    )

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_data,
        eval_dataset=val_data,
        compute_metrics=compute_metrics,
        tokenizer=tokenizer
    )

    trainer.train()

    final_path = f"./models/{args.output_dir}/final"
    trainer.save_model(final_path)
    tokenizer.save_pretrained(final_path)
    
    metrics = trainer.evaluate()
    print(f"Final Metrics: {metrics}")

if __name__ == "__main__":
    main()