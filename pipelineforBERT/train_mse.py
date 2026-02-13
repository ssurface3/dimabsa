import os    
import shutil
import logging
import argparse
import torch
import warnings
from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments,
    logging,
    AutoTokenizer
)
from dataloader import Dataloader
from custom_trainer_mse import CustomTrainer
from helper import SpaceSaverCallback , compute_metrics , save_training_history 
from tqdm import tqdm 
from transformers import ProgressCallback

logging.set_verbosity_error()
warnings.filterwarnings("ignore")

try:
    import torch._dynamo as _dynamo
    _dynamo.disable()
except Exception:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base")
parser.add_argument("--train_data_path", type=str, default="data/train.jsonl")
parser.add_argument("--eval_data_path", type=str, default="data/eval.jsonl")
parser.add_argument("--test_data_path", type=str, default="data/eval.jsonl")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--grad_accum", type=int, default=1)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--max_len", type=int, default=50)
# parser.add_argument("--normalize", type=str, default="standard")  # standard or normalized 
args = parser.parse_args()

def main():
    print(f"Training: {args.model_name}") # watch out for the already establiashed one ! to retrain on the new dataset fro example
    print(f"Train data: {args.train_data_path}")
    print(f"Eval data: {args.eval_data_path}") 

    train_list = Dataloader._parse_jsonl(args.train_data_path)
    eval_list = Dataloader._parse_jsonl(args.eval_data_path)
    train_dataset = Dataloader(train_list, args.model_name, max_len=args.max_len)
    eval_dataset = Dataloader(eval_list, args.model_name, max_len=args.max_len)
    print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

    model = AutoModelForSequenceClassification.from_pretrained(
        args.model_name, 
        num_labels=2, 
        problem_type="regression",
        use_safetensors=False,
    )
    training_args = TrainingArguments(
        output_dir=f"./models/{args.output_dir}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_strategy="steps",
        eval_steps=360, # approx 4 times per epoch for 23k data size
        save_strategy="steps",
        save_steps=360,
        save_total_limit=1,
        logging_steps=100,
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        report_to="none", 
        fp16=torch.cuda.is_available(),
        warmup_ratio=0.05 
    )

    space_saver = SpaceSaverCallback()
    trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
            )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        print("Starting training...")
        trainer.train()
        print("Training completed.")

    final_path = f"./models/{args.output_dir}/final"
    trainer.save_model(final_path)    
    save_training_history(trainer, args)
    
    for item in os.listdir(f"./models/{args.output_dir}"):
        if item.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(f"./models/{args.output_dir}", item))

if __name__ == "__main__":
    main()