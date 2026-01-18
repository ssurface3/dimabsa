import os    
import shutil
# import sys
# import logging

import argparse
import torch
import warnings
from transformers import (
    AutoModelForSequenceClassification, 
    TrainingArguments,
    # logging,
    # AutoTokenizer, 
    AutoConfig
)
from dataloader import Dataloader
from bertcls.custom_trainer_normalized import CustomTrainer
from helper import (
                    #  SpaceSaverCallback , 
                    compute_metrics , 
                    save_training_history 
                 )
# from tqdm import tqdm 
# from transformers import ProgressCallback
from Twohead import TwoheadModel

try:
    import torch._dynamo as _dynamo
    _dynamo.disable()
except Exception:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="bert-base-uncased")
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

    config = AutoConfig.from_pretrained(
        args.model_name,
        num_labels=32, # two head with 32 bins each
        problem_type= 'single_label_classification'
    )
    
    

    try:
        model = TwoheadModel.from_pretrained(
            args.model_name,
            config=config,
            ignore_mismatched_sizes=True
        )
    except Exception as e:
        print(f"Standard load failed: {e}")
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            config=config,
            ignore_mismatched_sizes=True,
            trust_remote_code=True 
        )
    print("Model loaded.")
    training_args = TrainingArguments(
        output_dir=f"./models/{args.output_dir}",
        num_train_epochs=args.epochs,
        per_device_train_batch_size=args.batch_size,
        per_device_eval_batch_size=args.batch_size,
        gradient_accumulation_steps=args.grad_accum,
        learning_rate=args.lr,
        eval_strategy="steps",
        eval_steps=50,
        save_strategy="epoch",
        save_total_limit=1,
        logging_steps=10,
        # load_best_model_at_end=True,
        greater_is_better=False,
        report_to="none", 
        fp16=torch.cuda.is_available(),
        warmup_ratio=0.05 # added warmup ratio 
    )

    # space_saver = SpaceSaverCallback()
    trainer = CustomTrainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                compute_metrics=compute_metrics,
                # callbacks=[ProgressCallback()]
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