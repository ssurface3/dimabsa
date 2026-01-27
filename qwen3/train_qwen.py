import os    
import shutil
import argparse
import torch
from transformers import (
    AutoModelForCausalLM,
    Trainer, 
    TrainingArguments,
    AutoConfig,
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    DefaultFlowCallback 
)
import bitsandbytes
os.environ["TORCHDYNAMO_DISABLE"] = "1" 
from datasett_without_prompts import QwenDataset
# import flash_attn
# from basic_new_mseloss import CustomTrainer
from helper import (
                    save_training_history,
                    PrinterCallback,
                    Cherrypiocker
                 )
# if we need some new model head 
# from Twohead import TwoheadModel
from peft import LoraConfig, get_peft_model, TaskType

try:
    import torch._dynamo as _dynamo
    _dynamo.disable()
except Exception:
    pass

parser = argparse.ArgumentParser()
parser.add_argument("--model_name", type=str, default="Qwen3d")
parser.add_argument("--train_data_path", type=str, default="data/train.jsonl")
parser.add_argument("--eval_data_path", type=str, default="data/eval.jsonl")
parser.add_argument("--test_data_path", type=str, default="data/eval.jsonl")
parser.add_argument("--output_dir", type=str, required=True)
parser.add_argument("--epochs", type=int, default=5)
parser.add_argument("--batch_size", type=int, default=16)
parser.add_argument("--lr", type=float, default=2e-5)
parser.add_argument("--grad_accum", type=int, default=1)
parser.add_argument("--resume_from_checkpoint", type=str, default=None)
parser.add_argument("--max_len", type=int, default=256)
parser.add_argument('--lora' , type = bool ,default = True)
args = parser.parse_args()

def main():
    print(f"Training: {args.model_name}") # watch out for the already establiashed one ! to retrain on the new dataset fro example
    print(f"Train data: {args.train_data_path}")
    print(f"Eval data: {args.eval_data_path}") 
    Dataloader = QwenDataset
    train_list = Dataloader._parse_jsonl(args.train_data_path)
    eval_list = Dataloader._parse_jsonl(args.eval_data_path)
    train_dataset = Dataloader(data = train_list, 
                               model = args.model_name, 
                               max_len=args.max_len
                               )
    eval_dataset = Dataloader(data = eval_list, 
                              model = args.model_name, 
                              max_len=args.max_len
                              )
    print(f"Train size: {len(train_dataset)} | Eval size: {len(eval_dataset)}")

    # config = AutoConfig.from_pretrained(args.model_name,
    #                                     num_labels = 2, 
    #                                     problem_type = 'regression',
    #                                     trust_remote_code = 'True',
    #                                     torch_dtype = torch.bfloat16, 
    #                                     attn_implementation="flash_attention_2"
                                    
    #                                     )
    print('loading model')
    model = AutoModelForCausalLM.from_pretrained(
            args.model_name, 
            # config=config , 
            torch_dtype = torch.bfloat16, 
            # attn_implementation="flash_attention_2", # need to import it
            trust_remote_code = 'True',
        )   
    print('loaded model ')
    #Lora part
    if args.lora:
        print('Lora activated')
        peft_config = LoraConfig(
                task_type=TaskType.CAUSAL_LM, # maybe question_ans
                inference_mode=False,
                r=16,           
                lora_alpha=32,    
                lora_dropout=0.1,
                target_modules=["q_proj", "v_proj"]
        )
        model = get_peft_model(model, peft_config)
        model.print_trainable_parameters() # telss how much parameters we are changing
        tokenizer = AutoTokenizer.from_pretrained(
            args.model_name, 
            return_tensors="pt",
        )
        collator = DataCollatorForLanguageModeling(
            tokenizer,
            mlm=False 

        )
        print("Model loaded.")
        training_args = TrainingArguments(
                        output_dir=args.output_dir,
                        
                        #Performance
                        per_device_train_batch_size=args.batch_size, 
                        gradient_accumulation_steps=args.grad_accum, 
                        learning_rate=args.lr,            
                        bf16=True,                    
                        
                        #Duration 
                        num_train_epochs=args.epochs,             
                        
                        #Logging and Saving
                        logging_steps=20,             
                        save_strategy="steps",
                        save_steps=40,
                        
                        #Evaluation
                        # evaluation_strategy="no", 
                        eval_strategy= 'steps', 
                        eval_steps = 20, 
                        load_best_model_at_end=True, # Save the best version, not the last version       
                        metric_for_best_model='loss',
                        #Optimizer
                        optim="paged_adamw_32bit",      
                        warmup_ratio=0.05,
                        lr_scheduler_type="cosine",
                    )

    # space_saver = SpaceSaverCallback()
    sample_callback = Cherrypiocker(
        tokenizer=tokenizer,
        eval_dataset=eval_dataset,
        num_samples=3
    )
    trainer = Trainer(
                model=model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset, 
                data_collator = collator,
                callbacks = [sample_callback, DefaultFlowCallback()]
            )

    if args.resume_from_checkpoint:
        trainer.train(resume_from_checkpoint=args.resume_from_checkpoint)
    else:
        print("Starting training...")
        trainer.train()
        print("Training completed.")

    final_path = f"./models/{args.output_dir}/final"
    trainer.save_model(final_path)    
    train_dataset.tokenizer.save_pretrained(final_path)
    save_training_history(trainer, args)
    
    for item in os.listdir(f"./models/{args.output_dir}"):
        if item.startswith("checkpoint-"):
            shutil.rmtree(os.path.join(f"./models/{args.output_dir}", item))

if __name__ == "__main__":
    main()