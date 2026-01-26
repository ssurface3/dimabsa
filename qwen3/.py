import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from peft import PeftModel
import json
from tqdm import tqdm
import numpy as np
from sklearn.metrics import mean_squared_error
import argparse
import csv
import os


parser = argparse.ArgumentParser()
parser.add_argument("--base_model", type=str, default="Qwen/Qwen3-0.6B", help="The base model used for training")
parser.add_argument("--checkpoint_path", type=str, required=True, help="Path to your saved adapter (e.g. results/checkpoint-500)")
parser.add_argument("--test_data", type=str, required=True, help="Path to dev.jsonl or test.jsonl")
args = parser.parse_args()


output_csv = "realtime_results.csv"


if not os.path.exists(output_csv):
    with open(output_csv, mode='w', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        writer.writerow(["Index", "Input_Text", "Real_VA", "Predicted_VA", "Raw_Output_Text"])

print(f"Logging results to: {output_csv}")

def parse_output(output_text):
    """
    Tries to extract 'Valence#Arousal' from the model output.
    """
    if "assistant" in output_text:

        response = output_text.split("assistant")[-1].strip()
    else:
        response = output_text
    response = response.replace("<|im_end|>", "").replace("<|im_start|>", "").strip()

    try:
        if "#" in response:
            v_str, a_str = response.split("#")
            return float(v_str), float(a_str), response
        else:
            return None, None, response
    except ValueError:
        return None, None, response

def main():
    print(f"Loading Base Model: {args.base_model}")
    tokenizer = AutoTokenizer.from_pretrained(args.base_model, trust_remote_code=True)
    
    base_model = AutoModelForCausalLM.from_pretrained(
        args.base_model,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    print(f"Loading LoRA Adapter: {args.checkpoint_path}")
    model = PeftModel.from_pretrained(base_model, args.checkpoint_path)
    model.eval()

    print(f"Loading Data: {args.test_data}")
    with open(args.test_data, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    predictions_v, predictions_a = [], []
    truths_v, truths_a = [], []
    

    system_instruction = (
        "You are an expert sentiment analyzer. "
        "Analyze the text and output the Valence and Arousal scores (1.0-9.0) "
        "formatted strictly as: Valence#Arousal"
    )

    print("Starting Inference...")
    
    for i, line in tqdm(enumerate(lines), total=len(lines)):
        entry = json.loads(line)
        
        text = entry.get('Text', '')

        target = "General"
        true_v, true_a = 5.0, 5.0
        if 'Quadruplet' in entry:
            q = entry['Quadruplet'][0]
            target = q.get('Aspect', 'General')
            try:
                va = q.get('VA', '5.0#5.0').split('#')
                true_v, true_a = float(va[0]), float(va[1])
            except: pass
        

        system_part = f"<|im_start|>system\n{system_instruction}<|im_end|>\n"
        user_content = f"Domain: general\nText: {text}\nTarget: {target}\nWhat is the valence and arousal score?"
        user_part = f"<|im_start|>user\n{user_content}<|im_end|>\n"
        assistant_header = "<|im_start|>assistant\n"
        
        prompt = system_part + user_part + assistant_header
        
        inputs = tokenizer(prompt, return_tensors="pt").to(model.device)


        with torch.no_grad():
            output_ids = model.generate(
                **inputs,
                max_new_tokens=20, 
                do_sample=False,   
                pad_token_id=tokenizer.eos_token_id
            )

        output_text = tokenizer.decode(output_ids[0], skip_special_tokens=False)
    
    
        pred_v, pred_a, raw_response = parse_output(output_text)
        
        pred_str = "FAIL"
        if pred_v is not None:
            pred_str = f"{pred_v}#{pred_a}"
            predictions_v.append(pred_v)
            predictions_a.append(pred_a)
            truths_v.append(true_v)
            truths_a.append(true_a)
        
    
        with open(output_csv, mode='a', newline='', encoding='utf-8') as f:
            writer = csv.writer(f)
    
            writer.writerow([i, text, f"{true_v}#{true_a}", pred_str, raw_response])

    
        if pred_v is None and i < 20:
            print(f"\n[FAIL] Raw Output: {raw_response}")

    
    if len(predictions_v) > 0:
        rmse_v = np.sqrt(mean_squared_error(truths_v, predictions_v))
        rmse_a = np.sqrt(mean_squared_error(truths_a, predictions_a))
        
        print("\n================RESULTS=================")
        print(f"Count: {len(predictions_v)} samples")
        print(f"Valence RMSE: {rmse_v:.4f}")
        print(f"Arousal RMSE: {rmse_a:.4f}")
        print(f"Detailed CSV saved to: {output_csv}")
        print("========================================")
    else:
        print("No valid predictions parsed.")

if __name__ == "__main__":
    main()