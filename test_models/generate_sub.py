import argparse
import os
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from dataloader import Dataloader
from scipy.special import expit
import numpy as np
parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--test_data", type=str, default="data/test.jsonl")
parser.add_argument("--output_file", type=str, default="submission.json")
parser.add_argument("--batch_size", type=int, default=32)
parser.add_argument("--max_len", type=int, default=50)
parser.add_argument("--model_name" , type=str , default = "jhu-clsp/mmBERT-base")
parser.add_argument("--base_tokenizer", type=str, default=None)
args = parser.parse_args()
def validate_io(batch, logits, tokenizer):
    ids = batch["input_ids"]
    v_size = tokenizer.vocab_size
    if torch.any(ids >= v_size) or torch.any(ids < 0):
        raise ValueError("Tokenizer index mismatch with model embeddings")

    unk_token = tokenizer.unk_token
    for i in range(ids.size(0)):
        raw_tokens = tokenizer.convert_ids_to_tokens(ids[i])
        if raw_tokens.count(unk_token) > (len(raw_tokens) * 0.7):
            print(f"Input verification failed for index {i}: Too many unknown tokens")

    if torch.any(torch.isnan(logits)):
        raise ValueError("Logits contain NaN")

    transformed_scores = (torch.sigmoid(logits) * 8.0) + 1.0
    
    if torch.any(transformed_scores < 1.0) or torch.any(transformed_scores > 9.0):
        print(f"Output out of range: Min {transformed_scores.min()}, Max {transformed_scores.max()}")

    return torch.clamp(transformed_scores, 1.0, 9.0)
def main():
    device = torch.device("cuda")

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    files_to_process = []
    if args.test_data and os.path.isdir(args.test_data):
        for p in sorted(os.listdir(args.test_data)):
            if p.lower().endswith('.jsonl') and 'test_task1' in p.lower():
                files_to_process.append(os.path.join(args.test_data, p))
    elif args.test_data:
        files_to_process = [args.test_data]

    if not files_to_process:
        raise SystemExit(f"No test files found at {args.test_data}")

    os.makedirs(os.path.dirname(args.output_file) or '.', exist_ok=True)

    for file_path in files_to_process:
        print(f"Processing {file_path} ...")
        dataset_list = Dataloader._parse_jsonl(file_path)
        
        dataset = Dataloader(dataset_list, args.base_tokenizer, max_len=args.max_len)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)

        raw_data = dataset.data
        results = []

        with torch.no_grad():
            for number , batch in enumerate(tqdm(loader, desc=f"Inferring {os.path.basename(file_path)}")):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                token_type_ids = batch.get('token_type_ids', None)

                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                else:
                    outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                if number ==1:
                    validate_io(
                        batch = batch ,
                        logits = outputs.logits,
                        tokenizer = AutoTokenizer.from_pretrained(
                            args.base_tokenizer, use_fast=True
                        )
                    )
                preds = outputs.logits.cpu().numpy()
                results.extend(preds)

        if len(results) != len(raw_data):
            print(f"Warning: predictions ({len(results)}) != examples ({len(raw_data)}). Truncating to min length.")
        n = min(len(results), len(raw_data))

        submission_map = {}
        csv_data = []

        for i in range(n):
            row = raw_data[i]
            
            val_logit = float(results[i][0])
            aro_logit = float(results[i][1])
            
            val_prob = expit(val_logit)
            aro_prob = expit(aro_logit)
            
            val_pred = (val_prob * 8.0) + 1.0
            aro_pred = (aro_prob * 8.0) + 1.0
            
            val_pred = np.clip(val_pred, 1.00, 9.00)
            aro_pred = np.clip(aro_pred, 1.00, 9.00)

            val_str = f"{val_pred:.2f}"
            aro_str = f"{aro_pred:.2f}"

            doc_id = row.get('ID', f'unk_{i}')
            target = row.get('Target', 'general')

            if doc_id not in submission_map:
                submission_map[doc_id] = {
                    "ID": doc_id,
                    "Aspect_VA": []
                }

            submission_map[doc_id]["Aspect_VA"].append({
                "Aspect": target,
                "VA": f"{val_str}#{aro_str}"
            })

            

        final_json = list(submission_map.values())

        out_json = args.output_file
        if os.path.isdir(args.test_data):
            base = os.path.splitext(os.path.basename(file_path))[0]
            model_tag = os.path.basename(args.model_path).replace('/', '-')
            out_json = os.path.join(os.path.dirname(args.output_file) or '.', f"{model_tag}_{base}.json")

        with open(out_json, 'w', encoding='utf-8') as f:
            for entry in final_json:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')
if __name__ == "__main__":
    main()