import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from scipy.special import expit

class InferenceDataset(Dataset):
    def __init__(self, data, tokenizer, max_len):
        self.data = data
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        row = self.data[index]
        encoding = self.tokenizer(
            str(row['Text']),
            str(row['Target']),
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        output = {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten()
        }
        if 'token_type_ids' in encoding:
            output['token_type_ids'] = encoding['token_type_ids'].flatten()
        return output

def parse_jsonl(path):
    data = []
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            entry_id = entry.get('ID')
            text = entry.get('Text')
            
            targets = []
            if 'Quadruplet' in entry:
                for q in entry['Quadruplet']:
                    aspect = q.get('Aspect', 'NULL')
                    if aspect == "NULL":
                        target = q.get('Category', 'general').replace("#", " ")
                    else:
                        target = aspect
                    targets.append(target)
            elif 'Aspect' in entry:
                raw = entry['Aspect']
                raw_list = [raw] if not isinstance(raw, list) else raw
                for item in raw_list:
                    # --- FIX START ---
                    # Handles ['text'] artifacts without deleting internal apostrophes
                    item_str = str(item).strip()
                    if item_str.startswith("['") and item_str.endswith("']"):
                        clean = item_str[2:-2]
                    elif item_str.startswith("[\"") and item_str.endswith("\"]"):
                        clean = item_str[2:-2]
                    else:
                        clean = item_str
                    # --- FIX END ---
                    targets.append(clean)
            
            for t in targets:
                data.append({'ID': entry_id, 'Text': text, 'Target': t})
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_file", type=str, required=True)
    parser.add_argument("--output_file", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base")

    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
   

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    raw_data = parse_jsonl(args.test_file)
    dataset = InferenceDataset(raw_data, tokenizer, args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    results = []

    with torch.no_grad():
        for batch in tqdm(loader, desc=f"Processing {os.path.basename(args.test_file)}"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)
            preds = outputs.logits.cpu().numpy()
            results.extend(preds)

    submission_map = {}
    
    for i, row in enumerate(raw_data):
        if i >= len(results): break
        val_pred = expit(float(results[i][0])) * 8 + 1 
        aro_pred = expit(float(results[i][1])) * 8 + 1 

        val_str = f"{val_pred:.2f}"
        aro_str = f"{aro_pred:.2f}"
        
        doc_id = row['ID']
        target = row['Target']

        if doc_id not in submission_map:
            submission_map[doc_id] = {"ID": doc_id, "Aspect_VA": []}
        
        submission_map[doc_id]["Aspect_VA"].append({
            "Aspect": target,
            "VA": f"{val_str}#{aro_str}"
        })

    os.makedirs(os.path.dirname(args.output_file), exist_ok=True)
    
    with open(args.output_file, 'w', encoding='utf-8') as f:
        for entry in submission_map.values():
            f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    print(f"Saved: {args.output_file}")

if __name__ == "__main__":
    main()