import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoConfig
from tqdm import tqdm
from Twohead import TwoheadModel
from bins_putiins import bin_to_float

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
    with open(path, 'r', encoding='utf-8') as f:
        for line in f:
            if not line.strip(): continue
            entry = json.loads(line)
            entry_id = entry.get('ID')
            text = entry.get('Text') or entry.get('Sentence')
            
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
                    clean = str(item).replace("['", "").replace("']", "").replace("'", "").strip()
                    targets.append(clean)
            
            for t in targets:
                data.append({'ID': entry_id, 'Text': text, 'Target': t})
    return data

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--test_data", type=str, default="data/test.jsonl")
    parser.add_argument("--output_file", type=str, default=None)
    parser.add_argument("--output_dir", type=str, default="results")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        except:
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    try:
        config = AutoConfig.from_pretrained(args.model_path)
        model = TwoheadModel.from_pretrained(args.model_path, config=config)
        model.to(device)
        model.eval()
    except Exception as e:
        print(e)
        return

    files_to_process = [args.test_data]

    for file_path in files_to_process:
        raw_data = parse_jsonl(file_path)
        dataset = InferenceDataset(raw_data, tokenizer, args.max_len)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
        
        final_preds_v = []
        final_preds_a = []

        with torch.no_grad():
            for batch in tqdm(loader):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                token_type_ids = batch.get('token_type_ids')
                if token_type_ids is not None:
                    token_type_ids = token_type_ids.to(device)
                    outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
                else:
                    outputs = model(input_ids, attention_mask=attention_mask)
                
                if isinstance(outputs, tuple):
                    logits_v, logits_a = outputs
                else:
                    logits_v = outputs[:, 0, :]
                    logits_a = outputs[:, 1, :]

                bins_v = torch.argmax(logits_v, dim=1).cpu().numpy()
                bins_a = torch.argmax(logits_a, dim=1).cpu().numpy()
                
                float_v = [bin_to_float(b) for b in bins_v]
                float_a = [bin_to_float(b) for b in bins_a]
                
                final_preds_v.extend(float_v)
                final_preds_a.extend(float_a)

        submission_map = {}
        csv_data = []
        
        if args.output_file:
            json_output_path = args.output_file
        else:
            base_name = os.path.basename(file_path).replace(".jsonl", "")
            os.makedirs(args.output_dir, exist_ok=True)
            json_output_path = os.path.join(args.output_dir, f"pred_{base_name}.json")
            
        csv_output_path = json_output_path.replace(".json", ".csv")

        for i, row in enumerate(raw_data):
            if i >= len(final_preds_v): break

            val_pred = float(np.clip(final_preds_v[i], 1.00, 9.00))
            aro_pred = float(np.clip(final_preds_a[i], 1.00, 9.00))

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

            csv_data.append({
                "ID": doc_id,
                "Target": target,
                "Valence": val_pred,
                "Arousal": aro_pred
            })

        os.makedirs(os.path.dirname(json_output_path), exist_ok=True)
        
        final_list = list(submission_map.values())
        with open(json_output_path, 'w', encoding='utf-8') as f:
            for entry in final_list:
                f.write(json.dumps(entry, ensure_ascii=False) + "\n")

        pd.DataFrame(csv_data).to_csv(csv_output_path, index=False)

if __name__ == "__main__":
    main()