import argparse
import os
import json
import torch
import numpy as np
from torch.utils.data import DataLoader, Dataset
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from scipy.stats import pearsonr
from sklearn.metrics import mean_squared_error

class LocalEvalDataset(Dataset):
    def __init__(self, data_list, tokenizer, max_len=128):
        self.data = data_list
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
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor([row['Valence'], row['Arousal']], dtype=torch.float)
        }
        
        if 'token_type_ids' in encoding:
            output['token_type_ids'] = encoding['token_type_ids'].flatten()
            
        return output

def load_eval_data(path):
    data_list = []
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            text = entry.get('Text')
            
            if 'Quadruplet' in entry:
                for quad in entry['Quadruplet']:
                    aspect = quad.get('Aspect', 'NULL')
                    if aspect == "NULL":
                        target = quad.get('Category', 'general').replace("#", " ")
                    else:
                        target = aspect
                    
                    try:
                        val, aro = map(float, quad.get('VA', '5.0#5.0').split('#'))
                    except:
                        val, aro = 5.0, 5.0

                    data_list.append({
                        'Text': text, 
                        'Target': str(target),
                        'Valence': val, 
                        'Arousal': aro
                    })
    return data_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--max_len", type=int, default=128)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Evaluating on {device}")

    try:
        tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    except:
        try:
            tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=False)
        except:
            print("Local tokenizer missing. Downloading bert-base-uncased.")
            tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")

    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        tokenizer.padding_side = "right"

    try:
        model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
        model.to(device)
        model.eval()
    except Exception as e:
        print(f"Error loading model weights: {e}")
        return

    print(f"Loading data from {args.data_path}")
    raw_data = load_eval_data(args.data_path)
    
    if len(raw_data) == 0:
        print("Error: No data found or file is empty.")
        return

    dataset = LocalEvalDataset(raw_data, tokenizer, args.max_len)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    print(f"Loaded {len(dataset)} items.")

    preds_list = []
    labels_list = []

    with torch.no_grad():
        for batch in tqdm(loader, desc="Inference"):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            if 'token_type_ids' in batch:
                token_type_ids = batch['token_type_ids'].to(device)
                outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids, attention_mask=attention_mask)

            preds_list.append(outputs.logits.cpu())
            labels_list.append(labels.cpu())

    preds = torch.cat(preds_list, dim=0).numpy()
    truth = torch.cat(labels_list, dim=0).numpy()

    preds = np.clip(preds, 1.0, 9.0)

    pcc_v, _ = pearsonr(preds[:, 0], truth[:, 0])
    pcc_a, _ = pearsonr(preds[:, 1], truth[:, 1])
    
    mse_v = mean_squared_error(truth[:, 0], preds[:, 0])
    mse_a = mean_squared_error(truth[:, 1], preds[:, 1])
    
    rmse_va = np.sqrt((mse_v + mse_a) / 2)

    print("-" * 30)
    print(f"Valence PCC: {pcc_v:.4f}")
    print(f"Arousal PCC: {pcc_a:.4f}")
    print(f"Combined RMSE: {rmse_va:.4f}")
    print("-" * 30)

if __name__ == "__main__":
    main()

# python /kaggle/working/slop/al.py \
#     --model_path "/kaggle/working/dimabsa/models/jhu-clsp/mmBERT-base-finetuned-dimabsa-laptop-alltasks/final" \
#     --data_path "/kaggle/working/dimabsa/data_split/test.jsonl"