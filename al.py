# import argparse
# import os
# import json
# import torch
# import numpy as np
# from torch.utils.data import DataLoader, Dataset
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from tqdm import tqdm
# from scipy.stats import pearsonr
# from scipy.special import expit
# from sklearn.metrics import mean_squared_error

# class LocalEvalDataset(Dataset):
#     def __init__(self, data_list, tokenizer, max_len=128):
#         self.data = data_list
#         self.tokenizer = tokenizer
#         self.max_len = max_len

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         row = self.data[index]
        
#         encoding = self.tokenizer(
#             str(row['Text']),
#             str(row['Target']),
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )

#         output = {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'labels': torch.tensor([row['Valence'], row['Arousal']], dtype=torch.float)
#         }
        
#         if 'token_type_ids' in encoding:
#             output['token_type_ids'] = encoding['token_type_ids'].flatten()
            
#         return output

# def load_eval_data(path):
#     data_list = []
#     with open(path, 'r') as f:
#         for line in f:
#             entry = json.loads(line)
#             text = entry.get('Text')
            
#             if 'Quadruplet' in entry:
#                 for quad in entry['Quadruplet']:
#                     aspect = quad.get('Aspect', 'NULL')
#                     if aspect == "NULL":
#                         target = quad.get('Category', 'general').replace("#", " ")
#                     else:
#                         target = aspect
                    
#                     try:
#                         val, aro = map(float, quad.get('VA', '5.0#5.0').split('#'))
#                     except:
#                         val, aro = 5.0, 5.0

#                     data_list.append({
#                         'Text': text, 
#                         'Target': str(target),
#                         'Valence': val, 
#                         'Arousal': aro
#                     })
#     return data_list

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--model_path", type=str, required=True)
#     parser.add_argument("--model_name", type=str, default="jhu-clsp/mmBERT-base")
#     parser.add_argument("--data_path", type=str, required=True)
#     parser.add_argument("--batch_size", type=int, default=32)
#     parser.add_argument("--max_len", type=int, default=128)
#     args = parser.parse_args()

#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Evaluating on {device}")

#     try:
#         tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
#     except:
#         print("Local tokenizer missing." + args.model_name + " will be used.")
#         tokenizer = AutoTokenizer.from_pretrained(args.model_name, use_fast=True)

#     if tokenizer.pad_token is None:
#         tokenizer.pad_token = tokenizer.eos_token
#         tokenizer.padding_side = "right"

#     try:
#         model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
#         model.to(device)
#         model.eval()
#     except Exception as e:
#         print(f"Error loading model weights: {e}")
#         return

#     print(f"Loading data from {args.data_path}")
#     raw_data = load_eval_data(args.data_path)
    
#     if len(raw_data) == 0:
#         print("Error: No data found or file is empty.")
#         return

#     dataset = LocalEvalDataset(raw_data, tokenizer, args.max_len)
#     loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
#     print(f"Loaded {len(dataset)} items.")

#     preds_list = []
#     labels_list = []

#     with torch.no_grad():
#         for batch in tqdm(loader, desc="Inference"):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
#             labels = batch['labels']
            
#             if 'token_type_ids' in batch:
#                 token_type_ids = batch['token_type_ids'].to(device)
#                 outputs = model(input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
#             else:
#                 outputs = model(input_ids, attention_mask=attention_mask)

#             preds_list.append(outputs.logits.cpu())
#             labels_list.append(labels.cpu())

#     preds = torch.cat(preds_list, dim=0).numpy()
#     truth = torch.cat(labels_list, dim=0).numpy()

#     preds = expit(preds) * 8 + 1 # scale to [1, 9]

#     pcc_v, _ = pearsonr(preds[:, 0], truth[:, 0])
#     pcc_a, _ = pearsonr(preds[:, 1], truth[:, 1])
    
#     mse_v = mean_squared_error(truth[:, 0], preds[:, 0])
#     mse_a = mean_squared_error(truth[:, 1], preds[:, 1])
    
#     rmse_va = np.sqrt((mse_v + mse_a) / 2)

#     print("-" * 30)
#     print(f"Valence PCC: {pcc_v:.4f}")
#     print(f"Arousal PCC: {pcc_a:.4f}")
#     print(f"Combined RMSE: {rmse_va:.4f}")
#     print("-" * 30)

# if __name__ == "__main__":
#     main()

# # python al.py \
# #     --model_path "/kaggle/working/dimabsa/models/jhu-clsp/mmBERT-base-finetuned-dimabsa-laptop-alltasks/final" \
# #     --data_path "/kaggle/working/dimabsa/data_split/test.jsonl"

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

# --- INTERNAL DATASET CLASS (To ensure it matches training) ---
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
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor([row['Valence'], row['Arousal']], dtype=torch.float)
        }

def load_eval_data(path):
    data_list = []
    with open(path, 'r') as f:
        for line in f:
            entry = json.loads(line)
            # Standardizing extraction logic
            text = entry.get('Text') or entry.get('Sentence')
            entry_id = entry.get('ID')
            
            # Logic for files with Ground Truth
            if 'Quadruplet' in entry:
                for quad in entry['Quadruplet']:
                    aspect = quad.get('Aspect', 'NULL')
                    if aspect == "NULL":
                        target = quad.get('Category', 'general').replace("#", " ")
                    else:
                        target = aspect
                    
                    try:
                        # Parse Raw 1-9 values
                        v, a = map(float, quad.get('VA', '5.0#5.0').split('#'))
                    except:
                        v, a = 5.0, 5.0

                    data_list.append({
                        'ID': entry_id, 'Text': text, 'Target': str(target),
                        'Valence': v, 'Arousal': a
                    })
    return data_list

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--data_path", type=str, required=True)
    parser.add_argument("--output_json", type=str, default="error_analysis.json")
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"--- Evaluating on {device} ---")

    # Load Tokenizer & Model
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    

    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    # Load Data
    raw_data = load_eval_data(args.data_path)
    dataset = LocalEvalDataset(raw_data, tokenizer)
    loader = DataLoader(dataset, batch_size=32, shuffle=False)

    preds_list = []
    labels_list = []

    # Inference
    print("Running Inference...")
    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels = batch['labels']
            
            outputs = model(input_ids, attention_mask=attention_mask)
            logits = outputs.logits
            
            # --- IF USING SIGMOID SCALING, UNCOMMENT THIS ---
            # probs = torch.sigmoid(logits)
            # preds = (probs * 8.0) + 1.0
            # ------------------------------------------------
            # --- IF USING STANDARD REGRESSION, KEEP THIS ---
            preds = logits
            # ------------------------------------------------

            preds_list.append(preds.cpu())
            labels_list.append(labels.cpu())

    preds = torch.cat(preds_list, dim=0).numpy()
    truth = torch.cat(labels_list, dim=0).numpy()
    
    # # Clip to valid range
    # preds = np.clip(preds, 1.0, 9.0)

    # --- ERROR ANALYSIS LOGGING ---
    error_log = []
    for i, row in enumerate(raw_data):
        p_v, p_a = preds[i]
        t_v, t_a = truth[i]
        
        # Calculate Error
        diff_v = abs(p_v - t_v)
        diff_a = abs(p_a - t_a)
        total_error = diff_v + diff_a
        
        error_log.append({
            "ID": row['ID'],
            "Text": row['Text'],
            "Target": row['Target'],
            "True_VA": f"{t_v:.2f}#{t_a:.2f}",
            "Pred_VA": f"{p_v:.2f}#{p_a:.2f}",
            "Error_V": float(f"{diff_v:.4f}"),
            "Error_A": float(f"{diff_a:.4f}"),
            "Total_Error": float(f"{total_error:.4f}")
        })

    # Sort by Biggest Error First
    error_log.sort(key=lambda x: x['Total_Error'], reverse=True)

    # Save
    with open(args.output_json, 'w', encoding='utf-8') as f:
        json.dump(error_log, f, indent=4, ensure_ascii=False)

    print(f"✅ Error analysis saved to: {args.output_json}")
    
    # Calculate Scores
    pcc_v, _ = pearsonr(preds[:, 0], truth[:, 0])
    pcc_a, _ = pearsonr(preds[:, 1], truth[:, 1])
    rmse = np.sqrt(mean_squared_error(truth, preds))
    print(f"Scores -> PCC_V: {pcc_v:.4f} | PCC_A: {pcc_a:.4f} | RMSE: {rmse:.4f}")

if __name__ == "__main__":
    main()