# # import argparse
# # import torch
# # import json
# # import pandas as pd
# # from transformers import AutoTokenizer, AutoModelForSequenceClassification
# # from torch.utils.data import DataLoader
# # from tqdm import tqdm

# # from dataloader import Dataloader

# # parser = argparse.ArgumentParser()
# # parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model folder")
# # parser.add_argument("--test_data", type=str, default="data/test.jsonl")
# # parser.add_argument("--output_file", type=str, default="submission.json")
# # args = parser.parse_args()

# # def predict():
# #     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# #     print(f"Inference on {device}")

# #     print(f"Loading model from: {args.model_path}")
# #     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
# #     model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
# #     model.to(device)
# #     model.eval() 


# #     print("Processing test data")
# #     test_dataset = Dataloader(args.test_data, tokenizer)
# #     raw_test_data = test_dataset.data 
    
# #     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
# #     results = []    

# #     print("Predicting")
# #     with torch.no_grad(): 
# #         for batch in tqdm(test_loader):
# #             input_ids = batch['input_ids'].to(device)
# #             attention_mask = batch['attention_mask'].to(device)
            
# #             model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
# #             if "token_type_ids" in batch and batch["token_type_ids"].sum() > 0:
# #                  if model.config.model_type in ['bert', 'deberta', 'deberta-v2', 'deberta-v3', 'mobilebert']:
# #                     model_inputs['token_type_ids'] = batch['token_type_ids'].to(device)
            
# #             outputs = model(**model_inputs)
# #             preds = outputs.logits.cpu().numpy()
# #             results.extend(preds)

# #     submission_map = {} 
# #     ensemble_data = [] 
    
# #     for i, item in enumerate(raw_test_data):
        
# #         val_raw = float(results[i][0])
# #         aro_raw = float(results[i][1])
        
        
# #         va_string = f"{val_raw:.2f}#{aro_raw:.2f}"
        
# #         sent_id = item['ID']
# #         aspect_term = item['Target'] 
        
        
# #         if sent_id not in submission_map:
# #             submission_map[sent_id] = {
# #                 "ID": sent_id,
# #                 "Aspect_VA": []
# #             }
        
# #         submission_map[sent_id]["Aspect_VA"].append({
# #             "Aspect": aspect_term,
# #             "VA": va_string
# #         })
# #         ensemble_data.append({
# #             "ID": sent_id,
# #             "Target": aspect_term,
# #             "Valence": val_raw,
# #             "Arousal": aro_raw
# #         })


# #     final_json_output = list(submission_map.values())


# #     with open(args.output_file, 'w') as f:
# #         for entry in final_json_output:
            
# #             f.write(json.dumps(entry) + "\n")

# #     csv_filename = args.output_file.replace(".json", ".csv")
# #     pd.DataFrame(ensemble_data).to_csv(csv_filename, index=False)
        
# #     print(f"JSON Submission saved to {args.output_file}")
# #     print(f"CSV Raw Scores saved to {csv_filename}")

# # if __name__ == "__main__":
# #     predict()
# import argparse
# import torch
# import json
# import re
# import numpy as np
# import pandas as pd
# from transformers import AutoTokenizer, AutoModelForSequenceClassification
# from torch.utils.data import DataLoader
# from tqdm import tqdm

# # Import our custom dataset logic
# from dataloader import Dataloader

# parser = argparse.ArgumentParser()
# parser.add_argument("--model_path", type=str, required=True, help="Path to the saved model folder")
# parser.add_argument("--test_data", type=str, default="data/test.jsonl")
# parser.add_argument("--output_file", type=str, default="submission.json")
# args = parser.parse_args()

# # --- HELPER FUNCTIONS (From your snippet) ---
# def extract_num(s):
#     """Helper to sort IDs numerically if needed"""
#     m = re.search(r"(\d+)$", str(s))
#     return int(m.group(1)) if m else -1

# def df_to_jsonl(df, out_path):
#     """
#     The Official Logic to convert DataFrame to Competition JSONL format.
#     Groups aspects by ID and writes line-by-line.
#     """
#     # 1. Sort by ID to keep things tidy
#     try:
#         df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
#     except:
#         df_sorted = df # Fallback if IDs are weird

#     # 2. Group by Sentence ID
#     grouped = df_sorted.groupby("ID", sort=False)

#     print(f"Saving {len(grouped)} groups to {out_path}...")

#     with open(out_path, "w", encoding="utf-8") as f:
#         for gid, gdf in grouped:
#             record = {
#                 "ID": gid,
#                 "Aspect_VA": []
#             }
#             for _, row in gdf.iterrows():
#                 record["Aspect_VA"].append({
#                     "Aspect": row["Target"], # We named it 'Target' in dataset.py
#                     "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
#                 })
#             # 3. Write as JSONL (No brackets, one object per line)
#             f.write(json.dumps(record, ensure_ascii=False) + "\n")

# # --- MAIN PREDICTION LOGIC ---
# def predict():
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#     print(f"Inference on {device}")

#     # 1. Load Model & Tokenizer
#     print(f"Loading model from: {args.model_path}")
#     tokenizer = AutoTokenizer.from_pretrained(args.model_path)
#     model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
#     model.to(device)
#     model.eval() 

#     # 2. Load Data (Using our robust Dataloader)
#     print("Processing test data...")
#     test_dataset = Dataloader(args.test_data, tokenizer)
    
#     # CRITICAL: shuffle=False ensures predictions match the dataframe order!
#     test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
#     # 3. Predict
#     all_preds = []
#     print("Predicting...")
#     with torch.no_grad(): 
#         for batch in tqdm(test_loader):
#             input_ids = batch['input_ids'].to(device)
#             attention_mask = batch['attention_mask'].to(device)
            
#             # Handle token_type_ids for models that need it (BERT/DeBERTa)
#             model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
#             if "token_type_ids" in batch and batch["token_type_ids"].sum() > 0:
#                  if model.config.model_type in ['bert', 'deberta', 'deberta-v2', 'deberta-v3', 'mobilebert']:
#                     model_inputs['token_type_ids'] = batch['token_type_ids'].to(device)
            
#             outputs = model(**model_inputs)
#             all_preds.append(outputs.logits.cpu().numpy())

#     # 4. Stack Results
#     preds = np.vstack(all_preds) # Shape: (N, 2)
    
#     # 5. Assign to DataFrame
#     # We convert the list of dicts (from dataset.py) back to a DataFrame
#     predict_df = pd.DataFrame(test_dataset.data)
    
#     predict_df["Valence"] = preds[:, 0]
#     predict_df["Arousal"] = preds[:, 1]

#     # 6. Save using the Official Function
#     df_to_jsonl(predict_df, args.output_file)
    
#     # 7. Save Raw CSV for Ensembling (Backup)
#     csv_filename = args.output_file.replace(".json", ".csv")
#     predict_df.to_csv(csv_filename, index=False)
#     print(f"✅ JSONL saved to {args.output_file}")
#     print(f"✅ CSV saved to {csv_filename}")

# if __name__ == "__main__":
#     predict()

import argparse
import torch
import json
import re
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataloader import Dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--test_data", type=str, default="data/test.jsonl")
parser.add_argument("--output_file", type=str, default="predictions.json") 
args = parser.parse_args()

def extract_num(s):
    m = re.search(r"(\d+)$", str(s))
    return int(m.group(1)) if m else -1

def predict():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Inference on {device}")

    # 1. Load Resources
    print(f"Loading model: {args.model_path}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval() 

    print("Processing test data")
    test_dataset = Dataloader(args.test_data, tokenizer)
    
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
    
    all_preds = []
    print("Predicting")
    with torch.no_grad(): 
        for batch in tqdm(test_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            
            model_inputs = {'input_ids': input_ids, 'attention_mask': attention_mask}
            if "token_type_ids" in batch and batch["token_type_ids"].sum() > 0:
                 if model.config.model_type in ['bert', 'deberta', 'deberta-v2', 'deberta-v3', 'mobilebert']:
                    model_inputs['token_type_ids'] = batch['token_type_ids'].to(device)
            
            outputs = model(**model_inputs)
            all_preds.append(outputs.logits.cpu().numpy())
    preds = np.vstack(all_preds)
    
    
    preds = np.clip(preds, 1.00, 9.00)


    
    df = pd.DataFrame(test_dataset.data)
    df["Valence"] = preds[:, 0]
    df["Arousal"] = preds[:, 1]

    
    try:
        df_sorted = df.sort_values(by="ID", key=lambda x: x.map(extract_num))
    except:
        df_sorted = df

    grouped = df_sorted.groupby("ID", sort=False)
    
    print(f"Saving {len(grouped)} unique IDs to {args.output_file}...")

    with open(args.output_file, "w", encoding="utf-8") as f:
        for gid, gdf in grouped:
            record = {
                "ID": gid,
                "Aspect_VA": []
            }
            for _, row in gdf.iterrows():
                record["Aspect_VA"].append({
                    "Aspect": row["Target"],
                    "VA": f"{row['Valence']:.2f}#{row['Arousal']:.2f}"
                })
            
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"File saved: {args.output_file}")

if __name__ == "__main__":
    predict()