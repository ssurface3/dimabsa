import argparse
import os
import json
import torch
import pandas as pd
import numpy as np
from transformers import AutoModelForSequenceClassification
from torch.utils.data import DataLoader
from tqdm import tqdm
from dataloader import Dataloader
import config 
def gen_sub(model_path, model_name, batch_size, max_len, output_file, test_data):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    model = AutoModelForSequenceClassification.from_pretrained(model_path)
    model.to(device)
    model.eval()

    files_to_process = []
    if test_data and os.path.isdir(test_data):
        for p in sorted(os.listdir(test_data)):
            if p.lower().endswith('.jsonl') and 'test_task1' in p.lower():
                files_to_process.append(os.path.join(test_data, p))
    elif test_data:
        files_to_process = [test_data]

    if not files_to_process:
        raise SystemExit("No test files found")

    os.makedirs(output_file if os.path.isdir(test_data) else os.path.dirname(output_file) or '.', exist_ok=True)

    for file_path in files_to_process:
        print(f"Processing {file_path} ...")
        
        filename = os.path.basename(file_path).lower()
        domain = "general"
        if "laptop" in filename: domain = "laptop"
        elif "restaurant" in filename: domain = "restaurant"
        elif "finance" in filename: domain = "finance"
        elif "hotel" in filename: domain = "hotel"

        raw_list = Dataloader._parse_jsonl(file_path)
        
        for item in raw_list:
            item['Text'] = "Domain:" + domain + "Text:" + str(item['Text'])

        dataset = Dataloader(raw_list, model_name, max_len=max_len)
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=False)

        results = []
        with torch.no_grad():
            for batch in tqdm(loader, desc=f"Inferring {os.path.basename(file_path)}"):
                input_ids = batch['input_ids'].to(device)
                attention_mask = batch['attention_mask'].to(device)
                
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                preds = outputs.logits.cpu().numpy()
                results.extend(preds)

        submission_map = {}
        csv_data = []

        for i, row in enumerate(raw_list):
            if i >= len(results): break
            
            val_pred = float(results[i][0])
            aro_pred = float(results[i][1])
            
            val_pred = (val_pred * 8.0) + 1.0
            aro_pred = (aro_pred * 8.0) + 1.0
            
            val_pred = np.clip(val_pred, 1.0, 9.0)
            aro_pred = np.clip(aro_pred, 1.0, 9.0)

            val_str = f"{val_pred:.2f}"
            aro_str = f"{aro_pred:.2f}"

            doc_id = row['ID']
            target = row['Target']

            if doc_id not in submission_map:
                submission_map[doc_id] = {"ID": doc_id, "Aspect_VA": []}

            submission_map[doc_id]["Aspect_VA"].append({"Aspect": target, "VA": f"{val_str}#{aro_str}"})
            csv_data.append({"ID": doc_id, "Target": target, "Valence": val_pred, "Arousal": aro_pred})

        final_json = list(submission_map.values())
        
        if os.path.isdir(test_data):
            base = os.path.splitext(os.path.basename(file_path))[0]
            out_json = os.path.join(output_file, f"pred_{base.replace('_test_task1', '')}.jsonl")
        else:
            out_json = output_file if output_file.endswith('.jsonl') else output_file + '.jsonl'

        with open(out_json, 'w', encoding='utf-8') as f:
            for entry in final_json:
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

if __name__ == "__main__":
    print("Starting inference...")
    gen_sub(
        config.model_path,
        config.model_name,
        config.batch_size,
        config.max_len,
        config.output_file,
        config.test_file
    )
    print("Inference complete.")