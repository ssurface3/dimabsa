import os
import json
import shutil
import torch
import pandas as pd
import numpy as np
import glob
from transformers import AutoTokenizer, AutoModelForSequenceClassification



MODEL_PATH = "/kaggle/working/models/deberta_v3_large_optimized/final" 

BASE_MODEL_ID = "roberta-large"

DOMAIN = "laptop"
DEV_URLS = {
    "laptop": "https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/subtask_1/eng/eng_laptop_dev_task1.jsonl",
    "restaurant": "https://raw.githubusercontent.com/DimABSA/DimABSA2026/refs/heads/main/task-dataset/track_a/subtask_1/eng/eng_restaurant_dev_task1.jsonl"
}

def get_best_model_path(root_dir):
    if not os.path.exists(root_dir):
        return root_dir
        
    final_path = os.path.join(root_dir, "final")
    if os.path.exists(os.path.join(final_path, "config.json")):
        return final_path
    checkpoints = glob.glob(os.path.join(root_dir, "checkpoint-*"))
    if checkpoints:
        return max(checkpoints, key=os.path.getctime)
    return root_dir

def safe_generate():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Processing: {DOMAIN}")
    
    print(f"Tokenizer: {BASE_MODEL_ID}")
    tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID)
    
    actual_path = get_best_model_path(MODEL_PATH)
    print(f"Weights: {actual_path}")
    model = AutoModelForSequenceClassification.from_pretrained(actual_path)
    model.to(device)
    model.eval()

    url = DEV_URLS[DOMAIN]
    df_dev = pd.read_json(url, lines=True)
    results = []

    print("   Predicting...")
    # 3. Prediction Loop
    for _, row in df_dev.iterrows():
        preds = []
        for aspect in row['Aspect']:
            # Prepare Input
            inputs = tokenizer(row['Text'], aspect, return_tensors="pt", padding=True, truncation=True).to(device)
            
            # --- CRITICAL FIX FOR ROBERTA ---
            # RoBERTa crashes/errors if you pass token_type_ids.
            # We explicitly check model type.
            model_inputs = {'input_ids': inputs['input_ids'], 'attention_mask': inputs['attention_mask']}
            
            # Only pass token_type_ids if it's BERT/DeBERTa/XLNet
            if model.config.model_type in ['bert', 'deberta', 'deberta-v2', 'deberta-v3', 'mobilebert', 'xlnet']:
                if "token_type_ids" in inputs:
                    model_inputs['token_type_ids'] = inputs['token_type_ids']
            # --------------------------------

            with torch.no_grad():
                outputs = model(**model_inputs)
                logits = outputs.logits.squeeze().cpu().tolist()
                
                if isinstance(logits, float): logits = [logits, logits]
                
                # --- SAFETY CHECK FOR NaNs ---
                v = logits[0]
                a = logits[1]
                
                # If NaN, default to Neutral (5.0) to prevent scorer crash
                if np.isnan(v) or np.isinf(v): v = 5.0
                if np.isnan(a) or np.isinf(a): a = 5.0
                
                # Clip to valid range
                v = max(1.0, min(9.0, v))
                a = max(1.0, min(9.0, a))
            
            preds.append({
                "Aspect": aspect, 
                "VA": f"{v:.2f}#{a:.2f}"
            })

        results.append({
            "ID": row['ID'], 
            "Aspect_VA": preds
        })

    # 4. Save to 'subtask_1' folder (Required by Competition)
    output_dir = "subtask_1"
    # Delete folder if exists to start fresh
    if os.path.exists(output_dir):
        shutil.rmtree(output_dir)
    os.makedirs(output_dir, exist_ok=True)
    
    filename = f"pred_eng_{DOMAIN}.jsonl"
    out_path = os.path.join(output_dir, filename)
    
    with open(out_path, 'w', encoding='utf-8') as f:
        for entry in results:
            # ensure_ascii=False fixes weird characters
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    # 5. Zip the folder
    # This creates 'submission.zip' containing the folder 'subtask_1'
    shutil.make_archive("submission", 'zip', root_dir=".", base_dir=output_dir)
    
    print(f"✅ SUCCESS! Created 'submission.zip'.")
    print(f"   Contents: {output_dir}/{filename}")

if __name__ == "__main__":
    safe_generate()