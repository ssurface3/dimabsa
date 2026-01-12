import argparse
import json
import torch
import pandas as pd
from torch.utils.data import DataLoader
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from tqdm import tqdm
from dataloader import Dataloader

parser = argparse.ArgumentParser()
parser.add_argument("--model_path", type=str, required=True)
parser.add_argument("--test_data", type=str, default="data/test.jsonl")
parser.add_argument("--output_file", type=str, default="submission.json")
parser.add_argument("--batch_size", type=int, default=32)
args = parser.parse_args()

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_path)
    model.to(device)
    model.eval()

    dataset = Dataloader(args.test_data, tokenizer)
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False)
    
    raw_data = dataset.data
    results = []

    with torch.no_grad():
        for batch in tqdm(loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            token_type_ids = batch.get('token_type_ids', None)
            
            if token_type_ids is not None:
                token_type_ids = token_type_ids.to(device)
                outputs = model(input_ids=input_ids, attention_mask=attention_mask, token_type_ids=token_type_ids)
            else:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
            
            preds = outputs.logits.cpu().numpy()
            results.extend(preds)

    submission_map = {}
    csv_data = []

    for i, row in enumerate(raw_data):
        val_pred = float(results[i][0])
        aro_pred = float(results[i][1])
        
        val_str = f"{val_pred:.2f}"
        aro_str = f"{aro_pred:.2f}"
        
        doc_id = row['ID']
        target = row['Target']

        if doc_id not in submission_map:
            submission_map[doc_id] = {
                "ID": doc_id,
                "Aspect_VA": []
            }
        
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

    final_json = list(submission_map.values())

    with open(args.output_file, 'w') as f:
        json.dump(final_json, f, indent=4)

    csv_filename = args.output_file.replace(".json", ".csv")
    pd.DataFrame(csv_data).to_csv(csv_filename, index=False)

if __name__ == "__main__":
    main()