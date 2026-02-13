import json
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm 

class Dataloader(Dataset):
    def __init__(self, data_source, model_name, max_len=128):
        self.max_len = max_len
        self.data = data_source
        self.model_name = 'jhu-clsp/mmBERT-base'
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name, 
            use_fast=True  
        )

    @staticmethod
    def _parse_jsonl(path):
        flattened_data = []
        with open(path, 'r', encoding='utf-8') as fh:
            total_lines = sum(1 for _ in fh)

        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, desc="Loading Data", unit="lines"):
                line = line.strip()
                if not line:
                    continue
                entry = json.loads(line)

                entry_id = entry.get('ID')
                text = entry.get('Text') or entry.get('Sentence') or ""
                
                quads = []
                if 'Quadruplet' in entry:
                    quads = entry['Quadruplet']
                elif 'Aspect_VA' in entry:
                    quads = entry['Aspect_VA']
                
                if quads:
                    for quad in quads:
                        aspect = quad.get('Aspect', 'NULL')
                        if aspect == "NULL":
                            raw_cat = quad.get('Category', 'general')
                            target = raw_cat.replace("#", " ")
                        else:
                            target = aspect
                        
                        va_string = quad.get('VA', '5.0#5.0')
                        try:
                            val, aro = map(float, va_string.split('#'))
                        except ValueError:
                            val, aro = 5.0, 5.0

                        flattened_data.append({
                            'ID': entry_id, 
                            'Text': text, 
                            'Target': str(target),
                            'Valence': val, 
                            'Arousal': aro
                        })

                elif 'Aspect' in entry:
                    raw_aspects = entry['Aspect']
                    if not isinstance(raw_aspects, list): 
                        raw_aspects = [raw_aspects]
                    
                    for single_aspect in raw_aspects:
                        item_str = str(single_aspect).strip()
                        if item_str.startswith("['") and item_str.endswith("']"):
                            clean_target = item_str[2:-2]
                        else:
                            clean_target = item_str.replace("['", "").replace("']", "").replace("'", "").strip()
                        
                        flattened_data.append({
                            'ID': entry_id, 
                            'Text': text, 
                            'Target': clean_target,
                            'Valence': 5.0, 
                            'Arousal': 5.0
                        })
        return flattened_data

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