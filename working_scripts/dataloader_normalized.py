import json
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer
from tqdm import tqdm 

class Dataloader(Dataset):
    def __init__(self, data_source, model, max_len=128):
        self.model = model
        self.max_len = max_len
        self.data = data_source
        self.tokenizer = AutoTokenizer.from_pretrained(
                    self.model, 
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
                            val = (val - 1.0) / 8.0
                            aro = (aro - 1.0) / 8.0
                        except ValueError:
                            val, aro = 0.5, 0.5


                        flattened_data.append({
                            'ID': entry_id, 'Text': text, 'Target': str(target),
                            'Valence': val, 'Arousal': aro
                        })

                elif 'Aspect' in entry:
                    raw_aspects = entry['Aspect']
                    if not isinstance(raw_aspects, list): raw_aspects = [raw_aspects]
                    
                    for single_aspect in raw_aspects:
                        clean_target = str(single_aspect).replace("['", "").replace("']", "").replace("'", "").strip()
                        flattened_data.append({
                            'ID': entry_id, 'Text': text, 'Target': clean_target,
                            'Valence': 0.5, 'Arousal': 0.5
                        })
        return flattened_data

    @classmethod
    def prepare_splits(cls, file_path, tokenizer, max_len=128, test_size=0.1):
        full_data = cls._parse_jsonl(file_path)
        
        train_list, val_list = train_test_split(full_data, test_size=test_size, random_state=42)
        
        return cls(train_list, tokenizer, max_len), cls(val_list, tokenizer, max_len)

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