import json
import torch
import ast
import re
import os
from torch.utils.data import Dataset
from transformers import AutoTokenizer
from tqdm import tqdm 

class Dataloader(Dataset):
    def __init__(self, data_source, tokenizer, max_len=128):
        if isinstance(tokenizer, str):
            try:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=True)
            except:
                self.tokenizer = AutoTokenizer.from_pretrained(tokenizer, use_fast=False)
        else:
            self.tokenizer = tokenizer
            
        self.max_len = max_len
        self.data = []
        
        if isinstance(data_source, str):
            self.data = self._parse_jsonl(data_source)
        elif isinstance(data_source, list):
            self.data = data_source

    @staticmethod
    def _parse_jsonl(path, filter_lang=None):
        flattened_data = []
        chinese_pattern = re.compile(r'[\u4e00-\u9fff]')

        if not os.path.exists(path):
            return []

        try:
            with open(path, 'r', encoding='utf-8') as f:
                total_lines = sum(1 for _ in f)
        except:
            total_lines = 0
            
        with open(path, 'r', encoding='utf-8') as f:
            for line in tqdm(f, total=total_lines, unit="lines"):
                line = line.strip()
                if not line: continue
                
                try:
                    entry = json.loads(line)
                except json.JSONDecodeError:
                    continue

                raw_text = entry.get('Text') or entry.get('Sentence')
                if not raw_text: continue
                
                text = str(raw_text)

                if filter_lang == 'zho':
                    if not chinese_pattern.search(text):
                        continue
                elif filter_lang == 'eng':
                    if chinese_pattern.search(text):
                        continue

                entry_id = entry.get('ID')
                
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
                    
                    for item in raw_aspects:
                        item_str = str(item).strip()
                        if item_str.startswith("['") and item_str.endswith("']"):
                            clean_target = item_str[2:-2]
                        elif item_str.startswith("[\"") and item_str.endswith("\"]"):
                            clean_target = item_str[2:-2]
                        else:
                            clean_target = item_str
                        
                        flattened_data.append({
                            'ID': entry_id, 'Text': text, 'Target': clean_target,
                            'Valence': 0.5, 'Arousal': 0.5
                        })
        
        return flattened_data

    @classmethod
    def prepare_splits(cls, file_path, tokenizer, max_len=128, test_size=0.1, filter_lang=None):
        full_data = cls._parse_jsonl(file_path, filter_lang=filter_lang)
        
        if not full_data:
            return None, None

        from sklearn.model_selection import train_test_split
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