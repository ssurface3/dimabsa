import json
import pandas as pd
import torch
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split

class Dataloader(Dataset):
    def __init__(self, pathx, tokenizer=None, max_len=128):
        self.path = pathx
        self.max_len = max_len
        self.tokenizer = tokenizer
        self.data = [] 
        self.load_data()
        self.split()

    def load_data(self):
        flattened_data = []

        with open(self.path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                entry_id = entry.get('ID')
                text = entry.get('Text')
                
                if 'Quadruplet' in entry:
                    quads = entry['Quadruplet']
                    
                    for quad in quads:

                        aspect = quad.get('Aspect', 'NULL')
                        if aspect == "NULL":
                            raw_cat = quad.get('Category', 'general')
                            target = raw_cat.replace("#", " ")
                        else:
                            target = aspect
                        
                        va_string = quad.get('VA', '5.0#5.0')
                        
                        try:
                            val_str, aro_str = va_string.split('#')
                            val_float = float(val_str)
                            aro_float = float(aro_str)
                        except ValueError:
                            val_float, aro_float = 5.0, 5.0

                        flattened_data.append({
                            'ID': entry_id,
                            'Text': text,
                            'Target': str(target),
                            'Valence': val_float,  
                            'Arousal': aro_float,
                            'VA' : quad.get('VA')
                        })

                elif 'Aspect' in entry:
                    raw_aspects = entry['Aspect']
                    if not isinstance(raw_aspects, list): 
                        raw_aspects = [raw_aspects]
                    
                    for single_aspect in raw_aspects:
                        clean_target = str(single_aspect).replace("['", "").replace("']", "").replace("'", "").strip()
                        
                        flattened_data.append({
                            'ID': entry_id,
                            'Text': text,
                            'Target': clean_target,
                            'Valence': 5.0, 
                            'Arousal': 5.0,
                            'VA' : '5.0#5.0'
                        })

        self.data = flattened_data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        """
        Returns one item for the model to process.
        """
        row = self.data[index]
        
        text = str(row['Text'])
        target = str(row['Target'])
        
       
        encoding = self.tokenizer(
            text,
            target,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )

        labels = torch.tensor([row['Valence'], row['Arousal']], dtype=torch.float)

        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'token_type_ids': encoding.get('token_type_ids', torch.tensor([])).flatten(),
            'labels': labels
        }
    def split(self, percentage = 0.9): 
        full_df = pd.DataFrame(self.data)
        self.train_df, self.val_df = train_test_split(full_df, test_size= 1- percentage, random_state=42)
        return None