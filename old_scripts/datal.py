import json
import pandas as pd
import torch
class Dataloader(): 
    def __init__(self,tokenizer=None, max_len=128):
        
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data = [] 
    def _load_from_jsonl(self, path):

        flattened_data = []
        try:
            with open(path, 'r') as f:
                for line_num, line in enumerate(f):
                    try:
                        entry = json.loads(line)
                    except:
                        continue

                    entry_id = str(entry.get('ID', ''))
                    text = entry.get('Text', "")
                    
        
                    quads = entry.get('Quadruplet', [])
                    
        
                    if quads:
                        for quad in quads:
                            aspect = quad.get('Aspect', 'NULL')
                            if aspect == "NULL":
                                target = quad.get('Category', 'general').replace("#", " ")
                            else:
                                target = aspect
                            
                            flattened_data.append({
                                'ID': entry_id,
                                'Text': text,
                                'Target': str(target),
                                'VA_Raw': quad.get('VA', '5.0#5.0')
                            })
                    
        
                    elif 'Aspect' in entry:
                        raw_aspects = entry['Aspect']
                        if not isinstance(raw_aspects, list): raw_aspects = [raw_aspects]
                        
                        for single_aspect in raw_aspects:
                            clean_target = str(single_aspect).replace("['", "").replace("']", "").replace("'", "").strip()
                            flattened_data.append({
                                'ID': entry_id,
                                'Text': text,
                                'Target': clean_target,
                                'VA_Raw': '5.0#5.0'
                            })
            
            self.data = flattened_data
            print(f"Loaded {len(self.data)} items from {path}")
            
        except Exception as e:
            print(f"Error loading file {path}: {e}")
            self.data = []

    def new_df(self): 
        if not self.data:
            raise ValueError('not executed or returned the error')
        i = 0 
        self.full_data = self.data
        for json_file in self.data: 
            self.full_data[i]['Valence'] , self.full_data[i]['Arousal'] = json_file['VA_Raw'].split('#')
            i+=1
        return self.full_data