
# import json
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from sklearn.model_selection import train_test_split
# from transformers import AutoTokenizer

# class Dataloader(Dataset):
#     def __init__(self, source, tokenizer=None, max_len=128):
#         """
#         :param source: Can be a string (file path) OR a pd.DataFrame
#         """
#         self.tokenizer = tokenizer
#         self.max_len = max_len
#         self.data = [] # We will store data as a list of dicts
        
#         # LOGIC: Load data based on input type
#         if isinstance(source, str):
#             # It's a file path, load it
#             self._load_from_jsonl(source)
#         elif isinstance(source, pd.DataFrame):
#             # It's already a DataFrame (from train_test_split)
#             # Convert directly to list to fix the Index/KeyError 3422
#             self.data = source.to_dict('records')
        
#     def _load_from_jsonl(self, path):
#         """Internal helper to parse JSONL"""
#         flattened_data = []
#         with open(path, 'r') as f:
#             for line in f:
#                 entry = json.loads(line)
                
#                 # Handle cases where Quadruplet is missing (inference)
#                 quads = entry.get('Quadruplet', [])
#                 # If no quads, add a dummy one so we still process the text
#                 if not quads:
#                      quads = [{'Aspect': 'NULL', 'Category': 'NULL', 'VA': '0#0'}]

#                 for quad in quads:
#                     aspect = quad.get('Aspect')
#                     category = quad.get('Category')
                    
#                     # NULL Aspect Logic
#                     if aspect == "NULL":
#                         target = category.replace("#", " ") if category else "general"
#                     else:
#                         target = aspect

#                     row = {
#                         'ID': entry.get('ID'),
#                         'Text': entry.get('Text'),
#                         'Target': target,
#                         'VA_Raw': quad.get('VA', '0.0#0.0')
#                     }
#                     flattened_data.append(row)
        
#         # Convert to list of dicts immediately
#         self.data = flattened_data

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         # Accessing by list index [0, 1, 2] is safe. 
#         # Pandas index [3422] is gone.
#         row = self.data[index]
        
#         text = str(row['Text'])
#         target = str(row['Target'])
        
#         # Parse Scores on the fly
#         if 'Valence' in row:
#             val = row['Valence']
#             aro = row['Arousal']
#         else:
#             # Parse from string if not pre-parsed
#             val, aro = map(float, row['VA_Raw'].split('#'))

#         # Tokenize
#         encoding = self.tokenizer(
#             text,
#             target,
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )

#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             # Safe get for models without token_type_ids
#             'token_type_ids': encoding.get('token_type_ids', torch.tensor([])).flatten(),
#             'labels': torch.tensor([val, aro], dtype=torch.float)
#         }
# import json
# import pandas as pd
# import torch
# from torch.utils.data import Dataset
# from transformers import AutoTokenizer

# class Dataloader(Dataset):
#     def __init__(self, pathx, tokenizer=None, max_len=128):
#         self.path = pathx
#         self.max_len = max_len
#         self.tokenizer = tokenizer
#         self.data = []
#         if isinstance(source, pd.DataFrame):
#             # It is a DataFrame (from train_test_split) -> Convert to list of dicts
#             self.data = source.to_dict('records')
#         else:
#             # It is a String (File Path) -> Load from file
#             self._load_from_jsonl(source)
#         # Load immediately
#         self._load_from_jsonl(pathx)

#     def _load_from_jsonl(self, path):
#         flattened_data = []
        
#         with open(path, 'r') as f:
#             for line_num, line in enumerate(f):
#                 try:
#                     entry = json.loads(line)
#                 except:
#                     continue # Skip broken lines

#                 # 1. Detect if this is Laptop or Restaurant data based on ID
#                 # This helps us guess the context if Aspect is missing
#                 entry_id = str(entry.get('ID', ''))
#                 is_laptop = 'lap' in entry_id.lower()
                
#                 # 2. Find the Quadruplets (or create dummy for inference)
#                 quads = entry.get('Quadruplet', [])
#                 # If quads is empty (some test sets), try to find 'Aspect' key directly
#                 if not quads and 'Aspect' in entry:
#                      # Handle flat format
#                      quads = [{'Aspect': entry['Aspect'], 'Category': entry.get('Category'), 'VA': '0#0'}]
                
#                 if not quads:
#                     # If absolutely nothing found, skip or create a general dummy
#                     continue 

#                 for quad in quads:
#                     aspect = str(quad.get('Aspect', 'NULL')).strip()
#                     category = str(quad.get('Category', 'NULL')).strip()
                    
#                     # --- THE FIX: INTELLIGENT TARGET MAPPING ---
#                     target = "general" # Default fallback
                    
#                     if aspect != "NULL" and aspect != "None" and aspect != "":
#                         # Best case: We have an explicit aspect (e.g., "battery")
#                         target = aspect
#                     elif category != "NULL" and category != "None" and category != "":
#                         # Fallback 1: We have a category (e.g., "RESTAURANT#PRICES")
#                         target = category.replace("#", " ").replace("_", " ").lower()
#                     else:
#                         # Fallback 2: Both are NULL. Guess based on dataset type.
#                         if is_laptop:
#                             target = "laptop"
#                         else:
#                             target = "restaurant"
                    
#                     # -------------------------------------------

#                     row = {
#                         'ID': entry_id,
#                         'Text': entry.get('Text', ""),
#                         'Target': target, # This will NEVER be "NULL" now
#                         'VA_Raw': quad.get('VA', '5.0#5.0')
#                     }
#                     flattened_data.append(row)
        
#         self.data = flattened_data
#         print(f"Loaded {len(self.data)} items from {path}")
#         # DEBUG: Print first item to prove it's fixed
#         if len(self.data) > 0:
#             print(f"DEBUG Sample: ID={self.data[0]['ID']} | Target='{self.data[0]['Target']}'")

#     def __len__(self):
#         return len(self.data)

#     def __getitem__(self, index):
#         row = self.data[index]
        
#         text = str(row['Text'])
#         target = str(row['Target'])
        
#         # Parse Scores
#         try:
#             val, aro = map(float, str(row['VA_Raw']).split('#'))
#         except:
#             val, aro = 5.0, 5.0

#         # Tokenize
#         encoding = self.tokenizer(
#             text,
#             target,
#             max_length=self.max_len,
#             padding='max_length',
#             truncation=True,
#             return_tensors='pt'
#         )

#         # Handle token_type_ids for Qwen/RoBERTa vs BERT
#         token_type_ids = encoding.get('token_type_ids', torch.tensor([0] * self.max_len))

#         return {
#             'input_ids': encoding['input_ids'].flatten(),
#             'attention_mask': encoding['attention_mask'].flatten(),
#             'token_type_ids': token_type_ids.flatten(),
#             'labels': torch.tensor([val, aro], dtype=torch.float)
#         }

# # import json
# # import pandas as pd
# # import torch
# # from torch.utils.data import Dataset
# # from transformers import AutoTokenizer

# # class Dataloader(Dataset):
# #     def __init__(self, pathx, tokenizer=None, max_len=128):
# #         self.path = pathx
# #         self.max_len = max_len
# #         self.tokenizer = tokenizer
# #         self.data = []
        
# #         # Load immediately
# #         self._load_from_jsonl(pathx)

# #     def _load_from_jsonl(self, path):
# #         flattened_data = []
        
# #         with open(path, 'r') as f:
# #             for line_num, line in enumerate(f):
# #                 try:
# #                     entry = json.loads(line)
# #                 except:
# #                     continue

# #                 text = entry.get('Text', "")
# #                 entry_id = str(entry.get('ID', ''))

# #                 # --- STRATEGY 1: Training Format (Quadruplet List) ---
# #                 if 'Quadruplet' in entry:
# #                     for quad in entry['Quadruplet']:
# #                         aspect = quad.get('Aspect', 'NULL')
# #                         # Logic: Use Aspect if available, else Category
# #                         if aspect == "NULL":
# #                             target = quad.get('Category', 'general').replace("#", " ")
# #                         else:
# #                             target = aspect
                            
# #                         flattened_data.append({
# #                             'ID': entry_id,
# #                             'Text': text,
# #                             'Target': str(target),
# #                             'VA_Raw': quad.get('VA', '5.0#5.0')
# #                         })

# #                 # --- STRATEGY 2: Test Format (Aspect List) ---
# #                 # THIS IS THE FIX FOR YOUR PROBLEM
# #                 elif 'Aspect' in entry:
# #                     raw_aspects = entry['Aspect']
                    
# #                     # 1. Ensure it is a list. If it's a string "cpu", make it ["cpu"]
# #                     if not isinstance(raw_aspects, list):
# #                         raw_aspects = [raw_aspects]
                        
# #                     # 2. ITERATE through the list (Explode)
# #                     # This splits ["screen", "touchscreen"] into two separate rows!
# #                     for single_aspect in raw_aspects:
                        
# #                         # Clean the string (remove brackets/quotes if they somehow exist in the text)
# #                         clean_target = str(single_aspect).replace("['", "").replace("']", "").replace("'", "").strip()
                        
# #                         flattened_data.append({
# #                             'ID': entry_id,
# #                             'Text': text,
# #                             'Target': clean_target, # Now it is just "screen"
# #                             'VA_Raw': '5.0#5.0'     # Dummy score
# #                         })
                
# #                 else:
# #                     continue

# #         self.data = flattened_data
# #         print(f"✅ Loaded {len(self.data)} items from {path}")

# #     def __len__(self):
# #         return len(self.data)

# #     def __getitem__(self, index):
# #         row = self.data[index]
        
# #         text = str(row['Text'])
# #         target = str(row['Target'])
        
# #         # Parse Scores
# #         try:
# #             val, aro = map(float, str(row['VA_Raw']).split('#'))
# #         except:
# #             val, aro = 5.0, 5.0

# #         encoding = self.tokenizer(
# #             text,
# #             target,
# #             max_length=self.max_len,
# #             padding='max_length',
# #             truncation=True,
# #             return_tensors='pt'
# #         )
        
# #         # Handle models without token_type_ids
# #         token_type_ids = encoding.get('token_type_ids', torch.tensor([0] * self.max_len))

# #         return {
# #             'input_ids': encoding['input_ids'].flatten(),
# #             'attention_mask': encoding['attention_mask'].flatten(),
# #             'token_type_ids': token_type_ids.flatten(),
# #             'labels': torch.tensor([val, aro], dtype=torch.float)
# #         }
