from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

class QwenDataset(Dataset):
    def __init__(self, model, data, system_instruction=None, max_len=256, inference_mode=False):
        self.inference_mode = inference_mode
        self.ignore_index = -100
        self.data = data
        self.max_len = max_len
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, 
            trust_remote_code=True, 
            padding_side='right'
        )
        self.tokenizer.pad_token = self.tokenizer.eos_token
        
        if not system_instruction:
            self.task = """Valence–Arousal (VA): Output two real-valued scores (1.00-9.00)."""
            self.system_instruction = self.task + "\nAnalyze the text, provide your reasoning inside <think> tags, and then output the Valence#Arousal scores."

    def __len__(self):
        return len(self.data)

    def get_collator(self):
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=True,
            label_pad_token_id=self.ignore_index
        )

    @staticmethod
    def domain_retrieval(path_name: str):
        path_name = path_name.lower()
        if "restaraunt" in path_name or "restaurant" in path_name:
            return 'restaurant'
        elif "laptop" in path_name:
            return "laptop"
        elif "finance" in path_name:
            return "finance"
        else:
            return "general"

    @staticmethod
    def _parse_jsonl(path):
        flattened_data = []
        current_domain = QwenDataset.domain_retrieval(path)
        print("Parsing JSONL data from:", path)
        
        with open(path, 'r', encoding='utf-8') as f:
            lines = f.readlines()
            
        for line in tqdm(lines, desc="Loading Data"):
            if not line.strip(): continue
            entry = json.loads(line)
            
            reasoning = entry.get('Reasoning', "Analysis of sentiment based on keywords and context.")

            if 'Quadruplet' in entry:
                for quad in entry['Quadruplet']:
                    aspect = quad.get('Aspect', 'NULL')
                    target = quad.get('Category', 'general').replace("#", " ") if aspect == "NULL" else aspect
                    try:
                        val, aro = map(float, quad.get('VA', '5.0#5.0').split('#'))
                    except:
                        val, aro = 5.0, 5.0
                    
                    flattened_data.append({
                        'Domain': current_domain, 'Text': entry.get('Text'), 'Target': str(target),
                        'Valence': val, 'Arousal': aro, 'Reasoning': reasoning
                    })
            else:
                flattened_data.append({
                    'Domain': current_domain, 'Text': entry.get('Text'), 'Target': "General",
                    'Valence': 5.0, 'Arousal': 5.0, 'Reasoning': reasoning
                })
        return flattened_data 

    def __getitem__(self, idx):
        row = self.data[idx]
        
        system_part = f"<|im_start|>system\n{self.system_instruction}<|im_end|>\n"
        user_content = f"Domain: {row['Domain']}\nText: {row['Text']}\nTarget: {row['Target']}"
        user_part = f"<|im_start|>user\n{user_content}<|im_end|>\n"
        assistant_header = "<|im_start|>assistant\n"
        
        prompt_str = system_part + user_part + assistant_header

        if self.inference_mode:
            encoding = self.tokenizer(
                prompt_str,
                truncation=True,
                max_length=self.max_len,
                padding=False,
                return_tensors="pt"
            )
            return {
                "input_ids": encoding["input_ids"].squeeze(0),
                "attention_mask": encoding["attention_mask"].squeeze(0)
            }

        reasoning_text = row.get('Reasoning', "Analyzing sentiment...")
        response_str = f"<think>\n{reasoning_text}\n</think>\n{row['Valence']}#{row['Arousal']}<|im_end|>"
        
        full_str = prompt_str + response_str
        
        encoding = self.tokenizer(
            full_str,
            truncation=True,
            max_length=self.max_len,
            padding="max_length",
            return_tensors="pt"
        )
        
        input_ids = encoding["input_ids"].squeeze(0)
        attention_mask = encoding["attention_mask"].squeeze(0)
        labels = input_ids.clone()

        prompt_enc = self.tokenizer(
            prompt_str,
            truncation=True,
            max_length=self.max_len,
            padding=False, 
            return_tensors="pt"
        )
        prompt_len = prompt_enc["input_ids"].shape[1]

        labels[:prompt_len] = self.ignore_index
        labels[attention_mask == 0] = self.ignore_index

        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels 
        }