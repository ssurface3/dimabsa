from torch.utils.data import Dataset
from tqdm import tqdm
import json
import torch
from transformers import AutoTokenizer, DataCollatorForSeq2Seq

class QwenDataset(Dataset):
    def __init__(self, model, data, system_instruction=None, max_len=512, inference_mode=False):
        self.inference_mode = inference_mode
        self.ignore_index = -100
        self.data = data
        self.max_len = max_len
        self.model = model
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, 
            trust_remote_code=True, 
            padding_side='right', 

        )
        self.tokenizer.pad_token = self.tokenizer.eos_token # ??
        
        if not system_instruction:
            self.task = """Valence–Arousal (VA): Output two real-valued scores (1.00-9.00). An Aspect Term is a specific opinion target linked to an abstract Entity#Attribute Aspect Category and a sentiment-bearing Opinion Term, characterized by Valence-Arousal scores ranging from 1.00 (negative/low) to 9.00 (positive/high) that measure the degree of positivity and emotional intensity."""
            self.system_instruction = self.task + "\nAnalyze the text and then output the Valence#Arousal scores."

    def __len__(self):
        return len(self.data)
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
                        'Valence': val, 'Arousal': aro , 'Aspect' : aspect 
                    })
            else:
                flattened_data.append({
                    'Domain': current_domain, 'Text': entry.get('Text'), 'Target': "General",
                    'Valence': 5.0, 'Arousal': 5.0 , 'Aspect' : aspect 
                })
        return flattened_data 

    def __getitem__(self, idx):
        row = self.data[idx]
        
        system_content = self.system_instruction

        user_content = f"Domain:{row['Domain']}\nText:{row['Text']}\nTarget:{row['Aspect']}"

        assistant_content  = f"Valence:{row['Valence']}Arousal:{row['Arousal']}"

        message = [
            {"role" :  "system" , "content" : system_content}, 
            {"role":   "user",    "content": user_content}, 
            {"role" :  "assistant" , "content" : assistant_content} , 
        ]
        prompt_message = [ # we need it in order to block model from learning the questions 
                {"role" :  "system" , "content" : system_content}, 
                {"role":   "user",    "content": user_content}
        ]

        if self.inference_mode: 
            message = [
                    {"role" : "system" , "content" : system_content},
                    {"role": "user", "content": user_content}# here we use the content from test to generate?  
                    ] 
            # we need the tokenized prompt to generate 
            prompt = self.tokenizer.apply_chat_template(
                message, 
                add_generation_prompt = True , 
                tokenizer = False, 
                enable_thinking = False
            )

            tokenized_prompt = self.tokenizer(
                prompt,
                truncation=True,
                max_length=self.max_len,
                padding=False, 
                return_tensors="pt"
            )

            return {
                "input_ids" :tokenized_prompt['input_ids'].squeeze(0), 
                "attention_masks" : tokenized_prompt['attention_mask'].squeeze(0)
            }

        full_message_chat = self.tokenizer.apply_chat_template(
            message ,
            enable_thinking = False, 
            add_generation_prompt = False , 
            tokenize = False
        )

        prompt_message_chat = self.tokenizer.apply_chat_template(
            prompt_message, 
            enable_thinking = False, 
            add_generation_prompt = True , 
            tokenize = False
        )

        tokenized_full_message_chat  = self.tokenizer(
            full_message_chat, 
            truncation = True, 
            max_length =self.max_len, 
            padding = 'max_length',
            return_tensors = 'pt'
        )

        attetion_mask = tokenized_full_message_chat['attention_mask'].squeeze(0)
        input_ids = tokenized_full_message_chat['input_ids'].squeeze(0)

        labels = input_ids.clone()

        tokenized_prompt_message_chat = self.tokenizer(
            prompt_message_chat,
            truncation = True, 
            max_length =self.max_len, 
            padding = 'max_length',
            return_tensors = 'pt'
        )
        input_ids_prompt_len = len(tokenized_prompt_message_chat['input_ids'])

        labels[:input_ids_prompt_len] = -100
        
        labels[attetion_mask == 0] = -100

        return {
                "input_ids": input_ids,
                "attention_mask": attetion_mask,
                "labels": labels
            } 

        


        

        