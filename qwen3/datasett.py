from torch.utils.data import Dataset
import tqdm
import json
from transformers import AutoTokenizer,DataCollatorForSeq2Seq
class QwenDataset(Dataset):
    def __init__(self,model , data, system_instruction = None,max_len= 256):
        self.model = model
        # self.__init_tokenizer()
        self.data = data
        self.max_len = max_len
        self.tokenizer = AutoTokenizer.from_pretrained(
            model, 
            trust_remoute = 'True', 
            padding_side = 'right'
        )
        if not system_instruction:
            self.task = """Valence–Arousal (VA):
                        A pair of real-valued scores, each ranging from 1.00 to 9.00, rounded to two decimal places:
                        Valence (V): Measures the degree of positivity or negativity
                        Arousal (A): Measures the intensity of emotion
                        A score of 1.00 indicates extremely negative valence or very low arousal, 9.00 indicates extremely positive valence or very high arousal, and 5.00 represents neutral valence or medium arousa"""
            self.system_instruction = self.task + """You are a person that has to output two numbers that represent Valence and Arousal as a continious value between 1 and 9 in a way like 5#5"""
        
    def _init_tokenizer(self,text): 
        """
        Helper function to ensure consistent tokenization settings
        """
        return self.tokenizer(
            text,
            truncation=True,   
            max_length = self.max_len      
            padding="max_length",  # ? cannout initialize?? at the start  
            return_tensors="pt",
        )
        self.tokenizer.pad_token  = self.tokenizer.eos_token # because i donnno if qwen 2 has the different one
    def __len__(self):
        return len(self.data)
    def get_collatoral(self):
        """returns the specific collator needed for this dataset"""
        return DataCollatorForSeq2Seq(
            tokenizer=self.tokenizer,
            padding=
        )
    @staticmethod
    def domain_retrieval(path_name:str) -> None:
        map_of_domains = {
            "general",
            "restaraunt", 
            "laptop", 
            "finance"
        }
        for domain in map_of_domains:
            if "restaraunt" in path_name:
                return 'restaraunt'
            elif  "laptop" in path_name:
                return  "laptop"
            elif "finance" in path_name:
                return  "finance"
            else:
                domain = "general"
                print('returned general,maybe there is a problem')
    @staticmethod
    def _parse_jsonl(path):
        flattened_data = []
        current_domain = QwenDataset.domain_retrieval(path)
        print("Parsing JSONL data from:", path)
        with open(path, 'r', encoding='utf-8') as fh:
            total_lines = sum(1 for _ in fh)
        print('loading data from:', path)
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
                        except ValueError:
                            val, aro = 5.0, 5.0
                        
                        flattened_data.append({
                            'ID': entry_id,'Domain' : current_domain , 'Text': text, 'Target': str(target),
                            'Valence': val, 'Arousal': aro
                        })

                elif 'Aspect' in entry:
                    raw_aspects = entry['Aspect']
                    if not isinstance(raw_aspects, list): raw_aspects = [raw_aspects]
                    
                    for single_aspect in raw_aspects:
                        clean_target = str(single_aspect).replace("['", "").replace("']", "").replace("'", "").strip()
                        flattened_data.append({
                            'ID': entry_id, 'Domain' : current_domain, 'Text': text, 'Target': clean_target,
                            'Valence': 5.0, 'Arousal': 5.0
                        })
        print(f"Loaded {len(flattened_data)} samples from {path}")
        return flattened_data 
    def __getitem__(self, idx ):
        if not self.inference_mode():
            row  = self.data[idx]
            # rows have this data:
            # ID , Domain , Text , Target , Valence , Arousal 
            system_part = "<|im_start|>system\n" +self.system_instruction + "<|im_end|>\n"
            
            user_content = f"Domain: {row['Domain']}\nText: {row['Text']}\nWhat is the valence and arousal score?"
            
            user_part = f"<|im_start|>user\n{user_content}<|im_end|>\n"
            
            assistant_header = "<|im_start|>assistant\n" # do not close as we need to generate a response
            
            prompt_str = system_part + user_part + assistant_header
            
            response_str = str(row['Valence']) + "#" + str(row['Arousal'])+ "<|im_end|>"
        
            full_str = prompt_str + response_str
            encoding = self._init_tokenizer(
                full_str, 
            )
            
            
            input_ids = encoding["input_ids"].squeeze(0)
            
            attention_mask = encoding["attention_mask"].squeeze(0)

            labels = input_ids.clone()

            prompt_tokens = self._init_tokenizer(
                prompt_str
            )["input_ids"]
            
            prompt_len = len(prompt_tokens)

            labels[:prompt_len] = self.ignore_index
            labels[attention_mask == 0] = self.ignore_index

            return {
                "input_ids": input_ids,
                "attention_mask": attention_mask,
                "labels": labels 
            }
        # apply chat temo
        if self.inference_mode():
            row = self.data[idx]

            system_part = "<|im_start|>system\n" + self.system_instruction +"<|im_end|>\n"
            user_content = f"Domain:{row['Domain']},text:{row['Text']}; output Valence and Arousal score" 
            user_part = "<|im_start|>user\n" + user_content + "<|im_start|>\n"
            assistant_header = "<|im_start|>assistant\n"
            prompt_str = system_part + user_part + assistant_header
            encoding = self._init_tokenizer(prompt_str
                                    )
            input_ids = encoding["input_ids"].squeeze(0)
            attention_mask = encoding["attention_mask"].squeeze(0)


            return {
                "input_ids" : input_ids ,
                "attention_mask" : attention_mask
            }


# next token pred? 
# lora on qwen is it training or not? 
# на маленькой лоре посмотреть
# 