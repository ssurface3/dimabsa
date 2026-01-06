import json
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import pandas as pd 
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split

class Dataloader(Dataset):
    def __init__(self,pathx):
        self.path = pathx
        self.max_len
    def load_data(self) -> None:
        """
        just creates pd.Series for each of them 
        """
        flattened_data = []
        

        with open(self.path, 'r') as f:
            for line in f:
                entry = json.loads(line)
                for quad in entry['Quadruplet']:
                    row = {
                        'ID': entry['ID'],
                        'Text': entry['Text'],
                        'Aspect': quad.get('Aspect'),
                        'Opinion': quad.get('Opinion'),
                        'Category': quad.get('Category'),
                        'VA_Raw': quad.get('VA')
                    }
                    flattened_data.append(row)

        df = pd.DataFrame(flattened_data)

        df[['Valence', 'Arousal']] = df['VA_Raw'].str.split('#', expand=True).astype(float)

        self.text = df.Text
        self.aspect = df.Aspect 
        self.opinion = df.Opinion
        self.val = df.Valence 
        self.aro = df.Arousal 
        self.df = df 
        return None 
    def __init_parmas(self, json_file): 
        """
        Initialuizes saved params for training if needed 
        
        :param self:
        :param json_file: json file with parameters 
        """
        with open(json_file + ".json", "r") as f:
            self.config = json.load(f)
    def tt_split(self, percent = 0.8): 
        self.load_data()
        return train_test_split(self.df, test_size=0.2, random_state=42)

    def encodimg(self,model_name):

        """
        Tokenize the data using the respectable tokenizer 
        
        :param self: Описание
        :param model_name: the name of the model that uses the tokiner 
        """
        self.load_data()

        self.tokenizer = AutoTokenizer.from_pretrained(model_name)

        self.encoding = self.tokenizer(
            self.text,
            self.aspect,
            max_length=self.max_len,
            padding='max_length',
            truncation=True,
            return_tensors='pt'
        )
        return {
            'input_ids': self.encoding['input_ids'].flatten(),
            'attention_mask': self.encoding['attention_mask'].flatten(),
            'token_type_ids': self.encoding['token_type_ids'].flatten(),
            'labels': torch.tensor(self.text.index, dtype=torch.float) 
        }
    