from transformers import PreTrainedModel, BertModel ,AutoModel 
import torch
import torch.nn as nn

class TwoHeadModel(PreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        self.valence_head = nn.Sequential(
            nn.Dropout(p = 0.1) ,
            nn.Linear(config.hidden_size , 768 ),
            nn.Tanh(), 
            nn.Linear(768 , 32)
        )
        self.arousal_head = nn.Sequential(
            nn.Dropout(p = 0.1) ,
            nn.Linear(config.hidden_size , 768 ),
            nn.Tanh(), 
            nn.Linear(768 , 32)
        )
        self.init_weights() # check if it even works 

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None,labels=None):
       
        model_inputs = {
            'input_ids': input_ids,
            'attention_mask': attention_mask
        }

        # if hasattr(self.bert.config, "type_vocab_size") and self.bert.config.type_vocab_size > 0:
        #     if token_type_ids is not None:
        #         model_inputs['token_type_ids'] = token_type_ids

        outputs = self.bert(**model_inputs)
        # outputs = self.bert(
        #     input_ids,
        #     attention_mask=attention_mask,
        #     token_type_ids=token_type_ids
        # )
        cls_output = outputs.last_hidden_state[: , 0  , :] # gets cls toekn

        valence_logits = self.valence_head(cls_output)
        arousal_logits = self.arousal_head(cls_output)

        # logits = torch.cat((valence_logits, arousal_logits), dim=-1)

        # return logits
        return (valence_logits, arousal_logits)