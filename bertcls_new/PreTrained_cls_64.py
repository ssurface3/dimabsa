from transformers import PreTrainedModel, AutoModel
import torch
import torch.nn as nn

class TwoHeadModel(PreTrainedModel):
    base_model_prefix = "model" #  ??????
    def __init__(self, config):
        super().__init__(config)
        self.bert = AutoModel.from_config(config)
        self.valence_head = nn.Sequential(
            nn.Dropout(p = 0.1) ,
            nn.Linear(config.hidden_size , 768 ),
            nn.GELU(),  # instead of Tanh because Bert uses GELU
            nn.Linear(768 , 32)
        )
        self.arousal_head = nn.Sequential(
            nn.Dropout(p = 0.1) ,
            nn.Linear(config.hidden_size , 768 ),
            nn.GELU(), 
            nn.Linear(768 , 32)
        )

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        inputs = {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
        }
        if token_type_ids is not None and hasattr(self.bert, "embeddings") and hasattr(self.bert.embeddings, "token_type_embeddings"):
             inputs["token_type_ids"] = token_type_ids

        outputs = self.bert(**inputs)
        cls_output = outputs.last_hidden_state[: , 0  , :] # gets cls toekn

        valence_logits = self.valence_head(cls_output)
        arousal_logits = self.arousal_head(cls_output)

        logits = torch.cat((valence_logits, arousal_logits), dim=-1)
        return logits

