import torch
import torch.nn as nn
from transformers import Trainer
class CustomTrainer(Trainer): 
    def __init__(self, *args , **kwargs):
        super().__init__(*args , **kwargs)

    def pearson_torch(slf, preds, targets):
        vx = preds - torch.mean(preds)
        vy = targets - torch.mean(targets)

        numerator = torch.sum(vx * vy)

        denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
        return numerator / (denominator + 1e-8)

    def compute_metrics(self,eval_pred):
        logits, labels = eval_pred
        
        pred_v = logits[:, 0]
        pred_a = logits[:, 1]
        
        gold_v = labels[:, 0]
        gold_a = labels[:, 1]


        pcc_v = self.pearson_torch(pred_v, gold_v)
        pcc_a = self.pearson_torch(pred_a, gold_a)

        sse_v = torch.sum((gold_v - pred_v) ** 2)
        sse_a = torch.sum((gold_a - pred_a) ** 2)
           
        total_sse = sse_v + sse_a
    
        n_samples = gold_v.shape[0]
        
        rmse_va = torch.sqrt(total_sse / n_samples)
        
        return {
            'PCC_V': pcc_v.item(),
            'PCC_A': pcc_a.item(),
            'RMSE_VA': rmse_va.item()
        }
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")

        logits_v = logits[:, 0:32]
        logits_a = logits[:, 32:64]

        labels_v = labels[:, 0]
        labels_a = labels[:, 1]


        loss_fct = nn.CrossEntropyLoss()
        
        loss_v = loss_fct(logits_v, labels_v)
        loss_a = loss_fct(logits_a, labels_a)

        loss = loss_v + loss_a

        return (loss, outputs) if return_outputs else loss
    def get_train_dataloader(self):
        """
        do no forget to rewerite it ;lul
        """
        return super().get_train_dataloader()