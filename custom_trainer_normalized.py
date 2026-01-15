import torch
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

        # here we normalized them as the can be out of boundary
        pred_v = torch.sigmoid(pred_v) * 8 + 1
        pred_a = torch.sigmoid(pred_a) * 8 + 1

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
    def loss_fct(self, logits, labels):
        
        pred_v = logits[:, 0]
        pred_a = logits[:, 1]
        
        gold_v = labels[:, 0] 
        gold_a = labels[:, 1] 

        pred_v = torch.sigmoid(pred_v) * 8 + 1
        pred_a = torch.sigmoid(pred_a) * 8 + 1

        pcc_v = self.pearson_torch(pred_v, gold_v)
        pcc_a = self.pearson_torch(pred_a, gold_a)

        sse_v = torch.sum((gold_v - pred_v) ** 2)
        sse_a = torch.sum((gold_a - pred_a) ** 2)
           
        total_sse = sse_v + sse_a
    
        n_samples = gold_v.shape[0]
        
        rmse_va = torch.sqrt(total_sse / n_samples)
        return rmse_va + 2 - pcc_v - pcc_a

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        loss = self.loss_fct(logits, labels)
        return (loss, outputs) if return_outputs else loss
    def get_train_dataloader(self):

        return 