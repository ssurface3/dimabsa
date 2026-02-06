import torch
import torch.nn as nn
from transformers import Trainer
class CustomTrainer(Trainer): 
    def __init__(self, *args , **kwargs):
        super().__init__(*args , **kwargs)

    def pearson_torch(self, preds, targets):
        vx = preds - torch.mean(preds)
        vy = targets - torch.mean(targets)

        numerator = torch.sum(vx * vy)

        denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
        return numerator / (denominator + 1e-8)

    # do not forget to pass the compute_metrics !
    def loss_fct(self, pred_v, pred_a, gold_v, gold_a):
        pcc_v = self.pearson_torch(pred_v, gold_v)
        pcc_a = self.pearson_torch(pred_a, gold_a)
        
        rmse_va = torch.sqrt(torch.mean((gold_v - pred_v)**2 + (gold_a - pred_a)**2))
        
        
        return rmse_va + (1 - pcc_v) + (1 - pcc_a)
    def convert_to_bin_indices(self, continuous_labels):
        return ((continuous_labels - 1.125) / 0.25).long().clamp(0, 31)

    def get_soft_predictions(self, logits):
        probs = torch.softmax(logits, dim=-1)
        bin_centers = torch.linspace(1.125, 8.875, 32).to(logits.device)
        return torch.sum(probs * bin_centers, dim=-1)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels") 
        

        outputs = model(**inputs)
        
        logits = outputs 

        pred_v = self.get_soft_predictions(logits[:, :32])
        pred_a = self.get_soft_predictions(logits[:, 32:])

        gold_v = labels[:, 0]
        gold_a = labels[:, 1]


        ce_v = nn.CrossEntropyLoss()(logits[:, :32], self.convert_to_bin_indices(gold_v))
        ce_a = nn.CrossEntropyLoss()(logits[:, 32:], self.convert_to_bin_indices(gold_a))
        
        custom_loss = self.loss_fct(pred_v, pred_a, gold_v, gold_a)

        total_loss = custom_loss * 0.9 + 0.1 * (ce_v + ce_a)

        return (total_loss, {"logits": logits}) if return_outputs else total_loss
    def get_train_dataloader(self):
        """
        do no forget to rewerite it ;lul
        """
        return super().get_train_dataloader()