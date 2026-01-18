import torch
import torch.nn as nn
import numpy as np
from transformers import Trainer

class CustomTrainer(Trainer): 
    def __init__(self, *args , **kwargs):
        super().__init__(*args , **kwargs)
        self.ce_loss = nn.CrossEntropyLoss()
        self.min_val = 1.0
        self.max_val = 9.0
        self.num_bins = 32
        self.bin_width = (self.max_val - self.min_val) / self.num_bins

    def score_to_bin(self, scores):
        bins = torch.floor((scores - self.min_val) / self.bin_width).long()
        return torch.clamp(bins, 0, self.num_bins - 1)

    def bin_to_score(self, bin_indices):
        return self.min_val + (bin_indices.float() * self.bin_width) + (self.bin_width / 2.0)

    def pearson(self, preds, targets):
        vx = preds - torch.mean(preds)
        vy = targets - torch.mean(targets)
        numerator = torch.sum(vx * vy)
        denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
        return numerator / (denominator + 1e-8)

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)

        if hasattr(outputs, "logits") and isinstance(outputs.logits, tuple):
            logits_v, logits_a = outputs.logits
        elif isinstance(outputs, tuple):
            logits_v, logits_a = outputs[0], outputs[1]
        else:
            logits_v, logits_a = outputs

        target_v = self.score_to_bin(labels[:, 0]).to(logits_v.device)
        target_a = self.score_to_bin(labels[:, 1]).to(logits_a.device)

        loss_v = self.ce_loss(logits_v, target_v)
        loss_a = self.ce_loss(logits_a, target_a)

        loss = loss_v + loss_a

        return (loss, (logits_v, logits_a)) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        predictions, labels = eval_pred
        
        logits_v, logits_a = predictions
        
        logits_v = torch.tensor(logits_v)
        logits_a = torch.tensor(logits_a)
        labels = torch.tensor(labels)

        pred_bin_v = torch.argmax(logits_v, dim=1)
        pred_bin_a = torch.argmax(logits_a, dim=1)

        pred_v = self.bin_to_score(pred_bin_v)
        pred_a = self.bin_to_score(pred_bin_a)
        
        gold_v = labels[:, 0]
        gold_a = labels[:, 1]
    
        pcc_v = self.pearson(pred_v, gold_v)
        pcc_a = self.pearson(pred_a, gold_a)

        sse_v = torch.sum((gold_v - pred_v) ** 2)
        sse_a = torch.sum((gold_a - pred_a) ** 2)
        
        rmse_va = torch.sqrt((sse_v + sse_a) / gold_v.shape[0])
        
        return {
            'PCC_V': pcc_v.item(),
            'PCC_A': pcc_a.item(),
            'RMSE_VA': rmse_va.item()
        }