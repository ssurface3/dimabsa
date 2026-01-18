import torch
import torch.nn as nn
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

        if hasattr(outputs, "logits"):
            preds = outputs.logits
        else:
            preds = outputs

        if isinstance(preds, (tuple, list)):
            logits_v, logits_a = preds
        else:
            # Fallback if model outputs one large tensor
            logits_v = preds[:, :32]
            logits_a = preds[:, 32:]

        target_v = self.score_to_bin(labels[:, 0]).to(logits_v.device)
        target_a = self.score_to_bin(labels[:, 1]).to(logits_a.device)

        loss_v = self.ce_loss(logits_v, target_v)
        loss_a = self.ce_loss(logits_a, target_a)
        
        loss = loss_v + loss_a

        # Concatenate back to (Batch, 64) so compute_metrics can slice it
        logits = torch.cat([logits_v, logits_a], dim=1)

        return (loss, logits) if return_outputs else loss

    def compute_metrics(self, eval_pred):
        logits, labels = eval_pred
        
        logits = torch.tensor(logits)
        labels = torch.tensor(labels)
        
        logits_v = logits[:, :32]
        logits_a = logits[:, 32:]

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