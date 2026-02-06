import torch
import torch.nn as nn
from transformers import Trainer
class CustomTrainer(Trainer): 
    def __init__(self, *args , **kwargs):
        super().__init__(*args , **kwargs)

    def pearson_torch(self, preds, targets):
        preds = preds.float()
        targets = targets.float()
        vx = preds - torch.mean(preds)
        vy = targets - torch.mean(targets)

        numerator = torch.sum(vx * vy)

        denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
        return numerator / (denominator + 1e-8)

    def loss_fct(self, pred_v, pred_a, gold_v, gold_a):
        pcc_v = self.pearson_torch(pred_v, gold_v)
        pcc_a = self.pearson_torch(pred_a, gold_a)

        rmse_va = torch.sqrt(torch.mean((gold_v - pred_v)**2 + (gold_a - pred_a)**2))

        return rmse_va + (1 - pcc_v) + (1 - pcc_a)
    def convert_to_bin_indices(self, continuous_labels):
        return torch.round((continuous_labels - 1.125) / 0.25).long().clamp(0, 31)

    def get_soft_predictions(self, logits):
        probs = torch.softmax(logits, dim=-1)
        bin_centers = torch.linspace(1.125, 8.875, 32).to(logits.device)
        return torch.sum(probs * bin_centers, dim=-1)
    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get("labels") 

        outputs = model(**inputs)

        if hasattr(outputs, "logits"):
            preds = outputs.logits
        else:
            preds = outputs

        if isinstance(preds, (tuple, list)):
            logits_v, logits_a = preds
        elif preds.dim() == 3:
            logits_v = preds[:, 0, :]
            logits_a = preds[:, 1, :]
        else:

            if preds.dim() == 1:
                preds = preds.unsqueeze(0)
            logits_v = preds[:, :32]
            logits_a = preds[:, 32:]

        pred_v = self.get_soft_predictions(logits_v)
        pred_a = self.get_soft_predictions(logits_a)

        gold_v = labels[:, 0]
        gold_a = labels[:, 1]

        ce_v = nn.CrossEntropyLoss()(logits_v, self.convert_to_bin_indices(gold_v))
        ce_a = nn.CrossEntropyLoss()(logits_a, self.convert_to_bin_indices(gold_a))

        custom_loss = self.loss_fct(pred_v, pred_a, gold_v, gold_a)

        total_loss = custom_loss * 0.9 + 0.1 * (ce_v + ce_a)

        if return_outputs:
            logits = torch.cat([logits_v, logits_a], dim=-1)

            if logits.dim() == 1:
                logits = logits.unsqueeze(0)
            elif logits.dim() == 0:
                logits = logits.unsqueeze(0).unsqueeze(0)

            return (total_loss, logits)

        return total_loss

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None):

        has_labels = all(inputs.get(k) is not None for k in ["labels"])
        inputs = self._prepare_inputs(inputs)

        with torch.no_grad():
            if has_labels:
                loss, outputs = self.compute_loss(model, inputs, return_outputs=True)
            else:
                loss = None
                with torch.no_grad():
                    outputs = model(**inputs)
                if hasattr(outputs, "logits"):
                    outputs = outputs.logits

            if prediction_loss_only:
                return (loss, None, None)

            if isinstance(outputs, torch.Tensor):
                if outputs.dim() == 1:
                    outputs = outputs.unsqueeze(0)

            labels = inputs.get("labels")
            if labels is not None and labels.dim() == 1:
                labels = labels.unsqueeze(0)

        return (loss, outputs, labels)

    def get_train_dataloader(self):

        return super().get_train_dataloader()