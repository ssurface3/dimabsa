import torch
import os
import pandas as pd
import numpy as np
from transformers import TrainerCallback
def pearson_torch(preds, targets):
    vx = preds - torch.mean(preds)
    vy = targets - torch.mean(targets)
    numerator = torch.sum(vx * vy)
    denominator = torch.sqrt(torch.sum(vx ** 2)) * torch.sqrt(torch.sum(vy ** 2))
    return numerator / (denominator + 1e-8)
def convert_to_cont(tensor):

    pred_bin = torch.argmax(tensor, dim=1)
    pred_cont = (pred_bin.float()* 0.25) + 1.125
    return pred_cont
def convert_to_bin(tensor):

    pred_bin = ((tensor - 1.125) / 0.25).long()
    return pred_bin.clamp(0, 31)

def compute_metrics_bin(eval_pred):

    logits, labels = eval_pred

    if isinstance(logits, np.ndarray):
        logits = torch.from_numpy(logits)
    if isinstance(labels, np.ndarray):
        labels = torch.from_numpy(labels)

    pred_v = logits[:, :32] # imagine an array if 32 numbers : these numbers are probabilites of each bin happening 
    pred_a = logits[:, 32:]

    gold_v = labels[:, 0] 
    gold_a = labels[:, 1]

    pred_cont_v = convert_to_cont(pred_v)
    pred_cont_a = convert_to_cont(pred_a)

    pcc_v = pearson_torch(pred_cont_v, gold_v)
    pcc_a = pearson_torch(pred_cont_a, gold_a)

    sse_v = torch.sum((gold_v - pred_cont_v) ** 2)
    sse_a = torch.sum((gold_a - pred_cont_a) ** 2)

    total_sse = sse_v + sse_a
    n_samples = gold_v.shape[0]

    rmse_va = torch.sqrt(total_sse / n_samples) 

    continous_metrics =   {
        'PCC_V': pcc_v.item(),
        'PCC_A': pcc_a.item(),
        'RMSE_VA': rmse_va.item()
    }

    gold_bins_v = convert_to_bin(gold_v)
    gold_bins_a = convert_to_bin(gold_a)
    ev_bins_v , ev_bins_a = get_continuous_from_bins(pred_v).mean().item(), get_continuous_from_bins(pred_a).mean().item()
    evalue = {
        "ev_bins_v" : ev_bins_v, 
        "ev_bins_a" : ev_bins_a
    }
    dic_v = check_bin_quality(pred_v , gold_bins_v , name = "V")
    dic_a = check_bin_quality(pred_a , gold_bins_a , name = "A")
    return continous_metrics|dic_v|dic_a|evalue
def get_continuous_from_bins(logits, num_bins=32, min_val=1.0, step=0.25):
    probs = torch.softmax(logits, dim=-1)
    bin_centers = torch.linspace(min_val + (step/2), (min_val + (num_bins * step)) - (step/2), num_bins).to(logits.device)
    expected_value = torch.sum(probs * bin_centers, dim=-1)
    return expected_value

def check_bin_quality(logits, labels, name):
    pred_bins = torch.argmax(logits, dim=-1)
    bin_diff = torch.abs(pred_bins - labels).float()

    mae_bins = torch.mean(bin_diff)
    within_1_bin = (bin_diff <= 1).float().mean()
    within_2_bins = (bin_diff <= 2).float().mean()

    return {
        "bin_mae_" + name: mae_bins.item(),
        "acc_plus_minus_1_" + name: within_1_bin.item(),
        "acc_plus_minus_2_"+name: within_2_bins.item()
    }
class BinSanityCheck(TrainerCallback):
    def __init__(self, tokenizer , eval_dataset):
        self.tokenizer = tokenizer
        self.eval_dataset = eval_dataset
    def on_evaluate(self, args, state, control,model, **kwargs):
        print("/n " +'=' * 50)
        print('Evaluation on first 5 samples from eval_dataset')

        device = next(model.parameters()).device
        model.eval()

        indices = [1,2,3,4,5]

        with torch.no_grad():
            for idx in indices:
                item = self.eval_dataset[idx] # row
                inputs = {
                    'input_ids': item['input_ids'].unsqueeze(0).to(device),
                    'attention_mask': item['attention_mask'].unsqueeze(0).to(device)
                }

                outputs = model(**inputs)

                if isinstance(outputs, torch.Tensor):
                    logits_v = outputs[:, :32]
                    logits_a = outputs[:, 32:]
                else:
                    logits_v, logits_a = outputs

                pred_bin_v = torch.argmax(logits_v, dim=1).item()
                pred_bin_a = torch.argmax(logits_a, dim=1).item()

                pred_v = (pred_bin_v * 0.25) + 1.125
                pred_a = (pred_bin_a * 0.25) + 1.125

                real_bin_v = item['labels'][0].item()
                real_bin_a = item['labels'][1].item()
                real_v = (real_bin_v * 0.25) + 1.125
                real_a = (real_bin_a * 0.25) + 1.125

                text = self.tokenizer.decode(item['input_ids'] , skip_special_tokens = True)

                print(f"\nText: {text[:80]}...")
                print(f"Real:  V={real_v:.2f} (Bin {real_bin_v}), A={real_a:.2f} (Bin {real_bin_a})")
                print(f"Pred:  V={pred_v:.2f} (Bin {pred_bin_v}), A={pred_a:.2f} (Bin {pred_bin_a})")
        print("="*50 + "\n")
        model.train()

def save_training_history(trainer, args):

    history = trainer.state.log_history
    df = pd.DataFrame(history)

    os.makedirs("logs", exist_ok=True)
    global_filename = f"logs/{args.output_dir}.csv"
    df.to_csv(global_filename, index=False)

    exp_log_dir = os.path.join(f"./models/{args.output_dir}", "experiment_info")
    os.makedirs(exp_log_dir, exist_ok=True)
    exp_filename = os.path.join(exp_log_dir, "training_history.csv")
    df.to_csv(exp_filename, index=False)
    print(f"Training history saved to {exp_filename}")

def create_experiment_dir(output_dir, data, folder_name="experiment_info"):

    target_dir = os.path.join(output_dir, folder_name)
    os.makedirs(target_dir, exist_ok=True)

    file_path = os.path.join(target_dir, "experiment_data.txt")
    with open(file_path, "w") as f:
        if isinstance(data, dict):
            for key, value in data.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write(str(data))

    print(f"Experiment data saved to {file_path}")
    return target_dir
