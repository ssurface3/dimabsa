import numpy as np 
import torch

def convert_task_1_data(self, true_ , predict_): 
    """


    """
    max_len  = len(true_)
    gold_v, gold_a, pred_v, pred_a= torch.zeros(max_len,1) ,torch.zeros(max_len,1) ,torch.zeros(max_len,1) , torch.zeros(max_len,1)



#_______________________________________________________________
def convert_task1_data(gold_data, pred_data):
    gold_data = {entry['ID']: entry for entry in gold_data}
    pred_data = {entry['ID']: entry for entry in pred_data}
    gold_v, gold_a, pred_v, pred_a=[], [], [], []
    for key, value in gold_data.items():
        gold_value = value["Aspect_VA"]
        if key not in pred_data:
            exit("Error: VA value is missing!")
        pred_value = pred_data[key]["Aspect_VA"]
        pred_value = {entry['Aspect']: entry for entry in pred_value}
        for item in gold_value:
            gold_va = item['VA'].split("#")
            gold_v.append(eval(gold_va[0]))
            gold_a.append(eval(gold_va[1]))
            if item['Aspect'] in pred_value:
                pred_va = pred_value[item['Aspect']]["VA"].split("#")
                pred_v.append(eval(pred_va[0]))
                pred_a.append(eval(pred_va[1]))
            else:
                # pred_v.append(0)
                # pred_a.append(0)
                exit("Error: VA value is missing!")
    return gold_v, gold_a, pred_v, pred_a
def evaluate_predictions_task1(gold_data, pred_data, is_norm = True):
    if not gold_data or not pred_data:
        print("Error: Failed to load one or both data files. Cannot perform evaluation.")
        return None
    
    gold_v, gold_a, pred_v, pred_a = convert_task1_data(gold_data, pred_data)
    if not (all(1 <= x <= 9 for x in pred_v) and all(1 <= x <= 9 for x in pred_a)):
        print(f"Warning: Some predicted values are out of the numerical range.")
    pcc_v = pearsonr(pred_v,gold_v)[0]
    pcc_a = pearsonr(pred_a,gold_a)[0]
    
    gold_va = gold_v + gold_a
    pred_va = pred_v + pred_a
    def rmse_norm(gold_va, pred_va, is_normalization = True):
        result = [(a - b)**2 for a, b in zip(gold_va, pred_va)]
        # if is_normalization:
        #     return math.sqrt(sum(result)/len(gold_v))/math.sqrt(128)
        return math.sqrt(sum(result)/len(gold_v))
    rmse_va = rmse_norm(gold_va, pred_va, is_norm)
    return {
        'PCC_V': pcc_v,
        'PCC_A': pcc_a,
        'RMSE_VA': rmse_va,
    }


# --- Main Program ---
if __name__ == "__main__":
    task = args.task
    # Specify the path to your JSONL file
    gold_file_path = args.gold_data_path  # Replace with your gold/standard file path
    pred_file_path = args.pred_data_path                # Replace with your prediction file path

    print("Loading gold data...")
    gold_data = read_jsonl_file(gold_file_path, task = task, data_type="gold")
    print("Loading prediction data...")
    pred_data = read_jsonl_file(pred_file_path, task = task)
    
    # Evaluate predictions
    if task == 1:
        results = evaluate_predictions_task1(gold_data, pred_data, is_norm = args.do_norm)
    else:
        results = evaluate_predictions(gold_data, pred_data, task = task)
    # You can use 'results' for further analysis or reporting
    if results:
        print(f"\nFinal Results: {results}")