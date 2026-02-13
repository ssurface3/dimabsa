import argparse
import json
import os
import glob
from tqdm import tqdm

def parse_gold_entry(entry):
    """
    Extracts (ID, Aspect) -> {Valence, Arousal, Text} from a gold line.
    Handles 'Quadruplet', 'Aspect_VA', and 'Aspect' (though 'Aspect' usually lists targets without scores).
    """
    extracted = []
    entry_id = entry.get('ID')
    text = entry.get('Text')
    
    if 'Quadruplet' in entry:
        for quad in entry['Quadruplet']:
            aspect = quad.get('Aspect', 'NULL')
            if aspect == "NULL":
                target = quad.get('Category', 'general').replace("#", " ")
            else:
                target = aspect
            
            try:
                val, aro = map(float, quad.get('VA', '5.0#5.0').split('#'))
            except ValueError:
                val, aro = 5.0, 5.0
                
            extracted.append({
                'key': (str(entry_id), str(target).strip()),
                'data': {'Valence': val, 'Arousal': aro, 'Text': text, 'Standard': 'Quadruplet'}
            })

    elif 'Aspect_VA' in entry:
        for item in entry['Aspect_VA']:
            target = item.get('Aspect', 'NULL')
            if target == "NULL":
                target = "general"
            
            val_str = item.get('VA', '5.0#5.0')
            if not val_str: val_str = '5.0#5.0'
            try:
                parts = val_str.split('#')
                if len(parts) >= 2:
                    val, aro = map(float, parts[:2])
                else:
                    val, aro = 5.0, 5.0
            except ValueError:
                val, aro = 5.0, 5.0

            extracted.append({
                'key': (str(entry_id), str(target).strip()),
                'data': {'Valence': val, 'Arousal': aro, 'Text': text, 'Standard': 'Aspect_VA'}
            })
            
    # If standard is just 'Aspect' without VA, we might not have gold labels for VA. 
    # But usually this script is for evaluating against gold that HAS labels.

    return extracted

def load_gold_data(gold_dir):
    print(f"Loading Gold Data from: {gold_dir}")
    gold_map = {}
    
    # Support both directory and single file
    if os.path.isfile(gold_dir):
        files = [gold_dir]
    else:
        files = glob.glob(os.path.join(gold_dir, "*.jsonl")) + \
                glob.glob(os.path.join(gold_dir, "*.json")) + \
                glob.glob(os.path.join(gold_dir, "**/*.jsonl"), recursive=True) + \
                glob.glob(os.path.join(gold_dir, "**/*.json"), recursive=True)
        # Remove duplicates
        files = list(set(files))
        
    print(f"Found {len(files)} gold files.")
    for fpath in files:
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    entry = json.loads(line)
                    items = parse_gold_entry(entry)
                    for item in items:
                        if item['key'] in gold_map:
                            # Warning: duplicate ID+Aspect tuple found
                            pass 
                        gold_map[item['key']] = item['data']
                except json.JSONDecodeError:
                    continue
    
    print(f"Loaded {len(gold_map)} unique gold entries.")
    return gold_map

def main():
    parser = argparse.ArgumentParser(description="Merge Predictions and Gold Data for Error Analysis")
    parser.add_argument("--pred_dir", type=str, required=True, help="Directory containing prediction JSONL files")
    parser.add_argument("--gold_dir", type=str, required=True, help="Directory containing gold JSONL files")
    parser.add_argument("--output_file", type=str, default="error_analysis.jsonl", help="Output JSONL file path")
    args = parser.parse_args()

    # 1. Load Gold Data
    gold_map = load_gold_data(args.gold_dir)

    # 2. Process Predictions
    print(f"Processing Predictions from: {args.pred_dir}")
    merged_results = []
    
    if os.path.isfile(args.pred_dir):
        pred_files = [args.pred_dir]
    else:
        # Search for both json and jsonl, recursive and flat
        pred_files = glob.glob(os.path.join(args.pred_dir, "*.jsonl")) + \
                     glob.glob(os.path.join(args.pred_dir, "*.json")) + \
                     glob.glob(os.path.join(args.pred_dir, "**/*.jsonl"), recursive=True) + \
                     glob.glob(os.path.join(args.pred_dir, "**/*.json"), recursive=True)
        pred_files = list(set(pred_files))

    print(f"Found {len(pred_files)} prediction files.")
    
    matches = 0
    mismatches = 0
    missing_gold = 0

    for fpath in pred_files:
        print(f"Reading: {os.path.basename(fpath)}")
        with open(fpath, 'r', encoding='utf-8') as f:
            for line in f:
                if not line.strip(): continue
                try:
                    pred_entry = json.loads(line)
                    pid = str(pred_entry.get('ID'))
                    
                    # Determine list of predictions in this line
                    predictions_in_line = []

                    if 'Aspect_VA' in pred_entry:
                        for item in pred_entry['Aspect_VA']:
                            t = item.get('Aspect', 'NULL')
                            v_str = item.get('VA')
                            predictions_in_line.append({'Target': t, 'VA': v_str})
                    
                    elif 'Quadruplet' in pred_entry:
                        for item in pred_entry['Quadruplet']:
                            t = item.get('Aspect', 'NULL')
                            if t == "NULL": t = item.get('Category', 'general').replace('#', ' ')
                            v_str = item.get('VA')
                            predictions_in_line.append({'Target': t, 'VA': v_str})
                            
                    else:
                        # Fallback for flat format
                        target = pred_entry.get('Target') or pred_entry.get('Aspect') or "NULL"
                        p_val = pred_entry.get('Valence') or pred_entry.get('Predicted_Valence')
                        p_aro = pred_entry.get('Arousal') or pred_entry.get('Predicted_Arousal')
                        p_va = pred_entry.get('VA') or pred_entry.get('Predicted_VA')
                        
                        if p_val is not None and p_aro is not None:
                            predictions_in_line.append({'Target': target, 'Val': float(p_val), 'Aro': float(p_aro)})
                        elif p_va:
                            predictions_in_line.append({'Target': target, 'VA': p_va})

                    # Process extracted predictions
                    for p_item in predictions_in_line:
                        target = str(p_item['Target']).strip()
                        
                        # Get float values
                        if 'Val' in p_item:
                            p_val, p_aro = p_item['Val'], p_item['Aro']
                        elif 'VA' in p_item and p_item['VA']:
                            try:
                                p_val, p_aro = map(float, p_item['VA'].split('#')[:2])
                            except:
                                continue
                        else:
                            continue

                        # Lookup in Gold
                        key = (pid, target)
                        gold_info = gold_map.get(key)
                        
                        result_item = {
                            'ID': pid,
                            'Target': target,
                            'Predicted_Valence': p_val,
                            'Predicted_Arousal': p_aro,
                            'Source_File': os.path.basename(fpath)
                        }

                        if gold_info:
                            g_val = gold_info['Valence']
                            g_aro = gold_info['Arousal']
                            result_item['Gold_Valence'] = g_val
                            result_item['Gold_Arousal'] = g_aro
                            result_item['Text'] = gold_info['Text']
                            
                            # Compute Errors
                            result_item['Error_Valence'] =  p_val - g_val
                            result_item['Abs_Error_Valence'] = abs(p_val - g_val)
                            result_item['Error_Arousal'] = p_aro - g_aro
                            result_item['Abs_Error_Arousal'] = abs(p_aro - g_aro)
                            result_item['Total_MSE_Contrib'] = ((p_val - g_val)**2 + (p_aro - g_aro)**2) / 2
                            
                            matches += 1
                        else:
                            result_item['Gold_Found'] = False
                            missing_gold += 1
                        
                        merged_results.append(result_item)

                except json.JSONDecodeError:
                    continue

    # 3. Write Output
    print(f"Writing {len(merged_results)} results to {args.output_file}")
    print(f"Stats: Matches: {matches}, Missing Gold: {missing_gold}")
    
    with open(args.output_file, 'w', encoding='utf-8') as f_out:
        for item in merged_results:
            f_out.write(json.dumps(item) + "\n")

if __name__ == "__main__":
    main()
