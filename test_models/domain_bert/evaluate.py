import json
import os
import glob
import math
import argparse
import pandas as pd
from scipy.stats import pearsonr
from logger import log_experiment
def load_json_data(path):
    data = {}
    if not os.path.exists(path):
        return data
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            if not line.strip():
                continue
            try:
                entry = json.loads(line)
                eid = entry["ID"]
                aspects = {}
                if "Aspect_VA" in entry:
                    items = entry["Aspect_VA"]
                elif "Quadruplet" in entry:
                    items = entry["Quadruplet"]
                elif "Aspect" in entry and "Intensity" in entry:
                    items = []
                    asps = entry["Aspect"]
                    ints = entry["Intensity"]
                    cats = entry.get("Category", [])
                    for i in range(len(asps)):
                        a_name = asps[i]
                        if a_name == "NULL" or a_name is None:
                            a_name = cats[i].replace("#", " ") if i < len(cats) else "general"
                        items.append({"Aspect": a_name, "VA": ints[i]})
                else:
                    items = []
                for item in items:
                    asp_name = item["Aspect"]
                    if asp_name == "NULL":
                        asp_name = item.get("Category", "general").replace("#", " ")
                    aspects[str(asp_name).replace(" ", "").lower()] = item["VA"]
                data[eid] = aspects
            except:
                continue
    return data

def compare_metrics(gold_dir, pred_path, output_dir):
    
    pred_files = glob.glob(os.path.join(pred_path, "*.json"))

    all_results = []
    csv_rows = []

    print(f"{'FILE':<45} | {'PCC_V':<10} | {'PCC_A':<10} | {'RMSE_VA':<10}")
    print("-" * 81)

    langs = {'eng', 'zho', 'zh', 'jpn', 'rus', 'ru', 'tat', 'ukr', 'uk'}
    doms = {'laptop', 'restaurant', 'rest', 'finance', 'fin', 'hotel'}
    
    for p_path in sorted(pred_files):
        fname = os.path.basename(p_path)
        f_low = fname.lower()
        
        target_lang = next((l for l in langs if l in f_low), None)
        target_dom = next((d for d in doms if d in f_low), None)

        gold_file = None
        if target_lang and target_dom:
            gold_candidates = glob.glob(os.path.join(gold_dir, "*.jsonl"))
    
            for cand in gold_candidates:
                c_low = os.path.basename(cand).lower()
                if target_lang in c_low and target_dom in c_low:
                    gold_file = cand
                    print(gold_file)
                    
                    break
        
        if not gold_file or not os.path.exists(gold_file):
            raise ValueError('nothing works you are retarded')
            

        g_data = load_json_data(gold_file)
        p_data = load_json_data(p_path)

        gv, ga, pv, pa = [], [], [], []
        for eid, g_aspects in g_data.items():
            if eid in p_data:
                p_aspects = p_data[eid]
                for asp_key, g_va in g_aspects.items():
                    if asp_key in p_aspects:
                        try:
                            p_va = p_aspects[asp_key]
                            g_split = g_va.split("#")
                            p_split = p_va.split("#")
                            gv.append(float(g_split[0]))
                            ga.append(float(g_split[1]))
                            pv.append(float(p_split[0]))
                            pa.append(float(p_split[1]))
                        except:
                            continue

        if len(gv) > 1:
            res_v = pearsonr(gv, pv)[0]
            res_a = pearsonr(ga, pa)[0]
            diff_sq = [(gv[i] - pv[i])**2 + (ga[i] - pa[i])**2 for i in range(len(gv))]
            rmse = math.sqrt(sum(diff_sq) / len(gv))
            
            all_results.append({"v": res_v, "a": res_a, "r": rmse})
            csv_rows.append({"file": fname[45:], "pcc_v": res_v, "pcc_a": res_a, "rmse": rmse})
            print(f"{fname[45:]:<45} | {res_v:<10.4f} | {res_a:<10.4f} | {rmse:<10.4f}")

    if all_results:
        avg_v = sum(x["v"] for x in all_results) / len(all_results)
        avg_a = sum(x["a"] for x in all_results) / len(all_results)
        avg_r = sum(x["r"] for x in all_results) / len(all_results)
        print("-" * 81)
        print(f"{'AVERAGE':<45} | {avg_v:<10.4f} | {avg_a:<10.4f} | {avg_r:<10.4f}")
        
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)
            df = pd.DataFrame(csv_rows)
            df.to_csv(os.path.join(output_dir, "metrics.csv"), index=False)
        # results_dict = {"avg_v": avg_v, "avg_a": avg_a, "avg_r": avg_r}
        # log_experiment(args, results_dict)
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--pred_file', type=str, required=True)
    parser.add_argument('--gold_file', type=str, required=True)
    parser.add_argument('--output_dir', type=str, default=None)
    args = parser.parse_args()
    compare_metrics(args.gold_file, args.pred_file, args.output_dir)