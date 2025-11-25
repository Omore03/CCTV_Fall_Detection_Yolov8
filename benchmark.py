import os
import sys
import time
import json
import subprocess
from glob import glob
from datetime import datetime
import csv
import pandas as pd
import matplotlib.pyplot as plt

CLIPS_DIR = 'clips'
LIVEFEED_DIR = 'livefeed'
CONFIG_FILE = 'config.json'
MAIN_SCRIPT = 'main.py'
INF_TIME_FILE = os.path.join(LIVEFEED_DIR, "last_inference_time.txt")
MODELS = [
    ('YOLOv8s.pt', 'YOLOv8s'),
    ('YOLOv8m.pt', 'YOLOv8m'),
    ('YOLOv8l.pt', 'YOLOv8l'),
    ('YOLOv8x.pt', 'YOLOv8x')
]

RESULTS_DIR = "benchmark_results"
os.makedirs(RESULTS_DIR, exist_ok=True)
RAW_CSV = os.path.join(RESULTS_DIR, "raw_results.csv")
STATS_CSV = os.path.join(RESULTS_DIR, "model_stats.csv")

RED = '\033[91m'
GREEN = '\033[92m'
RESET = '\033[0m'
BOLD = '\033[1m'

def get_power_usage():
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=power.draw --format=csv,noheader,nounits",
            shell=True
        )
        return float(out.decode().strip().split('\n')[0])
    except Exception:
        return None

def get_total_gpu_memory():
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits",
            shell=True
        ).decode()
        return float(out.strip().split('\n')[0])
    except Exception:
        return 0.0

def get_gpu_util():
    try:
        out = subprocess.check_output(
            "nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader,nounits",
            shell=True
        )
        return float(out.decode().strip().split('\n')[0])
    except Exception:
        return None

def print_heading(model_name):
    print(f"\n\n{BOLD}===== MODEL: {model_name} ====={RESET}")

def print_clip_result(idx, clip, prediction, vram_avg, vram_peak, inf_time, avg_power, avg_util):
    pred_col = GREEN if prediction == 'Fall' else RED
    print(f"[{idx}] {os.path.basename(clip)}:")
    print(f"      Prediction: {pred_col}{prediction}{RESET}")
    print(f"      VRAM (avg): {vram_avg:.2f} MB")
    print(f"      VRAM (peak): {vram_peak:.2f} MB")
    if inf_time is not None:
        print(f"      Inference time: {inf_time:.4f} sec")
    else:
        print(f"      Inference time: N/A")
    if avg_power is not None:
        print(f"      Power (avg): {avg_power:.2f} W")
    if avg_util is not None:
        print(f"      GPU Util (avg): {avg_util:.2f} %")

def print_summary(vram_avg_model, vram_peak_model, avg_inf, avg_power, avg_util):
    print(f"\n---> SUMMARY for the model (statistics per clip)")
    print(f"     Avg VRAM: {vram_avg_model:.2f} MB")
    print(f"     Peak VRAM: {vram_peak_model:.2f} MB")
    if avg_inf is not None:
        print(f"     Avg Inference time: {avg_inf:.4f} sec")
    else:
        print(f"     Avg Inference time: N/A")
    if avg_power is not None:
        print(f"     Avg Power: {avg_power:.2f} W")
    if avg_util is not None:
        print(f"     Avg GPU Util: {avg_util:.2f} %")

def run_and_monitor(clip, model_path, model_name):
    with open(CONFIG_FILE, 'r') as f:
        config = json.load(f)
    config['perception_engine']['model_name'] = model_path
    config['video']['type'] = 'local'
    config['video']['url'] = clip
    # ------ Ensure display/wait logic is always enabled ------
    if 'display' not in config['video']:
        config['video']['display'] = {}
    config['video']['display']['flag'] = True
    # ---------------------------------------------------------

    with open(CONFIG_FILE, 'w') as f:
        json.dump(config, f, indent=4)

    # Clean out previous outputs
    if os.path.exists(LIVEFEED_DIR):
        for f_ in os.listdir(LIVEFEED_DIR):
            try:
                os.remove(os.path.join(LIVEFEED_DIR, f_))
            except Exception:
                pass

    if os.path.exists(INF_TIME_FILE):
        try:
            os.remove(INF_TIME_FILE)
        except Exception:
            pass

    vram_baseline = get_total_gpu_memory()
    vram_samples = []
    power_samples = []
    util_samples = []
    p = subprocess.Popen([sys.executable, MAIN_SCRIPT],
                         stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    while p.poll() is None:
        vram_samples.append(get_total_gpu_memory())
        power = get_power_usage()
        util = get_gpu_util()
        if power is not None:
            power_samples.append(power)
        if util is not None:
            util_samples.append(util)
        time.sleep(0.25)
    vram_samples_deltas = [max(0, v - vram_baseline) for v in vram_samples]
    vram_avg = sum(vram_samples_deltas) / len(vram_samples_deltas) if vram_samples_deltas else 0
    vram_peak = max(vram_samples_deltas) if vram_samples_deltas else 0
    avg_power = sum(power_samples) / len(power_samples) if power_samples else None
    avg_util = sum(util_samples) / len(util_samples) if util_samples else None

    inf_time = None
    if os.path.exists(INF_TIME_FILE):
        try:
            with open(INF_TIME_FILE, "r") as f:
                times = json.load(f)
                inf_time = sum(times) / len(times) if times else None

        except Exception:
            inf_time = None

    # ======= BEST PRACTICE: Use most recent *_events.json after run finishes =======
    prediction = "No_Fall"
    event_files = sorted(
        glob(os.path.join(LIVEFEED_DIR, "*_events.json")),
        key=os.path.getmtime, reverse=True
    )
    if event_files:
        latest_events_file = event_files[0]
        with open(latest_events_file, "r") as f:
            events = json.load(f)
        if isinstance(events, list) and len(events) > 0:
            prediction = "Fall"

    return prediction, vram_avg, vram_peak, inf_time, avg_power, avg_util




def main():
    all_clips = sorted(glob(os.path.join(CLIPS_DIR, "*.mp4")))
    raw_results = []
    model_stats = []

    for model_path, model_label in MODELS:
        print_heading(model_label)
        results = []
        for idx, clip in enumerate(all_clips):
            prediction, vram_avg, vram_peak, inf_time, avg_power, avg_util = run_and_monitor(clip, model_path, model_label)
            results.append((prediction, vram_avg, vram_peak, inf_time, avg_power, avg_util, os.path.basename(clip)))
            print_clip_result(idx, clip, prediction, vram_avg, vram_peak, inf_time, avg_power, avg_util)
            raw_results.append({
                'model': model_label,
                'clip': os.path.basename(clip),
                'prediction': prediction,
                'vram_avg': vram_avg,
                'vram_peak': vram_peak,
                'inf_time': inf_time,
                'avg_power': avg_power,
                'avg_util': avg_util
            })
        # Model-level summary
        vram_avgs = [r[1] for r in results]
        vram_peaks = [r[2] for r in results]
        vram_avg_model = sum(vram_avgs) / len(vram_avgs) if vram_avgs else 0
        vram_peak_model = max(vram_peaks) if vram_peaks else 0
        inf_times = [r[3] for r in results if r[3] is not None]
        avg_inf = sum(inf_times) / len(inf_times) if inf_times else None
        avg_power = sum([r[4] for r in results if r[4] is not None]) / max(len([r for r in results if r[4] is not None]), 1) if results else 0
        avg_util = sum([r[5] for r in results if r[5] is not None]) / max(len([r for r in results if r[5] is not None]), 1) if results else 0
        print_summary(vram_avg_model, vram_peak_model, avg_inf, avg_power, avg_util)
        model_stats.append({
            'model': model_label,
            'avg_vram': vram_avg_model,
            'peak_vram': vram_peak_model,
            'avg_inf_time': avg_inf,
            'avg_power': avg_power,
            'avg_util': avg_util
        })

    # --- RESULTS CSV ---
    with open(RAW_CSV, "w", newline='') as f:
        fieldnames = ['model', 'clip', 'prediction', 'vram_avg', 'vram_peak', 'inf_time', 'avg_power', 'avg_util']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in raw_results:
            writer.writerow(row)
    print(f"\nRaw results saved to {RAW_CSV}")

    with open(STATS_CSV, "w", newline='') as f:
        fieldnames = ['model', 'avg_vram', 'peak_vram', 'avg_inf_time', 'avg_power', 'avg_util']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in model_stats:
            writer.writerow(row)
    print(f"Model summary stats saved to {STATS_CSV}")

    # --- PLOTTING ---
    df_raw = pd.read_csv(RAW_CSV)
    df_stats = pd.read_csv(STATS_CSV)
    metrics = ['vram_avg', 'vram_peak', 'avg_power', 'avg_util', 'inf_time']
    metric_labels = {
        'vram_avg': 'Average VRAM (MB)',
        'vram_peak': 'Peak VRAM (MB)',
        'avg_power': 'Average Power (W)',
        'avg_util': 'Average GPU Utilization (%)',
        'inf_time': 'Inference Time (s)'
    }

    # Bar charts: model summary metrics
    for metric in ['avg_vram', 'peak_vram', 'avg_power', 'avg_util', 'avg_inf_time']:
        plt.figure(figsize=(6,4))
        x = df_stats['model']
        y = df_stats[metric]
        plt.bar(x, y)
        plt.ylabel(metric_labels.get(metric, metric))
        plt.title(f"{metric_labels.get(metric, metric)} by Model")
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"bar_{metric}.png"))
        plt.close()
        print(f"Bar chart saved: {os.path.join(RESULTS_DIR, f'bar_{metric}.png')}")

    # Line graphs: per-clip, per-model
    for metric in metrics:
        plt.figure(figsize=(10,6))
        for model in df_raw['model'].unique():
            dfm = df_raw[df_raw['model'] == model]
            plt.plot(dfm['clip'], dfm[metric], marker='o', label=model)
        plt.ylabel(metric_labels.get(metric, metric))
        plt.xlabel("Clip")
        plt.title(f"{metric_labels.get(metric, metric)} per Clip (All Models)")
        plt.legend()
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(os.path.join(RESULTS_DIR, f"line_{metric}.png"))
        plt.close()
        print(f"Line chart saved: {os.path.join(RESULTS_DIR, f'line_{metric}.png')}")

if __name__ == '__main__':
    main()
