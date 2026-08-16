#!/usr/bin/env python3
"""Analyze WER evaluation results for adversarial audio."""

import json
import csv
import os
import sys
from collections import defaultdict
from pathlib import Path

def load_jsonl(path):
    rows = []
    with Path(path).open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return rows

def analyze_results(wer_file, summary_file):
    rows = load_jsonl(wer_file)
    print(f"\nLoaded {len(rows)} results from {wer_file}")
    
    if not rows:
        print("No results to analyze!")
        return
    
    ok_rows = [r for r in rows if r.get("status") == "ok"]
    
    # Overall statistics
    orig_wers = [r["original_wer"] for r in ok_rows]
    adv_wers = [r["adversarial_wer"] for r in ok_rows]
    wer_deltas = [r["wer_delta"] for r in ok_rows]
    
    print(f"\n=== Overall Statistics ===")
    print(f"Total samples: {len(ok_rows)}")
    print(f"Average original WER: {sum(orig_wers)/len(orig_wers):.4f}")
    print(f"Average adversarial WER: {sum(adv_wers)/len(adv_wers):.4f}")
    print(f"Average WER delta: {sum(wer_deltas)/len(wer_deltas):.4f}")
    print(f"Median original WER: {sorted(orig_wers)[len(orig_wers)//2]:.4f}")
    print(f"Median adversarial WER: {sorted(adv_wers)[len(adv_wers)//2]:.4f}")
    
    # Distribution of WER deltas
    print(f"\n=== WER Delta Distribution ===")
    bins = [(0, 0.05), (0.05, 0.1), (0.1, 0.2), (0.2, 0.5), (0.5, 1.01)]
    labels = ["0-5%", "5-10%", "10-20%", "20-50%", "50-100%"]
    for (low, high), label in zip(bins, labels):
        count = sum(1 for d in wer_deltas if low <= d < high)
        print(f"  WER delta {label}: {count} samples ({100*count/len(ok_rows):.1f}%)")
    
    # By risk category
    print(f"\n=== By Risk Category ===")
    categories = defaultdict(list)
    for r in ok_rows:
        cat = r.get("risk_category", "unknown")
        categories[cat].append(r)
    
    print(f"{'Category':<25} {'Count':<8} {'Avg Orig WER':<15} {'Avg Adv WER':<15} {'Avg Delta':<15}")
    print("-" * 75)
    for cat in sorted(categories.keys()):
        cat_rows = categories[cat]
        orig_wers_cat = [r["original_wer"] for r in cat_rows]
        adv_wers_cat = [r["adversarial_wer"] for r in cat_rows]
        deltas_cat = [r["wer_delta"] for r in cat_rows]
        print(f"{cat:<25} {len(cat_rows):<8} {sum(orig_wers_cat)/len(orig_wers_cat):<15.4f} {sum(adv_wers_cat)/len(adv_wers_cat):<15.4f} {sum(deltas_cat)/len(deltas_cat):<15.4f}")
    
    # Write summary CSV
    with Path(summary_file).open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["category", "count", "avg_orig_wer", "avg_adv_wer", "avg_delta", "median_orig_wer", "median_adv_wer", "median_delta"])
        
        # Overall
        writer.writerow([
            "overall", len(ok_rows),
            f"{sum(orig_wers)/len(orig_wers):.4f}",
            f"{sum(adv_wers)/len(adv_wers):.4f}",
            f"{sum(wer_deltas)/len(wer_deltas):.4f}",
            f"{sorted(orig_wers)[len(orig_wers)//2]:.4f}",
            f"{sorted(adv_wers)[len(adv_wers)//2]:.4f}",
            f"{sorted(wer_deltas)[len(wer_deltas)//2]:.4f}"
        ])
        
        # By category
        for cat in sorted(categories.keys()):
            cat_rows = categories[cat]
            orig_wers_cat = [r["original_wer"] for r in cat_rows]
            adv_wers_cat = [r["adversarial_wer"] for r in cat_rows]
            deltas_cat = [r["wer_delta"] for r in cat_rows]
            writer.writerow([
                cat, len(cat_rows),
                f"{sum(orig_wers_cat)/len(orig_wers_cat):.4f}",
                f"{sum(adv_wers_cat)/len(adv_wers_cat):.4f}",
                f"{sum(deltas_cat)/len(deltas_cat):.4f}",
                f"{sorted(orig_wers_cat)[len(orig_wers_cat)//2]:.4f}",
                f"{sorted(adv_wers_cat)[len(adv_wers_cat)//2]:.4f}",
                f"{sorted(deltas_cat)[len(deltas_cat)//2]:.4f}"
            ])
    
    print(f"\nSummary saved to {summary_file}")

def main():
    parser = sys.argv
    
    wer_files = [
        "results/adv_wer_matcha.jsonl",
        "results/adv_wer_cosyvoice.jsonl"
    ]
    
    summary_files = [
        "results/adv_wer_matcha_analysis.csv",
        "results/adv_wer_cosyvoice_analysis.csv"
    ]
    
    for wer_file, summary_file in zip(wer_files, summary_files):
        if os.path.exists(wer_file):
            analyze_results(wer_file, summary_file)
        else:
            print(f"File not found: {wer_file}")

if __name__ == "__main__":
    main()