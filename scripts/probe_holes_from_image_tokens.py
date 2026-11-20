"""Phase 1A: Probe holes directly from image tokens (CPU only).

Evaluates whether the frozen vision encoder separates 'hole' vs 'safe' cells.
Extracts per-cell pooled image-token features from the frozen Qwen2.5-VL vision tower,
labels each cell as hole (1) or safe (0), fits logistic regression with 5-fold CV,
and reports AUROC per grid size (3x3 to 6x6) and overall.
"""
import argparse
import json
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration
from qwen_vl_utils import process_vision_info

from spara.data import records, image_path


def extract_cell_features(vision_features, grid_size, spatial_shape):
    """
    vision_features: Tensor of shape (H_tokens * W_tokens, D)
    spatial_shape: (H_tokens, W_tokens)
    grid_size: N (for N x N map)
    Returns: (N * N, D) per-cell pooled features
    """
    H_t, W_t = spatial_shape
    feat_grid = vision_features.view(H_t, W_t, -1)  # (H_t, W_t, D)
    cell_feats = []
    
    for r in range(grid_size):
        r_start = int(round(r * H_t / grid_size))
        r_end = int(round((r + 1) * H_t / grid_size))
        r_start = min(r_start, H_t - 1)
        r_end = max(r_end, r_start + 1)
        
        for c in range(grid_size):
            c_start = int(round(c * W_t / grid_size))
            c_end = int(round((c + 1) * W_t / grid_size))
            c_start = min(c_start, W_t - 1)
            c_end = max(c_end, c_start + 1)
            
            # Spatial average pool over tokens in cell
            cell_pool = feat_grid[r_start:r_end, c_start:c_end, :].mean(dim=(0, 1))
            cell_feats.append(cell_pool.cpu().numpy())
            
    return np.stack(cell_feats, axis=0)  # (N*N, D)


def extract_features_and_labels(rows, split, data_root, model, processor, device="cpu"):
    X_list = []
    y_list = []
    grid_sizes = []
    cell_meta = []
    
    model.eval()
    for idx, row in enumerate(rows):
        map_desc = row["map_desc"]
        N = len(map_desc)
        img_p = image_path(data_root, split, row)
        
        with Image.open(img_p) as img:
            image = img.convert("RGB")
            
        messages = [{"role": "user", "content": [{"type": "image", "image": image}, {"type": "text", "text": "Plan"}]}]
        prompt = processor.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        image_inputs, _ = process_vision_info(messages)
        
        # Process image with Qwen vision processor
        inputs = processor(text=[prompt], images=image_inputs, padding=False, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(device)
        image_grid_thw = inputs["image_grid_thw"].to(device)
        
        with torch.no_grad():
            # Pass through frozen vision tower
            # Qwen2_5_VisionTransformerPretrainedModel forward produces visual tokens
            vis_tokens = model.visual(pixel_values, grid_thw=image_grid_thw)
            
        # Spatial shape of tokens
        # Qwen2.5-VL spatial merge factor is 2; spatial tokens: grid_thw[0, 1] // 2, grid_thw[0, 2] // 2
        H_tokens = int(image_grid_thw[0, 1].item()) // 2
        W_tokens = int(image_grid_thw[0, 2].item()) // 2
        
        cell_feats = extract_cell_features(vis_tokens, N, (H_tokens, W_tokens))
        
        # Labels: 1 if hole (-1), 0 otherwise
        for r in range(N):
            for c in range(N):
                is_hole = 1 if map_desc[r][c] == -1 else 0
                cell_idx = r * N + c
                X_list.append(cell_feats[cell_idx])
                y_list.append(is_hole)
                grid_sizes.append(N)
                cell_meta.append({"row_index": idx, "map_id": row.get("map_id", ""), "r": r, "c": c, "grid": N})
                
        if (idx + 1) % 100 == 0 or idx == len(rows) - 1:
            print(f"Extracted features for {idx + 1}/{len(rows)} {split} maps...", flush=True)
            
    return np.array(X_list), np.array(y_list), np.array(grid_sizes), cell_meta


def evaluate_probe(X, y, grid_sizes, cv_splits=5):
    results = {}
    unique_grids = sorted(np.unique(grid_sizes))
    
    print("\n" + "=" * 70)
    print("HOLE PROBING RESULTS FROM IMAGE TOKENS (5-Fold Cross Validation)")
    print("=" * 70)
    print(f"{'Grid':<8} {'Total Cells':<14} {'Holes (%)':<12} {'AUROC':<10} {'AP (PR-AUC)':<14} {'Bal Acc':<10}")
    print("-" * 70)
    
    # Per-grid evaluation
    for g in unique_grids:
        mask = (grid_sizes == g)
        X_g, y_g = X[mask], y[mask]
        
        n_holes = int(y_g.sum())
        pct_holes = 100.0 * n_holes / len(y_g)
        
        if n_holes < cv_splits or (len(y_g) - n_holes) < cv_splits:
            print(f"{g}x{g:<5} {len(y_g):<14} {pct_holes:>5.1f}%      insufficient samples for CV")
            continue
            
        skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
        aurocs, aps, bal_accs = [], [], []
        
        for train_idx, val_idx in skf.split(X_g, y_g):
            clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
            clf.fit(X_g[train_idx], y_g[train_idx])
            probs = clf.predict_proba(X_g[val_idx])[:, 1]
            preds = clf.predict(X_g[val_idx])
            
            aurocs.append(roc_auc_score(y_g[val_idx], probs))
            aps.append(average_precision_score(y_g[val_idx], probs))
            bal_accs.append(balanced_accuracy_score(y_g[val_idx], preds))
            
        mean_auroc = np.mean(aurocs)
        mean_ap = np.mean(aps)
        mean_bacc = np.mean(bal_accs)
        
        results[f"{g}x{g}"] = {
            "n_cells": len(y_g), "n_holes": n_holes, "pct_holes": pct_holes,
            "auroc": float(mean_auroc), "ap": float(mean_ap), "balanced_acc": float(mean_bacc)
        }
        print(f"{g}x{g:<5} {len(y_g):<14} {pct_holes:>5.1f}%       {mean_auroc:>7.4f}   {mean_ap:>11.4f}   {mean_bacc:>8.4f}")
        
    # Overall evaluation
    skf = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=42)
    aurocs, aps, bal_accs = [], [], []
    for train_idx, val_idx in skf.split(X, y):
        clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced")
        clf.fit(X[train_idx], y[train_idx])
        probs = clf.predict_proba(X[val_idx])[:, 1]
        preds = clf.predict(X[val_idx])
        aurocs.append(roc_auc_score(y[val_idx], probs))
        aps.append(average_precision_score(y[val_idx], probs))
        bal_accs.append(balanced_accuracy_score(y[val_idx], preds))
        
    overall_auroc = float(np.mean(aurocs))
    overall_ap = float(np.mean(aps))
    overall_bacc = float(np.mean(bal_accs))
    
    results["overall"] = {
        "n_cells": len(y), "n_holes": int(y.sum()), "pct_holes": float(100.0 * y.sum() / len(y)),
        "auroc": overall_auroc, "ap": overall_ap, "balanced_acc": overall_bacc
    }
    print("-" * 70)
    print(f"{'OVERALL':<8} {len(y):<14} {100.0 * y.sum() / len(y):>5.1f}%       {overall_auroc:>7.4f}   {overall_ap:>11.4f}   {overall_bacc:>8.4f}")
    print("=" * 70)
    
    print("\nComparison to Latent Probes (from Project Analysis):")
    print(f"  Latent Probe AUROC: 0.60 - 0.81 (0.65 on 7x7/8x8)")
    print(f"  Image Token Probe AUROC: {overall_auroc:.4f}")
    if overall_auroc > 0.85:
        print("  => CONCLUSION: Image tokens represent holes with high fidelity (AUROC > 0.85).")
        print("     The frozen vision encoder is intact. Bottleneck is downstream action readout/policy.")
    else:
        print("  => CONCLUSION: Image tokens weakly separate holes, suggesting vision-side limitation.")
        
    return results


def main():
    parser = argparse.ArgumentParser(description="Probe holes from image tokens")
    parser.add_argument("--data-root", default="data/vsp", help="Root of Mirage VSP dataset")
    parser.add_argument("--model-path", default="models/Qwen2.5-VL-3B-Instruct", help="Path to base model")
    parser.add_argument("--features-cache", default=None, help="Optional precomputed .npz features file")
    parser.add_argument("--save-features", default=None, help="Path to save extracted features (.npz)")
    parser.add_argument("--output", default="results/hole_probe_results.json", help="Path to save probe results JSON")
    parser.add_argument("--split", default="train", choices=["train", "test", "both"], help="Split to probe")
    parser.add_argument("--device", default="cpu", help="Device for feature extraction (cpu or cuda)")
    args = parser.parse_args()
    
    if args.features_cache and Path(args.features_cache).exists():
        print(f"Loading cached features from {args.features_cache}...")
        data = np.load(args.features_cache)
        X, y, grid_sizes = data["X"], data["y"], data["grid_sizes"]
    else:
        print(f"Loading processor and vision model from {args.model_path}...")
        processor = AutoProcessor.from_pretrained(args.model_path)
        model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            args.model_path, torch_dtype=torch.float32, device_map=args.device
        )
        
        all_X, all_y, all_grids = [], [], []
        splits = ["train", "test"] if args.split == "both" else [args.split]
        for s in splits:
            rows = records(args.data_root, s)
            print(f"Extracting features from {len(rows)} {s} rows...")
            X_s, y_s, grids_s, _ = extract_features_and_labels(rows, s, args.data_root, model, processor, args.device)
            all_X.append(X_s)
            all_y.append(y_s)
            all_grids.append(grids_s)
            
        X = np.concatenate(all_X, axis=0)
        y = np.concatenate(all_y, axis=0)
        grid_sizes = np.concatenate(all_grids, axis=0)
        
        if args.save_features:
            Path(args.save_features).parent.mkdir(parents=True, exist_ok=True)
            np.savez_compressed(args.save_features, X=X, y=y, grid_sizes=grid_sizes)
            print(f"Saved extracted features to {args.save_features}")
            
    results = evaluate_probe(X, y, grid_sizes)
    
    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(results, indent=2))
        print(f"\nSaved results to {args.output}")


if __name__ == "__main__":
    main()
