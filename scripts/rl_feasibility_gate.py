"""Phase 2B: Gated RL Feasibility Diagnostic.

Draws K (default 8) stochastic samples per map from a trained model checkpoint
(temperature=0.7, top_p=0.9) to measure the proportion of rollout groups that
have non-zero reward variance (mixed outcomes: at least one pass and at least one fail).

RL (GRPO / PPO) relies entirely on intra-group reward variance to compute advantages.
Groups with 0% success or 100% success yield advantage = 0.
Decision Gate: Full RL is justified ONLY if mixed groups exceed >10-15%.
"""
import argparse
import json
from pathlib import Path
import torch

from spara.data import records
from spara.evaluation import ReferenceEvaluator
from spara.runtime import json_write, load_model, read_config, seed_all
from spara.serialization import encode
from spara.text_rl import completion_mask, repeat_prompt
from spara.training import amp, to_device


def evaluate_rl_feasibility(model, processor, config, data_root, output_dir, samples=8, split="test", limit=None):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluator = ReferenceEvaluator(data_root)
    model.eval()
    
    eos = model.generation_config.eos_token_id
    eos = eos if isinstance(eos, list) else [eos]
    seed_all(config["seed"])
    
    rows = records(data_root, split)
    if limit:
        rows = rows[:limit]
        
    print(f"=== Running RL Feasibility Gate ({len(rows)} maps, {samples} samples/map) ===")
    results_list = []
    
    all_success_count = 0
    all_failure_count = 0
    mixed_count = 0
    
    detour_mixed_count = 0
    detour_total_count = 0
    
    with (output / "rollouts.jsonl").open("w", encoding="utf-8") as log:
        for idx, row in enumerate(rows):
            prompt = to_device(encode(processor, data_root, split, row, training=False,
                                     latents=getattr(model.config, "livr_latent_tokens", None)), model.device)
            
            with torch.no_grad(), amp(model.device):
                generated = model.generate(
                    **repeat_prompt(prompt, samples),
                    **config["generation"],
                    use_cache=True
                )
                
            completions = generated[:, prompt["input_ids"].shape[-1]:]
            lengths = completion_mask(completions, eos).sum(-1)
            
            eval_results = []
            for i in range(samples):
                full_ids = torch.cat([prompt["input_ids"][0], completions[i, :lengths[i]]])
                decoded_resp = processor.tokenizer.decode(full_ids, skip_special_tokens=False)
                res = evaluator.score(row, decoded_resp)
                eval_results.append(res)
                
            successes = [bool(r["success"]) for r in eval_results]
            num_success = sum(successes)
            
            is_all_success = (num_success == samples)
            is_all_failure = (num_success == 0)
            is_mixed = (not is_all_success and not is_all_failure)
            
            # Check if this map is a detour map
            # Simple Manhattan distance check vs shortest path
            map_desc = row["map_desc"]
            start, goal = None, None
            for r in range(len(map_desc)):
                for c in range(len(map_desc[0])):
                    if map_desc[r][c] == 1:
                        start = (r, c)
                    elif map_desc[r][c] == 2:
                        goal = (r, c)
                        
            manhattan = abs(start[0] - goal[0]) + abs(start[1] - goal[1])
            is_detour = len(row.get("text_output", "").split(",")) > manhattan
            if is_detour:
                detour_total_count += 1
                if is_mixed:
                    detour_mixed_count += 1
                    
            if is_all_success:
                all_success_count += 1
            elif is_all_failure:
                all_failure_count += 1
            else:
                mixed_count += 1
                
            row_summary = {
                "row_index": idx,
                "map_id": row["map_id"],
                "grid_size": len(map_desc),
                "is_detour": is_detour,
                "successes": successes,
                "success_rate": num_success / samples,
                "is_mixed": is_mixed,
                "statuses": [r["status"] for r in eval_results],
                "answers": [r["extracted_answer"] for r in eval_results]
            }
            results_list.append(row_summary)
            log.write(json.dumps(row_summary) + "\n")
            log.flush()
            
            if (idx + 1) % 25 == 0 or idx == len(rows) - 1:
                total_so_far = idx + 1
                mixed_pct = (mixed_count / total_so_far) * 100
                print(f"[{total_so_far}/{len(rows)}] All-Success: {all_success_count} | "
                      f"All-Failure: {all_failure_count} | Mixed: {mixed_count} ({mixed_pct:.1f}%)", flush=True)
                      
    total = len(rows)
    summary = {
        "total_maps": total,
        "samples_per_map": samples,
        "all_success_groups": all_success_count,
        "all_success_pct": all_success_count / total,
        "all_failure_groups": all_failure_count,
        "all_failure_pct": all_failure_count / total,
        "mixed_groups": mixed_count,
        "mixed_pct": mixed_count / total,
        "detour_maps_total": detour_total_count,
        "detour_maps_mixed": detour_mixed_count,
        "detour_maps_mixed_pct": (detour_mixed_count / detour_total_count) if detour_total_count else 0.0,
        "rl_decision_gate": {
            "threshold_pct": 10.0,
            "passed": (mixed_count / total) >= 0.10,
            "recommendation": "PROCEED WITH RL" if (mixed_count / total) >= 0.10 else "SKIP RL (EXPLORATION BOTTLENECK)"
        }
    }
    
    json_write(output / "summary.json", summary)
    
    print("\n" + "=" * 65)
    print("RL FEASIBILITY GATE RESULTS:")
    print(f"  Total Maps Evaluated:     {total}")
    print(f"  All-Success Groups:       {all_success_count} ({summary['all_success_pct']*100:.1f}%) -> Zero Advantage")
    print(f"  All-Failure Groups:       {all_failure_count} ({summary['all_failure_pct']*100:.1f}%) -> Zero Advantage")
    print(f"  Mixed Outcome Groups:     {mixed_count} ({summary['mixed_pct']*100:.1f}%) -> Non-Zero Advantage Gradient")
    if detour_total_count > 0:
        print(f"  Detour Maps Mixed Rate:   {detour_mixed_count}/{detour_total_count} ({summary['detour_maps_mixed_pct']*100:.1f}%)")
    print(f"  Decision Gate:            {summary['rl_decision_gate']['recommendation']}")
    print("=" * 65 + "\n")
    return summary


def main():
    parser = argparse.ArgumentParser(description="RL Feasibility Diagnostic")
    parser.add_argument("--checkpoint", required=True, help="Checkpoint to evaluate")
    parser.add_argument("--config", default="configs/livr.json", help="Config file")
    parser.add_argument("--data-root", default="data/vsp", help="Dataset root")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--samples", type=int, default=8, help="Rollouts per map")
    parser.add_argument("--split", default="test", help="Split to sample")
    parser.add_argument("--limit", type=int, default=None, help="Max maps to sample")
    args = parser.parse_args()
    
    config = read_config(args.config)
    seed_all(config["seed"])
    
    print(f"Loading model from {args.checkpoint}...")
    model, processor = load_model(config, args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    
    evaluate_rl_feasibility(
        model, processor, config, args.data_root, args.output,
        samples=args.samples, split=args.split, limit=args.limit
    )


if __name__ == "__main__":
    main()
