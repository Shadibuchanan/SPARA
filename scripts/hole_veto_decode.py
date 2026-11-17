"""Phase 1B: Greedy decode with hole moves blocked (Hole-Veto Decode).

At each step, mask out moves into a true adjacent hole before taking the top choice.
Evaluates on the benchmark test maps (and detour maps) to establish whether
avoiding adjacent holes is nearly the entire problem or if multi-step planning is required.
"""
import argparse
import copy
import json
from pathlib import Path
import torch

from spara.data import records
from spara.evaluation import ReferenceEvaluator, summarize
from spara.runtime import read_config, load_model, seed_all
from spara.serialization import encode
from spara.training import amp, to_device


ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def find_start_and_goal(map_desc):
    start = None
    goal = None
    H = len(map_desc)
    W = len(map_desc[0])
    for r in range(H):
        for c in range(W):
            if map_desc[r][c] == 1:
                start = (r, c)
            elif map_desc[r][c] == 2:
                goal = (r, c)
    return start, goal, H, W


def get_action_token_ids(tokenizer):
    """
    Find token IDs corresponding to cardinal action words in Qwen tokenizer.
    Handles variations like 'UP', ' UP', 'DOWN', ' DOWN', etc.
    """
    action_token_map = {}
    for action in ACTIONS:
        tokens = set()
        for variant in [action, " " + action, action.lower(), " " + action.lower(), action[0], " " + action[0]]:
            enc = tokenizer.encode(variant, add_special_tokens=False)
            if len(enc) == 1:
                tokens.add(enc[0])
        action_token_map[action] = list(tokens)
    return action_token_map


@torch.no_grad()
def decode_with_hole_veto(model, processor, prompt, map_desc, action_tokens, max_steps=40, veto_enabled=True):
    tokenizer = processor.tokenizer
    start, goal, H, W = find_start_and_goal(map_desc)
    curr_pos = start
    
    device = prompt["input_ids"].device
    eos_ids = model.generation_config.eos_token_id
    if not isinstance(eos_ids, (list, tuple, set)):
        eos_ids = [eos_ids]
    eos_ids = set(eos_ids)
    
    # Comma and closing tokens
    comma_token = tokenizer.encode(", ", add_special_tokens=False)
    close_brace_token = tokenizer.encode("}", add_special_tokens=False)[0]
    
    # Run prefill
    with amp(device):
        out = model(**prompt, use_cache=True)
    past_key_values = out.past_key_values
    logits = out.logits[:, -1, :].clone()
    
    generated_ids = []
    actions_taken = []
    
    # Check if we need to output \boxed{ first
    boxed_tokens = tokenizer.encode("\\boxed{", add_special_tokens=False)
    
    # Prefill the \boxed{ prefix
    for b_tok in boxed_tokens:
        generated_ids.append(b_tok)
        tok_tensor = torch.tensor([[b_tok]], device=device)
        with amp(device):
            out = model(input_ids=tok_tensor, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :].clone()
        
    for step in range(max_steps):
        # Determine adjacent holes
        r, c = curr_pos
        vetoed_actions = set()
        if veto_enabled:
            for action_name, (dr, dc) in ACTIONS.items():
                nr, nc = r + dr, c + dc
                if 0 <= nr < H and 0 <= nc < W:
                    if map_desc[nr][nc] == -1:  # Hole!
                        vetoed_actions.add(action_name)
                        
        # Mask out logits of vetoed actions
        step_logits = logits[0].clone()
        for act in vetoed_actions:
            for tok_id in action_tokens.get(act, []):
                step_logits[tok_id] = float("-inf")
                
        # If at goal, encourage closing brace
        if curr_pos == goal:
            step_logits[close_brace_token] += 10.0
            
        # Top choice (Greedy)
        next_tok = int(step_logits.argmax().item())
        generated_ids.append(next_tok)
        
        # Decode token to check what was emitted
        tok_str = tokenizer.decode([next_tok]).strip().upper()
        
        # Check if an action was chosen
        chosen_action = None
        for act in ACTIONS:
            if tok_str == act or tok_str == act[0]:
                chosen_action = act
                break
                
        if chosen_action is not None:
            actions_taken.append(chosen_action)
            dr, dc = ACTIONS[chosen_action]
            nr, nc = curr_pos[0] + dr, curr_pos[1] + dc
            if 0 <= nr < H and 0 <= nc < W:
                curr_pos = (nr, nc)
                
            # If reached goal, finish route
            if curr_pos == goal:
                # Add closing brace
                generated_ids.append(close_brace_token)
                break
                
            # Otherwise append comma ", " to guide sequence format
            for c_tok in comma_token:
                generated_ids.append(c_tok)
                tok_tensor = torch.tensor([[c_tok]], device=device)
                with amp(device):
                    out = model(input_ids=tok_tensor, past_key_values=past_key_values, use_cache=True)
                past_key_values = out.past_key_values
                logits = out.logits[:, -1, :].clone()
                continue
                
        # If model emitted closing brace or EOS, finish
        if next_tok in eos_ids or next_tok == close_brace_token:
            break
            
        tok_tensor = torch.tensor([[next_tok]], device=device)
        with amp(device):
            out = model(input_ids=tok_tensor, past_key_values=past_key_values, use_cache=True)
        past_key_values = out.past_key_values
        logits = out.logits[:, -1, :].clone()
        
    decoded_answer = tokenizer.decode(generated_ids, skip_special_tokens=False)
    # Ensure answer has \boxed{...} format if not present
    if "\\boxed{" not in decoded_answer:
        decoded_answer = "\\boxed{" + ", ".join(actions_taken) + "}"
        
    full_output = "<|im_start|>assistant\n" + decoded_answer + "<|im_end|>"
    return full_output, actions_taken


def run_hole_veto_benchmark(model, processor, config, data_root, output_dir, veto_enabled=True, split="test"):
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    evaluator = ReferenceEvaluator(data_root)
    model.eval()
    
    action_tokens = get_action_token_ids(processor.tokenizer)
    print(f"Action tokens mapped: {action_tokens}")
    print(f"Running evaluation on {split} set (veto_enabled={veto_enabled})...")
    
    rows = records(data_root, split)
    outputs = []
    
    with (output / "responses.jsonl").open("w", encoding="utf-8") as log:
        for idx, row in enumerate(rows):
            prompt = to_device(encode(processor, data_root, split, row, training=False,
                                     latents=getattr(model.config, "livr_latent_tokens", None)), model.device)
            raw_response, actions = decode_with_hole_veto(
                model, processor, prompt, row["map_desc"], action_tokens, veto_enabled=veto_enabled
            )
            result = evaluator.score(row, raw_response)
            result["row_index"] = idx
            result["actions_taken"] = actions
            result["veto_enabled"] = veto_enabled
            outputs.append(result)
            log.write(json.dumps(result) + "\n")
            log.flush()
            
            if (idx + 1) % 50 == 0 or idx == len(rows) - 1:
                stats = summarize(outputs)
                print(f"{idx + 1}/{len(rows)}: Accuracy = {stats['overall']['accuracy']*100:.2f}% "
                      f"({stats['overall']['successes']}/{stats['overall']['count']})", flush=True)
                
    summary = summarize(outputs)
    
    # Calculate detailed failure modes
    failures = [x for x in outputs if not x["success"]]
    fell_in_hole = sum(1 for x in failures if "hole" in x.get("status", "").lower())
    not_reached = sum(1 for x in failures if "reach" in x.get("status", "").lower() and "hole" not in x.get("status", "").lower())
    
    summary["failure_breakdown"] = {
        "total_failures": len(failures),
        "fell_in_hole": fell_in_hole,
        "did_not_reach_goal": not_reached,
        "other": len(failures) - fell_in_hole - not_reached,
    }
    
    (output / "summary.json").write_text(json.dumps(summary, indent=2))
    
    print("\n" + "=" * 60)
    print(f"RESULTS (Veto Enabled = {veto_enabled}):")
    print(f"  Overall Success: {summary['overall']['successes']}/{summary['overall']['count']} ({summary['overall']['accuracy']*100:.2f}%)")
    print("  Per Grid:")
    for g, g_stats in summary["per_grid"].items():
        print(f"    {g}x{g}: {g_stats['successes']}/{g_stats['count']} ({g_stats['accuracy']*100:.2f}%)")
    print(f"  Failures in holes: {fell_in_hole} (vs {len(failures)} total failures)")
    print("=" * 60)
    return summary


def main():
    parser = argparse.ArgumentParser(description="Hole-veto greedy decode")
    parser.add_argument("--checkpoint", default=None, help="Checkpoint to evaluate (e.g. Stage 2 LIVR or E2)")
    parser.add_argument("--base-model", default=None, help="Base model path if evaluating base model")
    parser.add_argument("--config", default="configs/livr.json", help="Model config")
    parser.add_argument("--data-root", default="data/vsp", help="Dataset root")
    parser.add_argument("--output", required=True, help="Output directory")
    parser.add_argument("--split", default="test", choices=["train", "test"], help="Dataset split")
    parser.add_argument("--no-veto", action="store_true", help="Disable veto (run standard greedy baseline)")
    args = parser.parse_args()
    
    config = read_config(args.config)
    seed_all(config["seed"])
    
    if args.base_model:
        config["local_model_path"] = str(Path(args.base_model).resolve())
    elif args.checkpoint and (Path(args.checkpoint) / "download-manifest.json").is_file():
        config["local_model_path"] = str(Path(args.checkpoint).resolve())
        args.checkpoint = None
        
    print(f"Loading model (checkpoint={args.checkpoint}, base={config.get('local_model_path')})...")
    model, processor = load_model(config, args.checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
    
    run_hole_veto_benchmark(
        model, processor, config, args.data_root, args.output,
        veto_enabled=not args.no_veto, split=args.split
    )


if __name__ == "__main__":
    main()
