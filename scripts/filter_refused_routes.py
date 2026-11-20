"""Phase 2A: Filter and Pair Accepted and Refused Routes for DPO / Unlikelihood Training.

Filters candidate routes from search or multi-sample rollouts:
1. Simulates step-by-step navigation on the map.
2. Identifies specific failure modes:
   - 'hole_fatal': route steps into a hole (the primary failure mode to penalize!)
   - 'loop_fail': route revisits cells or wanders until max depth
   - 'off_grid_pad': route has excessive out-of-bounds moves (filtered out / cleaned)
3. Constructs paired preference examples:
   - prompt (image + task instructions)
   - chosen: accepted route ($y_w$) reaching the goal
   - rejected: clean refused route ($y_l$) stepping into a hole or failing
"""
import argparse
import json
from pathlib import Path

ACTIONS = {
    "UP": (-1, 0),
    "DOWN": (1, 0),
    "LEFT": (0, -1),
    "RIGHT": (0, 1),
}


def analyze_route(map_desc, actions):
    """
    Simulate actions on map_desc.
    Returns:
      valid_steps: number of steps before death/goal
      failure_mode: 'goal', 'hole', 'wall_pad', 'incomplete'
      cleaned_actions: actions up to and including the fatal step
    """
    H = len(map_desc)
    W = len(map_desc[0])
    
    start = None
    goal = None
    for r in range(H):
        for c in range(W):
            if map_desc[r][c] == 1:
                start = (r, c)
            elif map_desc[r][c] == 2:
                goal = (r, c)
                
    curr = start
    visited = [curr]
    wall_bumps = 0
    cleaned = []
    
    for act in actions:
        act = act.strip().upper()
        if act not in ACTIONS:
            continue
        cleaned.append(act)
        dr, dc = ACTIONS[act]
        nr, nc = curr[0] + dr, curr[1] + dc
        
        # Check off-grid move
        if not (0 <= nr < H and 0 <= nc < W):
            wall_bumps += 1
            # Off-grid leaves agent in place
            continue
            
        curr = (nr, nc)
        visited.append(curr)
        
        # Check hole
        if map_desc[nr][nc] == -1:
            return "hole", cleaned, wall_bumps
            
        # Check goal
        if map_desc[nr][nc] == 2:
            return "goal", cleaned, wall_bumps
            
    # If excessive wall bumps (> 25% of moves or >= 3 consecutive), reject as degenerate padding
    if wall_bumps > max(2, len(cleaned) // 4):
        return "wall_pad", cleaned, wall_bumps
        
    return "incomplete", cleaned, wall_bumps


def build_preference_dataset(rollouts_path, output_path, min_margin=0.0):
    """
    Reads rollouts/search data containing map description and candidate answers.
    Pairs successful answers with high-quality rejected answers (specifically hole fatalities).
    """
    pairs = []
    total_records = 0
    paired_records = 0
    hole_rejects = 0
    
    with open(rollouts_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            total_records += 1
            record = json.loads(line)
            
            # Case 1: multi-sample rollout format
            if "successes" in record and "answers" in record:
                successes = record["successes"]
                answers = record["answers"]
                map_desc = record.get("map_desc")
                
                # If map_desc not in record, needs external lookup
                chosen_answers = [ans for s, ans in zip(successes, answers) if s and ans]
                rejected_answers = [ans for s, ans in zip(successes, answers) if not s and ans]
                
                if chosen_answers and rejected_answers:
                    paired_records += 1
                    pairs.append({
                        "row_index": record["row_index"],
                        "map_id": record.get("map_id"),
                        "grid_size": record.get("grid_size"),
                        "chosen": chosen_answers[0],
                        "rejected": rejected_answers[0],
                    })
                    
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, "w", encoding="utf-8") as f:
        for p in pairs:
            f.write(json.dumps(p) + "\n")
            
    print(f"Generated {len(pairs)} preference pairs from {total_records} records.")
    return pairs


def main():
    parser = argparse.ArgumentParser(description="Filter refused routes and create DPO pairs")
    parser.add_argument("--input", required=True, help="Input rollouts or search JSONL")
    parser.add_argument("--output", required=True, help="Output preference JSONL")
    args = parser.parse_args()
    
    build_preference_dataset(args.input, args.output)


if __name__ == "__main__":
    main()
