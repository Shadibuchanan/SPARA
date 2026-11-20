"""Analyze detour vs non-detour maps in the dataset.

A map requires a detour if the shortest path around holes is longer than
the Manhattan distance, or if any valid shortest path must step away from the goal.
"""
import argparse
from collections import deque, Counter
import json
from pathlib import Path

from spara.data import records


ACTIONS = [(-1, 0), (1, 0), (0, -1), (0, 1)]


def find_start_and_goal(map_desc):
    start, goal = None, None
    H, W = len(map_desc), len(map_desc[0])
    for r in range(H):
        for c in range(W):
            if map_desc[r][c] == 1:
                start = (r, c)
            elif map_desc[r][c] == 2:
                goal = (r, c)
    return start, goal, H, W


def analyze_map(map_desc):
    start, goal, H, W = find_start_and_goal(map_desc)
    manhattan = abs(goal[0] - start[0]) + abs(goal[1] - start[1])
    
    # BFS
    queue = deque([(start[0], start[1], 0)])
    visited = {start}
    shortest_len = None
    
    while queue:
        r, c, d = queue.popleft()
        if (r, c) == goal:
            shortest_len = d
            break
        for dr, dc in ACTIONS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < H and 0 <= nc < W and (nr, nc) not in visited:
                if map_desc[nr][nc] != -1:
                    visited.add((nr, nc))
                    queue.append((nr, nc, d + 1))
                    
    is_detour = (shortest_len is not None) and (shortest_len > manhattan)
    solvable = shortest_len is not None
    extra_steps = (shortest_len - manhattan) if solvable else None
    
    return {
        "solvable": solvable,
        "manhattan": manhattan,
        "shortest_path": shortest_len,
        "extra_steps": extra_steps,
        "is_detour": is_detour,
        "grid_size": H,
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", default="data/vsp")
    args = parser.parse_args()
    
    for split in ["train", "test"]:
        rows = records(args.data_root, split)
        analyses = [analyze_map(r["map_desc"]) for r in rows]
        
        detours = sum(1 for a in analyses if a["is_detour"])
        no_detours = len(analyses) - detours
        
        print(f"\n=== Split: {split} (Total {len(rows)} maps) ===")
        print(f"  No Detour needed: {no_detours} ({no_detours/len(rows)*100:.1f}%)")
        print(f"  Detour needed:    {detours} ({detours/len(rows)*100:.1f}%)")
        
        by_grid = Counter((a["grid_size"], a["is_detour"]) for a in analyses)
        for g in sorted(set(a["grid_size"] for a in analyses)):
            n_det = by_grid[(g, True)]
            n_nodet = by_grid[(g, False)]
            tot = n_det + n_nodet
            print(f"  Grid {g}x{g}: Detour = {n_det}/{tot} ({n_det/tot*100:.1f}%), No Detour = {n_nodet}/{tot} ({n_nodet/tot*100:.1f}%)")


if __name__ == "__main__":
    main()
