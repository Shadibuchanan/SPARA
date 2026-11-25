# Hole check: writing out the holes around each cell before each move

Status, 2026-11-16: run and read. Writing the check harms (65 against EB3's 83 of 400,
p = 0.006); the lines show the model misreads the map, and it never steps into a hole it
has listed. See [Result](#result-job-7108600).

This follows [expert iteration round 3](EXPERT_ITERATION_ROUND3_SPEC.md#result-jobs-70881347088136).
Round 3 added nothing detectable, and its fixed table named the next lever as the search
or the policy's form. This changes the policy's form.

## Why

The policy fails by walking into holes, and early.
- **Holes are three failures in four.** EB3 walks into a hole on 292 of the 400 new
  held-out maps; B on 308.
- **The fatal move comes early.** Of B's 308, the median is move 3; 61 are the first
  move and 79 the second. EB3's are the same: median 3, with 50 and 73.
- **Knowing the map does not stop it.** With the map also given exactly as text, O2
  walks into a hole on 307 of 400 ([perception ceiling](PERCEPTION_CEILING_SPEC.md)).
- **Knowing its cell does not stop it.** Step grounding trained position into the
  states the moves are written from (89% decodable, against 39%), and the state often
  marks the hole ahead. Walks into holes did not fall
  ([step grounding](STEP_GROUNDING_SPEC.md#result)).

So the move is chosen badly from a state that often holds what it needs. The model
answers with the moves alone, so nothing makes each move depend on a check of the cells
around it. This experiment makes the check part of the answer: before each move, the
answer names the current cell and the directions that lead into a hole. The move is then
written after, and conditioned on, its own check.

The check is also a readout. Where a move goes into a hole, the line before it says
whether the model listed that hole and moved into it anyway, misread the cell, or lost
track of where it was.

## Design

**S3: EB3's recipe with the hole check in every answer.** It differs from EB3 in the
answer text and the generation cap only:
- **The same start:** B (job 7088134), the base of round 3.
- **The same rows:** the 3,000 rows EB3 trained on (round 1's routes and rounds 2 and 3's
  searches), shuffled with seed 42 into the same two phases. Phase 1 trains under LIVR's
  Stage 1 mask, phase 2 unmasked, each with a fresh optimizer, learning rate 1e-5,
  cosine schedule with 5% warmup, accumulation 8.
- **The same prompt.** It already asks for the final answer in `\boxed{}`, and the
  reference evaluator reads the last `\boxed{}`, so scoring is unchanged.
- **The answer:** one line per move, then the released answer unchanged. Each line is
  the cell (row, column, counted from 0 at the top left), the directions from it that
  lead into a hole (or `-`), and the move:

  ```
  3,2 hole ULR > D
  4,2 hole - > L
  4,1 hole U > L
  4,0 hole - > U
  \boxed{DOWN, LEFT, LEFT, UP}
  ```

  The lines come from the map and the route by the reference evaluator's rules. A move
  off the grid repeats the cell on the next line.
- **Length:** on round 2's 199 solved 7 x 7 and 8 x 8 maps, the lines take a median of
  48 tokens and at most 88. The released maps take a median of 24 and at most 112.
- **Generation cap:** 256 new tokens for S3, against 128 for EB3, whose answers are
  under 40 tokens.

**Evaluation:** as EB3. Unmasked greedy on the 400 maps of `detour-heldout-78`, round 1's
240 held-out maps and the first 98 maps of `detour-pool-78b`; masked greedy on the 400;
latent states; the reference benchmark. The generated text is kept for the readout.
S3's weights are saved.

## Readings (fixed before the run)

**Primary: S3 against EB3** on the 400 maps of `detour-heldout-78`. Paired by map, exact
McNemar, two-sided.

| Result | Reading | Next |
| --- | --- | --- |
| S3 solves more, p < 0.05 | Writing the check helps | The check becomes the policy's form. Build the search on it and run the next round of expert iteration with it. |
| No detectable difference | Writing the check adds nothing detectable | Decided by the fatal moves (below). |
| S3 solves fewer, p < 0.05 | Writing the check harms | Check the format and the training loss before anything else. |

**The fatal moves** (descriptive, with the next step fixed now). For each S3 answer on
the 400 that walks into a hole, the line before the fatal move is one of:
- **Listed and taken:** the line names the right cell and lists the hole's direction,
  and the move goes there anyway. The model ignores its own check.
- **Missed:** the line names the right cell but does not list the hole. The model
  misread what lies next to it.
- **Lost:** the line names the wrong cell.
- **No line:** the answer does not follow the format at that move.

Without a detectable difference, the next step follows the largest class:
- **Missed:** reading cells limits once the check is explicit. Run S3 with the map also
  given as text, as O2 did.
- **Listed and taken:** the model ignores what it writes. Train on failures as well as
  successes (preferences between the policy's failed route and the search's route on
  the same map).
- **Lost or no line:** the format is not learned well enough to read. Report and hold.

**Power,** as round 3: a gain of 16 from about 50 discordant maps is detected; half that
is not.

**Validity (the readings are withheld if any check fails):**
1. **S3 trained on EB3's rows:** the same counts (472 new maps, 387 old maps, 3,000 rows).
2. **Every training answer's lines** replay to its `\boxed{}` route, and every listed hole
   set matches the map.
3. **S3 and EB3 answer the same maps.**
4. **The cap:** at most 2% of S3's answers on the 400 reach 256 new tokens. Otherwise the
   cap, not the form, may be what is measured.

**Descriptive (no verdicts):**
- **Format:** answers whose lines all parse and replay to their `\boxed{}` route.
- **Per line, on the 400:** the cell named right; the hole set listed exactly; holes
  missed and holes invented.
- **S3 against B,** and against EB3 masked and on round 1's 240 held-out maps.
- **Failures:** into a hole, stops short, passes the goal.
- **The benchmark:** S3 against EB3 and against LIVR, overall, detour and no-detour.
- **Training loss** per phase, and GPU-seconds.

## Budget

One job on one H200, about 60 minutes: EB3's 30 minutes, plus about 50 new tokens on
each of about 1,600 greedy answers. The time limit is 120 minutes. The weights take
15 GB on scratch.

## Implementation

In `spara/expert_round2.py`, with `configs/hole_check.json` (round 3's settings with a cap
of 256):
- **`with_hole_check`** writes a row's lines in front of its answer; arm `s3` applies it
  to every training row.
- **Greedy records** keep the generated answer text.
- **`hole_check`** applies the readings above.

## Running

On PACE, with `R=$SPARA_REMOTE_ROOT/results`:

```bash
SPARA_STAGE=train SPARA_ARM=s3 SPARA_SAVE_WEIGHTS=1 SPARA_ROUND=3 SPARA_SETTINGS=configs/hole_check.json \
  SPARA_E_CHECKPOINT=$R/round3-eb2-7088134/phase2/final SPARA_ROUND1_ROUTES=$R/expert-routes-7063156/routes.jsonl \
  SPARA_SEARCH_RUN="$R/round2-search-7065168 $R/round3-search-7088135"
```

## Limits

- **One training run.** Two runs of the same recipe disagree on about 20 maps.
- **The answer gets longer,** so S3 differs from EB3 in compute per answer as well as in
  form.
- **The lines teach the hole set for every cell on the route,** which is supervision EB3
  never sees. A gain shows the form plus that supervision helps, not the form alone.

## Result (job 7108600)

Evidence: [docs/experiments/hole-check-7108600/](experiments/hole-check-7108600/), with
`hole_check.json` holding every fixed reading and `reading.json` the post hoc reading of
the lines. S3's weights are on PACE.

**Job.** 3,406 GPU-seconds (57 min) by the job's own clock, against EB3's 1,617.

**Validity: all checks passed.**
- **S3 trained on EB3's rows:** 472 new maps, 387 old maps, 3,000 rows.
- **Every training answer's lines replay** to its `\boxed{}` route.
- **S3 and EB3 answer the same maps.**
- **The cap:** no answer on the 400 reaches 256 new tokens. The longest takes 144, the
  median 64.

**Primary: writing the check harms.** On the 400 maps of `detour-heldout-78`, S3 solves
65 and EB3 83: 11 maps only S3 solves, 29 only EB3, p = 0.006. By size: 39 against 51 at
7 x 7 (6 against 18, p = 0.02) and 26 against 32 at 8 x 8 (5 against 11, p = 0.21). The
loss is larger than the rerun noise floor of about 10 maps each way.

**The fixed step: the format and the training loss.** Neither is broken.
- **Format:** 392 of 400 answers have lines that all parse and replay to their `\boxed{}`
  route.
- **Loss:** phase 1 starts at 7.6 (the new lines) and ends its last quarter at 0.18;
  phase 2 ends at 0.11, against EB3's 0.08 and 0.07. The extra loss sits on the line
  tokens, which carry the map reading that EB3's answers never write.

**The fatal moves.** S3 walks into a hole on 322 of 400 maps, against EB3's 292.
- **Missed:** 210. The line names the right cell and leaves out the hole.
- **Lost:** 112. The line names the wrong cell.
- **Listed and taken:** 0. No fatal move goes into a hole its own line listed.
- **No line:** 0.

**How the lines read the map** (post hoc, `scripts/hole_reading.py`).
- **The start cell:** right on 269 of 400 first lines. 116 more are one cell off in one
  direction, both ways about equally (35 right, 35 down, 26 left, 20 up). Each later line
  carries the offset, which is most of the lost class.
- **Holes next to the agent,** on the 959 lines naming the right cell: 277 of the 797
  hole directions listed (35%), with 149 directions invented. The exact set is right on
  451 lines (47%), against 370 (39%) for writing `-` every time.
- **Lines naming the wrong cell** match the true cell's holes on 206 of 442 and the
  named cell's on 148. Some offsets are labels on a correct reading, some are not.

**Descriptive.**
- **S3 against B:** 65 against 76, 20 against 31, p = 0.16.
- **Masked:** S3 62 against EB3 93, p < 0.0001.
- **Round 1's 240 held-out maps:** S3 77 against EB3 111, p < 0.0001.
- **Failures:** S3 322 into a hole, 13 stop short; EB3 292 and 24.
- **The benchmark:** S3 313, EB3 330 (p = 0.02), LIVR 323 (p = 0.17). On the 40 detour
  maps S3 solves 13 against LIVR's 7; on the 360 others, 300 against 316 (p = 0.01).

**What this says.** When the model writes a hole down, it avoids it every time. When it
must write out what lies next to it, it reads a third of the holes and misplaces the
start on a third of the maps. EB3, which never writes the reading, does better, so the
explicit lines expose less than the policy uses. Both ways point at reading the image,
not at choosing the move. That is the opposite of this spec's premise, which read step
grounding and O2 as showing a move-choice failure.

**Next.** The fixed table ends at the format and loss check, so the next step is not
fixed. The fatal-move rule fixed for a null names the same step as the largest class
here (missed): S3 with the map also given as text. Paired with EB3's recipe on the text
map without the check, it separates the two parts. If the check helps once the reading is
exact, the form is sound and image reading is the limit, which is Raymond's perception
lever.

**Follow-up.** That pair ran as the [hole check on the exact map](HOLE_CHECK_TEXT_SPEC.md#result-jobs-7125957-and-7125958).
The check still harms, and the hole lists are as poor from the exact text map (33%) as
from the image (35%). So the lines above measure the working out of neighbouring cells,
not image reading.
