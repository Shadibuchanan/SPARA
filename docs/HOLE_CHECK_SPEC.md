# Hole check: writing out the holes around each cell before each move

Status, 2026-11-16: run and read. Writing the check harms (65 against EB3's 83 of 400,
p = 0.006); the lines show the model misreads the map, and it never steps into a hole it
has listed. See [Result](#result-job-7108600).

Reading rows (S3R), 2026-11-17: run and read. The check still harms after a phase of
reading rows (64 against EB3's 83 of 400, p = 0.018). The model lists the holes around
the drawn agent (65%) and writes `-` for cells named by coordinates (0.8% listed). See
[Result](#result-job-7132192).

Oracle check, 2026-11-18: run as jobs 7154267 and 7155886. Three validity checks fail, so
the fixed reading is withheld and the numbers are descriptive. With the true hole lists
written for it S3 solves 132 of 400, and with the cells as well 166, against 65 unforced
and EB3's 83. See [Result](#result-jobs-7154267-and-7155886).

Glimpse latents (G3), 2026-11-18: run as job 7154420 and valid. No detectable difference
from EB3 (81 against 83 of 400, p = 0.88), and the lines list 45% of the holes, so they do
not read. G3 does beat S3 (65, p = 0.020) and its own blind pass (68, p = 0.004). See
[Result](#result-job-7154420).

Reading rows with glimpses (G3R), 2026-11-18: run as job 7159270 and valid. After the
reading phase the probe lists 9 of 2,403 hole directions around named cells (0.4%), the
same as with the glimpses blind and as S3R's 0.8%, so the latents do not read under
labels for every cell. Held. See [Result](#result-job-7159270).

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

## Reading rows: S3R

Both hole checks listed a third of the holes next to the agent, from the image (35%) and
from exact text (33%). Each arm saw about 18,000 check lines in 375 optimizer steps, all on
route cells, and every line also had to choose a move. Raymond's EGD reached the same
place from the latents: the model steps into holes its block latents flag
([latent JEPA spec](PERCEPTION_JEPA_SPEC.md#result-egd-jobs-7124258-and-7125522)).

**Question: can the model learn to list the holes next to a cell from the image, given
labels for every cell, and does the hole check then help?**

### Design

**S3R: a reading phase, then S3's recipe unchanged.**
- **Start:** B (job 7088134), as S3.
- **Phase 0, new:** reading rows on the 500 maps of `detour-pool-78b`, unmasked, with a
  fresh optimizer on S3's schedule (learning rate 1e-5, cosine, 5% warmup, accumulation 8).
- **Phases 1 and 2:** S3's 3,000 rows, order, mask and schedule.
- **A reading row** keeps the released prompt up to the image and replaces the request
  for a plan. It asks for the start cell, which the prompt does not name, and then up to 8
  named cells. The answer is one line per cell, in the check's own form without the move:

  ```
  6,4 hole R
  3,2 hole UL
  4,2 hole -
  ```

- **Coverage:** each map's safe cells are shuffled (seed 42) and cut into groups of 8, so
  the rows ask about every cell the agent can stand on. That is 2,951 rows and 24,825
  lines, none of them on a held-out map. Writing `-` every time is right on 43% of them.
- **The probe:** one reading row per map of `detour-heldout-78`, answered greedily after
  phase 0 and again after phase 2.
- **Evaluation:** as S3. No weights are saved; scratch is at 93%.

### Readings (fixed before the run)

**Primary: S3R against EB3** on the 400 maps of `detour-heldout-78`. Paired by map, exact
McNemar, two-sided.

**Reads** means at least 90% in both shares, the threshold the exact-map check used:
- **The probe reads** if, after phase 0, its first lines name the start on 90% of maps and
  its lines list 90% of the hole directions.
- **The answers read** if, after phase 2, S3R's first check line names the start on 90% of
  the 400 and its lines naming the right cell list 90% of the hole directions.

| Result | Reading | Next |
| --- | --- | --- |
| S3R solves more, p < 0.05 | The check helps once the reading is trained | The check becomes the policy's form. Run the next round of expert iteration with it. |
| No difference or fewer, and the answers read | The check does not help even when read right | Drop the form and keep EB3's. |
| No difference or fewer, the probe reads and the answers do not | The reading is learned and then not used in the answers | Train the reading rows together with the routes, not before them. |
| No difference or fewer, and the probe does not read | Labels for every cell do not teach the holes next to a cell | Report and hold. |

**Validity (the readings are withheld if any check fails):**
1. **S3R's phases 1 and 2 train on EB3's rows:** the same counts (472 new maps, 387 old
   maps, 3,000 rows).
2. **Every S3 training answer's lines** replay to its `\boxed{}` route.
3. **S3R, S3 and EB3 answer the same maps.**
4. **The cap:** at most 2% of S3R's answers on the 400, and of its probe answers, reach 256
   new tokens.

**Descriptive (no verdicts):** S3R against S3; the fatal-move classes; the probe's shares
after phase 0 and after phase 2, with writing `-` every time as the baseline; training
loss per phase; GPU-seconds.

**Budget:** one job on one H200, about 80 minutes: S3's 57, about 12 for phase 0 and about
10 for the two probes. The time limit is 120 minutes.

**Limits:** one run; the reading rows change what the model has seen as well as what it
can read, so a gain shows the two together.

**Implementation:** `reading_rows`, `reading_score` and `probe_lines` in `spara/hole_check.py`; arm `s3r`
and `reading_check` in `spara/expert_round2.py`. Run as S3 with `SPARA_ARM=s3r` and
`SPARA_SETTINGS=configs/hole_check_text.json`.

### Result (job 7132192)

Evidence: [docs/experiments/hole-check-reading-7132192/](experiments/hole-check-reading-7132192/),
with `reading_check.json` holding every reading below except the one marked post hoc.

**What changed from the plan.** Nothing in the design. The job took 5,701 GPU-seconds
(95 minutes) by its own clock against the 80 estimated: phase 0 took 1,121 seconds, not 12
minutes.

**Validity: all checks passed.**
- **Phases 1 and 2 trained on EB3's rows:** 472 new maps, 387 old maps, 3,000 rows.
- **Every training answer's lines replay** to its `\boxed{}` route.
- **S3R, S3 and EB3 answer the same maps.**
- **The cap:** no answer and no probe answer reaches 256 new tokens. The longest answer
  takes 124.

**Primary: the check still harms.** On the 400 maps of `detour-heldout-78`, S3R solves 64
and EB3 83: 20 maps only S3R solves, 39 only EB3, p = 0.018. By size: 35 against 51 at
7 x 7 (9 against 25, p = 0.009) and 29 against 32 at 8 x 8 (11 against 14, p = 0.69).

**The probe does not read.** The table's fourth row applies.
- **After phase 0:** the first line names the start on 379 of 400 maps (95%). The lines
  list 263 of the 2,796 hole directions (9%) and invent 131. The exact set is right on
  1,631 of 3,600 cells (45%), against 1,555 (43%) for writing `-` every time.
- **Not a format failure:** 3,579 of the 3,600 cells are answered in the form.
- **The start line reads and the other lines do not.** `probe_by_line`, added to the
  readout after the run, splits the lines.
  - **The start line,** where the agent is drawn: 246 of 378 hole directions listed
    (65%), 108 invented.
  - **The other lines,** for cells that only their coordinates name: 20 of 2,400 listed
    (0.8%). 3,170 of these 3,221 lines write `-`.
- **After phase 2:** no answer has a line in the reading form. Phases 1 and 2 replaced it
  with the plan's form, so this probe says nothing about what the model kept.

**The answers do not read either,** though the start improved.
- **The start cell:** right on 349 of 400 first lines (87%), against S3's 269 (67%).
- **Holes next to the agent,** on the 1,297 lines naming the right cell: 486 of 1,082
  hole directions listed (45%), against S3's 35%.
- **By line:** 78% of holes listed on the first line (277 of 353), against S3's 45%.
  Later lines list 29% (209 of 729), the same as S3's 29%.
- **Listed and taken:** 0 again. Of 310 falls, 268 are missed holes, 41 are on lines
  naming the wrong cell (S3: 112), and 1 has no line.
- **Format:** 396 of 400 answers have lines that parse and replay to their route.

**Descriptive.**
- **S3R against S3:** 64 against 65, 23 against 24, p = 1.0.
- **S3R against B,** its start: 64 against 76, 22 against 34, p = 0.14.
- **Where the falls happen** (post hoc, from the routes in `greedy.jsonl`): S3R falls on
  its first move on 37 maps and later on 273. S3: 64 and 258. EB3: 50 and 242. The better
  start line removes first-move falls, and the same maps then fail on a later move.
- **Round 1's 240 held-out maps:** 82 against EB3's 111, p < 0.001.
- **The benchmark's 400 test maps:** 310 against EB3's 330, p = 0.015.
- **Masked read:** 56 against EB3's 93 of 400.
- **Training loss:** phase 0 falls from 0.95 in its first quarter to 0.37, 0.33 and 0.30.
  Last quarter of phase 2: 0.10, against S3's 0.11.

**What this says.**
- **The model reads the cells around the drawn agent and not around a cell it must find
  by its coordinates.** Phase 0 taught the first and left the second at `-`. The plan
  needs the second from its second move on, and those lines list 29% of the holes in S3,
  S3T and S3R alike.
- **This agrees with the exact-map check and with EGD.** The text map did not help the
  later lines, and EGD's answers step into holes that its block latents flag
  ([latent JEPA spec](PERCEPTION_JEPA_SPEC.md#result-egd-jobs-7124258-and-7125522)). In
  all three the failing step is tying a position worked out in text to the map.
- **A better first line does not raise success.** It moves the fall one or more moves
  later.

**Limits of this reading.**
- **One run, and 369 optimizer steps for a new skill.** The phase 0 loss was still
  falling. The result says this much training does not teach the named cells, not that
  no amount does.
- **Writing `-` is right on 45% of the named lines,** so the loss reaches that answer
  early and can stay there.
- **The probe asks 8 named cells per map,** once.
- **No weights were saved,** so the phase 0 model cannot be probed further.

**Next: report and hold,** as the table fixes. The open question is whether the named-cell
reading is short of training or out of reach in this form. A longer phase 0 on named
cells alone, probed as it trains, would separate the two. That is a proposal, not a fixed
step.

## Oracle check: does a right reading make a safe route?

S3, S3T and S3R each tried to make the lines read the map, and each assumed that a right
reading would then give a safe route. None got the reading right, so that assumption has
never been tested. Per-move latents that read the cells around the agent rest on it too.
This check tests it directly, with no training.

**Design.** S3's saved weights (job 7108600) answer the 400 maps of `detour-heldout-78`
line by line, greedily. The model always writes each line's move and the final
`\boxed{}` route. Three passes differ in what the map supplies:
- **Free:** nothing. The model writes every line itself.
- **Holes forced:** the model names the cell, and the true hole list of that cell is
  written for it.
- **Cell forced:** the cell is also written for it, as the cell its moves so far lead to.

S3 trained on lines that were always right, so forced lines are what it saw in training.

### Readings (fixed before the run)

**Primary: the cell-forced pass against EB3** on the 400. Paired by map, exact McNemar,
two-sided.

| Result | Reading | Next |
| --- | --- | --- |
| Cell-forced solves more, p < 0.05 | A right reading makes safe routes | Build per-move reading latents: before each move, one latent per direction that may attend only to that neighbouring cell. The cell-forced count bounds what they can buy. |
| No difference or fewer | The policy does not turn right readings into routes | Reading latents cannot help in this form. The limit is choosing the route over several moves. Report, and take that up instead. |

**Validity (the readings are withheld if any check fails):**
1. **The free pass reproduces S3:** its route equals S3's recorded greedy route on at
   least 95% of the 400. This shows that writing line by line does not change the model's
   answers.
2. **The cap:** at most 2% of answers in each pass reach 256 new tokens.
3. **The forced lines are right:** in the cell-forced pass every line names the cell the
   moves lead to and lists its holes.

**Descriptive (no verdicts):** the holes-forced pass against EB3 and against the free
pass; each pass by map size; how the forced passes fail (into a listed hole, stops
short, passes the goal).

**Power,** as round 3: a gain of 16 from about 50 discordant maps is detected; half that
is not.

**Budget:** one job on one H200, about 45 minutes: three passes of about 6,000 short
generations each. The limit is 90 minutes. Nothing is saved beyond the records.

**Limits:** one model; an oracle is a ceiling, not a method, because it reads the map.

**Implementation:** `forced_answer` in `spara/hole_check.py`; `oracle_check` in
`spara/expert_round2.py`, run with `SPARA_STAGE=oracle-check`, `SPARA_S3_CHECKPOINT` and
`SPARA_SETTINGS=configs/hole_check.json`. `SPARA_ORACLE_OUTPUT` resumes a run cut short.

### Result (jobs 7154267 and 7155886)

Evidence: [docs/experiments/oracle-check-7154267/](experiments/oracle-check-7154267/), with
`oracle_read.json` holding every reading below except those marked post hoc.

**What changed from the plan.** The three passes took 31, 27 and about 32 minutes, not 15 each,
because every line re-encodes the image. Job 7154267 reached its 90-minute limit with 396
of the cell-forced pass's 400 maps written. `oracle_check` was made to resume, and job
7155886 wrote the last 4 maps in 49 seconds. The first 396 rows are byte for byte the ones
the first job wrote.

**Validity: three checks fail, so the fixed primary reading is withheld.** Each is listed
with what was found when it was looked into.
1. **The free pass reproduces S3 on 377 of the 400 routes, and the rule needs 380.**
   - 360 answers are identical, and on each of them the token count equals S3's less the
     end token, so re-tokenizing the answer text gives the tokens the model generated.
   - The other 40 first differ in one hole list (29), one move (8) or the boxed route (3).
   - Both solve 65 maps: one map only S3 solves, one only the free pass.
   - S3's recorded answers are one cached generation each. The free pass starts a new
     generation at every stop. No further cause was established.
2. **The cap: 19 cell-forced answers reach 256 new tokens, and the rule allows 8.** The
   free pass has none and the holes-forced pass 4. In 18 of the 19 the route returns to a
   cell or repeats a move into the map's edge (post hoc). A capped answer counts as a
   failure, so the cap can only lower the forced counts.
3. **The forced lines: the readout finds every line right on 396 of 400 maps.** On the
   other 4 the lines are the forced ones, and the `\boxed{}` route the model wrote is not
   the lines' moves (in 3 it miscounts a repeated move). The check aligns the lines with
   the scored route, so it cannot tell these apart.

**The numbers, descriptive only.** On the 400 maps of `detour-heldout-78`, paired by map:

| Pass | Solved | Against EB3 (83) | 7 x 7 (198) | 8 x 8 (202) |
| --- | --- | --- | --- | --- |
| Free (S3's own lines) | 65 | | 39 | 26 |
| Holes forced | 132 | 75 only forced, 26 only EB3, p = 1e-6 | 71 | 61 |
| Cell forced | 166 | 98 only forced, 15 only EB3, p = 4e-16 | 87 | 79 |

- **Cell forced against the free pass:** 106 maps only the forced pass solves, 5 only the
  free pass.
- **Cell forced against holes forced:** 35 against 1.
- **Checks 1 and 2 do not threaten the direction.** The free pass and S3 differ by one map
  each way, against gaps of 67 and 101 maps, and the cap can only lower the forced counts.

**A right reading more than doubles S3's routes, and 234 of 400 maps still fail.** In the
cell-forced pass:
- **152 stop short of the goal.** 18 of them are capped answers. Of the other 134, 97
  write fewer moves than the shortest safe route, 57 of them exactly one fewer, and 73 end
  one step from the goal (post hoc).
- **79 fall into a hole.** 69 are "listed and taken": the line lists the hole and the move
  goes into it. No earlier run has one, because S3's own lists agreed with its moves.
- **The steps into a listed hole are refused detours (post hoc).** Of the 76 lines that
  step into a hole they list, in 59 the hole lies toward the goal and every safe move
  leads away from it, counting distance in rows and columns. In 13 a safe move toward the
  goal was open, and in 4 the hole lies away from the goal.
- **3 pass the goal and move on.**
- **Routes get longer:** the median is 8 moves, against 6 in the free pass and 7 for EB3.

**What this supports.** Reading the holes is worth about 100 maps of 400 to this policy,
which is why per-move reading latents are worth testing (G3 below). It is not sufficient:
with every line right, the policy still fails where the route must turn away from the
goal, and it ends routes early. Aarya's hole-veto decoding (PR #9: no deaths, 181 of 400)
found the same limit from the other side.

**Limits.** No verdict from the fixed table, for the three reasons above. One model, one
set of weights. The oracle reads the map, so 166 is a ceiling for S3's policy with these
lines, not for a model trained with reliable readings.

## Glimpse latents: G3

Per-move latents for the hole check. S3, S3T and S3R list a third of the holes on lines
after the first, because the model cannot look up the map at a cell that only its
coordinates name ([reading rows](#result-job-7132192)). G3 gives that lookup to the
attention mask and leaves the latent one job: say whether one cell is a hole.

**Question: with a latent that sees each neighbouring cell before every move, do the lines
read the map, and do the routes then get safer?**

### Design

**G3: S3 with four latents after each line's cell.**
- **The line:** `5,3<U><D><L><R> hole U > L`. The four tokens are LIVR's own latent tokens,
  one per direction. The model writes them itself, and they are dropped when the answer is
  read, so the text is S3's.
- **The glimpse:** of the image, each written latent may attend only to the cell next to
  the named cell in its direction. A latent whose cell is off the grid, or that follows
  no cell, sees no image token. It sees all the text before it. The rule holds in both
  stages and at test time.
- **Where the cell comes from:** the three tokens the model wrote before the latents. No
  map data reaches the mask.
- **Where the grid comes from:** the image. Maps are drawn at 64 pixels per cell, so the
  number of image tokens gives the grid size. Each image token belongs to the cell that
  covers most of it.
- **Everything else is S3's:** the start (B), the 3,000 rows and their order, phase 1
  under LIVR's Stage 1 mask and phase 2 unmasked, the schedule and the evaluation. In
  phase 1 the answer's text reaches the image only through the prompt latents and the
  glimpses.
- **The cap is 384 new tokens,** not S3's 256. A line is 12 tokens with its latents and
  8 without, so the same number of lines needs half as many tokens again. A 16-move
  training answer takes 228.
- **The ablation:** after training, the 400 held-out maps again with every glimpse blind.
- **No helper images,** and no weights saved.

### Readings (fixed before the run)

**Primary: G3 against EB3** on the 400 maps of `detour-heldout-78`. Paired by map, exact
McNemar, two-sided.

**The lines read** if, on lines naming the right cell, they list at least 90% of the
directions that lead into a hole. S3 listed 35% and S3R 45%.

| Result | Reading | Next |
| --- | --- | --- |
| G3 solves more, p < 0.05 | Glimpse latents help | Let RL adjust the glimpse latents, which is the project's question on latents the answer uses. Run a round of expert iteration with this form. |
| No difference or fewer, and the lines read | The reading is right and the routes are not | The limit is choosing the route over several moves. The oracle check above bounds the same thing without training. |
| No difference or fewer, and the lines do not read | The glimpses did not teach the lines in 375 steps | Compare with the blind pass, check the mask on real images, and hold. |

**Validity (the readings are withheld if any check fails):**
1. **G3 trains on EB3's rows,** and every training answer's lines replay to its route,
   checked before the latents are added.
2. **G3, S3 and EB3 answer the same maps.**
3. **The cap:** at most 2% of G3's answers on the 400 reach 384 new tokens.
4. **The latents are written:** at least 98% of G3's answers on the 400 carry exactly
   four latents per line.

**Descriptive (no verdicts):** hole directions listed on first and later lines, for G3,
the blind pass and S3; G3 against the blind pass and against S3; the masked read; the
fatal-move classes; the benchmark; training loss.

**Budget:** one job on one H200, about 75 minutes: S3's 57, longer answers, and the
blind pass. The limit is 120 minutes.

**Limits:**
- **One run.**
- **The mask knows the grid.** It relies on square maps drawn at a fixed cell size. That
  is a fact about this benchmark, as it is for the block latents of the latent JEPA line.
- **A wrong cell gives a wrong glimpse.** S3R named the right cell on 88% of lines.

**Implementation:** `spara/glimpse.py` (the mask and the row format); arm `g3` and
`glimpse_check` in `spara/expert_round2.py`. Run as S3 with `SPARA_ARM=g3` and
`SPARA_SETTINGS=configs/glimpse.json` (the hole check's settings with the cap at 384 and no
saved weights).

### Result (job 7154420)

Evidence: [docs/experiments/glimpse-latents-7154420/](experiments/glimpse-latents-7154420/),
with `glimpse_check.json` holding every reading below except the one marked post hoc.

**What changed from the plan.** Nothing in the design. The job took 5,155 seconds (86
minutes) by its own clock against the 75 estimated.

**Validity: all checks passed.**
- **G3 trained on EB3's rows:** 472 new maps, 387 old maps, 3,000 rows, and every training
  answer's lines replay to its route.
- **G3, S3 and EB3 answer the same maps.**
- **The cap:** no answer reaches 384 new tokens. The longest takes 172.
- **The latents are written:** all 400 answers carry four latents on every line.

**Primary: no detectable difference.** On the 400 maps of `detour-heldout-78`, G3 solves
81 and EB3 83: 22 maps only G3 solves, 24 only EB3, p = 0.88. By size: 51 against 51 at
7 x 7 and 30 against 32 at 8 x 8.

**The lines do not read,** so the table's third row applies. On lines naming the right
cell, G3 lists 351 of the 787 directions that lead into a hole (45%), against the 90%
that counts as reading.

| Hole directions listed | G3 | G3 blind | S3 |
| --- | --- | --- | --- |
| All lines naming the right cell | 351 of 787 (45%) | 254 of 749 (34%) | 277 of 797 (35%) |
| First line | 159 of 263 (60%) | 124 of 263 (47%) | 132 of 292 (45%) |
| Later lines | 192 of 524 (37%) | 130 of 486 (27%) | 145 of 505 (29%) |
| Up | 68 of 185 (37%) | 59 of 175 (34%) | 57 of 194 (29%) |
| Down | 114 of 219 (52%) | 40 of 214 (19%) | 63 of 225 (28%) |
| Left | 76 of 173 (44%) | 72 of 164 (44%) | 74 of 173 (43%) |
| Right | 93 of 210 (44%) | 83 of 196 (42%) | 83 of 205 (40%) |

**The blind pass: the glimpses carry part of the reading.**
- **G3 against its blind pass:** 81 against 68. 16 maps only the sighted pass solves, 3
  only the blind pass, p = 0.004.
- **G3 against S3:** 81 against 65 (29 against 13, p = 0.020). G3 is the first hole-check
  arm that is not below EB3: S3 solved 65, S3T 70 and S3R 64.
- **Almost all of the gain in listing is one direction.** Blinding takes down from 52% to
  19% and leaves left and right where they were. The model learned to use the down
  latent, used the up latent a little, and did not learn the other two in 375 steps.
- **Invented directions:** 113, against 114 blind and 149 for S3.

**The mask on real images (post hoc).** On 50 of the 400 maps with the real processor,
each cell's mean patch was taken from the image tokens the mask assigns to it, and classed
hole or safe by the nearer class mean, with the means fitted on half the 2,695 cells. On
the other half it is right on 1,347 of 1,347. With rows and columns swapped it is right
on 943, and finds 93 of the 287 holes. So the mask points each latent at the right
pixels. That the model keeps the processor's token order is read from its code, not
measured here.

**A wrong cell gives a wrong glimpse on a third of lines.** G3 names the right cell on
982 of 1,522 lines (65%), and the first line names the start on 262 of 400 maps. S3 named
it on 269.

**Descriptive.**
- **Falls:** 303 routes fall, 194 on a hole the line missed and 109 on a line naming the
  wrong cell. None is listed and taken.
- **The masked read:** 59 against EB3's 93 (7 against 41).
- **The benchmark:** 314 against EB3's 330 of 400 (15 against 31, p = 0.026), and 86
  against 111 on the 240 round-1 held-out maps (p = 0.002). The hole-check form still
  costs on maps that need no detour.
- **Training loss,** last quarter of phase 2: 0.076, against EB3's 0.065.

**What this supports.** Glimpse latents do part of the job: they lift the hole check from
below EB3 to level with it, and blinding them takes that back. They are far from the
reading the [oracle check](#result-jobs-7154267-and-7155886) forced (132 to 166 of 400).
One reading, not tested: the route loss alone gives the latents little reason to report
what they see. The table says hold. The cheapest next test is the reading
rows of [S3R](#reading-rows-s3r) with glimpses: dense labels on every cell, where S3R
listed 0.8% around named cells because the lookup failed.

**Limits.** One run. The mask relies on the benchmark's fixed cell size. Why three
latents did not learn was not established.

## Reading rows with glimpses: G3R

G3's lines list 45% of the holes, and one of its four latents does most of that
([result](#result-job-7154420)). G3 trained on routes only, so nothing asked each latent
to report its cell. S3R's reading rows label every cell, and without glimpses the model
listed 0.8% of the holes around cells named by coordinates
([result](#result-job-7132192)). G3R gives those rows the glimpses.

**Question: with a label on every cell, do glimpse latents read the holes around a cell
that only its coordinates name?**

### Design

**G3R: S3R's reading phase with glimpses, and nothing after it.**
- **Start:** B (job 7088134), as S3R and G3.
- **Training:** S3R's phase 0 unchanged: the same 2,951 reading rows on the 500 maps of
  `detour-pool-78b`, in the same order, unmasked, on the same schedule. The one
  difference is G3's four latents after each line's cell, under G3's mask:

  ```
  6,4<U><D><L><R> hole R
  3,2<U><D><L><R> hole UL
  ```

- **The probe:** S3R's, one reading row per map of `detour-heldout-78`, answered greedily.
  Then the same probe with every glimpse blind.
- **No route phases, no benchmark, no weights saved.** The run stops after the probes.

### Readings (fixed before the run)

**Primary: the share of hole directions the probe lists on lines for named cells,** which
are every line but a first line naming the start. S3R listed 20 of 2,400 (0.8%).

| Share listed | Reading | Next |
| --- | --- | --- |
| 90% or more | Glimpse latents read under labels for every cell, so G3 lacked supervision | Run the reading phase and then S3's phases with glimpses, and read it against EB3 and the oracle check's 132 to 166. |
| 50% to 90% | They read part of the map | Report by direction and against the blind probe, and hold. |
| Under 50% | They do not read even under labels for every cell | The form is the limit. Report and hold. |

**Validity (the reading is withheld if any check fails):**
1. **The probe answers all 400 maps.**
2. **The cap:** at most 2% of probe answers reach 384 new tokens.
3. **The latents are written:** at least 98% of probe answers carry four latents on every
   line.

**Descriptive (no verdicts):** the blind probe; the start line; the share by direction;
directions invented; phase 0's loss; S3R's probe beside it.

**Budget:** one job on one H200, about 30 minutes: S3R's phase 0 took 19, and each probe
about 5. The limit is 50 minutes.

**Limits:**
- **One run.**
- **The probe names its cells in the prompt.** In a route's lines the model must also name
  its own cell, which G3 got right on 65% of lines.
- **A reading phase says nothing about routes.** That is the first row's next step.

**Implementation:** arm `g3r` in `spara/expert_round2.py`, which with `reading_only` set
stops after the probes; `glimpse_reading` reads it. Run as G3 with `SPARA_ARM=g3r` and
`SPARA_SETTINGS=configs/glimpse_reading.json` (G3's settings with `reading_only`).

### Result (job 7159270)

Evidence: [docs/experiments/glimpse-reading-7159270/](experiments/glimpse-reading-7159270/),
with `glimpse_reading.json` holding every reading below except those marked post hoc.

**Validity: all three checks pass.** The probe answers all 400 maps, no answer reaches the
cap (the longest is 92 new tokens), and all 400 answers carry four latents on every line.

**Primary: the probe lists 9 of 2,403 hole directions around named cells (0.4%), so the
table's third row applies.** Glimpse latents do not read even with a label on every cell.
Report and hold.

| Probe | Named cells: holes listed | Named lines that write "-" | Start line: holes listed |
| --- | --- | --- | --- |
| G3R | 9 of 2,403 (0.4%) | 3,202 of 3,221 | 247 of 379 (65%) |
| G3R, glimpses blind | 9 of 2,403 (0.4%) | 3,199 of 3,221 | 250 of 379 (66%) |
| S3R (no glimpses) | 20 of 2,400 (0.8%) | 3,170 of 3,221 | 246 of 378 (65%) |

**Descriptive.**
- **The glimpses change nothing the probe can see.** The blind probe lists the same 9
  directions by count, and 383 of its 400 answers are identical to the seeing probe's
  (post hoc). The other 17 show that the blind setting was applied.
- **By direction, around named cells:** up 7 of 625, left 2 of 593, down 0 of 590, right 0
  of 595. The blind probe has the same counts.
- **Invented directions on named lines:** 18 (blind 21, S3R 43).
- **The start line is unchanged:** 65%, as in S3R and in the blind probe.
- **Phase 0's loss** falls from 11.3 to 0.25 within the first eighth of its 369 steps and
  then slowly to 0.17 (post hoc, by eighths: 1.98, 0.25, 0.22, 0.21, 0.19, 0.19, 0.18,
  0.17). S3R's ended at 0.30. The two are not comparable, because each G3R line carries
  four latent tokens that are easy to predict.
- **Time:** the job took 47 minutes of its 50, against 30 planned. Training took 15, as
  planned. Each probe took about 15, against 5 planned. The cause was not measured.

**What this supports.** Two reading phases with a label on every cell, one without glimpses
(S3R) and one with them (G3R), end at the same place: the model writes "-" for a named
cell. Handing each latent the right patch of the image did not change that. In G3 the
route phases did teach one glimpse latent to carry a reading, so the latents can carry
one. This reading phase did not teach them to.

**Limits.**
- **One run, 369 steps.** The loss was still falling slowly at the end. The table fixed
  "the form is the limit" for this outcome before the run, but the run cannot tell that
  apart from a reading phase too short for the latents to learn. No run with more steps
  was made.
- **No state probe.** Whether a glimpse latent's state separates hole from safe was not
  measured, and no weights were saved. So the failure may lie in the latent not encoding
  its cell or in the line not using it.
- **The probe names its cells in the prompt,** as in the design's limits.
