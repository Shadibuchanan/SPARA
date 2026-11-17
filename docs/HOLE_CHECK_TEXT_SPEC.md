# Hole check on the exact map: does the check work once the reading is right?

Status, 2026-11-16: run and read. The check harms even when the map is exact (S3T 70
against O3's 92 of 400, p = 0.005). With the exact text map the lines name the start on
92% of maps but still list a third of the holes next to the agent, as from the image.
See [Result](#result-jobs-7125957-and-7125958).

This follows the [hole check](HOLE_CHECK_SPEC.md#result-job-7108600). S3 wrote the holes
around each cell before each move and solved fewer maps than EB3. Its lines showed why:
the model never stepped into a hole it had listed, but it listed 35% of the holes next
to it and misplaced the start on a third of the maps.

## Why

Two earlier results pull against each other.
- **The exact map alone changes nothing.** With the map also given as text, O2 solved 77
  maps against 72 without it, and T2, with the text map and a blank image, solved 69
  ([perception ceiling](PERCEPTION_CEILING_SPEC.md)). Both still walked into a hole on
  more than 300 of 400 maps.
- **A hole written down is a hole avoided.** In S3, 0 of 322 falls went into a hole the
  line before had listed.

O2 and T2 answered with the moves alone, so nothing made a move depend on a reading of
the cells around it. S3 made the reading explicit, from the image, and the reading was
poor. Nobody has put the two together: the exact map and the written check. That run
answers three questions at once.
- **Does the check work when the reading can be exact?** If so, the form is sound.
- **How much could better image reading buy?** The gain over EB3 bounds it.
- **Can the model read cells and neighbours from an exact text map at all?** If its
  lines are wrong even then, the limit is working out positions, not reading the image.

## Design

Two arms, each an existing arm with the text map added to every prompt, in training and
in evaluation.
- **S3T: S3 with the text map.** The hole check leads every answer.
- **O3: EB3 with the text map.** The control, with the released answer alone. O2 is not
  the control, because it started from E under E2's recipe.

Everything else is S3's and EB3's recipe.
- **The same start:** B (job 7088134).
- **The same rows:** the 3,000 rows EB3 and S3 trained on, shuffled with seed 42 into the
  same two phases. Phase 1 trains under LIVR's Stage 1 mask, phase 2 unmasked, each with
  a fresh optimizer, learning rate 1e-5, cosine schedule with 5% warmup, accumulation 8.
  The mask hides the image from the answer, not the text map.
- **The text map:** O2's, written after the image, one row per line, `S`, `G`, `H` and
  `F`. The image stays.
- **The generation cap:** 256 new tokens for both arms.
- **No weights saved.** Neither arm is deployable, because the prompt holds the layout.

**Evaluation:** as O2. Before training, B's zero-shot answers with the text map on the
400 maps of `detour-heldout-78`. After training, unmasked greedy with the text map on
the 400, round 1's 240 held-out maps and the first 98 maps of `detour-pool-78b`. No
masked read, latent states or benchmark.

## Readings (fixed before the run)

**Primary: S3T against O3** on the 400 maps of `detour-heldout-78`. Paired by map, exact
McNemar, two-sided.

**The lines read the text map** if, on the 400, S3T's first line names the start on at
least 90% of maps and its lines naming the right cell list at least 90% of the directions
that lead into a hole.

| Result | Reading | Next |
| --- | --- | --- |
| S3T solves more, p < 0.05 | The check helps when the map is exact | Image reading is the limit. Train the model to transcribe the map from the image, on generated maps where the labels are free. S3T's gain over EB3 bounds what that can buy. |
| No difference or fewer, and the lines read the text map | The check does not help even when read right | Drop the form and keep EB3's. Look at how S3T fails. |
| No difference or fewer, and the lines do not read the text map | The model cannot place itself or its neighbours from an exact text map | Neither image reading nor move choice limits; working out positions does. Report and hold. |

**Power,** as round 3: a gain of 16 from about 50 discordant maps is detected; half that
is not.

**Validity (the readings are withheld if any check fails):**
1. **Both arms trained on EB3's rows:** the same counts (472 new maps, 387 old maps, 3,000
   rows).
2. **Every S3T training answer's lines** replay to its `\boxed{}` route, and every listed
   hole set matches the map.
3. **All four arms answer the same maps** (S3T, O3, S3, EB3).
4. **The cap:** at most 2% of S3T's answers on the 400 reach 256 new tokens.
5. **The same base:** the two jobs' zero-shot outcomes, both B on the same prompts, agree
   on at least 95% of the 400.

**Descriptive (no verdicts):**
- **S3T against S3:** what the text map adds to the check. **O3 against EB3:** what it
  adds without the check, which repeats O2 from B. **S3T against EB3.**
- **S3T's lines,** beside S3's: the start cell, the hole directions listed and invented,
  and the fatal-move classes.
- **Format:** answers whose lines all parse and replay to their `\boxed{}` route.
- **Failures** per arm: into a hole, stops short, passes the goal.
- **Round 1's 240 held-out maps:** S3T against O3.
- **Zero-shot success, training loss** per phase, and GPU-seconds.

## Budget

Two jobs on one H200 each, run together: about 45 minutes for S3T and 30 for O3, against
O2's 26 and S3's 57. The time limit is 90 minutes each. Nothing is written to scratch
beyond the records.

## Implementation

In `spara/expert_round2.py`, with `configs/hole_check_text.json` (the hole check's
settings without saved weights):
- **Arms `s3t` and `o3`** join the text arms; `s3t` also joins the hole-check arms, so its
  training rows pass the replay check.
- **`line_reading`** in `spara/hole_check.py` reads a run's lines against the map.
- **`text_check`** applies the readings above.

## Running

On PACE, with `R=$SPARA_REMOTE_ROOT/results`, once per arm:

```bash
SPARA_STAGE=train SPARA_ARM=s3t SPARA_ROUND=3 SPARA_SETTINGS=configs/hole_check_text.json \
  SPARA_E_CHECKPOINT=$R/round3-eb2-7088134/phase2/final SPARA_ROUND1_ROUTES=$R/expert-routes-7063156/routes.jsonl \
  SPARA_SEARCH_RUN="$R/round2-search-7065168 $R/round3-search-7088135"
```

## Limits

- **One training run per arm.** Two runs of the same recipe disagree on about 20 maps.
- **A ceiling, not a model.** Both arms read the layout from the prompt.
- **A text grid is not perfect perception of neighbours.** The model must still find its
  row and column in the text. The lines measure that step directly, which is why the
  table splits on them.
- **The 90% line threshold is a judgment,** set before the run from S3's 35% and 67%.

## Result (jobs 7125957 and 7125958)

Evidence: [docs/experiments/hole-check-text-7125957/](experiments/hole-check-text-7125957/),
with `text_check.json` holding every reading below.

**What changed from the plan.** The jobs were submitted several hours late: PACE could
not be reached until the VPN was restarted. Nothing in the design changed.

**Jobs.** 3,596 GPU-seconds (60 min) by the jobs' own clocks: S3T 2,218 and O3 1,378.

**Validity: all checks passed.**
- **Both arms trained on EB3's rows:** 472 new maps, 387 old maps, 3,000 rows.
- **Every S3T training answer's lines replay** to its `\boxed{}` route.
- **All four arms answer the same maps.**
- **The cap:** no S3T answer on the 400 reaches 256 new tokens. The longest takes 114.
- **The same base:** the two jobs' zero-shot outcomes agree on 400 of 400.

**Primary: the check harms even when the map is exact.** On the 400 maps of
`detour-heldout-78`, S3T solves 70 and O3 92: 18 maps only S3T solves, 40 only O3,
p = 0.005. By size: 40 against 56 at 7 x 7 (10 against 26, p = 0.01) and 30 against 36
at 8 x 8 (8 against 14, p = 0.29).

**The lines do not read the text map.** The table's third row applies.
- **The start cell:** right on 369 of 400 first lines (92%), against S3's 269. The text
  map fixes where the model thinks it starts. The 31 misses are one column off.
- **Holes next to the agent,** on the 1,285 lines naming the right cell: 358 of the
  1,073 hole directions listed (33%), against S3's 35% from the image. The exact set is
  right on 603 lines (47%), against 482 (38%) for writing `-` every time.
- **By direction:** left 36% and right 38%, up 34% and down 26%. Left and right
  neighbours sit beside the agent's cell in the text, so the misses are not a matter of
  lining up rows.
- **By line:** 42% of holes listed on the first line, 29% on later lines. S3: 45% and 29%.
- **Listed and taken:** 0 again. Of 319 falls, 285 are missed holes and 34 are on lines
  naming the wrong cell.
- **Format:** 399 of 400 answers have lines that parse and replay to their route.

**Descriptive.**
- **S3T against S3:** 70 against 65, 19 against 14, p = 0.49. The text map does not
  rescue the check.
- **O3 against EB3:** 92 against 83, 21 against 12, p = 0.16. The text map alone adds
  nothing detectable from B, as O2 found from E.
- **S3T against EB3:** 70 against 83, p = 0.09.
- **Round 1's 240 held-out maps:** S3T 87 against O3 110, p = 0.003.
- **Failures:** S3T 319 into a hole and 11 stop short; O3 288 and 20.
- **Zero-shot:** B with the text map and no training on it solves 71 of 400, in both jobs.
- **Training loss,** last quarter of phase 2: S3T 0.10, S3 0.11, O3 0.06, EB3 0.07. The
  line tokens stay as hard to predict with the text map as without it.

**What this says.**
- **The hole lists are equally poor from pixels and from an exact text map.** So what
  the lines measured in S3 was not image reading. The step that fails is working out
  which cells lie next to the agent, from either form of the map.
- **The check costs success in both settings,** by 18 maps from the image and 22 with
  the text. A model that writes a wrong reading and then obeys it does worse than one
  that never writes it.
- **This removes the case for transcription training as the next step.** A perfect
  transcription is what the text map already is, and the model does not use it for its
  neighbours.
- **The perception ceiling's reading stands** for image reading: O2, T2 and now O3 all
  find that the exact map adds nothing detectable.

**Limits of this reading.**
- **375 optimizer steps.** The lines are a new skill, and neither arm had long to learn
  it. The result says this recipe does not learn to read neighbours, not that the model
  cannot.
- **The lines are on the route only,** about six cells per map, so most cells are never
  supervised.

**Next: report and hold,** as the table fixes. The one question these runs leave open is
whether the neighbour reading can be learned at all with more examples. Labels for it
are free on generated maps. That is a proposal, not a fixed step.
