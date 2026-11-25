# Seed extension and crossed-view choice scoring (V20)

**Date fixed:** 13 November 2026, before any V20 GPU work. **Approved by the author:** both experiments listed in
`deliverables/20261113/后续实验选项_20261113.md` ("两项都跑").
**Goal:** (A) measure two-candidate choice on the crossed views, where only generation has been measured;
(B) add three seeds to the comparison of compressed SFT with the weakest-margin + locality objective, so that seed
variance is estimated from five seeds; (C) check on the three new SFT readers that the readout gap, the digit
structure of wrong codes and the slot rule do not depend on the two original seeds.
**Architecture:** new files only, under `temp/v20`, `runs/v20-seed-extension-20261113`,
`results/v20-seeds-crossed-choice-20261113` and `logs/v20-*` on the host (E:). No file under `src/` or any
earlier run is edited. One serial queue; no retries; nothing is done to other processes or terminals.
The reserved test splits are not opened, hashed or evaluated. Confirmation units are not used.

## Status of the data

- Crossed development views (144 units, 8 scenes) and crossed new-scene views (144 units, 8 scenes): generation
  on these units has been inspected (V16, V19). Part A is therefore a **planned analysis on data already seen**.
- New scenes, original layout (96 units, 32 scenes): evaluated in V19 with the two original SFT readers. They are
  **not unseen**. Part C uses them only to ask whether three readers that do not exist yet, trained by the
  unchanged protocol with no tuning, show the same contrasts. It is a replication across seeds on previously
  evaluated scenes and must be described that way.
- Development original units (96 units, 32 scenes, 24 articles): used throughout the study.

## Part A. Choice scoring on crossed views (inference only)

- Readers: the four 512-update SFT readers pinned in `results/v19-budget-layout-20261108/prospective/protocol.json`
  (compressed seeds 73 and 97, full-context seeds 73 and 97), checked by file hash and adapter identity.
- Cohorts: `data/v16-role-balance-20261106/development.jsonl` and `data/v19-prospective-20261108/crossed.jsonl`.
- Rows: both target questions in all four contexts, 8 rows per unit, 1,152 per reader and cohort (64 on the 8
  identity views, 1,088 on the 136 transformed views).
- Scoring, unchanged from the earlier choice study: the two candidates are the two target values of the unit in
  the current context, with leading blank and EOS; score = mean token log probability; token lengths equal;
  exact tie = error; the candidates are not shown in the prompt.
- Generation on the same rows: the stored greedy outputs of the V16 role diagnostic and of the V19 crossed
  evaluation, pinned by hash. Sixteen rows per reader and cohort (the first identity view and the first transformed
  view) are decoded again and must reproduce the stored strings and token IDs.
- Main measures, per compressed reader and cohort, on transformed views: choice accuracy, generation accuracy, and
  their difference on the same rows, each with a 95% percentile interval from 10,000 scene-cluster bootstrap
  resamples (RNG 20261113; ratio of totals). There are eight scenes per cohort, so the intervals are rough; the
  per-scene minimum and maximum are reported with them. No threshold is set: chance is 50%, the original-layout
  level is 98% to 99%, and both a high and a low value are informative. A value near the original-layout level
  places the layout failure in what the reader generates; a value near chance means that the conditional
  assignment itself does not survive the change of layout.
- Secondary, descriptive: identity views; the choice version of the target joint; choice accuracy and mean
  candidate margin by rendered slot of the queried value, by queried label, by whether the queried value sits in
  the slot the slot rule reads, by label rotation, field-order rotation, team exchange and context, and by the
  category of the generated answer on the same row (correct, other candidate, same-row substitution, other record
  value, absent code, other). Full-context readers are the reference.

## Part B. Three more seeds for compressed SFT and weakest margin + locality

- Runs: seeds 101, 131 and 151 (fixed here), two arms, 512 updates, development evaluation at 128, 256 and 512
  updates as coded. Data, source order algorithm, optimizer, learning-rate schedule, loss coefficients, decoding
  and metrics are those of the original runs; the code that implements them is called unchanged
  (`binding.v13_training.train` for SFT, `binding.v14_training.train` for weakest margin + locality).
- Two necessary differences, made in the calling process only (`temp/v20/seed_train.py`): the output folder is
  new, and the seed registry inside `binding.v11_training.schedule`, which admits only 73 and 97, is replaced by a
  copy of the same algorithm that also admits 101, 131 and 151. The preflight checks that the copy reproduces the
  recorded source orders of the four original runs. The process environment is that of the original queue
  (`PYTHONHASHSEED=73`, offline flags, caches on E:).
- Positive control before any new seed: the first 32 updates of the original seed-73 run of each arm are replayed
  through the new entry point (the order is cut to 32; the learning-rate schedule is still computed for 512).
  Initial reader identity, frozen-memory digest, source order and learning rate must equal the original log
  exactly. Losses and gradient norms are expected to be exactly equal; a relative difference up to 1e-4 is
  recorded and tolerated; a larger one stops part B before the new seeds.
- Each seed runs SFT first and weakest margin second; the two must share the initial reader identity and the
  memory digest. No confirmation or new-scene evaluation of the weakest-margin readers.
- Fixed analysis (development, 512 updates). Per seed: target-joint groups of both arms, their difference in
  percentage points, the differences in natural F1 and in unaffected joint correctness, and the original
  continuation rule (gain of at least 5 points with at most 2 points lost on each of the other two).
  The criterion is **met in the extension only if it holds at all three new seeds**. Summary over the five seeds
  (73, 97, 101, 131, 151) and separately over the three new seeds: mean difference, standard deviation across
  seeds, and a 95% t interval for the mean with the seed as the unit (4 and 2 degrees of freedom). A five-point
  mean gain is taken as excluded if the upper bound of the five-seed interval is below 5 points. The three new
  seeds are the part not seen before this plan; the pooled figures include the two seeds that prompted it.
  The standard deviation of SFT target success across seeds, the 128- and 256-update counts, and the pass
  probabilities of a both-seeds rule under a normal model with the five-seed standard deviation are descriptive.
- Each run stays within the authorised 1,024 updates (512 are used). No coefficient is changed and no further
  seeds are added after seeing results.

## Part C. Readouts of the three new SFT readers (added to the approved scope; inference only, about 75 minutes)

Reason: the main findings rest on two seeds, and three new SFT readers come out of part B anyway.
Runs after parts A and B, so it cannot delay or affect them. Weakest-margin readers are not evaluated here.

- Development original units: choice scoring on the target rows; generation is the run's own stored 512-update
  evaluation (16 rows replayed).
- New scenes, original layout: greedy generation of all 16 answers and choice scoring on the target rows.
- Crossed development views and crossed new-scene views: greedy generation of all 16 answers and choice scoring
  on the target rows.
- Tests on the new scenes (original layout), fixed before the readers are trained: for each new seed, choice
  minus generation accuracy on the target rows, and the digit advantage of four-digit record-absent outputs
  (positions shared with the correct code minus the mean over the other record values), must both have a positive
  lower bound in a 99.17% interval (Bonferroni over three seeds and two contrasts) from 10,000 scene-cluster
  bootstrap resamples, RNG 20261113. Undefined resamples are not dropped; an undefined contrast is a failure.
- Slot rule, fixed here as it was stated in the v5 manuscript (a question about badge or route is answered from the
  first rendered value of the queried row, a question about locker from the second): the share of same-row
  substitutions that follow the rule is tabulated per new seed and cohort with the same scene-cluster interval and
  per-scene range. No threshold; the rule itself is the prediction.
- Everything else (accuracy on identity and transformed views, joint successes, categories) is descriptive.

## Execution

- [ ] Local: scripts, this plan and `temp/v20/lock.json` (hashes of the V20 scripts, this plan, and the `src/binding`
  modules they call) are created before upload. The frozen analysis (`temp/v20/analysis.py`) was checked on earlier
  data: it reproduces the v5 figures for the new scenes and the crossed views.
- [ ] Host: check GPU, RAM, disk and running processes; upload with SHA-256 verification; `preflight.py --dry`
  (CPU); then start `run_queue.py` in the background with timestamped logs and recorded PIDs.
- [ ] Queue: preflight (CPU), smoke (GPU; two training-split units; exercises generation and scoring before any
  study inference), part A, replays, part B, part C, collect (CPU: frozen analysis and an evidence archive without
  adapter tensors). A resource gate (GPU free >= 23,500 MiB, RAM >= 12 GiB, disk >= 20 GiB; waits at most 30
  minutes) precedes each GPU job. Parts A and B are independent; a failed job is not retried.
- [ ] Afterwards: fetch the archive with checksum, rerun the analysis offline, update the manuscript and package
  as a new dated version. Results are reported whatever their direction.

Expected wall time on the RTX 5090: about 6.7 hours (A 15 min, replays 13 min, B 4.9 h, C 75 min).
