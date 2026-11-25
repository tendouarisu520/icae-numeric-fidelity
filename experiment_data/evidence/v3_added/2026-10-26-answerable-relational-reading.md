# Answerable relational reading implementation plan

Execute inline with writing-plans, test-driven-development and executing-plans. The user approved the narrowed scope and requested execution. Local changes only; no commits, pushes, worktrees or new agents.

**Goal:** Test whether assignment and locality supervision improve condition-sensitive reading of one fixed query-independent soft memory, without researching refusal.

**Evidence:** V12 is complete: five of six readers regress from512 to1024. Relation sources do not consistently improve target-all-four. Existing A/B misbinding is0/96 in every V11 arm. Old four-way scores include deleted-evidence UNKNOWN and are retained as historical diagnostics.

**Architecture:** One completely frozen official ICAE produces128 context-only memory slots. A fresh matched rank16 q/v reader is trained on a mixture of existing natural answerable QA and new controlled multi-condition records. Full-context SFT is a capacity-matched reference. Only the reading adapter changes; this experiment does not claim a new memory writer.

## Constraints

- Remote E:/Research/ICAE-ConditionalMemory-20261021, RTX5090, dedicated UU command terminal10 only. Local D:/CodexStorage/research/memory-binding-20261021.
- Preserve all V11/V12 code, data, checkpoints and original endpoints. New V13 files only.
- Open only existing512 training and32 development units. Do not open confirmation examples or enumerate/read/stat reserved test. Derived development scenes remain development, not a fresh independent validation set.
- Main512 updates, diagnostics at128/256/512. Overfit <=1024 with early gate; smoke2. No automatic retries or unbounded coefficient sweep.
- CPU RAM memory cache only. No new model/dependency downloads. Efree initially >=700MB and before each cell >=450MB. Do not delete old artifacts or stop user applications.
- UNKNOWN outputs in answerable evaluation count as errors. No abstention thresholds, forced answers or answer-based row filtering.

## Data and metrics

`v13_data.make_probe(source,split,level)` builds four contexts from source UID/article: original; swap two values for one team across two modes; edit one unqueried fact; reorder records. Every value is present in every context. Four questions: two swapped target relations and two unaffected facts. Sixteen known labels. Interference levels2/3/4 teams x2modes x3fields=12/18/24 facts. Team names and4digit value pools differ across train/development; controlled records are explicitly synthetic, with a short source-paragraph background. No relation-record truncation permitted.

`v13_data.natural_rows(source)` retains exactly seven known labels: target in original/answer-edit/irrelevant-edit; unaffected question in all four contexts. Existing deleted target and genuine unknown are excluded by fixed task definition, before looking at predictions. Natural rows are never called controlled binding probes.

Primary structured metrics: all-original/edit target answers correct (four answers), unaffected-all-four correct (eight answers), irrelevant/reorder preservation, known EM/F1. Candidate-swap errors are counted across all target predictions; refusal remains an ordinary incorrect prediction. Natural metrics: known EM/F1, target original/edit pair, unaffected-all-four. Report article/source-grouped outcomes, not independent-row significance claims.

## Training

Every optimizer update uses one source's natural7 + structured16 answers; each family has weight.5. All arms share source order, initial reader identity, model/context/memory budgets and labels. Main LR5e-5,32step warmup, cosine decay to5e-6, AdamW(.9,.999),eps1e-8,weight_decay0,clip1,eval-mode deterministic training. Sources cycle balanced interference levels; evaluation reports all three.

Arms per seed73/97: compressed SFT; compressed paired assignment; compressed assignment+locality; full-context SFT. Fixed512 endpoint. Full reference may use greater inference compute and is not a requirement for beating compressed SFT.

Assignment loss: for each context, obtain2x2 mean-loglikelihood scores for the two condition questions and their two gold values. Minimize softplus(.5 - (S00+S11-S01-S10)/2). Weight.1. All candidates have the same four-digit representation; generation remains open-ended and is primary.

Locality: teacher-force the two unaffected answers; detach original-context token probabilities and minimize KL into each edited/reordered-context distribution. Weight.05. SFT and assignment-only arms receive the same answer supervision but no locality term. This isolates supervision components.

## Task1 Data and evaluation protocol

- [ ] Add `src/tests/test_v13_protocol.py`: deterministic reconstruction, exact swaps, unaffected gold correctness, disjoint name/value pools, noUNKNOWN, prefix-independent data construction, always-complete groups, all-wrong consistency cannot count as correctness, natural7 rule.
- [ ] Run missing-module tests and retain RED output.
- [ ] Implement `src/binding/v13_data.py` and `v13_protocol.py`; run tests GREEN.

## Task2 Reader objectives

- [ ] Add actual-neural tests `src/tests/test_v13_neural.py`: score orientation, loss gradient direction, locality zero at identical logits, actual frozen-prefix gradients and zero frozen-base gradients. Local neural tests may skip withoutTorch; remote gate must execute all tests with zero skips.
- [ ] Implement `src/binding/v13_objectives.py` using sequential branch replay. Reuse tested V12 RAM cache and greedy generation; no changes to those modules.
- [ ] Validate no-gradient objective values against a direct tiny-model loss and ensure adapter flags/cache bytes remain unchanged.

## Task3 Training and recorded queue

- [ ] Implement `src/binding/v13_training.py`: deterministic data admission with frozen input hashes, completely frozen official encoder, matched fresh reader, mixed-family updates, fixed checkpoints, exact reload and cache/source/frozen checks.
- [ ] Implement `temp/v13_queue_20261026.py`: all remote tests; two-step smoke; eight-source overfit gate for full and compressed; eight main cells. Every child background/logged with process record, bounded monitoring and no retries.
- [ ] Freeze source-plan hashes before deployment. Verify upload SHA and whitelist extraction. Record actual PIDs, output/log paths immediately.
- [ ] Start main jobs only after real7B calibration and overfit gates pass. Fetch actual timings and learning curves; do not infer completion from submission.

## Task4 Paper

- [ ] Retrieve all six V12 results/checks; summarize original fixed endpoints separately from retrospective512 preferences.
- [ ] Produce a formal local Word manuscript with problem, related work, model/objectives, reproducible protocol, complete historical tables, failure analysis and limitations. Keep unobserved V13 outcomes out of empirical claims.
- [ ] Update the existing proposal with the narrowed scope and complete V12 evidence. Preserve existing equations/tables/history.
- [ ] Verify primary references, render documents, inspect changed pages and deliver only verified Word artifacts.

## Selection gate

At fixed512 compare relation against compressedSFT for both seeds and separately by interference. Require >=5percentage-point gain in joint target correctness, no greater than2point loss in natural known F1 and unaffected correctness. These are continuation gates, not guarantees of publication. Keep all ablations; do not select seed-specific objectives. Independent confirmation remains frozen until a complete configuration has been chosen.
