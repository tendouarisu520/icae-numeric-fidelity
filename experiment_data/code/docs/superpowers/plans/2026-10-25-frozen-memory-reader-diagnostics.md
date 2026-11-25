# Frozen memory reader and refusal diagnostics implementation plan

> Execute task by task with the executing-plans workflow. User approved the experiment order and constraints in this conversation. Local edits only; no commits, pushes, new worktrees or delegated agents.

**Goal:** Determine whether late SFT regression comes from memory representations, reader adaptation, or refusal decisions, then choose the next training change from development results.

**Architecture:** Preserve completed V11 artifacts. Freeze each source compressor, encode context-only memories into CPU RAM, unload the compressor and train an independently initialized matched rank16 reader. Compare normal, shuffled and null memory; score gold, UNKNOWN and generated alternatives on the same prefixes. Match full-context reader checkpoints as a reference.

**Tech stack:** Existing remote Torch/Transformers/PEFT environment and RTX5090 through a dedicated UU command terminal. Local Python for pure tests and manuscript preparation.

## Constraints

- Remote root E:/Research/ICAE-ConditionalMemory-20261021. Local root D:/CodexStorage/research/memory-binding-20261021. No new model downloads, dependencies, C-drive task outputs, or user-process termination.
- Each run at most1024 optimizer updates. Six main reader runs plus one2step calibration run. No automatic retries or loss-weight sweep.
- Only existing512 training units and32 development units may be opened. Confirmation and reserved test remain unopened. Keep completed V11 endpoints and source files immutable.
- RAM cache rather than disk: Efree1.85GB; available physical memory about22GiB. Save rank16 adapters at256/512/1024, no optimizer/base checkpoints.

## Fixed cells and comparisons

- Seeds73/97 each use SFT source512, SFT source1024, relation source1024: six independent reader runs.
- Reader rank16 q/v, alpha32, eval-mode training with dropout disabled,6,815,744 trainable parameters. Initialization must exactly match the corresponding V11 full reader.
- Same512unit schedule,12 answer+EOS targets per update, AdamW LR1e-4/warmup32/betas(.9,.999)/eps1e-8/weight_decay0/clip1,1024updates. Reader training is plain SFT; source memory is constant.
- Preserve rawgreedy metrics at fixed endpoints. Show checkpoints256/512/1024; development checkpoint preference, if used later, is maximum QA F1 then earlier checkpoint and must be labeled separately from the fixed endpoint.
- Replay the source's development predictions with the fresh zero reader before training. Abort on token mismatches, changed source files, nonfinite memories, missing gradients or changed frozen tensors.

## Tasks

- [x] Retrieve and checksum-verify all six completed V11 cells and their256/512/1024 curves.
- [ ] Add pure protocol tests for article-disjoint shuffled donors, sequence-level UNKNOWN bans, and article-held-out threshold calibration. Watch tests fail before implementation.
- [ ] Add tiny actual-neural tests for frozen context-only RAM cache, reader-only gradients, autoregressive scoring alignment and preservation of non-UNKNOWN words sharing a token prefix.
- [ ] Implement fixed-cache reader training and diagnostics in new V12 modules. No changes to pinned V11 modules.
- [ ] Run all V12 tests on the remote installed Torch runtime with zero skips; run2step actual7B calibration before main training.
- [ ] Execute the six cells sequentially in a logged background controller, recording PIDs, artifact paths, memory/step speed, exact source/adapter identities and cache digests.
- [ ] Collect complete paired metrics and diagnose regression. Choose subsequent tuning only after these results.
- [ ] Update local proposal with complete V11 results, V12 design and observed V12 outcomes; render and visually verify the document.

## UNKNOWN diagnostics

Score answer+EOS using both mean and sum log probability, recording token count and first-token/EOS scores. Report gold-versus-UNKNOWN margins as oracle diagnostics only. Generate a non-UNKNOWN alternative without consulting the gold label: retain an already non-UNKNOWN greedy output, otherwise ban complete UNKNOWN token sequences during generation. Preserve legitimate words sharing their prefixes. Keep ordinary greedy accuracy primary.

For optional threshold analysis, use generated alternatives only. Fit on one half of development articles and score the other half, then swap folds; never split rows of one article between folds. Select the threshold by mixed QA F1 with fixed tie rules. Report these as development out-of-fold diagnostics, together with the all-development fitted threshold for any later frozen use. No confirmation/test evaluation.

## Memory controls and interpretation

Shuffled donors are a deterministic bijection over development units from different articles, retaining the context-variant index; question and reference answer remain unchanged. Null memory preserves128 slots and the same FT delimiter. Report answerable accuracy, falseUNKNOWN, wrong-when-answering, genuine/deleted UNKNOWN and all-four target/locality metrics separately.

If a trained reader recovers the gap from unchanged memory, prioritize reader adaptation and subsequent fair relation-loss comparisons. If full reading remains much stronger after adaptation, inspect representation preservation/teacher supervision. If score calibration rescues refusals but increases wrong answers, retain the complete tradeoff rather than treating forced answering as task accuracy. Numeric checks and source replay precede scientific interpretation.
