# experiment_design
Turn a research question into one falsifiable claim, one pack, one comparable confirm_ppl number.

## The claim (write_hypothesis)
Three fields, one sentence each:
- `claim`: what changes and the predicted direction. "Halving lr to 1.5e-3 at 64 steps lowers confirm_ppl below ep-0004's 9.81."
- `why`: the mechanism. "ep-0004 train_loss oscillated; 3e-3 is above the stable range for hidden=32."
- `falsify`: the observation that kills it. "confirm_ppl >= 9.81 or job fails."
Bad claim: "try a different lr". It cannot fail, so it teaches nothing.

## One factor at a time
Change exactly one config key per pack. If `lr` and `steps` both change you cannot attribute the result. Exception: `steps` and `budgets.max_steps` move together.

## Always compare on confirm_ppl
- Frozen holdout, lower is better. Read it from the episode card (`eval.confirm_ppl`, `eval.confirm_source`).
- Ignore trainer `val_ppl` for ranking; it is one random training batch.
- `confirm_source=trainer_val_proxy` means no checkpoint was scored. Treat as "no result".
- A win is a lower confirm_ppl than the parent episode with the same data mix and architecture.

## Read before proposing
1. `list_episodes` once (n=12). Note the best confirm_ppl and its config.
2. `read_episode` on the best one and on the most recent one. Read `verdict` and `next_hint`.
3. Never write a pack whose `config` + `data_manifest` equals an indexed episode. Same content hashes to the same pack; the harness learns nothing.
4. Then `write_hypothesis`, `write_pack`, `enter_train`. Do not loop on reads; the research budget is finite.

## Resuming from a parent
`parent_checkpoint = observation.last_checkpoint`. Weights load only if `hidden, layers, heads, seq_len` (and vocab=256) match the parent exactly. Changing any of them silently trains from scratch, so an architecture sweep is a from-scratch comparison, not a continuation. Say which one you are doing in the hypothesis.

## When to kill a hypothesis
- Two consecutive packs in the same direction did not lower confirm_ppl.
- The job failed for a reason in the config (nan loss, mismatch), not the harness.
- The mechanism in `why` was contradicted by the training log.
Write `write_hypothesis {"status": "killed"}` plus a `write_note` with the reason and the next direction.

## Good hypothesis strings for the pack
Include factor, value, baseline episode, expected direction:
- "steps 32->128 at lr 3e-3, resume ep-0002 (confirm_ppl 11.2); expect < 10.5"
- "from scratch: hidden 32->64, heads 1->2, 64 steps; compare to ep-0005 9.9"
Under 120 chars; it becomes the episode title.

## Checklist before write_pack
- [ ] `list_episodes` read this cycle; config differs from every indexed episode
- [ ] `claim`, `why`, `falsify` written; `falsify` names a confirm_ppl threshold
- [ ] exactly one factor changed vs the baseline episode
- [ ] `trainer` = `lab`; `eval_suite_id` = `core`; `eval_suite_version` = 1
- [ ] `hidden % heads == 0`; `steps == budgets.max_steps`; `max_hours <= 3.5`
- [ ] `parent_checkpoint` = `observation.last_checkpoint` (or the subject) when resuming
- [ ] `data_manifest.sources` = default mix unless `prefetch_data` cached more allowlisted text
- [ ] lr in 1e-4..1e-2, steps in 8..512

## Five hypothesis -> pack pairs (baseline ep-0003: lr 3e-3, steps 32, hidden 32, layers 1, heads 1, seq_len 32, batch 8, confirm_ppl 10.4)
Shared fields: `"trainer": "lab", "data_manifest": {"sources": ["hf:roneneldan/TinyStories:train:10000"]}, "eval_suite_id": "core", "eval_suite_version": 1`.

1. lr sweep (resume): "lr 3e-3->1e-3 at 32 steps, resume ep-0003; expect confirm_ppl < 10.4"
   `config: {lr: 0.001, steps: 32, hidden: 32, layers: 1, heads: 1, seq_len: 32, batch: 8}`, `parent_checkpoint: <last_checkpoint>`, `budgets: {max_hours: 0.1, max_steps: 32}`
2. steps sweep (resume): "steps 32->128 at lr 3e-3, resume ep-0003; expect < 9.5"
   `config: {lr: 0.003, steps: 128, hidden: 32, layers: 1, heads: 1, seq_len: 32, batch: 8}`, `budgets: {max_hours: 0.2, max_steps: 128}`
3. width (from scratch): "hidden 32->64 (heads 1), 128 steps from scratch; compare to ep-0003"
   `config: {lr: 0.003, steps: 128, hidden: 64, layers: 1, heads: 1, seq_len: 32, batch: 8}`, `parent_checkpoint: "none-from-scratch"` (any non-empty string that is not a file)
4. depth (from scratch): "layers 1->2 at hidden 32, 128 steps from scratch; compare to ep-0003"
   `config: {lr: 0.003, steps: 128, hidden: 32, layers: 2, heads: 1, seq_len: 32, batch: 8}`
5. seq_len (from scratch): "seq_len 32->64, 128 steps from scratch; expect lower confirm_ppl from longer context"
   `config: {lr: 0.003, steps: 128, hidden: 32, layers: 1, heads: 1, seq_len: 64, batch: 8}`

Pairs 1-2 keep the architecture so weights continue. Pairs 3-5 change it, so they are from-scratch runs; give them enough steps to be fair (>= 128) and label them as such.
