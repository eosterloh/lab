# experiment_design
Turn a research question into one falsifiable claim, one pack, one comparable confirm_ppl number.

## The claim (write_hypothesis)
Three fields, one sentence each:
Values in angle brackets below are yours to fill from the observation. Never emit a bracket.
- `claim`: what changes and the predicted direction. "<knob> <old>-><new> at <other knob>, vs <baseline ep>'s confirm_ppl <ppl>; expect lower."
- `why`: the mechanism. "<what the baseline episode's log or metrics show> implies <old> is <too high / too low> for <this config>."
- `falsify`: the observation that kills it. "confirm_ppl >= <baseline ppl> or the job fails."
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
Include factor, value, baseline episode, expected direction. Shape, not text to copy:
- "steps <old>-><new> at lr <lr>, resume <ep> (confirm_ppl <ppl>); expect < <target>"
- "from scratch: hidden <old>-><new>, heads <old>-><new>, <n> steps; compare to <ep> <ppl>"
Under 120 chars; it becomes the episode title. Every bracket must be replaced by a number or id from your observation.

## Checklist before write_pack
- [ ] `list_episodes` read this cycle; config differs from every indexed episode
- [ ] `claim`, `why`, `falsify` written; `falsify` names a confirm_ppl threshold
- [ ] exactly one factor changed vs the baseline episode
- [ ] `trainer` = `lab`; `eval_suite_id` = `core`; `eval_suite_version` = 1
- [ ] `hidden % heads == 0`; `steps == budgets.max_steps`; `max_hours <= 3.5`
- [ ] `parent_checkpoint` = `observation.last_checkpoint` (or the subject) when resuming
- [ ] `data_manifest.sources` = default mix unless `prefetch_data` cached more allowlisted text
- [ ] lr in 1e-4..1e-2, steps in 8..512

## Five angles: which key moves
Your baseline is the best episode in YOUR observation, not any number written here. Copy nothing from this table; it says which key to move and what must stay fixed.

| angle | key you move | keep fixed | resumes? |
|---|---|---|---|
| learning rate | `config.lr` (stay in 1e-4..1e-2) | steps, architecture | yes |
| training steps | `config.steps` and `budgets.max_steps` together | lr, architecture | yes |
| width | `config.hidden` (keep `hidden % heads == 0`) | lr, steps | no |
| depth | `config.layers` | lr, steps | no |
| context | `config.seq_len` | lr, steps | no |

The first two keep the architecture, so weights continue from `parent_checkpoint` and the comparison is a continuation. The last three change it: the run starts from scratch even when a parent is named, so give it enough steps to be a fair comparison and say "from scratch" in the hypothesis.
