# Evaluation Report

Evaluation of the `physicsnemo-functional-builder` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

> **Status: pending.** The results, Tier-1/Tier-2 findings, and verdict below are
> populated by an NVSkills-Eval run prior to publication. The evaluation dataset
> (`evals/evals.json`) and target agents are committed; run the harness and
> refresh this file before publishing.

## Evaluation Summary

- Skill: `physicsnemo-functional-builder`
- Evaluation date: _pending_
- NVSkills-Eval profile: `external`
- Environment: `local`
- Dataset: 4 evaluation tasks (`evals/evals.json`)
- Attempts per task: 2
- Pass threshold: 50%
- Overall verdict: _pending_

## Agents Used

- `claude-code`
- `codex`

## Metrics Used

Reported benchmark dimensions:

- Security: checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access.
- Correctness: checks whether the agent follows the expected workflow and produces the correct final output.
- Discoverability: checks whether the agent loads the skill when relevant and avoids using it when irrelevant.
- Effectiveness: checks whether the agent performs measurably better with the skill than without it.
- Efficiency: checks whether the agent uses fewer tokens and avoids redundant work.

Underlying evaluation signals used in this run:

- `security` (Security): checks for unsafe operations, secret leakage, and unauthorized access.
- `skill_execution` (Skill Execution): verifies that the agent loaded the expected skill and workflow.
- `skill_efficiency` (Efficiency): checks routing quality, decoy avoidance, and redundant tool usage.
- `accuracy` (Accuracy): grades final-answer correctness against the reference answer.
- `goal_accuracy` (Goal Accuracy): checks whether the overall user task completed successfully.
- `behavior_check` (Behavior Check): verifies expected behavior steps, including safety expectations.
- `token_efficiency` (Token Efficiency): compares token usage with and without the skill.

## Test Tasks

The benchmark dataset contained 4 evaluation tasks:

- Positive tasks: 2 tasks where the skill was expected to activate (add a new functional op with a Warp backend; add an optional cuML/SciPy backend to an existing op).
- Negative tasks: 2 tasks where the functional-builder skill was not expected (a reusable-layer/model request that belongs to `physicsnemo-model-builder`; an out-of-scope request such as a datapipe or a "which op should I use" usage question).
- Unlabeled tasks: 0.

Entries with `expected_skill` set are treated as positive skill-activation cases; entries with `expected_skill: null` are treated as negative activation cases.

## Results

_Pending NVSkills-Eval run._

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | — | — | — |
| Correctness | — | — | — |
| Discoverability | — | — | — |
| Effectiveness | — | — | — |
| Efficiency | — | — | — |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

_Pending NVSkills-Eval run._

## Tier 2: Deduplication Summary

_Pending NVSkills-Eval run._ Note: this skill is intentionally distinct from
`physicsnemo-model-builder` (authoring `nn.functional` ops/backends vs. models and
`nn.Module` layers); the negative eval tasks guard that routing boundary.

## Publication Recommendation

_Pending NVSkills-Eval run._ Refresh this file with the harness output (results
table, Tier-1/Tier-2 findings, verdict) before publishing, and keep it with the
skill; re-run when the evaluation dataset, skill behavior, or target agents
materially change.
