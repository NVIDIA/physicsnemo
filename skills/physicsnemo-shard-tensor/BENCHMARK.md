# Evaluation Report

Evaluation of the `physicsnemo-shard-tensor` skill before publication through NVSkills-Eval.

This benchmark summarizes 3-Tier Evaluation from NVSkills-Eval results for the skill. The goal is to document whether the skill is safe, discoverable, effective, and useful for agents before it is published for broader workflow use.

## Evaluation Summary

- Skill: `physicsnemo-shard-tensor`
- Evaluation date: 2026-07-21
- Environment: `local`
- Dataset: 8 evaluation tasks
- Attempts per task: 2
- Pass threshold: 50%
- Overall verdict: PASS

## Agents Used

- `claude-code`

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

The benchmark dataset contained 4 evaluation tasks (8 trials per condition at 2 attempts per task):

- Positive tasks: 3 tasks where the skill was expected to activate (script conversion to ddp x domain parallelism, enabling an unsupported op via shard-patch registration, diagnosing the swallowed-TypeError pitfall).
- Negative tasks: 1 task where no skill was expected (generic HuggingFace/PyTorch FSDP fine-tuning with no domain parallelism).

Task composition is derived from the evaluation dataset. Entries with `expected_skill` set are treated as positive skill-activation cases, while entries with `expected_skill: null` are treated as negative activation cases. All 32 recorded trajectories additionally passed the repo's deterministic citation-discipline check (`scripts/check_citations.py`): no ungrounded absolute paths.

## Results

| Dimension | Num | `claude-code` |
|---|---:|---:|
| Security | 8 | 100% (+0%) |
| Correctness | 8 | 100% (+16%) |
| Discoverability | 8 | 100% (+48%) |
| Effectiveness | 8 | 97% (+0%) |
| Efficiency | 8 | 91% (+44%) |

Score values show skill-assisted performance. Values in parentheses show uplift versus the no-skill baseline when baseline data is available.

## Tier 1: Static Validation Summary

Tier 1 validation passed (`nv-base validate`, 2026-07-21): all validators passed, quality score 93.8/100 (Grade A). No critical or gating findings.

## Tier 2: Deduplication Summary

This tier was not run or did not produce findings in this report.

## Publication Recommendation

The skill is suitable to proceed toward NVSkills-Eval publication based on this benchmark. Skill owners should keep this file with the skill and refresh it when the evaluation dataset, skill behavior, or target agents materially change.
