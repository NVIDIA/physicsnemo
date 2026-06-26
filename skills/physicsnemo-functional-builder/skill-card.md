## Description: <br>
Official NVIDIA-authored workflow for adding a new functional op (or a new optimized backend for an existing op) to `physicsnemo.nn.functional`. Scaffolds a `FunctionSpec` with multi-backend dispatch (a torch reference plus optional Warp, cuML, or SciPy backends), wires re-exports, writes cross-backend equivalence tests, and runs the local CI gates. <br>

This skill is ready for commercial/non-commercial use. <br>

## Owner
NVIDIA <br>

### License/Terms of Use: <br>
Apache-2.0 <br>
## Use Case: <br>
Contributors and researchers adding a new functional op to the PhysicsNeMo package, or adding an accelerated backend (Warp CUDA kernels, cuML, or SciPy) to an existing op, so it follows the repository's functional conventions (`FunctionSpec` registration and dispatch, `torch.library.custom_op` wrapping, placement, typing, validation, cross-backend tests) and passes CI. <br>

### Deployment Geography for Use: <br>
Global <br>

## Known Risks and Mitigations: <br>
Risk: The skill scaffolds and edits source files, including GPU kernels; generated code could be incorrect, incomplete, or place files in the wrong location if the live repository structure differs from assumptions. <br>
Mitigation: The skill reads `physicsnemo/core/function_spec.py` and an existing exemplar op before scaffolding, verifies paths against the live repo, runs the CI gates (ruff, interrogate, pytest) and cross-backend equivalence tests, and runs an independent code-review pass over the diff before completion. It defers the op's algorithm and any hand-written kernel to the human. Review the diff, the cross-backend parity tests, and the CI result before merging. <br>

## Reference(s): <br>
- [placement.md](references/placement.md) <br>
- [dispatch.md](references/dispatch.md) <br>
- [backends.md](references/backends.md) <br>
- [testing.md](references/testing.md) <br>
- [scaffolds.md](references/scaffolds.md) <br>
- [lessons.md](references/lessons.md) <br>
- [PhysicsNeMo GitHub Repository](https://github.com/NVIDIA/physicsnemo) <br>


## Skill Output: <br>
**Output Type(s):** [Code scaffolding, File edits, Analysis] <br>
**Output Format:** [Python, Markdown] <br>
**Output Parameters:** [N/A] <br>
**Other Properties Related to Output:** [Generated files are standards-compliant skeletons completed by the contributor; the skill does not author the op's algorithm or hand-written kernels.] <br>

## Evaluation Agents Used: <br>
- Claude Code (`claude-code`) <br>
- Codex (`codex`) <br>



## Evaluation Tasks: <br>
Evaluated against 4 internal evaluation tasks (2 positive skill-activation, 2 negative) with 2 attempts per task via NVSkills-Eval. <br>

## Evaluation Metrics Used: <br>
Reported benchmark dimensions: <br>
- Security: Checks whether skill-assisted execution avoids unsafe behavior such as secret leakage, destructive commands, or unauthorized access. <br>
- Correctness: Checks whether the agent follows the expected workflow and produces the correct final output. <br>
- Discoverability: Checks whether the agent loads the skill when relevant and avoids using it when irrelevant. <br>
- Effectiveness: Checks whether the agent performs measurably better with the skill than without it. <br>
- Efficiency: Checks whether the agent uses fewer tokens and avoids redundant work. <br>

Underlying evaluation signals used in this run: <br>
- `security`: Checks for unsafe operations, secret leakage, and unauthorized access. <br>
- `skill_execution`: Verifies that the agent loaded the expected skill and workflow. <br>
- `skill_efficiency`: Checks routing quality, decoy avoidance, and redundant tool usage. <br>
- `accuracy`: Grades final-answer correctness against the reference answer. <br>
- `goal_accuracy`: Checks whether the overall user task completed successfully. <br>
- `behavior_check`: Verifies expected behavior steps, including safety expectations. <br>
- `token_efficiency`: Compares token usage with and without the skill. <br>



## Evaluation Results: <br>
_Pending — populated by NVSkills-Eval prior to publication (see `BENCHMARK.md`)._ <br>

| Dimension | Num | `claude-code` | `codex` |
|---|---:|---:|---:|
| Security | — | — | — |
| Correctness | — | — | — |
| Discoverability | — | — | — |
| Effectiveness | — | — | — |
| Efficiency | — | — | — |

## Skill Version(s): <br>
0.1.0 (source: pyproject.toml) <br>

## Ethical Considerations: <br>
NVIDIA believes Trustworthy AI is a shared responsibility and we have established policies and practices to enable development for a wide array of AI applications. When downloaded or used in accordance with our terms of service, developers should work with their internal team to ensure this skill meets requirements for the relevant industry and use case and addresses unforeseen product misuse. <br>

(For Release on NVIDIA Platforms Only) <br>
Please report quality, risk, security vulnerabilities or NVIDIA AI Concerns [here](https://app.intigriti.com/programs/nvidia/nvidiavdp/detail). <br>
