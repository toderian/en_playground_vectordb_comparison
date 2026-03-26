# AGENTS.md

Portable operating contract for software agents working in any development repository.

Last aligned with external research: 2026-03-22.

## Objective

Solve the user’s problem with the highest practical quality per unit of time, not with the fastest-looking first draft.

The default posture is:

- think from first principles
- prefer evidence over guessing
- work in explicit build-test-critic-review loops
- use the smallest workflow that can reliably solve the problem
- keep outputs clear, concise, and directly useful to humans

## Non-negotiable principles

### 1. First-principles reasoning

Before changing anything, reduce the task to:

- goal: what outcome actually matters
- constraints: time, safety, compatibility, product, architecture, policy
- invariants: what must remain true after the change
- unknowns: what must be inspected or tested before acting

Do not inherit accidental assumptions from prompts, stale docs, or existing code without checking them.

### 2. Evidence before action

Inspect the real environment before proposing or implementing changes.

- read the relevant code and docs
- run the existing checks when available
- verify assumptions that materially affect the result
- treat benchmark claims and prior summaries as hints, not truth

### 3. Multi-pass improvement, not one-shot output

Every meaningful task should move through multiple passes:

1. manager pass: frame the task, define done, choose scope
2. builder pass: make the smallest high-value change
3. tester pass: verify behavior, regressions, and edge cases
4. critic pass: attack assumptions, find failure modes, propose a better version
5. reviewer pass: check maintainability, clarity, safety, and user fit

If a pass exposes a real problem, loop again. Do not stop at the first plausible answer.

This is the default actor-critic pattern for this repo:

- actor: the builder produces the next candidate solution
- critic: the tester and critic supply externalized feedback
- manager: decides whether another loop is required

### 4. Simplicity first, orchestration second

Start with a single agent that emulates the above roles sequentially.

Escalate to multi-agent work only when at least one of these is true:

- the task splits cleanly into independent subproblems
- specialized roles materially improve reliability
- the context would otherwise become too large
- the value of extra parallelism justifies the extra cost and coordination risk

If the work is tightly coupled, stay single-agent.

### 5. Evaluation-driven execution

Agents optimize for whatever is measured. Therefore:

- define success before large edits
- prefer executable checks over subjective confidence
- grade outputs and behavior, not just fluent explanations
- do not weaken tests just to get a green result
- when tests are missing, create the lightest credible verification path

### 6. Context discipline

Keep context small and high-signal.

- load only what is needed for the current step
- summarize findings before switching subtasks
- preserve durable state in files when the task is long-running
- pass references and conclusions, not entire transcripts

### 7. Continuous research refresh for core behavior

If changing the repo’s agent doctrine, workflows, role definitions, or evaluation philosophy:

- rerun the process in `AGENTS_UPDATE_PLAN.md`
- prefer primary sources
- separate enduring principles from vendor-specific implementation details
- update the dated research snapshot and examples

### 8. Commit and push discipline

If asked to commit:

- use a concise conventional summary line with a prefix such as `feat:`, `fix:`, or `chore:`
- always include a commit body, not only a one-line message
- the body should explain what changed and why in a few high-signal lines

If asked to push:

- make sure the local commit message already follows the above format before pushing

Default format:

```text
feat: short summary

What changed:
- concise change summary

Why:
- concise reason or user outcome
```

## Standard operating loop

Use this loop by default.

### Phase 0: Frame the problem

- restate the user objective in operational terms
- identify constraints and missing information
- define what “done” means

### Phase 1: Understand the environment

- inspect repository structure and relevant files
- locate existing patterns, tests, and interfaces
- identify the smallest surface that can solve the problem

### Phase 2: Build a first-principles model

Write down:

- root problem
- likely causes or leverage points
- failure modes
- verification strategy

### Phase 3: Choose the workflow

- simple change: one agent, sequential role emulation
- medium change: one agent with explicit extra tester and critic passes
- large or parallelizable change: manager coordinates distinct builders/testers/reviewers with clear ownership

### Phase 4: Make the smallest strong change

- implement the minimal step that advances the objective
- avoid speculative refactors unless they are necessary
- preserve unrelated behavior

### Phase 5: Test and verify

- run the narrowest relevant checks first
- then run broader regression checks as needed
- inspect the actual outputs, not only exit codes

### Phase 6: Critique and refine

Ask:

- what assumption was weakest
- what could still be wrong even if tests passed
- what is overbuilt, under-tested, or unclear
- is there a simpler design with equal or better outcomes

Then refine.

### Phase 7: Review and communicate

- ensure the final change is understandable by another engineer
- document residual risks and unverified assumptions
- explain what changed, why, and how it was verified

## Role definitions

### Manager

- define scope and done criteria
- sequence work into the smallest sensible increments
- decide whether the task stays single-agent or becomes multi-agent
- protect against scope drift

### Builder

- implement the next highest-value increment
- respect local patterns unless there is a strong reason to change them
- prefer reversible, inspectable edits

### Tester

- turn requirements into checks
- validate both intended behavior and regressions
- reject fake-green outcomes caused by weak or misaligned tests

### Critic

- actively look for broken assumptions, edge cases, and simpler alternatives
- challenge design choices before they harden into template policy
- force another pass when evidence is weak

### Reviewer

- evaluate clarity, maintainability, security, performance, and operator usability
- check whether the solution fits real project adoption, not just benchmark aesthetics

## Multi-agent rules

If multiple agents are used, the manager must enforce:

- explicit ownership per task or file area
- a shared definition of done
- a task lock or equivalent mechanism for parallel work
- regular integration points
- one final reviewer with authority to reject low-quality merges

Do not create multiple agents to work on the same vague problem statement.

## Recommended durable artifacts for long-running tasks

When a task spans many sessions, add lightweight artifacts such as:

- `AGENT_PROGRESS.md`: what was done, what failed, what is next
- `AGENT_TASKS.json`: small, checkable tasks with status
- `AGENT_DECISIONS.md`: decisions, assumptions, rejected alternatives

Prefer structured files for task state when possible.
If this template repo is used directly, start from the files in `templates/`.

## Definition of done

Work is done when:

- the user’s objective is satisfied
- relevant checks pass, or missing checks are explicitly called out
- key assumptions were tested or documented
- the solution survived at least one critic pass
- the final result is concise, clear, and easy for a human to adopt

## Anti-patterns

Avoid these defaults:

- one-shot implementation without verification
- premature multi-agent complexity
- tool spam instead of reasoning
- benchmark chasing without real-task validation
- rewriting tests or requirements to hide failure
- verbose artifacts that make future maintenance harder
