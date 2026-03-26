# RESEARCH_SNAPSHOT.md

Current research snapshot for this template.

Reviewed on: 2026-03-22.

## What changed this iteration

The current template now leans harder into:

- explicit multi-pass loops instead of one-shot output
- a default single-agent workflow that emulates manager, builder, tester, critic, and reviewer roles
- evaluation quality as a first-class concern
- durable state and incremental progress for long-running work
- stricter rules for when multi-agent parallelism is worth the coordination cost

## Current conclusions

### 1. Simple, composable workflows remain the strongest default

Recent provider guidance still converges on the same operational pattern: start with a single agent, add tools and structure, and introduce multi-agent orchestration only when the task clearly benefits from decomposition or parallel reasoning.

Template impact:

- `AGENTS.md` defaults to sequential role emulation first
- multi-agent use is treated as an escalation path, not a starting point

### 2. Critique/refinement loops are still core, but they need external verification

Foundational work like Self-Refine and Reflexion still supports iterative self-feedback. More recent work on process reward models suggests actor-critic style improvement remains an active direction. Operationally, recent engineering guidance reinforces that critique is most useful when tied to tests, transcripts, or explicit evaluators rather than free-form “think harder” prompting.

Template impact:

- every substantial task includes tester and critic passes
- critic output is not trusted by itself; it must drive another concrete verification step

### 3. Evaluation quality is now a bigger differentiator than raw prompt cleverness

Recent evaluation guidance emphasizes clean tasks, reference solutions, balanced problem sets, and grading the produced result rather than the exact path taken. This is reinforced by benchmark-integrity work showing that contaminated or weak benchmarks can mislead teams badly.

Template impact:

- stronger definition-of-done requirements
- more emphasis on executable checks and reference behavior
- explicit skepticism toward benchmark-only optimization

### 4. Long-running agents need durable handoffs and incremental progress

Recent harness work shows agents do better when they start each session by reading progress artifacts, choose one well-scoped task at a time, and leave behind structured state for the next pass. Unstructured “continue until perfect” loops are less reliable without these artifacts.

Template impact:

- `AGENT_PROGRESS.md`, `AGENT_TASKS.json`, and `AGENT_DECISIONS.md` are recommended durable artifacts
- incremental progress is preferred over broad one-shot implementation

### 5. Multi-agent systems help on parallelizable work, but coordination cost is real

Recent production and research writeups show multi-agent systems can unlock more reasoning capacity and specialization, but they also consume many more tokens, introduce coordination failure modes, and fit research-style breadth tasks better than tightly coupled coding tasks.

Template impact:

- explicit ownership and task locks are required for multi-agent work
- the template avoids prescribing multi-agent flows for ordinary repo changes

### 6. Tool ergonomics matter as much as raw model capability

Recent tool-design guidance shows that clearer tool purpose, better descriptions, and evaluation-driven iteration materially improve agent outcomes. More tools are not automatically better.

Template impact:

- agents should prefer a small set of well-defined tools
- repo maintainers should update behavior and artifacts based on observed failure transcripts, not intuition alone

### 7. Benchmarks must be interpreted cautiously

2026 evidence from OpenAI and Anthropic argues that benchmark scores can be distorted by contamination, grader flaws, or infrastructure differences. Small leaderboard gaps are not enough to justify template changes.

Template impact:

- do not change default agent behavior because of a small benchmark delta alone
- require stronger operational evidence before promoting a new pattern

## Sources reviewed

| Date | Source | Why it mattered | Repo consequence |
| --- | --- | --- | --- |
| 2026-02-23 | [Why SWE-bench Verified no longer measures frontier coding capabilities](https://openai.com/index/why-we-no-longer-evaluate-swe-bench-verified/) | Strong warning on contamination and flawed tests in coding benchmarks | Added explicit benchmark skepticism and stronger evaluation rules |
| 2026-01-09 | [Demystifying evals for AI agents](https://www.anthropic.com/engineering/demystifying-evals-for-ai-agents) | Concrete guidance on solvable tasks, reference solutions, balanced evals, and grader design | Tightened definition of done and testing guidance |
| 2026-01-21 | [Designing AI-resistant technical evaluations](https://www.anthropic.com/engineering/AI-resistant-technical-evaluations) | Reinforces eval integrity as an adversarial problem | Kept evaluation integrity as an explicit update-plan concern |
| 2026-02-05 | [Building a C compiler with a team of parallel Claudes](https://www.anthropic.com/engineering/building-c-compiler) | High-signal lessons on task locks, specialization, and test quality for autonomous teams | Added multi-agent task-lock and ownership rules |
| 2026 | [Quantifying infrastructure noise in agentic coding evals](https://www.anthropic.com/engineering/infrastructure-noise) | Shows infra configuration can move scores by more than leaderboard gaps | Added caution against over-reading small benchmark differences |
| 2025-11-26 | [Effective harnesses for long-running agents](https://www.anthropic.com/engineering/effective-harnesses-for-long-running-agents) | Best recent source on progress artifacts and incremental long-running execution | Added recommended durable artifacts and incremental progress rules |
| 2025-09-29 | [Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) | Strong framing for context as a systems-design problem | Strengthened context-discipline rules |
| 2025-09-11 | [Writing effective tools for agents — with agents](https://www.anthropic.com/engineering/writing-tools-for-agents) | Clear evidence that tool ergonomics and tool evals matter | Added tool-quality guidance |
| 2025-06-13 | [How we built our multi-agent research system](https://www.anthropic.com/engineering/built-multi-agent-research-system) | Good evidence on when orchestrator-worker systems pay off and when they are expensive | Multi-agent kept as an escalation path only |
| 2025-03-11 | [New tools for building agents](https://openai.com/index/new-tools-for-building-agents/) | Useful general framing: tools, instructions, models, and reliable agent foundations | Kept template portable and tool-oriented instead of SDK-shaped |
| 2025 | [A practical guide to building AI agents](https://cdn.openai.com/business-guides-and-resources/a-practical-guide-to-building-agents.pdf) | Strong general guidance to start simple, define guardrails, and scale orchestration only when needed | Reinforced single-agent-first design |
| 2025-02 | [Process Reward Models for LLM Agents: Practical Framework and Directions](https://arxiv.org/abs/2502.10325) | Recent actor-critic direction relevant to the repo’s requested principles | Justified explicit actor-critic language in the template |
| 2023-03-21 | [Reflexion: Language Agents with Verbal Reinforcement Learning](https://arxiv.org/abs/2303.11366) | Foundational support for reflection loops with feedback memory | Preserved critic/refine passes as a core pattern |
| 2023-03-30 | [Self-Refine: Iterative Refinement with Self-Feedback](https://arxiv.org/abs/2303.17651) | Foundational support for iterative self-feedback and revision | Preserved multi-pass refinement as a default behavior |

## Open questions for the next refresh

- Do newer benchmark families like SWE-bench Pro or SWE-Lancer materially change what “good agent behavior” should look like in a portable dev template?
- Is there enough cross-provider evidence yet to recommend a stronger explicit memory protocol beyond simple progress/task/decision files?
