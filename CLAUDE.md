# CSCI3033 LLM Reasoner — Final Project

## First-Principles Standards

**1. Clarify before acting.**
If the goal is unclear, stop and ask. Do not infer intent silently.

**2. Shortest path wins.**
If a better approach exists, say so and wait for a decision.

**3. Fix root causes, not symptoms.**
Find out why before touching code. No defensive patches.

**4. Output only what changes decisions.**
Skip preamble, summaries, and obvious observations.

---

## Context Management

State lives in structured files — each concern has one authoritative source.

**Session start — read in order:**
1. `CLAUDE.md` — this file (map + principles)
2. `.claude/features.json` — feature list and status
3. `.claude/sp-harness.json` — config (dev_mode, hygiene counter, external_codebase flag)
4. `.claude/codebase-context.md` — only if sp-harness.json has `external_codebase: true`
5. `.claude/todos.json` — idea backlog
6. `.claude/memory.md` — short-term session memory (undecided observations)
7. `git log --oneline -20` — recent activity
8. `git status` — uncommitted work (where you physically left off)

**Rules:**
- commits use `[module]: description` format
- Decided ideas → `.claude/todos.json` (manage-todos)
- Decided requirements → `.claude/features.json` (manage-features)
- Undecided observations → `.claude/memory.md`
- Design rationale → `docs/design-docs/`
- Reusable patterns → raise via sp-feedback (agent-memory)

Each concern has ONE home. Never duplicate across sources.

---

## Project Map

Stack: Python + PyTorch + HuggingFace. Master-level LLM course final project.

### Design Docs
docs/
├── design-docs/     — specs from brainstorming (empty)
├── plans/
│   ├── active/      — plans currently being executed (empty)
│   └── completed/   — finished plans (empty)
└── reports/         — sp-feedback optimization reports (empty)

### Codebase
(empty — to be populated as the project develops)
