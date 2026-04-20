#!/bin/bash
cat <<'EOF'
CONTEXT CHECK: If this task surfaced anything worth remembering, route it:
  - Decided idea/direction → sp-harness:manage-todos add (.claude/todos.json)
  - Decided bug to fix → sp-harness:manage-features add with fix_feature
  - Undecided observation (bug/hypothesis/concern/in-flight) → append to
    .claude/memory.md (before adding, triage existing entries against git log
    and other state sources — remove any already-resolved or already-tracked)

Each concern has ONE home. Never duplicate across sources.
EOF
