---
name: Task
about: Any other improvement (a tool, the workflow, a brief, a doc), worked on in rounds by agents
title: "area: what to improve"
labels: ready
---
Area: {what it touches, e.g. a tool, a brief, a doc}
After: none (or issues whose task must be merged first: #n, #m)
Rounds: 1 (3 if it is speed work)

Seen: {what is slow, fragile or wasteful, with the evidence: times, failures, how often}
Task: {what to do; how to tell it worked}

Agents work on this in rounds (see `tools/prompt.md`), like a kernel issue. Each round ends with a
comment whose last line is `Result: gain` or `Result: no gain`.
