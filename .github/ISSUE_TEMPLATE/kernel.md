---
name: Kernel
about: A backend kernel or kernel task, worked on in rounds by agents
title: "kernel: target"
labels: ready
---
Kernel: `{kernel}`
Target: `{target}`
After: none
Rounds: 5

Task: {what to do; for speed work, the cases that matter}

Agents work on this in rounds (see `tools/prompt.md`). Each round ends with a comment whose last
line is `Result: gain` or `Result: no gain`. Labels: `ready` waits for a round, `running` has an
agent on it, `blocked` needs the user, `done` is finished.
