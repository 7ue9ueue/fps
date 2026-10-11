---
description: Run backend rounds on ready kernel issues with subagents, one per issue
argument-hint: "[agents at once, default 3]"
---
Read `AGENTS.md`. Work through the open GitHub issues labeled `ready` but not `build` (those are
`/build`'s), fewest rounds first, then oldest, with up to $ARGUMENTS subagents at once (3 if no
number is given). Never two subagents on the same issue. Start an issue only after every issue it
lists under `After:` has merged its task: it has a `Result: gain` or `Result: done` comment, or is
closed as done. It need not be closed.

For each round:
1. Relabel the issue from `ready` to `running`. The round number is 1 plus the number of `Result:`
   comments on the issue.
2. Start a subagent in its own git worktree, in the background, briefed with `tools/prompt.md`.
   Fill in `{issue}`, `{round}`, and `{kernel}` and `{target}` from the issue's `Kernel:` and
   `Target:` lines.
3. When it returns, read the issue's newest `Result:` line. If the round left none, comment what
   happened, ending with `Result: no gain`.
4. Relabel the issue. Leave it `blocked` if the agent asked for the user. Mark it `done` and close
   it after its round limit (`Rounds:` line, default 5) or 2 `no gain` results in a row. Otherwise
   set it back to `ready`.
5. Remove the round's worktree once its pull request is merged or closed.

Keep starting rounds until no issue is `ready`, then report each issue's result in a few lines.

To add work, open an issue from `.github/ISSUE_TEMPLATE/kernel.md` with label `ready`.
