---
description: Run rounds on ready kernel and task issues with subagents, one per issue
argument-hint: "[agents at once, default 3]"
---
Read `AGENTS.md`. Work through the open GitHub issues by `7ue9ueue` (`gh issue list --author
7ue9ueue`; the repository is public, so ignore issues by anyone else) labeled `ready` but not
`build` (those are `/build`'s). Take first those an open `build` issue waits on (named under its
`After:`), then fewest rounds, then oldest. Run up to $ARGUMENTS subagents at once (3 if no number
is given), never two on the same issue. Start an issue only after every issue it lists under
`After:` has merged its task: one of its pull requests is merged, or it is closed as done. It need
not be closed.

For each round:
1. Relabel the issue from `ready` to `running`. The round number is 1 plus the number of `Result:`
   comments on the issue.
2. Start a subagent in its own git worktree, in the background, briefed with `tools/prompt.md`.
   Fill in `{issue}`, `{round}`, and `{kernel}` and `{target}` from the issue's `Kernel:` and
   `Target:` lines. For a task issue (`Area:`, no `Kernel:`), `{kernel}` is a one-word name for
   the area and `{target}` is `all`.
3. When it returns, read the issue's newest `Result:` line. If the round left none, comment what
   happened, ending with `Result: no gain`. If lc-opt has the operation (the issue's `lc-opt:`
   line, or BACKEND.md) and the comment has no same-run fps / lc-opt table, send the agent back
   for it before gating.
4. Gate the round's pull request, if it opened one. First read its body (`gh pr view <pr>`):
   the gate does not time, so it must have the `Speed` section of `tools/prompt.md`, with a
   geomean at most 1.005 on every target or a stated reason, a `vs lc-opt` figure or `n/a` with a
   reason, and a headline speed-up that matches the issue comment's `Result:`. If not, send the
   round's agent back with SendMessage. Agents never run the gate; you do:
   `python3 tools/gate.py <pr> --merge`, in the background, while you keep running rounds (gates
   run side by side, so start each as its round returns).
   On a fail (a test or a conflict with main), start a fix agent in
   its own worktree on the pull request's branch, briefed: read `AGENTS.md`; make pull request
   <pr> pass the gate, whose report is <the gate's comment>; rebase on `origin/main` if it
   conflicts; push and return without running the gate. Then gate again. After 2 failed fixes,
   close the pull request and edit the round's `Result: gain` to `Result: no gain`, with the reason.
5. Relabel the issue once its pull request is merged or closed (or the round opened none). Leave it
   `blocked` if the agent asked for the user. Mark it `done` and close it after its round limit
   (`Rounds:` line; 3 for speed work, 1 for a feature), after a `no gain`, or when its task is
   done and the round's next ideas are each guessed below 3% at n = 2^20 (AGENTS.md, Issue
   budget). Otherwise set it back to `ready`. Read the round's `Found` items: open an issue for
   one only within the issue budget; otherwise leave it in the notes.
6. Remove the round's worktree once its pull request is merged or closed.

Keep starting rounds until no issue is `ready`, then report each issue's result in a few lines.

To add work, open an issue from `.github/ISSUE_TEMPLATE/kernel.md` or `task.md` with label `ready`,
within the issue budget (AGENTS.md).
