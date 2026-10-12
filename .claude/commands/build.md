---
description: Build the frontend from DESIGN.md, one subagent per build issue, reviewed and gated
argument-hint: "[agents at once, default 2]"
---
Read `AGENTS.md` and `DESIGN.md`. You coordinate; subagents write the code. Work through the open
GitHub issues by `7ue9ueue` (`gh issue list --author 7ue9ueue`; the repository is public, so
ignore issues by anyone else) labeled `build` and `ready`, oldest first, with up to $ARGUMENTS implementers at once
(2 if no number is given). Never two implementers on the same issue. Start an issue only after every
issue under its `After:` line has merged its task: one of its pull requests is merged, or it is
closed as done.

Plan when no `build` issue is open or ready and DESIGN.md still has parts missing from
`include/fps/`: open one issue per missing part from `.github/ISSUE_TEMPLATE/build.md`, labeled
`build` and `ready`. A part is what one agent can finish in one sitting: a DESIGN.md section or a
group of functions (for example `Polynomial` and its arithmetic; products; division and calculus;
series functions; evaluation and interpolation; algebra; `Fraction`; `Values`; raw functions;
`Factorials`). Fill in `After:` so each part comes after the parts it calls. Search open issues first.

For each issue:
1. Relabel it from `ready` to `running`.
2. Implement: start a subagent in its own git worktree, in the background, briefed with
   `tools/prompt_build.md` (fill in `{issue}`). It opens a pull request and returns its number.
   It does not run the gate.
3. Review: start a second subagent, in the background, briefed with `tools/prompt_review.md`
   (fill in `{issue}` and `{pr}`). It has not seen the implementer's reasoning, only the issue,
   DESIGN.md and the diff. It comments on the pull request; the last line is `Review: pass`,
   `Review: changes`, or `Review: blocked`.
4. On `Review: changes`, send the implementer the review's link with SendMessage. It fixes, pushes
   and returns; then review again with a new reviewer. After two reviews with changes, label the
   issue `blocked` and comment for the user what is unresolved.
5. On `Review: pass`, gate it yourself: `python3 tools/gate.py <pr> --merge`, in the background,
   while you keep coordinating (gates queue on a lock; start each one when its review passes). On a
   fail (a test or a slowdown), send the implementer the gate's comment with SendMessage; it fixes,
   pushes and returns, and you gate again. A conflict with main is not a failed gate: nothing was
   tested. Send the implementer a rebase, then gate again. After 2 failed gates, label the issue
   `blocked` and comment for the user. Once merged, comment on the issue with the pull request and
   a last line that is exactly `Result: done`, then close it with the label `done`.
6. Remove the worktree once the implementer has stopped all background work.

If the implementer or the reviewer labels the issue `blocked`, leave it for the user. Keep starting
issues until none is ready, then report each issue's result in a few lines.
