---
description: Build the frontend from DESIGN.md, one subagent per build issue, reviewed and gated
argument-hint: "[agents at once, default 2]"
---
Read `AGENTS.md` and `DESIGN.md`. You coordinate; subagents write the code. Work through the open
GitHub issues labeled `build` and `ready`, oldest first, with up to $ARGUMENTS implementers at once
(2 if no number is given). Never two implementers on the same issue. Start an issue only after every
issue under its `After:` line has merged its task: it has a `Result: done` or `Result: gain`
comment, or is closed as done.

Plan when no `build` issue is open or ready and DESIGN.md still has parts missing from
`include/fps/`: open one issue per missing part from `.github/ISSUE_TEMPLATE/build.md`, labeled
`build` and `ready`. A part is what one agent can finish in one sitting: a DESIGN.md section or a
group of functions (for example `Polynomial` and its arithmetic; products; division and calculus;
series functions; evaluation and interpolation; algebra; `Fraction`; `Values`; raw functions;
`Factorials`). Fill in `After:` so each part comes after the parts it calls. Search open issues first.

For each issue:
1. Relabel it from `ready` to `running`.
2. Implement: start a subagent in its own git worktree, in the background, briefed with
   `tools/prompt_build.md` (fill in `{issue}`). It opens a pull request, passes the gate without
   merging, and returns the pull request's number.
3. Review: start a second subagent, in the background, briefed with `tools/prompt_review.md`
   (fill in `{issue}` and `{pr}`). It has not seen the implementer's reasoning, only the issue,
   DESIGN.md and the diff. It comments on the pull request; the last line is `Review: pass` or
   `Review: changes`.
4. On `Review: changes`, send the implementer the review's link with SendMessage. It fixes, passes
   the gate again and returns; then review again with a new reviewer. After two reviews with
   changes, label the issue `blocked` and comment for the user what is unresolved.
5. On `Review: pass`, tell the implementer to merge (`tools/gate.py <pr> --merge`) and finish.
   Check that the issue's last comment ends with `Result: done`, then close it with the label `done`.
6. Remove the worktree once the implementer has stopped all background work.

If the implementer labels the issue `blocked`, leave it for the user. Keep starting issues until
none is ready, then report each issue's result in a few lines.
