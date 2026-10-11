You are working on backend kernel(s) {kernel}, target `{target}`, round {round} of issue #{issue}.
You are unattended: the user is offline, so do not ask questions. Decide and act.

1. Read `AGENTS.md`, `BACKEND.md`, issue #{issue} with its comments, and each kernel's
   `backend/<kernel>/notes.md` if it exists.
2. Start branch `agent/issue{issue}-r{round}` from a fresh `origin/main`.
3. This round: do the issue's task. Once it is done, make the kernels faster on `{target}` (on
   every target if `{target}` is `all`). Choose your own approach. Do not repeat a logged attempt
   without a new reason.
4. Explore and measure with `tools/vm.py` and `tools/kernel.py`. Use each target's own test and
   timing machines (`TARGETS` in `tools/kernel.py`), and `lc-intel` for `perf`.
5. Log the attempt in each touched kernel's `notes.md`. Commit, push, and open a pull request.
6. Run `python3 tools/gate.py <pr> --merge`. If it fails, fix and rerun it, or close the pull request
   and log why.
7. Finish with one comment on issue #{issue}: what you tried, the numbers, and a last line that is
   exactly `Result: gain` or `Result: no gain`. A gain is a merged speed-up, or the issue's task
   merged.

If something only the user can fix stops you, comment why on issue #{issue} and add the `blocked` label.
