# frontend

The user API in `include/fps/`, planned in `DESIGN.md`, built issue by issue with `/build`.
Tests: `frontend/test_*.cpp`, one per area. Bench: `frontend/bench.cpp` (API calls as `<op>/<size>`,
the backend call they wrap as `<op>@kernel/<size>`).

Prototype: a local, unpushed branch (`claude/frontend-lc-opt-backend-8360f6`, worktree
`.claude/worktrees/frontend-lc-opt-backend-8360f6`) implements most of DESIGN.md over a private copy
of lc-opt's kernels. Read it and move code from it; do not copy its `include/fps/lcopt/` (that work
belongs in `backend/`).

## Log

- 2026-10-11: scalars (`mod_mul`, `mod_pow`, `mod_inverse`) from the prototype's `modular.hpp`,
  as the first piece through the pipeline. Tests against brute force on 6 primes from 2 to 2^32 - 5.
