You are reviewing pull request #{pr}, which builds part of the fps frontend for issue #{issue}.
You are unattended: do not ask questions. Do not edit code or push.

1. Read `AGENTS.md`, `DESIGN.md`, issue #{issue}, and the pull request with its comments and diff
   (`gh pr view {pr} --comments`, `gh pr diff {pr}`). Read the touched files in full where needed
   (`git fetch origin pull/{pr}/head` and read them from `FETCH_HEAD`).
2. Check the code against the spec, not against your taste:
   - Every function the issue asks for exists, with DESIGN.md's signature, defaults (precision n),
     preconditions and errors (`std::domain_error`, `std::optional`), and its Conventions:
     canonical values, `size()` against `degree()`, no trimming, aliasing, thread safety.
   - If the pull request changes DESIGN.md, the change has a reason that holds (AGENTS.md).
   - Tests: a simple reference, sizes 0, 1, powers of two and neighbours, random inputs, every error.
   - Speed: no copies, allocations or checks on hot paths beyond what DESIGN.md requires; bench
     cases `<op>/<lg>` and `<op>@kernel/<lg>` exist, and the reported ratios are within noise.
   - AGENTS.md's Code rules: readable, small functions, no dead code, short and rare comments,
     start simple.
3. Comment once on the pull request (`gh pr comment {pr}`): a numbered list of findings, each with
   file and line, what is wrong, and what DESIGN.md or AGENTS.md says. Only real problems; no
   style preferences. The last line is exactly `Review: pass` (no findings) or `Review: changes`.
4. Return the comment's last line and the number of findings.
