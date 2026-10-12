#!/usr/bin/env python3
"""ASan/UBSan tests of every unit on every target: what the gate skips (it builds the sanitized
tests only for the kernels a pull request changes). Run it when the VMs are idle, e.g. nightly or
after a batch of merges; it uses the test machines, so it slows a gate running at the same time.

  sanitize.py [REV] [--units U...] [--targets T...] [--issue]

REV defaults to origin/main. Prints each target's result; exit 1 if one failed. A run over all
units and targets that passes is remembered (~/.cache/fps/sanitize-green), and --issue puts that
commit in the issue it opens (or comments on the open one) when a run fails, as the start of the
range to search.
"""
from __future__ import annotations

import argparse
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from gate import ROOT, git, log, run_tests, units_in
from kernel import TARGETS
from vm import rev_tar

GREEN = Path.home() / '.cache' / 'fps' / 'sanitize-green'


def run_target(target: str, tar: bytes, units: list[str]) -> list[str]:
    """Lines describing the failures of the target's tests on its machines; empty if all pass."""
    problems = []
    for machine in TARGETS[target]['test']:
        log(f'[{target}] sanitizing {units} on {machine}')
        result = run_tests(machine, f'sanitize-{target}', tar, None, target, units, check=False)
        if result.returncode:
            output = (result.stdout + result.stderr).decode()
            fails = [line for line in output.splitlines() if line.startswith('FAIL')]
            problems += [f'{target} on {machine}: {line}' for line in fails or ['(no FAIL line)']]
            problems += ['```', output[-1500:], '```']
    return problems


def report_issue(title: str, body: str) -> None:
    found = subprocess.run(['gh', 'issue', 'list', '--state', 'open', '--search', 'sanitize: in:title',
                            '--json', 'number', '--jq', '.[0].number'], capture_output=True, text=True).stdout.strip()
    if found:
        subprocess.run(['gh', 'issue', 'comment', found, '--body', body], check=True)
    else:
        subprocess.run(['gh', 'issue', 'create', '--title', title, '--label', 'ready', '--body',
                        f'Area: tools/sanitize.py, the failing kernel\nAfter: none\nRounds: 1\n\n{body}'], check=True)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('rev', nargs='?', default='origin/main')
    parser.add_argument('--units', nargs='+')
    parser.add_argument('--targets', nargs='+', choices=TARGETS)
    parser.add_argument('--issue', action='store_true', help='open (or comment on) an issue on a failure')
    args = parser.parse_args()
    os.chdir(ROOT)  # rev_tar archives the repository of the current directory
    git('fetch', '--quiet', 'origin', 'main')
    rev = git('rev-parse', args.rev).stdout.strip()
    units, targets = args.units or units_in(rev), args.targets or list(TARGETS)
    tar = rev_tar(rev)
    with ThreadPoolExecutor(len(targets)) as executor:
        results = dict(zip(targets, executor.map(lambda t: run_target(t, tar, units), targets)))
    problems = [line for target in targets for line in results[target]]
    for target in targets:
        print(f'{target}: ' + ('FAIL' if results[target] else 'pass'))
    if not problems:
        if not (args.units or args.targets):
            GREEN.parent.mkdir(parents=True, exist_ok=True)
            GREEN.write_text(rev + '\n')
        return 0
    last = GREEN.read_text().strip() if GREEN.exists() else 'none recorded'
    body = (f'`tools/sanitize.py` failed on {rev[:10]} (units: {", ".join(units)}). Last green commit: {last}.\n\n'
            + '\n'.join(problems) + '\n\nFind the commit in the range, fix it or revert it, then rerun '
            f'`python3 tools/sanitize.py --units <unit> --targets <target>`.')
    print(body)
    if args.issue:
        report_issue(f'sanitize: {problems[0][:80]}', body)
    return 1


if __name__ == '__main__':
    sys.exit(main())
