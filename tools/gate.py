#!/usr/bin/env python3
"""Merge gate. Runs on the Mac; needs git, gh and gcloud.

  gate.py PR [--merge]          a pull request, merged into origin/main
  gate.py REV [--base REV]      a local revision, merged into --base (default origin/main)

For every target in tools/kernel.py, on the target's machines:
  - test: kernel.py check, then kernel.py test for every affected kernel, on each test machine
    ('mac' is this Mac, with clang);
  - time, if the target has a timing machine: kernel.py bench against the base, 21 rounds;
    on a slowdown, 21 more, pooled.
Units are the backend kernels and the frontend (see tools/kernel.py). Affected units: those
changed, those that include them, or all if a shared file changed.
Passes if every test passes and, per target, the geomean over units of new/base (each unit's
geomean over its gated cases) is at most TOLERANCE (1.005). Byte-identical bench binaries count
as 1. The tolerance absorbs code-layout and same-process effects on unchanged code (0.1-1% seen).

For a pull request it posts the report as a comment and sets the commit status `gate` on its
head, which `main` requires; with --merge it squash-merges on a pass. Merges hold a lock on this
Mac, so two gates never both merge against the same main. If main moved during the run, the gate
runs again only when the new commits on main touch a unit this pull request affects (with the same
include closure) or a shared file; otherwise the result still holds and it merges. The last line of the report
is "Gate: pass" or "Gate: fail".

The repository is public and the gate runs a pull request's code on the VMs and this Mac, so it
refuses pull requests by any account but the repository's owner, and pull requests from forks.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import re
import io
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kernel import TARGETS, summarize, table, unit_dir
from vm import rev_tar, ship, ssh

ROUNDS = 21
RAW = Path.home() / 'Documents' / 'hpc_library-explore' / 'gate'  # raw times, kept on the Mac
TOLERANCE = 1.005  # new/base per target; above it, ROUNDS more rounds are pooled and judged again
SHARED = re.compile(r'^(backend/[^/]+|tools/kernel\.py)$')  # files outside a kernel folder


ROOT = Path(__file__).resolve().parent.parent  # paths below are relative to the repository root


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(['git', *args], check=check, capture_output=True, text=True, cwd=ROOT)


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def unit_paths(unit: str) -> list[str]:
    """Folders whose changes change the unit's code."""
    return ['frontend/', 'include/'] if unit == 'frontend' else [f'backend/{unit}/']


def units_in(tree: str) -> list[str]:
    names = git('ls-tree', '--name-only', tree, 'backend/', check=False).stdout.split()
    units = [n.split('/')[1] for n in names
             if git('cat-file', '-e', f'{tree}:{n}/test.cpp', check=False).returncode == 0]
    frontend = git('ls-tree', '--name-only', tree, 'frontend/', check=False).stdout.split()
    if any(re.fullmatch(r'frontend/test[^/]*\.cpp', n) for n in frontend):
        units.append('frontend')
    return sorted(units)


def affected(base: str, tree: str, includes: str | None = None) -> list[str]:
    """Units the changes from base to tree affect. The units and their includes are read from
    `includes` (default: tree)."""
    includes = includes or tree
    units = units_in(includes)
    changed = git('diff', '--name-only', base, tree).stdout.split()
    if any(SHARED.match(path) for path in changed):
        return units
    hit = {u for u in units if any(path.startswith(tuple(unit_paths(u))) for path in changed)}
    uses = {u: set(git('grep', '-h', '-o', '-E', r'backend/[A-Za-z0-9_]+/', includes, '--',
                       *unit_paths(u), check=False).stdout.split()) for u in units}
    while True:  # add units that include an affected kernel, until nothing changes
        more = {u for u in units if u not in hit and any(f'backend/{h}/' in uses[u] for h in hit)}
        if not more:
            return sorted(hit)
        hit |= more


def bench(vm: str, path: str, base_path: str, kernel: str, target: str) -> dict:
    command = (f'cd {path} && python3 tools/kernel.py bench {kernel} --target {target} '
               f'--base $HOME/{base_path} --rounds {ROUNDS} --json gate.json >/dev/null && cat gate.json')
    result = ssh(vm, command, capture=True)
    if result.returncode:
        raise RuntimeError(f'bench {kernel} on {vm} failed:\n{result.stderr.decode()}')
    return json.loads(result.stdout)


def pool(report: dict, again: dict) -> None:
    """Add a second bench run's rounds to report. A case holds 'base' and 'new' lists of times
    per round and an 'items' count."""
    for case, times in again['cases'].items():
        entry = report['cases'].setdefault(case, {'items': times.get('items', 0)})
        for side in ('base', 'new'):
            entry.setdefault(side, []).extend(times.get(side, []))
    report['rounds'] += again['rounds']


def kernel_ratio(report: dict) -> float | None:
    return 1.0 if report['identical'] else summarize(report['cases'])[1]


def geomean(values: list[float]) -> float | None:
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def run_tests(machine: str, name: str, new_tar: bytes, target: str, kernels: list[str]) -> subprocess.CompletedProcess:
    tests = [f'python3 tools/kernel.py test {k} --target {target}' for k in kernels]
    if machine == 'mac':  # generators are checked on the VMs
        with tempfile.TemporaryDirectory() as tree:
            tarfile.open(fileobj=io.BytesIO(new_tar), mode='r:gz').extractall(tree)
            return subprocess.run(' && '.join(tests), shell=True, cwd=tree, capture_output=True)
    path = ship(machine, name, new_tar)
    result = ssh(machine, f'cd {path} && ' + ' && '.join(['python3 tools/kernel.py check'] + tests), capture=True)
    ssh(machine, f'rm -rf {path}')
    return result


def run_target(target: str, label: str, new_tar: bytes, base_tar: bytes, base: str,
               kernels: list[str]) -> tuple[bool, str]:
    spec = TARGETS[target]
    name = f'gate-{label}-{target}'
    out = [f'### {target}: `{spec["flags"] or "no flags"}`', '']
    for machine in spec['test']:
        log(f'[{target}] testing on {machine}')
        result = run_tests(machine, name, new_tar, target, kernels)
        lines = [line for line in result.stdout.decode().splitlines() if line.startswith(('pass', 'FAIL'))]
        out.append(f'Tests on {machine}: ' + ('pass' if result.returncode == 0 else 'FAIL'))
        out += [f'- {line}' for line in lines]
        if result.returncode:
            out += ['', '```', (result.stdout + result.stderr).decode()[-3000:], '```']
            return False, '\n'.join(out)
    if spec['time'] is None:
        return True, '\n'.join(out + ['', f'{target}: tested, not timed'])

    timed = [k for k in kernels if git('cat-file', '-e', f'{base}:{unit_dir(k)}/bench.cpp', check=False).returncode == 0]
    path = ship(spec['time'], name, new_tar)
    base_path = ship(spec['time'], f'{name}.base', base_tar)
    reports = {}
    for k in timed:
        log(f'[{target}] timing {k} on {spec["time"]}')
        reports[k] = bench(spec['time'], path, base_path, k, target)
        ratio = kernel_ratio(reports[k])
        if ratio is not None and ratio > TOLERANCE:
            log(f'[{target}] {k}: {ratio:.4f}, timing {ROUNDS} more rounds')
            again = bench(spec['time'], path, base_path, k, target)
            pool(reports[k], again)
    ssh(spec['time'], f'rm -rf {path} {base_path}')
    for k, report in reports.items():  # every round's times, pooled rounds included
        out_path = RAW / label / f'{target}-{k}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1))

    ratios = {k: kernel_ratio(r) for k, r in reports.items()}
    overall = geomean([r for r in ratios.values() if r is not None])
    cpu = next(iter(reports.values()))['cpu'] if reports else spec['time']
    out += ['', f'Timing on {spec["time"]} ({cpu}), new/base, below 1 is faster. '
            f'Raw times per round: `{RAW / label}/{target}-<unit>.json` on the Mac.']
    for k in kernels:
        if k not in reports:
            out.append(f'- {k}: no bench in base, not timed')
        elif reports[k]['identical']:
            out.append(f'- {k}: byte-identical bench binaries')
        else:
            rows, ratio = summarize(reports[k]['cases'])
            value = f'{ratio:.4f}' if ratio is not None else 'no gated cases'
            out += [f'- {k}: **{value}** ({reports[k]["rounds"]} rounds)', '',
                    '<details><summary>cases</summary>', '', table(rows), '', '</details>', '']
    passed = overall is None or overall <= TOLERANCE
    out.append(f'{target}: ' + ('no timed kernels' if overall is None else f'{overall:.4f}')
               + ('' if passed else f' SLOWER (limit {TOLERANCE})'))
    return passed, '\n'.join(out)


def gh_json(*args: str) -> dict:
    return json.loads(subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout)


def gate(args) -> tuple[bool, str, str, str, list[str]]:
    """Returns (passed, report, the base commit it ran against, the head commit, affected units)."""
    git('fetch', '--quiet', 'origin', 'main')
    pr = args.target if args.target.isdigit() else None
    if pr:
        info = gh_json('pr', 'view', pr, '--json', 'headRefOid,state,author,isCrossRepository')
        owner = gh_json('repo', 'view', '--json', 'owner')['owner']['login']
        if info['author']['login'] != owner or info['isCrossRepository']:
            sys.exit(f'refusing pull request {pr}: it is by {info["author"]["login"]}'
                     f'{" from a fork" if info["isCrossRepository"] else ""}, and the gate runs its code. '
                     f'Only {owner}\'s own pull requests are gated.')
        if info['state'] != 'OPEN':
            sys.exit(f'pull request {pr} is {info["state"].lower()}')
        for _ in range(12):  # right after a push, GitHub can report the old head for a few seconds
            git('fetch', '--quiet', 'origin', f'pull/{pr}/head')
            head = git('rev-parse', 'FETCH_HEAD').stdout.strip()
            if head == info['headRefOid']:
                break
            time.sleep(5)
            info = gh_json('pr', 'view', pr, '--json', 'headRefOid,state,author,isCrossRepository')
        else:
            sys.exit(f'pull request {pr}: GitHub reports head {info["headRefOid"][:10]}, fetched {head[:10]}')
        label = f'pr{pr}'
    else:
        head = git('rev-parse', args.target).stdout.strip()
        label = head[:10]
    base = git('rev-parse', args.base).stdout.strip()
    merged = git('merge-tree', '--write-tree', base, head, check=False)
    if merged.returncode:
        return False, f'Conflicts with {args.base} ({base[:10]}): rebase first.\n\nGate: fail', base, head, []
    tree = merged.stdout.split()[0]

    kernels = affected(base, tree)
    lines = [f'Gate for {label} ({head[:10]}) merged into {args.base} ({base[:10]}).', '']
    if not kernels:
        return True, '\n'.join(lines + ['No unit affected.', '', 'Gate: pass']), base, head, []
    lines += [f'Units: {", ".join(kernels)}', '']
    new_tar = subprocess.run(['git', 'archive', '--format=tar.gz', tree], check=True, capture_output=True).stdout
    base_tar = rev_tar(base)
    with ThreadPoolExecutor(len(TARGETS)) as pool:
        futures = [pool.submit(run_target, t, label, new_tar, base_tar, base, kernels) for t in TARGETS]
        results = [f.result() for f in futures]
    passed = all(ok for ok, _ in results)
    lines += [text + '\n' for _, text in results]
    lines.append('Gate: ' + ('pass' if passed else 'fail'))
    return passed, '\n'.join(lines), base, head, kernels


def set_status(head: str, passed: bool, url: str) -> None:
    """The commit status `gate`, which branch protection on main requires."""
    repo = gh_json('repo', 'view', '--json', 'nameWithOwner')['nameWithOwner']
    subprocess.run(['gh', 'api', f'repos/{repo}/statuses/{head}', '-f', f'state={"success" if passed else "failure"}',
                    '-f', 'context=gate', '-f', f'description=Gate: {"pass" if passed else "fail"}',
                    '-f', f'target_url={url}'], check=True, capture_output=True)


def merge_lock():
    """Held from the last look at main to the merge."""
    path = Path.home() / '.cache' / 'fps' / 'gate-merge.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(path, 'w')
    fcntl.flock(lock, fcntl.LOCK_EX)
    return lock


def stale(base: str, tip: str, merged: str, units: list[str]) -> bool:
    """Whether commits that landed on main between base and tip can change this gate's result.
    merged: the pull request merged into tip, whose includes decide what depends on what."""
    return bool(set(affected(base, tip, merged)) & set(units))


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('target', help='pull request number or git revision')
    parser.add_argument('--base', default='origin/main')
    parser.add_argument('--merge', action='store_true', help='squash-merge the pull request on a pass')
    args = parser.parse_args()
    pr = args.target if args.target.isdigit() else None
    for _ in range(3):
        passed, report, base, head, units = gate(args)
        print(report)
        if pr:
            url = subprocess.run(['gh', 'pr', 'comment', pr, '--body-file', '-'], input=report, text=True,
                                 check=True, capture_output=True).stdout.strip()
            set_status(head, passed, url)
        if not (passed and args.merge and pr):
            return 0 if passed else 1
        with merge_lock():
            git('fetch', '--quiet', 'origin', 'main')
            tip = git('rev-parse', 'origin/main').stdout.strip()
            if tip != base:
                merged = git('merge-tree', '--write-tree', tip, head, check=False)
                if merged.returncode:
                    log(f'main moved to {tip[:10]} and now conflicts: rebase, then run the gate again')
                    return 1
                if stale(base, tip, merged.stdout.split()[0], units):
                    log(f'main moved to {tip[:10]} and touches {", ".join(units)}: running the gate again')
                    continue
                log(f'main moved to {tip[:10]} but touches none of {", ".join(units) or "no units"}: merging')
            subprocess.run(['gh', 'pr', 'merge', pr, '--squash'], check=True)
            return 0
    log('main kept changing the affected units; not merged')
    return 1

if __name__ == '__main__':
    sys.exit(main())
