#!/usr/bin/env python3
"""Merge gate. Runs on the Mac; needs git, gh and gcloud.

  gate.py PR [--merge]          a pull request, merged into origin/main
  gate.py REV [--base REV]      a local revision, merged into --base (default origin/main)

For every target in tools/kernel.py, all at once, on the target's machines:
  - test: kernel.py test of every affected unit on each test machine, at once ('mac' is this Mac,
    with clang); kernel.py check once, on the first target's first test machine;
  - time, if the target has a timing machine: kernel.py bench of the gated cases against the base,
    21 rounds; on a slowdown, 21 more, pooled. `<op>@<variant>` cases are not timed. Timing runs
    alongside the tests unless the timing machine also tests the target (its bench lock would
    queue the two anyway); it stops at the next unit once any test has failed.
Units are the backend kernels and the frontend (see tools/kernel.py). Affected units: those
changed, those that include them, or all if a shared file changed.
Passes if every test passes and, per target, the geomean over units of new/base (each unit's
geomean over its gated cases) is at most TOLERANCE (1.005). Byte-identical bench binaries count
as 1. The tolerance absorbs code-layout and same-process effects on unchanged code (0.1-1% seen).

For a pull request it posts the report as a comment and sets the commit status `gate` on its head,
which `main` requires; with --merge it squash-merges that head on a pass (GitHub refuses if the
head moved). The full report is kept with the raw times; the comment drops the case tables if it
would pass GitHub's limit. Merges hold a lock on this Mac, so two gates never both merge against
the same main. If main moved during the run, the gate runs again only when the new commits on main
touch a unit this pull request affects (with the same include closure) or a shared file; otherwise
the result still holds and it merges. The last line of the report is "Gate: pass" or "Gate: fail".

It runs as origin/main's copy of tools/ (gate.py, kernel.py, vm.py), whatever the checkout it is
started from: a coordinator's checkout can lag main by hours, and the gate's rules are main's.
FPS_GATE_HERE=1 runs this checkout's copy instead, to try a change to the gate.

The repository is public and the gate runs a pull request's code on the VMs and this Mac, so it
refuses pull requests by any account but the repository's owner, and pull requests from forks.
"""
from __future__ import annotations

import argparse
import fcntl
import io
import json
import math
import os
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kernel import TARGETS, summarize, table, unit_dir
from vm import rev_tar, ship, ssh

ROUNDS = 21
RAW = Path.home() / 'Documents' / 'hpc_library-explore' / 'gate'  # raw times, kept on the Mac
TOLERANCE = 1.005  # new/base per target; above it, ROUNDS more rounds are pooled and judged again
SHARED = re.compile(r'^(backend/[^/]+|tools/kernel\.py)$')  # files outside a kernel folder
GATED = '^[^@]*$'  # the cases the verdict uses (kernel.gated); the others took 30-50% of bench time
COMMENT_LIMIT = 65536  # characters in a GitHub comment


# paths below are relative to the repository root; FPS_GATE_ROOT: set by run_from_main
ROOT = Path(os.environ.get('FPS_GATE_ROOT') or Path(__file__).resolve().parent.parent)


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
               f"--base $HOME/{base_path} --rounds {ROUNDS} --filter '{GATED}' --json gate.json >/dev/null "
               f'&& cat gate.json')
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


def run_tests(machine: str, name: str, new_tar: bytes, target: str, units: list[str],
              check: bool) -> subprocess.CompletedProcess:
    command = ('python3 tools/kernel.py check && ' if check else '') + \
        f'python3 tools/kernel.py test {" ".join(units)} --target {target}'
    if machine == 'mac':
        with tempfile.TemporaryDirectory() as tree:
            tarfile.open(fileobj=io.BytesIO(new_tar), mode='r:gz').extractall(tree)
            return subprocess.run(command, shell=True, cwd=tree, capture_output=True)
    share = sum(machine in spec['test'] for spec in TARGETS.values())  # targets testing here share its cores
    command += f' -j $(( ($(nproc) + {share - 1}) / {share} ))'
    path = ship(machine, name, new_tar)
    result = ssh(machine, f'cd {path} && {command}', capture=True)
    ssh(machine, f'rm -rf {path}')
    return result


def time_units(vm: str, name: str, new_tar: bytes, base_tar: bytes, target: str, units: list[str],
               failed: threading.Event) -> dict:
    """Bench reports per unit, one unit at a time; stops before the next unit once a test failed."""
    reports = {}
    if not units or failed.is_set():
        return reports
    path = ship(vm, name, new_tar)
    base_path = ship(vm, f'{name}.base', base_tar)
    try:
        for k in units:
            if failed.is_set():
                break
            log(f'[{target}] timing {k} on {vm}')
            reports[k] = bench(vm, path, base_path, k, target)
            ratio = kernel_ratio(reports[k])
            if ratio is not None and ratio > TOLERANCE and not failed.is_set():
                log(f'[{target}] {k}: {ratio:.4f}, timing {ROUNDS} more rounds')
                pool(reports[k], bench(vm, path, base_path, k, target))
    finally:
        ssh(vm, f'rm -rf {path} {base_path}')
    return reports


def run_target(target: str, label: str, new_tar: bytes, base_tar: bytes, units: list[str], timed: list[str],
               check: bool, failed: threading.Event) -> tuple[bool, str]:
    spec = TARGETS[target]
    name = f'gate-{label}-{target}'
    out = [f'### {target}: `{spec["flags"] or "no flags"}`', '']

    def test_on(machine: str) -> subprocess.CompletedProcess:
        log(f'[{target}] testing on {machine}')
        result = run_tests(machine, f'{name}-test', new_tar, target, units, check and machine == spec['test'][0])
        if result.returncode:
            failed.set()
        return result

    vm = spec['time']
    with ThreadPoolExecutor() as executor:
        tests = {m: executor.submit(test_on, m) for m in spec['test']}
        if vm in spec['test']:
            for future in tests.values():
                future.result()
        reports = time_units(vm, name, new_tar, base_tar, target, timed, failed) if vm else {}
        results = {m: future.result() for m, future in tests.items()}

    tests_ok = True
    for machine, result in results.items():
        lines = [line for line in result.stdout.decode().splitlines() if line.startswith(('pass', 'FAIL'))]
        if result.returncode == 0:
            out.append(f'Tests on {machine}: pass ({len(lines)} checks)')
            continue
        tests_ok = False
        out += [f'Tests on {machine}: FAIL', *[f'- {line}' for line in lines if line.startswith('FAIL')],
                '', '```', (result.stdout + result.stderr).decode()[-3000:], '```']
    if not tests_ok:
        return False, '\n'.join(out)
    if vm is None:
        return True, '\n'.join(out + ['', f'{target}: tested, not timed'])
    if failed.is_set():
        return True, '\n'.join(out + ['', f'{target}: timing stopped, a test failed'])

    for k, report in reports.items():  # every round's times, pooled rounds included
        out_path = RAW / label / f'{target}-{k}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(report, indent=1))
    ratios = {k: kernel_ratio(r) for k, r in reports.items()}
    overall = geomean([r for r in ratios.values() if r is not None])
    cpu = next(iter(reports.values()))['cpu'] if reports else vm
    out += ['', f'Timing on {vm} ({cpu}), new/base, below 1 is faster. '
            f'Raw times per round: `{RAW / label}/{target}-<unit>.json` on the Mac.']
    for k in units:
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
        info = gh_json('pr', 'view', pr, '--json', 'state,author,isCrossRepository')
        owner = gh_json('repo', 'view', '--json', 'owner')['owner']['login']
        if info['author']['login'] != owner or info['isCrossRepository']:
            sys.exit(f'refusing pull request {pr}: it is by {info["author"]["login"]}'
                     f'{" from a fork" if info["isCrossRepository"] else ""}, and the gate runs its code. '
                     f'Only {owner}\'s own pull requests are gated.')
        if info['state'] != 'OPEN':
            sys.exit(f'pull request {pr} is {info["state"].lower()}')
        git('fetch', '--quiet', 'origin', f'pull/{pr}/head')  # the merge names this head
        head = git('rev-parse', 'FETCH_HEAD').stdout.strip()
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
    timed = [k for k in kernels if git('cat-file', '-e', f'{base}:{unit_dir(k)}/bench.cpp', check=False).returncode == 0]
    failed = threading.Event()  # a test failed somewhere: stop timing
    first = next(iter(TARGETS))
    with ThreadPoolExecutor(len(TARGETS)) as executor:
        futures = [executor.submit(run_target, t, label, new_tar, base_tar, kernels, timed, t == first, failed)
                   for t in TARGETS]
        results = [f.result() for f in futures]
    passed = all(ok for ok, _ in results)
    lines += [text + '\n' for _, text in results]
    lines.append('Gate: ' + ('pass' if passed else 'fail'))
    return passed, '\n'.join(lines), base, head, kernels


def run_from_main() -> None:
    """Replace this process with origin/main's tools/gate.py, extracted once per commit."""
    if os.environ.get('FPS_GATE_HERE'):
        return
    git('fetch', '--quiet', 'origin', 'main')
    main = git('rev-parse', 'origin/main').stdout.strip()
    tools = Path.home() / '.cache' / 'fps' / 'gate' / main
    if not tools.is_dir():
        tools.parent.mkdir(parents=True, exist_ok=True)
        tmp = tempfile.mkdtemp(dir=tools.parent)
        archive = subprocess.run(['git', 'archive', main, 'tools/'], check=True, capture_output=True, cwd=ROOT)
        subprocess.run(['tar', '-xf', '-', '-C', tmp], input=archive.stdout, check=True)
        try:
            os.rename(tmp, tools)
        except OSError:  # another gate extracted it first
            shutil.rmtree(tmp)
    log(f'running tools/gate.py of origin/main ({main[:10]})')
    os.chdir(ROOT)
    os.execve(sys.executable, [sys.executable, str(tools / 'tools' / 'gate.py'), *sys.argv[1:]],
              {**os.environ, 'FPS_GATE_HERE': '1', 'FPS_GATE_ROOT': str(ROOT)})


def comment(label: str, report: str) -> str:
    """The report as a comment: whole, kept on the Mac too; without case tables past the limit."""
    path = RAW / label / 'report.md'
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(report)
    if len(report) <= COMMENT_LIMIT:
        return report
    short = re.sub(r'<details>.*?</details>\n', '', report, flags=re.S)
    return short.replace('\nGate: ', f'\nCase tables dropped for length: `{path}` on the Mac.\n\nGate: ')


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
    run_from_main()
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
            url = subprocess.run(['gh', 'pr', 'comment', pr, '--body-file', '-'], input=comment(f'pr{pr}', report),
                                 text=True, check=True, capture_output=True).stdout.strip()
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
            if subprocess.run(['gh', 'pr', 'merge', pr, '--squash', '--match-head-commit', head]).returncode:
                log(f'merge of {head[:10]} refused: if the pull request moved on, run the gate again')
                return 1
            return 0
    log('main kept changing the affected units; not merged')
    return 1

if __name__ == '__main__':
    sys.exit(main())
