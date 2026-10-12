#!/usr/bin/env python3
"""Merge gate. Runs on the Mac; needs git, gh and gcloud.

  gate.py PR [--merge]          a pull request, merged into origin/main
  gate.py REV [--base REV]      a local revision, merged into --base (default origin/main)

For every target in tools/kernel.py, all at once, on the target's machines:
  - test: kernel.py test of every affected unit on each test machine, at once ('mac' is this Mac,
    with clang); kernel.py check once, on the first target's first test machine. A test whose
    preprocessed text for the target is identical to the base's is skipped (kernel.py --base):
    the base passed it already, and the same text, compiler and flags give the same binary;
  - time, if the target has a timing machine: kernel.py bench of the gated cases against the base,
    21 rounds; on a slowdown, a second run of 21, pooled with the first. If the two runs disagree
    (a case's new/base moved by more than SHIFT and out of both runs' IQRs: a state of the run,
    not of the code, as in PR 99's zen5 ntt), the report says so, a third run is timed, and the
    verdict pools the pair of runs that disagrees least. `<op>@<variant>` cases are not timed.
    Timing runs alongside the tests unless the timing machine also tests the target (its bench
    lock would queue the two anyway); it stops at the next unit once any test has failed.
Units are the backend kernels and the frontend (see tools/kernel.py). Affected units: those
changed, those that #include them ("backend/<kernel>/..."), or all if a shared file changed.
Changes to .md files (notes) affect no unit: no build reads them.
Passes if every test passes and, per target, the geomean over units of new/base (each unit's
geomean over its gated cases) is at most TOLERANCE (1.005). Byte-identical bench binaries count
as 1. The tolerance absorbs code-layout and same-process effects on unchanged code (0.1-1% seen).

For a pull request it posts the report as a comment and sets the commit status `gate` on its head,
which `main` requires; with --merge it squash-merges that head on a pass (GitHub refuses if the
head moved). The full report is kept with the raw times; the comment drops the case tables if it
would pass GitHub's limit. The last line of the report is "Gate: pass" or "Gate: fail".

One gate runs at a time on this Mac: a gate waits for the lock, then holds it from its first look
at main to the merge, and gates do not slow each other down on the VMs (PR 83: 44 minutes beside PR
82's gate; PR 77, alone: at most 21). A pull request that affects no unit (notes, tools/gate.py,
...) or conflicts with main takes no lock: nothing runs for it, so it is settled and merged at
once, even while another gate runs (PR 115 waited 28 min for a gate that took 0.0). Main changes
only through the gate, so it cannot move under a running gate, apart from such merges, which touch
no unit. If main moved anyway, the gate runs again only when the new commits on main touch a
unit this pull request affects (with the same include closure) or a shared file; otherwise the
result still holds and it merges.

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
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kernel import TARGETS, gated, summarize, table, unit_dir
from vm import rev_tar, ship, ssh

ROUNDS = 21
RAW = Path.home() / 'Documents' / 'hpc_library-explore' / 'gate'  # raw times, kept on the Mac
TOLERANCE = 1.005  # new/base per target; above it, ROUNDS more rounds are pooled and judged again
SHIFT = 1.05  # a case's new/base moving by more than this between two runs (and out of both IQRs): they disagree
SHARED = re.compile(r'^(backend/[^/]+|tools/kernel\.py)$')  # files outside a kernel folder
INCLUDE = re.compile(r'#include\s*"(backend/[A-Za-z0-9_]+/)')  # how one unit uses another
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
    changed = [path for path in git('diff', '--name-only', base, tree).stdout.split() if not path.endswith('.md')]
    if any(SHARED.match(path) for path in changed):
        return units
    hit = {u for u in units if any(path.startswith(tuple(unit_paths(u))) for path in changed)}
    # Notes name other units' folders too: only #include lines say what a unit builds from.
    uses = {u: set(INCLUDE.findall(git('grep', '-h', '-F', '#include', includes, '--', *unit_paths(u),
                                       check=False).stdout)) for u in units}
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


def pooled(runs: list[dict]) -> dict:
    report = {**runs[0], 'cases': {}, 'rounds': 0}
    for run in runs:
        pool(report, run)
    return report


def disagreements(a: dict, b: dict) -> list[str]:
    """Gated cases whose new/base moved between bench runs a and b by more than SHIFT, each run's
    ratio outside the other's IQR. Both sides moving together does not count: the pairing cancels
    it (PR 82, zen5 frontend: both sides 5-7% slower at 2^22 in run 2, the ratios within 1.4%)."""
    x, y = ({r['case']: r for r in summarize(run['cases'])[0] if gated(r['case'])} for run in (a, b))
    return [case for case, r in x.items() if case in y
            and max(r['ratio'], y[case]['ratio']) > SHIFT * min(r['ratio'], y[case]['ratio'])
            and not r['q1'] <= y[case]['ratio'] <= r['q3'] and not y[case]['q1'] <= r['ratio'] <= y[case]['q3']]


def judged(runs: list[dict]) -> list[int]:
    """The runs the verdict uses: all of one or two; of three, the pair that disagrees least
    (on a tie, the later pair)."""
    if len(runs) < 3:
        return list(range(len(runs)))
    pairs = [[1, 2], [0, 2], [0, 1]]
    return min(pairs, key=lambda p: len(disagreements(runs[p[0]], runs[p[1]])))


def disagreement_note(k: str, runs: list[dict], use: list[int]) -> list[str]:
    """Report lines for runs 1 and 2 of unit k disagreeing: the cases, each run's medians."""
    found = disagreements(runs[0], runs[1])
    if not found:
        return []
    rows = [{r['case']: r for r in summarize(run['cases'])[0]} for run in runs]
    cell = lambda r: f"{r['new_ns']:.0f} / {r['base_ns']:.0f} = {r['ratio']:.4f}" if r else ''
    return [f'- {k}: runs 1 and 2 disagree on {len(found)} cases (new/base moved by more than {SHIFT - 1:.0%}, '
            f'outside both IQRs); judged on runs {" and ".join(str(i + 1) for i in use)}', '',
            '<details><summary>disagreeing cases: new ns / base ns = new/base, per run</summary>', '',
            '| case | ' + ' | '.join(f'run {i + 1}' for i in range(len(runs))) + ' |', '|---|' + '---:|' * len(runs),
            *[f'| {case} | ' + ' | '.join(cell(r.get(case)) for r in rows) + ' |' for case in found],
            '', '</details>', '']


def kernel_ratio(report: dict) -> float | None:
    return 1.0 if report['identical'] else summarize(report['cases'])[1]


def minutes(start: float) -> float:
    return (time.monotonic() - start) / 60


def geomean(values: list[float]) -> float | None:
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def run_tests(machine: str, name: str, new_tar: bytes, base_tar: bytes, target: str, units: list[str],
              check: bool) -> subprocess.CompletedProcess:
    """Tests that build to the same preprocessed text as in base are skipped (kernel.py --base)."""
    command = ('python3 tools/kernel.py check && ' if check else '') + \
        f'python3 tools/kernel.py test {" ".join(units)} --target {target}'
    if machine == 'mac':
        with tempfile.TemporaryDirectory() as tree, tempfile.TemporaryDirectory() as base:
            tarfile.open(fileobj=io.BytesIO(new_tar), mode='r:gz').extractall(tree)
            tarfile.open(fileobj=io.BytesIO(base_tar), mode='r:gz').extractall(base)
            return subprocess.run(f'{command} --base {base}', shell=True, cwd=tree, capture_output=True)
    share = sum(machine in spec['test'] for spec in TARGETS.values())  # targets testing here share its cores
    command += f' -j $(( ($(nproc) + {share - 1}) / {share} ))'
    path, base_path = ship(machine, name, new_tar), ship(machine, f'{name}-base', base_tar)
    result = ssh(machine, f'cd {path} && {command} --base $HOME/{base_path}', capture=True)
    ssh(machine, f'rm -rf {path} {base_path}')
    return result


def time_units(vm: str, name: str, new_tar: bytes, base_tar: bytes, target: str, units: list[str],
               failed: threading.Event) -> dict[str, list[dict]]:
    """Bench runs per unit, one unit at a time; stops before the next unit once a test failed."""
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
            runs = reports[k] = [bench(vm, path, base_path, k, target)]
            ratio = kernel_ratio(runs[0])
            if ratio is not None and ratio > TOLERANCE and not failed.is_set():
                log(f'[{target}] {k}: {ratio:.4f}, timing {ROUNDS} more rounds')
                runs.append(bench(vm, path, base_path, k, target))
                if disagreements(*runs) and not failed.is_set():
                    log(f'[{target}] {k}: the two runs disagree, timing {ROUNDS} more rounds')
                    runs.append(bench(vm, path, base_path, k, target))
    finally:
        ssh(vm, f'rm -rf {path} {base_path}')
    return reports


def run_target(target: str, label: str, new_tar: bytes, base_tar: bytes, units: list[str], timed: list[str],
               check: bool, failed: threading.Event) -> tuple[bool, str]:
    spec = TARGETS[target]
    name = f'gate-{label}-{target}'
    out = [f'### {target}: `{spec["flags"] or "no flags"}`', '']

    def test_on(machine: str) -> tuple[subprocess.CompletedProcess, float]:
        log(f'[{target}] testing on {machine}')
        start = time.monotonic()
        result = run_tests(machine, f'{name}-test', new_tar, base_tar, target, units,
                           check and machine == spec['test'][0])
        if result.returncode:
            failed.set()
        return result, minutes(start)

    vm = spec['time']
    with ThreadPoolExecutor() as executor:
        tests = {m: executor.submit(test_on, m) for m in spec['test']}
        if vm in spec['test']:
            for future in tests.values():
                future.result()
        start = time.monotonic()
        runs = time_units(vm, name, new_tar, base_tar, target, timed, failed) if vm else {}
        timing_minutes = minutes(start)
        results = {m: future.result() for m, future in tests.items()}

    tests_ok = True
    for machine, (result, took) in results.items():
        lines = [line for line in result.stdout.decode().splitlines() if line.startswith(('pass', 'FAIL', 'skip'))]
        if result.returncode == 0:
            skipped = sum(line.startswith('skip') for line in lines)
            out.append(f'Tests on {machine}: pass ({len(lines) - skipped} checks'
                       + (f', {skipped} skipped as identical to base' if skipped else '') + f', {took:.1f} min)')
            continue
        tests_ok = False
        out += [f'Tests on {machine}: FAIL ({took:.1f} min)',
                *[f'- {line}' for line in lines if line.startswith('FAIL')],
                '', '```', (result.stdout + result.stderr).decode()[-3000:], '```']
    if not tests_ok:
        return False, '\n'.join(out)
    if vm is None:
        return True, '\n'.join(out + ['', f'{target}: tested, not timed'])
    if failed.is_set():
        return True, '\n'.join(out + ['', f'{target}: timing stopped, a test failed'])

    reports, notes = {}, {}
    for k, unit_runs in runs.items():  # raw: every run's rounds, in order, and the runs judged
        use = judged(unit_runs)
        reports[k] = pooled([unit_runs[i] for i in use])
        notes[k] = disagreement_note(k, unit_runs, use)
        out_path = RAW / label / f'{target}-{k}.json'
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps({**pooled(unit_runs), 'run_rounds': ROUNDS, 'judged': use}, indent=1))
    ratios = {k: kernel_ratio(r) for k, r in reports.items()}
    overall = geomean([r for r in ratios.values() if r is not None])
    cpu = next(iter(reports.values()))['cpu'] if reports else vm
    out += ['', f'Timing on {vm} ({cpu}), {timing_minutes:.1f} min, new/base, below 1 is faster. '
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
                    '<details><summary>cases</summary>', '', table(rows), '', '</details>', '', *notes[k]]
    passed = overall is None or overall <= TOLERANCE
    out.append(f'{target}: ' + ('no timed kernels' if overall is None else f'{overall:.4f}')
               + ('' if passed else f' SLOWER (limit {TOLERANCE})'))
    return passed, '\n'.join(out)


def gh_json(*args: str) -> dict:
    return json.loads(subprocess.run(['gh', *args], check=True, capture_output=True, text=True).stdout)


def gate(args, locked: bool) -> tuple[bool, str, str, str, list[str]] | None:
    """Returns (passed, report, the base commit it ran against, the head commit, affected units).
    Without the lock it only looks at the diff: it returns None if a unit is affected (the tests
    and timing need the lock), and the verdict if none is (nothing runs, so nothing to protect)."""
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
    if not locked:
        return None
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


def gate_lock(label: str):
    """Held by one gate at a time, from its first look at main to its merge. The file is the one
    gates before the queue held around their merge, so a gate still running that code waits too.
    It names the holder, for those waiting."""
    path = Path.home() / '.cache' / 'fps' / 'gate-merge.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    lock = open(path, 'a+')  # not 'w': that would erase the holder's name before the wait
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError:
        lock.seek(0)
        log(f'waiting for the gate lock, held by {lock.read().strip() or "another gate"}')
        fcntl.flock(lock, fcntl.LOCK_EX)
    lock.seek(0)
    lock.truncate()
    lock.write(f'{label}, pid {os.getpid()}, since {time.strftime("%H:%M:%S")}\n')
    lock.flush()
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
    start = time.monotonic()
    result = gate(args, locked=False)  # a conflict or no affected unit is settled without the lock
    waited = None  # None: no lock taken
    if result is None:
        lock = gate_lock(f'pr{pr}' if pr else args.target)  # held until this process exits
        waited = minutes(start)
    for _ in range(3):
        if result is None:
            start = time.monotonic()
            result = gate(args, locked=True)
        passed, report, base, head, units = result
        result = None
        before, _, verdict = report.rpartition('\nGate: ')
        took = f'Ran in {minutes(start):.1f} min, ' + ('no unit affected, so no gate lock.' if waited is None else
                                                      f'after {waited:.1f} min waiting for the gate lock.')
        report = f'{before}\n{took}\n\nGate: {verdict}'
        waited = 0.0
        print(report)
        if pr:
            url = subprocess.run(['gh', 'pr', 'comment', pr, '--body-file', '-'], input=comment(f'pr{pr}', report),
                                 text=True, check=True, capture_output=True).stdout.strip()
            set_status(head, passed, url)
        if not (passed and args.merge and pr):
            return 0 if passed else 1
        # main moves only through the gate, and this gate holds the lock: tip == base, unless
        # someone merged by hand, a gate from before the queue merged, or a pull request that
        # affects no unit merged without the lock (which stale() lets pass).
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
