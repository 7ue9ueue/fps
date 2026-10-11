#!/usr/bin/env python3
"""Merge gate. Runs on the Mac; needs git, gh and gcloud.

  gate.py PR [--merge]

Tests run in GitHub Actions (.github/workflows/test.yml) on the pull request merged into main:
the generators, and every unit on every target. This script adds the timing, on the VMs: for
every target with a timing machine (tools/kernel.py), kernel.py bench of each affected unit
against main, 21 rounds; on a slowdown, 21 more, pooled.
Units are the backend kernels and the frontend (see tools/kernel.py). Affected units: those
changed, those that include them, or all if a shared file changed.
The timing passes if, per target, the geomean over units of new/base (each unit's geomean over
its gated cases) is at most 1. Byte-identical bench binaries count as 1.

It posts the timing and the checks' current results as a comment, and sets the commit status
`gate` (the timing) on the head. Then it waits for the checks: the gate passes if the timing and
every check pass. With --merge it first merges main into the branch if it is behind, so the
checks test what lands; on a pass it squash-merges (auto-merge) and waits until merged. If main
moves first, it runs again. The last line is "Gate: pass" or "Gate: fail".

The repository is public and the gate runs a pull request's code on the VMs, so it refuses pull
requests by any account but the repository's owner, and pull requests from forks.
"""
from __future__ import annotations

import argparse
import json
import math
import re
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor

from kernel import TARGETS, summarize, table, unit_dir
from vm import rev_tar, ship, ssh

ROUNDS = 21
SHARED = re.compile(r'^(backend/[^/]+|tools/kernel\.py)$')  # files outside a kernel folder
POLL = 30  # seconds between looks at the checks
WAIT = 90 * 60  # longest wait for the checks to finish, in seconds
MAIN_MOVED = 'main moved'


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    return subprocess.run(['git', *args], check=check, capture_output=True, text=True)


def gh(*args: str, check: bool = True, input: str | None = None) -> subprocess.CompletedProcess:
    return subprocess.run(['gh', *args], check=check, capture_output=True, text=True, input=input)


def gh_json(*args: str) -> dict:
    return json.loads(gh(*args).stdout)


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


def affected(base: str, tree: str) -> list[str]:
    units = units_in(tree)
    changed = git('diff', '--name-only', base, tree).stdout.split()
    if any(SHARED.match(path) for path in changed):
        return units
    hit = {u for u in units if any(path.startswith(tuple(unit_paths(u))) for path in changed)}
    uses = {u: set(git('grep', '-h', '-o', '-E', r'backend/[A-Za-z0-9_]+/', tree, '--',
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
        raise RuntimeError(f'bench {kernel} on {vm} failed:\n{result.stderr.decode()[-3000:]}')
    return json.loads(result.stdout)


def kernel_ratio(report: dict) -> float | None:
    return 1.0 if report['identical'] else summarize(report['cases'])[1]


def geomean(values: list[float]) -> float | None:
    return math.exp(sum(map(math.log, values)) / len(values)) if values else None


def time_target(target: str, name: str, new_tar: bytes, base_tar: bytes, base: str,
                kernels: list[str]) -> tuple[bool, str]:
    spec = TARGETS[target]
    out = [f'### {target}: `{spec["flags"]}`', '']
    timed = [k for k in kernels if git('cat-file', '-e', f'{base}:{unit_dir(k)}/bench.cpp', check=False).returncode == 0]
    path = ship(spec['time'], name, new_tar)
    base_path = ship(spec['time'], f'{name}.base', base_tar)
    reports = {}
    try:
        for k in timed:
            log(f'[{target}] timing {k} on {spec["time"]}')
            reports[k] = bench(spec['time'], path, base_path, k, target)
            ratio = kernel_ratio(reports[k])
            if ratio is not None and ratio > 1:
                log(f'[{target}] {k}: {ratio:.4f}, timing {ROUNDS} more rounds')
                again = bench(spec['time'], path, base_path, k, target)
                for case, times in again['cases'].items():
                    for side, values in times.items():
                        reports[k]['cases'].setdefault(case, {}).setdefault(side, []).extend(values)
                reports[k]['rounds'] += again['rounds']
    except RuntimeError as error:
        return False, '\n'.join(out + ['```', str(error), '```', f'{target}: FAIL'])
    finally:
        ssh(spec['time'], f'rm -rf {path} {base_path}')

    ratios = {k: kernel_ratio(r) for k, r in reports.items()}
    overall = geomean([r for r in ratios.values() if r is not None])
    cpu = next(iter(reports.values()))['cpu'] if reports else spec['time']
    out += [f'Timing on {spec["time"]} ({cpu}), new/base, below 1 is faster:']
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
    passed = overall is None or overall <= 1
    out.append(f'{target}: ' + ('no timed kernels' if overall is None else f'{overall:.4f}')
               + ('' if passed else ' SLOWER'))
    return passed, '\n'.join(out)


def timing(pr: str, head: str) -> tuple[bool, str, str]:
    """Times the head merged into origin/main. Returns (passed, report, the base commit)."""
    git('fetch', '--quiet', 'origin', 'main')
    base = git('rev-parse', 'origin/main').stdout.strip()
    lines = [f'Gate for #{pr} ({head[:10]}) merged into main ({base[:10]}).', '']
    merged = git('merge-tree', '--write-tree', base, head, check=False)
    if merged.returncode:
        return False, '\n'.join(lines + ['Conflicts with main: rebase first.']), base
    tree = merged.stdout.split()[0]
    kernels = affected(base, tree)
    if not kernels:
        return True, '\n'.join(lines + ['No unit affected: nothing to time.']), base
    lines += [f'Units: {", ".join(kernels)}', '']
    new_tar = subprocess.run(['git', 'archive', '--format=tar.gz', tree], check=True, capture_output=True).stdout
    base_tar = rev_tar(base)
    timed = [t for t, spec in TARGETS.items() if spec['time']]
    with ThreadPoolExecutor(len(timed)) as pool:
        futures = [pool.submit(time_target, t, f'gate-pr{pr}-{t}', new_tar, base_tar, base, kernels)
                   for t in timed]
        results = [f.result() for f in futures]
    lines += [text + '\n' for _, text in results]
    return all(ok for ok, _ in results), '\n'.join(lines), base


def fetch_head(pr: str, stale: str | None = None) -> str:
    """The pull request's head, fetched. Refuses pull requests the gate must not run. GitHub can
    report the old head for a few seconds after a push; with stale, waits until the head moves on."""
    for _ in range(24):
        info = gh_json('pr', 'view', pr, '--json', 'headRefOid,state,author,isCrossRepository')
        owner = gh_json('repo', 'view', '--json', 'owner')['owner']['login']
        if info['author']['login'] != owner or info['isCrossRepository']:
            sys.exit(f'refusing pull request {pr}: it is by {info["author"]["login"]}'
                     f'{" from a fork" if info["isCrossRepository"] else ""}, and the gate runs its code. '
                     f'Only {owner}\'s own pull requests are gated.')
        if info['state'] != 'OPEN':
            sys.exit(f'pull request {pr} is {info["state"].lower()}')
        git('fetch', '--quiet', 'origin', f'pull/{pr}/head')
        head = git('rev-parse', 'FETCH_HEAD').stdout.strip()
        if head == info['headRefOid'] and head != stale:
            return head
        time.sleep(5)
    sys.exit(f'pull request {pr}: GitHub reports head {info["headRefOid"][:10]}, fetched {head[:10]}')


def contains(head: str, rev: str) -> bool:
    return git('merge-base', '--is-ancestor', rev, head, check=False).returncode == 0


def update_branch(pr: str, head: str) -> str:
    """Merges main into the pull request's branch if it is behind, so that its checks test what
    merging lands. Returns the new head."""
    git('fetch', '--quiet', 'origin', 'main')
    if contains(head, 'origin/main') or git('merge-tree', '--write-tree', 'origin/main', head, check=False).returncode:
        return head  # up to date, or conflicts, which the timing reports
    log(f'merging main into the branch of #{pr}; pull before you push to it again')
    gh('pr', 'update-branch', pr)
    return fetch_head(pr, stale=head)


def checks(pr: str) -> list[dict]:
    """The checks on the pull request's head but `gate`: name, bucket (pass, fail, pending,
    skipping or cancel), link."""
    result = gh('pr', 'checks', pr, '--json', 'name,bucket,link', check=False)  # exits nonzero unless all pass
    return [c for c in json.loads(result.stdout or '[]') if c['name'] != 'gate']


def checks_report(found: list[dict], head: str) -> str:
    if not found:
        return f'Checks on {head[:10]}: none yet.'
    return '\n'.join([f'Checks on {head[:10]}:'] + [f'- {c["name"]}: {c["bucket"]}'
                     + ('' if c['bucket'] == 'pass' else f' ({c["link"]})') for c in found])


def wait(pr: str, head: str, merge_base: str | None) -> tuple[bool, str]:
    """Waits until every check on head has finished and, with merge_base (the main the timing ran
    against), until the pull request is merged. Returns (passed, what happened), or MAIN_MOVED."""
    start, merging = time.time(), False
    while time.time() - start < WAIT:
        info = gh_json('pr', 'view', pr, '--json', 'state,headRefOid')
        if info['state'] == 'MERGED':
            return True, f'Merged #{pr}.'
        if info['state'] != 'OPEN' or info['headRefOid'] != head:
            return False, f'#{pr} changed while the gate waited ({info["state"].lower()}, head '\
                          f'{info["headRefOid"][:10]}): run the gate again.'
        found = checks(pr)
        failed = [c for c in found if c['bucket'] not in ('pass', 'pending')]
        if failed:
            return False, checks_report(failed, head)
        if not found and time.time() - start > 300:
            return False, f'No checks on {head[:10]} after 5 minutes: push to the branch to run them.'
        if found and all(c['bucket'] == 'pass' for c in found):
            if merge_base is None:
                return True, checks_report(found, head)
            git('fetch', '--quiet', 'origin', 'main')
            main = git('rev-parse', 'origin/main').stdout.strip()
            if main != merge_base or not contains(head, main):  # the checks or the timing saw another main
                if merging:
                    gh('pr', 'merge', pr, '--disable-auto', check=False)
                return False, MAIN_MOVED
            if not merging:
                # Not before every check passed: without branch protection, gh merges at once.
                result = gh('pr', 'merge', pr, '--auto', '--squash', '--match-head-commit', head, check=False)
                if result.returncode:  # auto-merge unavailable: every check passed, so merge directly
                    result = gh('pr', 'merge', pr, '--squash', '--match-head-commit', head, check=False)
                if result.returncode:
                    return False, f'gh pr merge failed:\n{result.stderr}'
                merging = True
        time.sleep(POLL)
    return False, f'Gave up after {WAIT // 60} minutes: checks still pending or not merged.'


def set_status(head: str, passed: bool, url: str) -> None:
    """The commit status `gate`: the timing, which branch protection on main requires."""
    gh('api', f'repos/{{owner}}/{{repo}}/statuses/{head}', '-f', f'state={"success" if passed else "failure"}',
       '-f', 'context=gate', '-f', f'description=Timing: {"pass" if passed else "fail"}', '-f', f'target_url={url}')


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('pr', help='pull request number')
    parser.add_argument('--merge', action='store_true', help='squash-merge the pull request on a pass')
    args = parser.parse_args()
    for _ in range(3):
        head = fetch_head(args.pr)
        if args.merge:
            head = update_branch(args.pr, head)
        passed, report, base = timing(args.pr, head)
        report += '\n\n' + checks_report(checks(args.pr), head) + f'\n\nTiming: {"pass" if passed else "fail"}'
        url = gh('pr', 'comment', args.pr, '--body-file', '-', input=report).stdout.strip()
        set_status(head, passed, url)
        print(report, flush=True)
        if not passed:
            print('\nGate: fail')
            return 1
        log('waiting for the checks' + (' and the merge' if args.merge else ''))
        passed, outcome = wait(args.pr, head, base if args.merge else None)
        if outcome != MAIN_MOVED:
            print(f'\n{outcome}\n\nGate: {"pass" if passed else "fail"}')
            return 0 if passed else 1
        log('main moved during the gate: running it again')
    log('main kept moving; not merged')
    print('\nGate: fail')
    return 1


if __name__ == '__main__':
    sys.exit(main())
