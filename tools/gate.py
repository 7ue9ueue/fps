#!/usr/bin/env python3
"""Merge gate. Runs on the Mac; needs git, gh and gcloud.

  gate.py PR [--merge]          a pull request, merged into origin/main
  gate.py REV [--base REV]      a local revision, merged into --base (default origin/main)

For every target in tools/kernel.py, all at once, on the target's test machines ('mac' is this
Mac, with clang): kernel.py test of every affected unit; kernel.py check once, on the first
target's first test machine. A test whose preprocessed text for the target is identical to the
base's is skipped (kernel.py --base): the base passed it already, and the same text, compiler and
flags give the same binary. ASan/UBSan builds only for the kernels whose own folder changed (all,
if a shared file did); tools/sanitize.py runs them for every unit on every target, apart from the
gate.
Units are the backend kernels and the frontend (see tools/kernel.py). Affected units: those
changed, those that #include them ("backend/<kernel>/..."), or all if a shared file changed.
Changes to .md files (notes) affect no unit: no build reads them.

The gate does not time. A pull request's speed is the author's measurement, in its body (the
`Speed` section of tools/prompt.md and tools/prompt_build.md), which the reviewer or coordinator
checks. Nothing here runs a benchmark, so gates run side by side and nothing waits for another.
A timing machine that also tests a target (lc-arm) is safe: kernel.py takes /tmp/bench.lock for
every command on a timing machine, so a gate's tests wait for a running benchmark and no benchmark
starts while they run. Every other test machine is never timed.

Passes if every test passes. For a pull request it posts the report as a comment and sets the
commit status `gate` on its head, which `main` requires; with --merge it squash-merges that head
on a pass (GitHub refuses if the head moved). The last line of the report is "Gate: pass" or
"Gate: fail". A test or a copy that takes longer than TEST_TIMEOUT fails, killed on the machine.
On a VM the tests run detached from ssh and the gate polls them, so a dropped connection does not
restart them.

The only lock is the merge lock, held for seconds: from the last look at main to the merge. If
main moved while the tests ran, the tests run again only when the new commits touch a unit this
pull request affects (with the same include closure) or a shared file; otherwise the result
still holds and it merges.

It runs as origin/main's copy of tools/ (gate.py, kernel.py, vm.py), whatever the checkout it is
started from: a coordinator's checkout can lag main by hours, and the gate's rules are main's.
FPS_GATE_HERE=1 runs this checkout's copy instead, to try a change to the gate.

The repository is public and the gate runs a pull request's code on the VMs and this Mac, so it
refuses pull requests by any account but the repository's owner, and pull requests from forks.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import io
import json
import os
import random
import re
import secrets
import shlex
import shutil
import signal
import subprocess
import sys
import tarfile
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from kernel import TARGETS
from vm import ROOT as ROOT_ON_VM, rev_tar, ship, slug, ssh

TEST_TIMEOUT = 60 * 60  # seconds for one machine's tests, a wait for the bench lock on lc-arm included
POLL = 30  # seconds between looks at a VM's detached tests
MERGE_LOCK_WAIT = 30 * 60  # seconds to wait for the merge lock
NETWORK_TIMEOUT = 300  # seconds for one git or gh call
SHARED = re.compile(r'^(backend/[^/]+|tools/kernel\.py)$')  # files outside a kernel folder
INCLUDE = re.compile(r'#include\s*"(backend/[A-Za-z0-9_]+/)')  # how one unit uses another

# paths below are relative to the repository root; FPS_GATE_ROOT: set by run_from_main
ROOT = Path(os.environ.get('FPS_GATE_ROOT') or Path(__file__).resolve().parent.parent)


def git(*args: str, check: bool = True) -> subprocess.CompletedProcess:
    """A fetch is retried: gates run side by side in one checkout, and two fetches at once can fail
    on a ref lock (#220: PR 203's and PR 217's gates, started in the same second)."""
    for attempt in range(5 if args[0] == 'fetch' else 1):
        if attempt:
            log(f'git {args[0]} failed, retrying: {result.stderr.strip()[-200:]}')
            time.sleep(1 + 4 * random.random())
        result = subprocess.run(['git', *args], capture_output=True, text=True, cwd=ROOT, timeout=NETWORK_TIMEOUT)
        if result.returncode == 0:
            break
    if check and result.returncode:
        raise subprocess.CalledProcessError(result.returncode, result.args, result.stdout, result.stderr)
    return result


def log(message: str) -> None:
    print(message, file=sys.stderr, flush=True)


def minutes(start: float) -> float:
    return (time.monotonic() - start) / 60


def unit_paths(unit: str) -> list[str]:
    """Folders whose changes change the unit's code."""
    return ['frontend/', 'include/'] if unit == 'frontend' else [f'backend/{unit}/']


def units_in(tree: str) -> list[str]:
    """Units with a test*.cpp, as kernel.py test builds them."""
    files = git('ls-tree', '-r', '--name-only', tree, 'backend/', 'frontend/', check=False).stdout.split()
    found = (re.fullmatch(r'(?:backend/([^/]+)|(frontend))/test[^/]*\.cpp', f) for f in files)
    return sorted({m.group(1) or m.group(2) for m in found if m})


def changed_files(base: str, tree: str) -> list[str]:
    return [path for path in git('diff', '--name-only', base, tree).stdout.split() if not path.endswith('.md')]


def affected(base: str, tree: str, includes: str | None = None) -> list[str]:
    """Units the changes from base to tree affect. The units and their includes are read from
    `includes` (default: tree)."""
    includes = includes or tree
    units = units_in(includes)
    changed = changed_files(base, tree)
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


def touched(base: str, tree: str, units: list[str]) -> list[str]:
    """Units to build with ASan/UBSan: the backend kernels whose own folder changed, or every unit
    if a shared file changed. The frontend and the units that only include a changed kernel are
    built without: tools/sanitize.py runs everything, apart from the gate."""
    changed = changed_files(base, tree)
    if any(SHARED.match(path) for path in changed):
        return units
    return [u for u in units if u != 'frontend' and any(path.startswith(tuple(unit_paths(u))) for path in changed)]


def timed_out(command: str) -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(command, 124, b'', f'timed out after {TEST_TIMEOUT // 60} min'.encode())


def run_here(command: str, cwd: str) -> subprocess.CompletedProcess:
    """Run a shell command on this Mac; on a timeout, kill it with everything it started."""
    proc = subprocess.Popen(command, shell=True, cwd=cwd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
                            start_new_session=True)
    try:
        out, err = proc.communicate(timeout=TEST_TIMEOUT)
    except subprocess.TimeoutExpired:
        os.killpg(proc.pid, signal.SIGKILL)
        out, err = proc.communicate()
        return subprocess.CompletedProcess(command, 124, out, err + timed_out(command).stderr)
    return subprocess.CompletedProcess(command, proc.returncode, out, err)


def run_tests(machine: str, name: str, new_tar: bytes, base_tar: bytes | None, target: str, units: list[str],
              check: bool, sanitized: list[str] | None = None) -> subprocess.CompletedProcess:
    """Tests that build to the same preprocessed text as in base_tar are skipped (kernel.py --base);
    only the units in `sanitized` are built with ASan/UBSan (None: all). Exit code 124: timed out."""
    command = ('python3 tools/kernel.py check && ' if check else '') + \
        f'python3 tools/kernel.py test {" ".join(units)} --target {target}'
    if sanitized is not None:
        command += f' --sanitize {" ".join(sanitized)}'
    if machine == 'mac':
        with tempfile.TemporaryDirectory() as tree, tempfile.TemporaryDirectory() as base:
            tarfile.open(fileobj=io.BytesIO(new_tar), mode='r:gz').extractall(tree)
            if base_tar:
                tarfile.open(fileobj=io.BytesIO(base_tar), mode='r:gz').extractall(base)
                command += f' --base {base}'
            return run_here(command, tree)
    # the test runs on this machine share its cores: this gate's targets here, or, counted when the
    # run starts, every live gate run on it if more (gates run side by side: two gates' 6 runs on
    # lc-amd's 4 cores each took a quarter of them, and neither finished in 80 min on 2026-10-16)
    share = sum(machine in spec['test'] for spec in TARGETS.values())
    live = ('$(for d in $HOME/' + ROOT_ON_VM + '/gate-*.run-*; do [ -e "$d/exit" ] || '
            '! kill -0 "$(cat "$d/pid" 2>/dev/null)" 2>/dev/null || echo; done | wc -l)')
    command += f' -j $(( s = {live}, s = s > {share} ? s : {share}, ($(nproc) + s - 1) / s ))'
    path, base_path = f'{ROOT_ON_VM}/{slug(name)}', f'{ROOT_ON_VM}/{slug(name + "-base")}' if base_tar else ''
    run = f'{path}.run-{secrets.token_hex(4)}'  # the detached run's pid, out, err and exit; new per call
    try:
        ship(machine, name, new_tar)
        if base_tar:
            ship(machine, f'{name}-base', base_tar)
        command = f'cd {path} && {command}' + (f' --base $HOME/{base_path}' if base_path else '')
        return run_detached(machine, run, command)
    finally:
        # kill the run if it is still going (the gate stopped waiting, or was interrupted)
        stop = (f'sid=$(cat {run}/pid 2>/dev/null) && [ ! -e {run}/exit ] && '
                f'{{ pkill -TERM -s $sid; sleep 5; pkill -KILL -s $sid; }}; ')
        with contextlib.suppress(subprocess.TimeoutExpired):
            ssh(machine, stop + f'rm -rf {path} {base_path} {run}', timeout=120)


def run_detached(machine: str, run: str, command: str) -> subprocess.CompletedProcess:
    """Run a shell command on a VM, detached from ssh: one short ssh starts it in its own session,
    writing pid, out, err and exit into the folder `run`; short ssh calls every POLL seconds look
    for exit. A dropped connection costs a poll, not a rerun (on 2026-10-16 every ssh from this
    Mac to two VMs dropped at once, 50 min into PR 203's and PR 217's gates, and both restarted
    their tests). `timeout` on the machine ends the tests after TEST_TIMEOUT and releases the bench
    lock; the gate stops waiting 5 min later, and the caller kills what is left."""
    at = f'$HOME/{run}'
    job = (f'echo $$ > {at}/pid; timeout -k 30 {TEST_TIMEOUT} bash -c {shlex.quote(command)} > {at}/out 2> {at}/err; '
           f'echo $? > {at}/exit.tmp && mv {at}/exit.tmp {at}/exit')
    # mkdir is the guard: a start whose ssh dropped after it ran is retried by ssh() and starts nothing
    detach = f'setsid nohup bash -c {shlex.quote(job)} </dev/null >/dev/null 2>&1 &'
    start = ssh(machine, f'if mkdir {run}; then {detach} fi', capture=True, timeout=120)
    if start.returncode:
        return subprocess.CompletedProcess(command, 1, b'', f'could not start the tests on {machine}: '.encode()
                                           + start.stderr)
    # exit code, or "gone" if the run ended without writing one (the VM restarted, or it was killed)
    look = f'cat {run}/exit || kill -0 $(cat {run}/pid) || cat {run}/exit || echo gone'
    deadline = time.monotonic() + TEST_TIMEOUT + 300
    while time.monotonic() < deadline:
        time.sleep(POLL)
        try:
            status = ssh(machine, look, capture=True, timeout=120)
            if status.returncode:  # `look` itself exits 0: the connection failed
                log(f'no answer from {machine} (exit {status.returncode}); the tests run on, polling again in {POLL} s')
                continue
            code = status.stdout.decode().strip()
            if code == 'gone':
                return subprocess.CompletedProcess(command, 1, b'', f'the tests on {machine} stopped without '
                                                   'an exit code (killed, or the machine restarted)'.encode())
            if not code.isdigit():
                continue
            out = ssh(machine, f'cat {run}/out', capture=True, timeout=300)
            err = ssh(machine, f'cat {run}/err', capture=True, timeout=300)
            if out.returncode == 0 and err.returncode == 0:
                return subprocess.CompletedProcess(command, int(code), out.stdout, err.stdout)
        except subprocess.TimeoutExpired:
            log(f'no answer from {machine}; the tests run on, polling again in {POLL} s')
    return timed_out(command)


def run_target(target: str, label: str, new_tar: bytes, base_tar: bytes, units: list[str], sanitized: list[str],
               check: bool) -> tuple[bool, str]:
    spec = TARGETS[target]
    out = [f'### {target}: `{spec["flags"] or "no flags"}`', '']

    def test_on(machine: str) -> tuple[subprocess.CompletedProcess, float]:
        log(f'[{target}] testing on {machine}')
        start = time.monotonic()
        try:
            result = run_tests(machine, f'gate-{label}-{target}-test', new_tar, base_tar, target, units,
                               check and machine == spec['test'][0], sanitized)
        except (Exception, SystemExit) as error:  # a copy that failed or hung: this machine fails, the others finish
            result = subprocess.CompletedProcess('', 1, b'', f'{type(error).__name__}: {error}'.encode())
        return result, minutes(start)

    with ThreadPoolExecutor() as executor:
        futures = {m: executor.submit(test_on, m) for m in spec['test']}
        results = {m: future.result() for m, future in futures.items()}
    passed = True
    for machine, (result, took) in results.items():
        lines = [line for line in result.stdout.decode().splitlines() if line.startswith(('pass', 'FAIL', 'skip'))]
        if result.returncode == 0:
            skipped = sum(line.startswith('skip') for line in lines)
            out.append(f'Tests on {machine}: pass ({len(lines) - skipped} checks'
                       + (f', {skipped} skipped as identical to base' if skipped else '') + f', {took:.1f} min)')
            continue
        passed = False
        out += [f'Tests on {machine}: FAIL' + (' (timed out)' if result.returncode == 124 else '') + f' ({took:.1f} min)',
                *[f'- {line}' for line in lines if line.startswith('FAIL')],
                '', '```', (result.stdout + result.stderr).decode()[-3000:], '```']
    return passed, '\n'.join(out)


def gh_json(*args: str) -> dict:
    return json.loads(subprocess.run(['gh', *args], check=True, capture_output=True, text=True,
                                     timeout=NETWORK_TIMEOUT).stdout)


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

    units = affected(base, tree)
    lines = [f'Gate for {label} ({head[:10]}) merged into {args.base} ({base[:10]}).', '']
    if not units:
        return True, '\n'.join(lines + ['No unit affected.', '', 'Gate: pass']), base, head, []
    sanitized = touched(base, tree, units)
    lines += [f'Units: {", ".join(units)}', f'ASan/UBSan: {", ".join(sanitized) or "none"} '
              '(the others -O2 only; tools/sanitize.py runs all)', '']
    new_tar = subprocess.run(['git', 'archive', '--format=tar.gz', tree], check=True, capture_output=True,
                             cwd=ROOT).stdout
    base_tar = rev_tar(base)
    first = next(iter(TARGETS))
    with ThreadPoolExecutor(len(TARGETS)) as executor:
        futures = [executor.submit(run_target, t, label, new_tar, base_tar, units, sanitized, t == first)
                   for t in TARGETS]
        results = [f.result() for f in futures]
    passed = all(ok for ok, _ in results)
    lines += [text + '\n' for _, text in results]
    lines.append('Gate: ' + ('pass' if passed else 'fail'))
    return passed, '\n'.join(lines), base, head, units


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


def set_status(head: str, passed: bool, url: str) -> None:
    """The commit status `gate`, which branch protection on main requires."""
    repo = gh_json('repo', 'view', '--json', 'nameWithOwner')['nameWithOwner']
    subprocess.run(['gh', 'api', f'repos/{repo}/statuses/{head}', '-f', f'state={"success" if passed else "failure"}',
                    '-f', 'context=gate', '-f', f'description=Gate: {"pass" if passed else "fail"}',
                    '-f', f'target_url={url}'], check=True, capture_output=True, timeout=NETWORK_TIMEOUT)


@contextlib.contextmanager
def merge_lock(label: str):
    """Held from the last look at main to the merge, seconds. The file is the one the gates that
    held it for a whole run used, so a gate still running that code makes this wait. It names the
    holder, for those waiting, and is emptied on release; a name left in it is from a gate that
    died holding the lock, which the kernel released with it: only flock says who holds it."""
    path = Path.home() / '.cache' / 'fps' / 'gate-merge.lock'
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, 'a+') as lock:  # not 'w': that would erase the holder's name before the wait
        deadline = time.monotonic() + MERGE_LOCK_WAIT
        while True:
            try:
                fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except BlockingIOError:
                if time.monotonic() > deadline:
                    lock.seek(0)
                    sys.exit(f'merge lock held by {lock.read().strip() or "another gate"} for {MERGE_LOCK_WAIT // 60} '
                             f'min: run the gate again')
                time.sleep(2)
        lock.seek(0)
        lock.truncate()
        lock.write(f'{label}, pid {os.getpid()}, since {time.strftime("%H:%M:%S")}\n')
        lock.flush()
        try:
            yield
        finally:
            lock.seek(0)
            lock.truncate()


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
        start = time.monotonic()
        passed, report, base, head, units = gate(args)
        before, _, verdict = report.rpartition('\nGate: ')
        report = f'{before}\nRan in {minutes(start):.1f} min.\n\nGate: {verdict}'
        print(report)
        if pr:
            url = subprocess.run(['gh', 'pr', 'comment', pr, '--body-file', '-'], input=report, text=True, check=True,
                                 capture_output=True, timeout=NETWORK_TIMEOUT).stdout.strip()
            set_status(head, passed, url)
        if not (passed and args.merge and pr):
            return 0 if passed else 1
        with merge_lock(f'pr{pr}'):
            git('fetch', '--quiet', 'origin', 'main')
            tip = git('rev-parse', 'origin/main').stdout.strip()
            if tip != base:  # another pull request merged while the tests ran
                merged = git('merge-tree', '--write-tree', tip, head, check=False)
                if merged.returncode:
                    log(f'main moved to {tip[:10]} and now conflicts: rebase, then run the gate again')
                    return 1
                if stale(base, tip, merged.stdout.split()[0], units):
                    log(f'main moved to {tip[:10]} and touches {", ".join(units)}: running the gate again')
                    continue
                log(f'main moved to {tip[:10]} but touches none of {", ".join(units) or "no units"}: merging')
            if subprocess.run(['gh', 'pr', 'merge', pr, '--squash', '--match-head-commit', head],
                              timeout=NETWORK_TIMEOUT).returncode:
                log(f'merge of {head[:10]} refused: if the pull request moved on, run the gate again')
                return 1
            return 0
    log('main kept changing the affected units; not merged')
    return 1


if __name__ == '__main__':
    sys.exit(main())
