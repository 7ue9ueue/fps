#!/usr/bin/env python3
"""Run a command on a VM in a copy of this checkout. Runs on the Mac; needs gcloud.

  vm.py VM [--base REV] [--name NAME] [--no-sync] COMMAND
      Copy the working tree (uncommitted changes included) to ~/hpc_library/NAME on VM, then run
      COMMAND there with bash. With --base, also copy REV (e.g. origin/main) to
      ~/hpc_library/NAME.base; COMMAND sees its path as $BASE. NAME defaults to the branch name.

  python3 tools/vm.py lc-amd 'python3 tools/kernel.py test ntt'
  python3 tools/vm.py lc-bench --base origin/main 'python3 tools/kernel.py bench ntt --base $BASE'

Trees travel as tar streams over ssh: the VMs need no git access, and run uncommitted changes too.

The command runs detached on the VM (its own session, output to a log beside the folder), and this
script streams the log: a dropped connection reconnects and resumes the stream instead of running
the command again; a run killed on the VM ends with its exit code instead of restarting; Ctrl-C
here stops the run there. A folder is replaced only under ~/hpc_library/NAME.lock, which a running
command holds, so a second command to the same folder waits for the first instead of deleting the
tree under it (#199); use --name for commands that should run side by side.
"""
from __future__ import annotations

import argparse
import secrets
import signal
import io
import re
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ZONES = {'lc-intel': 'europe-west2-c', 'lc-intel-test': 'europe-west2-c'}  # every other VM is in europe-west2-b
ROOT = 'hpc_library'  # relative to the VM user's home


def zone(vm: str) -> str:
    return ZONES.get(vm, 'europe-west2-b')


def ssh_argv(vm: str, command: str) -> list[str]:
    return ['gcloud', 'compute', 'ssh', vm, f'--zone={zone(vm)}', '--quiet', f'--command={command}',
            '--', '-o', 'LogLevel=ERROR', '-o', 'ConnectTimeout=20',
            '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4']


def ssh(vm: str, command: str, stdin: bytes | None = None, capture: bool = False,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run command on vm. A connection failure (exit 255, a connection that went silent for two
    minutes, or gcloud itself crashing on a timeout to the Compute API, seen in PR 134's gate) is
    retried twice: every command the tools send (copy, test, bench) can safely run again.
    `timeout` (seconds, per attempt) raises subprocess.TimeoutExpired."""
    cmd = ssh_argv(vm, command)
    for attempt in range(3):
        # stderr always read, so a gcloud crash is seen on uncaptured calls too (ship: PR 156's gate)
        result = subprocess.run(cmd, input=stdin, stdout=subprocess.PIPE if capture else None,
                                stderr=subprocess.PIPE, timeout=timeout)
        if not capture and result.stderr:
            sys.stderr.buffer.write(result.stderr)
            sys.stderr.flush()
        crashed = b'gcloud crashed' in (result.stderr or b'')
        if result.returncode != 255 and not crashed:
            return result
        print(f'ssh to {vm} failed (exit {result.returncode}), attempt {attempt + 1} of 3', file=sys.stderr)
        time.sleep(15)
    return result


def git(*args: str) -> str:
    return subprocess.run(['git', *args], check=True, capture_output=True, text=True).stdout.strip()


def slug(text: str) -> str:
    return re.sub(r'[^A-Za-z0-9._-]+', '-', text).strip('-.') or 'tree'


def working_tree_tar() -> bytes:
    root = Path(git('rev-parse', '--show-toplevel'))
    files = git('-C', str(root), 'ls-files', '--cached', '--others', '--exclude-standard', '-z').split('\0')
    data = io.BytesIO()
    with tarfile.open(fileobj=data, mode='w:gz') as tar:
        for name in files:
            path = root / name
            if name and path.is_file():
                tar.add(path, arcname=name)
    return data.getvalue()


def rev_tar(rev: str) -> bytes:
    return subprocess.run(['git', 'archive', '--format=tar.gz', rev], check=True, capture_output=True).stdout


def locked(path: str, command: str) -> str:
    """command under path's lock (~/hpc_library/<name>.lock), saying so when it has to wait."""
    lock, quoted = f'{path}.lock', shlex.quote(command)
    # -E 75: a busy lock exits 75, told apart from the command's own failure (which is not rerun)
    return (f'( flock -n -E 75 {lock} -c {quoted}; rc=$?; [ $rc -ne 75 ] && exit $rc; '
            f'echo "waiting for the run in {path}" >&2; exec flock {lock} -c {quoted} )')


def ship(vm: str, name: str, data: bytes) -> str:
    """Replace ~/hpc_library/<name> on vm with the tar stream; return its path from home. The tree
    is unpacked beside the folder and swapped in under the folder's lock, so it never replaces a
    tree a running command (vm.py's, which holds the lock) is using (#199)."""
    path = f'{ROOT}/{slug(name)}'
    incoming = f'{path}.incoming-{secrets.token_hex(4)}'
    unpack = f'mkdir -p {incoming} && tar -xzmf - -C {incoming}'
    swap = f'rm -rf {path} && mv {incoming} {path}'
    result = ssh(vm, f'mkdir -p {ROOT} && {unpack} && {{ {locked(path, swap)}; }} || {{ rm -rf {incoming}; false; }}',
                 stdin=data, timeout=900)
    if result.returncode:
        sys.exit(f'copy to {vm}:{path} failed')
    return path


def run_streamed(vm: str, path: str, command: str) -> int:
    """Run command on vm detached (its own session, under path's lock), stream its output here, and
    return its exit code. A dropped connection resumes the stream at the byte it reached; a run
    killed on the VM returns its code (no rerun); SIGINT or SIGTERM here stops the run there."""
    run = f'{path}.run-{secrets.token_hex(4)}'
    job = (f'echo $$ > $HOME/{run}/pid; {{ {locked(path, command)}; }} > $HOME/{run}/out 2>&1; '
           f'echo $? > $HOME/{run}/exit.tmp && mv $HOME/{run}/exit.tmp $HOME/{run}/exit')
    detach = f'setsid nohup bash -c {shlex.quote(job)} </dev/null >/dev/null 2>&1 &'
    # mkdir guards the start: a retried start whose first attempt ran starts nothing
    if ssh(vm, f'if mkdir -p {ROOT} && mkdir {run}; then {detach} fi', timeout=120).returncode:
        print(f'could not start the command on {vm}', file=sys.stderr)
        return 1

    def stop(signum, frame):
        print(f'stopping the run on {vm}', file=sys.stderr)
        ssh(vm, f'sid=$(cat {run}/pid 2>/dev/null) && [ ! -e {run}/exit ] && '
                f'{{ pkill -TERM -s $sid; sleep 5; pkill -KILL -s $sid; }}; rm -rf {run}', timeout=120)
        sys.exit(130)
    signal.signal(signal.SIGINT, stop)
    signal.signal(signal.SIGTERM, stop)

    seen = 0  # bytes of the log already written here
    while True:
        # stream the log from byte seen until the job's session leader exits
        follow = (f'for i in $(seq 300); do [ -s {run}/pid ] && break; sleep 0.2; done; '
                  f'tail -c +{seen + 1} -f --pid=$(cat {run}/pid) {run}/out 2>/dev/null')
        proc = subprocess.Popen(ssh_argv(vm, follow), stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
        while chunk := proc.stdout.read1(65536):
            sys.stdout.buffer.write(chunk)
            sys.stdout.buffer.flush()
            seen += len(chunk)
        proc.wait()
        # the exit code; "running" if only the connection ended; "gone" if the run died without one
        look = f'cat {run}/exit 2>/dev/null || {{ kill -0 $(cat {run}/pid) 2>/dev/null && echo running; }} || cat {run}/exit 2>/dev/null || echo gone'
        status = ssh(vm, look, capture=True, timeout=120)
        code = status.stdout.decode().strip() if status.returncode == 0 else 'running'
        if code.isdigit() or code == 'gone':
            # what the last stream may have missed before the run ended, then clean up
            rest = ssh(vm, f'tail -c +{seen + 1} {run}/out; rm -rf {run}', capture=True, timeout=300)
            sys.stdout.buffer.write(rest.stdout)
            sys.stdout.buffer.flush()
            if code == 'gone':
                print(f'the run on {vm} stopped without an exit code (killed, or the VM restarted)', file=sys.stderr)
                return 1
            return int(code)
        print(f'lost the connection to {vm}; the run goes on, reconnecting', file=sys.stderr)
        time.sleep(15)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('vm')
    parser.add_argument('command')
    parser.add_argument('--base', help='a git revision to copy alongside, e.g. origin/main')
    parser.add_argument('--name', help='folder under ~/hpc_library (default: the branch name)')
    parser.add_argument('--no-sync', action='store_true', help='reuse the copies already there')
    args = parser.parse_args()
    name = slug(args.name or git('rev-parse', '--abbrev-ref', 'HEAD'))
    path, base = f'{ROOT}/{name}', f'{ROOT}/{name}.base'
    if not args.no_sync:
        ship(args.vm, name, working_tree_tar())
        if args.base:
            ship(args.vm, f'{name}.base', rev_tar(args.base))
    command = f'cd $HOME/{path} && BASE=$HOME/{base} bash -c {shlex.quote(args.command)}'
    return run_streamed(args.vm, path, command)


if __name__ == '__main__':
    sys.exit(main())
