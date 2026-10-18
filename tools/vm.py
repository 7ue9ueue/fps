#!/usr/bin/env python3
"""Run a command on a VM in a copy of this checkout. Runs on the Mac; needs gcloud.

  vm.py VM [--base REV] [--name NAME] [--no-sync] [--refresh] COMMAND
      Copy the working tree (uncommitted changes included) to ~/hpc_library/NAME on VM, then run
      COMMAND there with bash. With --base, also copy REV (e.g. origin/main) to
      ~/hpc_library/NAME.base; COMMAND sees its path as $BASE. NAME defaults to the branch name.
  vm.py --refresh
      Rewrite the ssh aliases now (one `gcloud compute instances list`) and print them.

  python3 tools/vm.py lc-amd 'python3 tools/kernel.py test ntt'
  python3 tools/vm.py lc-bench --base origin/main 'python3 tools/kernel.py bench ntt --base $BASE'

Trees travel as tar streams over ssh: the VMs need no git access, and run uncommitted changes too.

ssh is plain `ssh -F ~/.cache/hpc_library/vm_ssh_config VM`, to host aliases written from one
`gcloud compute instances list` with gcloud's key and host keys: a call makes no Compute API call
(#160). A failed connection starts the VM if it is stopped and rewrites them (a VM stopped and
started has a new IP); the last retry goes through `gcloud compute ssh`, as does a VM with no alias
(stopped, or never reached by gcloud, which records its host key). FPS_VM_SSH=gcloud uses
`gcloud compute ssh` only.

The command runs detached on the VM (its own session, output to a log beside the folder), and this
script streams the log: a dropped connection reconnects and resumes the stream instead of running
the command again; a run killed on the VM ends with its exit code instead of restarting; Ctrl-C
here stops the run there. A folder is replaced only under ~/hpc_library/NAME.lock, which a running
command holds, so a second command to the same folder waits for the first instead of deleting the
tree under it (#199); use --name for commands that should run side by side.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import getpass
import io
import json
import os
import re
import secrets
import signal
import shlex
import subprocess
import sys
import tarfile
import time
from pathlib import Path

ZONES = {'lc-intel': 'europe-west2-c', 'lc-intel-test': 'europe-west2-c'}  # every other VM is in europe-west2-b
ROOT = 'hpc_library'  # relative to the VM user's home

# Host aliases for plain ssh, from one `gcloud compute instances list` (#160): `gcloud compute ssh`
# asks the Compute API for the instance on every call, which costs about 1 s and has hung for 300 s
# (PRs 134 and 156). Shared by every checkout and gate; used with `ssh -F`, so ~/.ssh/config is
# neither read nor changed. Rewritten when a connection through it fails (a VM that was stopped and
# started has a new external IP) or by `vm.py --refresh`.
SSH_CONFIG = Path(os.environ.get('FPS_VM_SSH_CONFIG') or Path.home() / '.cache' / 'hpc_library' / 'vm_ssh_config')
GCLOUD_KEY = Path.home() / '.ssh' / 'google_compute_engine'
GCLOUD_KNOWN_HOSTS = Path.home() / '.ssh' / 'google_compute_known_hosts'
LIST_TIMEOUT = 60  # seconds for the one API call; past it the old aliases stay
SSH_OPTIONS = ['-o', 'LogLevel=ERROR', '-o', 'ConnectTimeout=20',
               '-o', 'ServerAliveInterval=30', '-o', 'ServerAliveCountMax=4']


def zone(vm: str) -> str:
    return ZONES.get(vm, 'europe-west2-b')


def aliases() -> set[str]:
    """The VMs SSH_CONFIG has a Host block for."""
    try:
        text = SSH_CONFIG.read_text()
    except OSError:
        return set()
    return {line.split()[1] for line in text.splitlines() if line.startswith('Host ')}


def write_aliases() -> bool:
    """List the VMs (one Compute API call) and rewrite SSH_CONFIG: a Host block per running VM with
    a host key gcloud has recorded. The options are those `gcloud compute ssh` passes: its key, its
    known_hosts, and HostKeyAlias compute.<id>, so the host key is checked by instance, not by IP,
    and an IP now used by another machine is refused. False if the list failed (the file stays)."""
    try:
        listed = subprocess.run(['gcloud', 'compute', 'instances', 'list', '--format=json', '--quiet'],
                                capture_output=True, text=True, timeout=LIST_TIMEOUT)
        vms = json.loads(listed.stdout) if listed.returncode == 0 else None
    except (subprocess.TimeoutExpired, ValueError, OSError):
        vms = None
    if not isinstance(vms, list):
        print('could not list the VMs; ssh aliases not refreshed', file=sys.stderr)
        return False
    try:
        known = GCLOUD_KNOWN_HOSTS.read_text()
    except OSError:
        known = ''
    user = getpass.getuser()  # gcloud's default user name, as in its ssh command
    lines = [f'# written by tools/vm.py from `gcloud compute instances list` at {time.strftime("%Y-%m-%d %H:%M:%S %z")}']
    for vm in sorted(vms, key=lambda vm: vm.get('name', '')):
        name, status = vm.get('name', ''), vm.get('status', '')
        ip = next((c['natIP'] for n in vm.get('networkInterfaces', []) for c in n.get('accessConfigs', [])
                   if c.get('natIP')), None)
        alias = f'compute.{vm.get("id")}'
        if status != 'RUNNING' or not ip or f'{alias} ' not in known:
            # stopped, no external IP, or never reached by gcloud (which records the host key)
            lines.append(f'# {name}: {status.lower() or "unknown"}, no alias (vm.py uses gcloud compute ssh)')
            continue
        lines += [f'Host {name}', f'  HostName {ip}', f'  User {user}', f'  IdentityFile {GCLOUD_KEY}',
                  '  IdentitiesOnly yes', f'  HostKeyAlias {alias}', f'  UserKnownHostsFile {GCLOUD_KNOWN_HOSTS}',
                  '  StrictHostKeyChecking yes', '  CheckHostIP no', '  HashKnownHosts no']
    SSH_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    tmp = SSH_CONFIG.with_name(f'{SSH_CONFIG.name}.{os.getpid()}.{secrets.token_hex(4)}')
    tmp.write_text('\n'.join(lines) + '\n')
    os.replace(tmp, SSH_CONFIG)  # atomic: a concurrent reader sees the old file or the new one
    return True


def refresh_aliases(since: float) -> None:
    """Rewrite SSH_CONFIG unless some process started a list after `since` (when the caller saw the
    aliases fail, or saw its VM running): callers that fail together list the VMs once."""
    SSH_CONFIG.parent.mkdir(parents=True, exist_ok=True)
    with open(SSH_CONFIG.with_name(SSH_CONFIG.name + '.lock'), 'a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)  # the lock file holds when the last list started
        lock.seek(0)
        try:
            last = float(lock.read().strip() or 0)
        except ValueError:
            last = 0
        if last > since:
            return
        lock.truncate(0)
        lock.write(f'{time.time()}\n')
        lock.flush()
        write_aliases()


def alias_target(vm: str) -> str | None:
    """vm's HostName in SSH_CONFIG, or None."""
    try:
        text = SSH_CONFIG.read_text()
    except OSError:
        return None
    block = re.search(rf'^Host {re.escape(vm)}\n  HostName (\S+)', text, re.M)
    return block[1] if block else None


def ssh_argv(vm: str, command: str, via: str = 'auto') -> list[str]:
    """The command line that runs command on vm: plain ssh to its alias in SSH_CONFIG, or (via
    'gcloud', no alias, or FPS_VM_SSH=gcloud) `gcloud compute ssh`."""
    if via == 'auto':
        via = 'ssh' if os.environ.get('FPS_VM_SSH') != 'gcloud' and vm in aliases() else 'gcloud'
    if via == 'ssh':
        return ['ssh', '-F', str(SSH_CONFIG), '-T', *SSH_OPTIONS, vm, '--', command]
    return ['gcloud', 'compute', 'ssh', vm, f'--zone={zone(vm)}', '--quiet', f'--command={command}',
            '--', *SSH_OPTIONS]


up: set[str] = set()  # VMs seen running by this process; a failed connection looks again


def gcloud_instance(vm: str, *args: str, timeout: float = 600) -> subprocess.CompletedProcess:
    return subprocess.run(['gcloud', 'compute', 'instances', *args, vm, f'--zone={zone(vm)}', '--quiet'],
                          capture_output=True, text=True, timeout=timeout)


def ensure_running(vm: str) -> None:
    """Start vm if it is stopped (VMs power off after 30 minutes idle: tools/idle_shutdown.sh), and
    return once it is running and takes ssh. A running VM costs one status call. A VM another
    process is starting, or one still powering off, is waited for, never started twice: a start
    that fails (someone else's start won the race) is followed by another look, not an error.
    A VM it waited for has a new external IP, so the ssh aliases are rewritten."""
    if vm in up:
        return
    waited = False
    deadline = time.monotonic() + 600
    while time.monotonic() < deadline:
        try:
            look = gcloud_instance(vm, 'describe', '--format=value(status)', timeout=120)
        except subprocess.TimeoutExpired:
            print(f'could not read the status of {vm}: no answer from the Compute API', file=sys.stderr)
            return  # let ssh try and report
        status = look.stdout.strip()
        if look.returncode or not status:
            print(f'could not read the status of {vm}: {look.stderr.strip()}', file=sys.stderr)
            return  # let ssh try and report
        if status == 'RUNNING':
            break
        waited = True
        if status in ('TERMINATED', 'SUSPENDED'):
            action = 'start' if status == 'TERMINATED' else 'resume'
            print(f'{vm} is stopped; {action}ing it', file=sys.stderr)
            with contextlib.suppress(subprocess.TimeoutExpired):
                gcloud_instance(vm, action)
        else:  # STAGING, PROVISIONING, STOPPING, SUSPENDING, REPAIRING: someone else's start or stop
            print(f'{vm} is {status}; waiting', file=sys.stderr)
            time.sleep(10)
    else:
        print(f'{vm} did not reach RUNNING in 10 minutes', file=sys.stderr)
        return
    if waited:  # a fresh boot: a new IP, and sshd comes up a little after RUNNING
        refresh_aliases(time.time())
        for _ in range(18):
            if subprocess.run(ssh_argv(vm, 'true'), capture_output=True, timeout=120).returncode == 0:
                break
            time.sleep(10)
    up.add(vm)


def ssh(vm: str, command: str, stdin: bytes | None = None, capture: bool = False,
        timeout: float | None = None) -> subprocess.CompletedProcess:
    """Run command on vm. A connection failure (exit 255, a connection that went silent for two
    minutes, or gcloud itself crashing on a timeout to the Compute API, seen in PR 134's gate) is
    retried twice: every command the tools send (copy, test, bench) can safely run again.
    The first two attempts use plain ssh to vm's alias, which needs no Compute API call; a failure
    starts vm if it is stopped (ensure_running) and rewrites the aliases (a VM stopped and started
    has a new IP). The last attempt, and any without an alias, go through `gcloud compute ssh`,
    after ensure_running. `timeout` (seconds, per attempt) raises subprocess.TimeoutExpired."""
    plain = os.environ.get('FPS_VM_SSH') != 'gcloud'
    for attempt in range(3):
        via = 'ssh' if plain and attempt < 2 else 'gcloud'
        if via == 'gcloud' or vm not in aliases():
            ensure_running(vm)
            if via == 'ssh' and vm not in aliases():
                refresh_aliases(time.time())  # first use, or vm was stopped or new at the last list
                if vm not in aliases():
                    via = 'gcloud'
        started = time.time()
        # stderr always read, so a gcloud crash is seen on uncaptured calls too (ship: PR 156's gate)
        result = subprocess.run(ssh_argv(vm, command, via), input=stdin,
                                stdout=subprocess.PIPE if capture else None,
                                stderr=subprocess.PIPE, timeout=timeout)
        if not capture and result.stderr:
            sys.stderr.buffer.write(result.stderr)
            sys.stderr.flush()
        crashed = b'gcloud crashed' in (result.stderr or b'')
        if result.returncode != 255 and not crashed:
            return result
        print(f'ssh to {vm} failed (exit {result.returncode}, {via}), attempt {attempt + 1} of 3', file=sys.stderr)
        up.discard(vm)  # it may have powered off since
        if via == 'ssh':
            before = alias_target(vm)
            ensure_running(vm)  # a VM that powered off is started, and the aliases rewritten
            refresh_aliases(started)  # unless that, or another caller, did it since the failure
            if alias_target(vm) != before:
                continue  # a new address (or none: gcloud next), so no need to wait
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


def refresh() -> int:
    """vm.py --refresh: rewrite the aliases and say what they are."""
    if not write_aliases():
        return 1
    print(f'{SSH_CONFIG}:', *(f'{vm} {alias_target(vm)}' for vm in sorted(aliases())),
          *(line[2:] for line in SSH_CONFIG.read_text().splitlines()[1:] if line.startswith('# ')),
          sep='\n  ')
    return 0


def main() -> int:
    if sys.argv[1:] == ['--refresh']:
        return refresh()
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('vm')
    parser.add_argument('command')
    parser.add_argument('--base', help='a git revision to copy alongside, e.g. origin/main')
    parser.add_argument('--name', help='folder under ~/hpc_library (default: the branch name)')
    parser.add_argument('--no-sync', action='store_true', help='reuse the copies already there')
    parser.add_argument('--refresh', action='store_true',
                        help=f'rewrite the ssh aliases ({SSH_CONFIG}) first; alone, only that')
    args = parser.parse_args()
    if args.refresh:
        refresh()
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
