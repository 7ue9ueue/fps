#!/usr/bin/env python3
"""Run a command on a VM in a copy of this checkout. Runs on the Mac; needs gcloud.

  vm.py VM [--base REV] [--name NAME] [--no-sync] COMMAND
      Copy the working tree (uncommitted changes included) to ~/hpc_library/NAME on VM, then run
      COMMAND there with bash. With --base, also copy REV (e.g. origin/main) to
      ~/hpc_library/NAME.base; COMMAND sees its path as $BASE. NAME defaults to the branch name.

  python3 tools/vm.py lc-amd 'python3 tools/kernel.py test ntt'
  python3 tools/vm.py lc-bench --base origin/main 'python3 tools/kernel.py bench ntt --base $BASE'

The VMs cannot fetch the private repository, so trees travel as tar streams over ssh.
"""
from __future__ import annotations

import argparse
import io
import re
import shlex
import subprocess
import sys
import tarfile
from pathlib import Path

ZONES = {'lc-intel': 'europe-west2-c', 'lc-intel-test': 'europe-west2-c'}  # every other VM is in europe-west2-b
ROOT = 'hpc_library'  # relative to the VM user's home


def zone(vm: str) -> str:
    return ZONES.get(vm, 'europe-west2-b')


def ssh(vm: str, command: str, stdin: bytes | None = None, capture: bool = False) -> subprocess.CompletedProcess:
    cmd = ['gcloud', 'compute', 'ssh', vm, f'--zone={zone(vm)}', '--quiet', f'--command={command}',
           '--', '-o', 'LogLevel=ERROR']
    return subprocess.run(cmd, input=stdin, capture_output=capture)


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


def ship(vm: str, name: str, data: bytes) -> str:
    """Replace ~/hpc_library/<name> on vm with the tar stream; return its path from home."""
    path = f'{ROOT}/{slug(name)}'
    result = ssh(vm, f'rm -rf {path} && mkdir -p {path} && tar -xzmf - -C {path}', stdin=data)
    if result.returncode:
        sys.exit(f'copy to {vm}:{path} failed')
    return path


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
    command = f'cd {path} && BASE=$HOME/{base} bash -c {shlex.quote(args.command)}'
    return ssh(args.vm, command).returncode


if __name__ == '__main__':
    sys.exit(main())
