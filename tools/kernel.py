#!/usr/bin/env python3
"""Test and time backend kernels, per target (see TARGETS and BACKEND.md).

  kernel.py test UNIT... [--target T] [-j N] [--base DIR] [--sanitize [UNIT...]]
      Build every test*.cpp of the units for target T at -O2 and with ASan/UBSan, at most N at a
      time (default: every core), fewer if the machine is busy (build_slots); run each as soon as
      it is built, one run at a time, in order. With --base (another checkout that passed these
      tests, e.g. of main), skip a build whose preprocessed text is identical there: the same
      compiler, flags and text give the same binary, so the same result. A skipped one prints
      `skip` instead of `pass`. With --sanitize, only the listed units are built with ASan/UBSan
      (none, if the list is empty); the gate uses it, and tools/sanitize.py runs them all.
  kernel.py bench UNIT [--target T] [--base DIR] [--rounds N] [--filter RE] [--json FILE]
      Time the unit's bench*.cpp, each its own binary (e.g. frontend/bench_values.cpp, whose
      code changes how GCC compiles bench.cpp's). Case names must be unique across a unit's
      bench files: a duplicate within one round is an error. With --base (another checkout,
      e.g. of main), time both in the same run, alternating, and print each case's paired
      ratio new/base: below 1 is faster.

A unit is a backend kernel (folder backend/<kernel>/) or `frontend` (folder frontend/, headers in
include/fps/).
  kernel.py check
      Run every backend/*/gen_*.py --check: generated files match their generators; and
      tools/tune.py --check: tuned tables (backend/*/tuned.hpp) match their measurements.

On Linux (the VMs) it builds in the pinned GCC 15.2 image with Docker. On the Mac it builds with
the local clang++ and runs the targets that list 'mac'. Run from the checkout's root.
bench holds /tmp/bench.lock and pins one core; on a timing machine every command holds the lock,
so builds never disturb another agent's timing.
"""
from __future__ import annotations

import argparse
import contextlib
import fcntl
import hashlib
import json
import math
import os
import platform
import shlex
import socket
import statistics
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

IMAGE = 'mirror.gcr.io/library/gcc:15.2.0@sha256:3ae15afe768b06d0c0fe088d822ba5f8045c26630bdacc8d8e7713cf5d8e7289'
MAC = sys.platform == 'darwin'
CXX = ('clang++' if MAC else 'g++') + ' -std=c++23 -Wall -Wextra -I. -Iinclude'

# Targets as in BACKEND.md: compiler flags, the machines that test them ('mac' is the local Mac) and
# the one that times them (None: never timed). needs: /proc/cpuinfo flags a machine must have.
# mac_flags: flags for the Mac instead (its M4 has NEON but no SVE).
TARGETS = {
    'scalar': {'flags': '', 'needs': [], 'test': ['lc-amd', 'mac'], 'time': None, 'mac_flags': ''},
    'sse': {
        'flags': '-march=x86-64-v2',
        'needs': ['ssse3', 'sse4_1', 'sse4_2', 'popcnt'],
        'test': ['lc-amd'],
        'time': None,
    },
    'avx2': {
        'flags': '-march=x86-64-v3',
        'needs': ['avx2', 'bmi2', 'fma'],
        'test': ['lc-amd'],
        'time': 'lc-bench',
    },
    'avx512': {
        'flags': '-march=x86-64-v4 -mavx512ifma -mavx512vbmi',
        'needs': ['avx512f', 'avx512bw', 'avx512dq', 'avx512vl', 'avx512ifma', 'avx512vbmi'],
        'test': ['lc-intel-test'],
        'time': 'lc-intel',
    },
    'zen5': {
        'flags': '-march=znver5',
        'needs': ['avx512f', 'avx512ifma', 'avx512vbmi', 'avx512_vbmi2', 'avx512_vp2intersect'],
        'test': ['lc-zen5-test'],
        'time': 'lc-zen5',
    },
    'arm': {
        'flags': '-mcpu=neoverse-v2',
        'needs': ['asimd', 'sve2'],
        'test': ['lc-arm-test', 'mac'],
        'time': 'lc-arm',
        'mac_flags': '-mcpu=apple-m4',
    },
}


@contextlib.contextmanager
def bench_lock():
    with open('/tmp/bench.lock', 'w') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        yield


def gated(case: str) -> bool:
    """Cases "<op>@<variant>/<size>" time one variant and are not gated."""
    return '@' not in case


def summarize(cases: dict) -> tuple[list[dict], float | None]:
    """Rows of paired ratios new/base per case, and their geometric mean over gated cases.

    cases: name -> {'base': [ns per round], 'new': [ns per round], 'items': n}, rounds aligned.
    """
    rows = []
    for name, times in cases.items():
        pairs = [(b, n) for b, n in zip(times.get('base', []), times.get('new', [])) if b and n]
        if not pairs:
            continue
        ratios = sorted(n / b for b, n in pairs)
        q1, _, q3 = statistics.quantiles(ratios, n=4) if len(ratios) > 1 else (ratios[0],) * 3
        rows.append({'case': name, 'items': times.get('items', 0),
                     'base_ns': statistics.median(b for b, _ in pairs),
                     'new_ns': statistics.median(n for _, n in pairs),
                     'ratio': statistics.median(ratios), 'q1': q1, 'q3': q3})
    logs = [math.log(r['ratio']) for r in rows if gated(r['case'])]
    return rows, (math.exp(sum(logs) / len(logs)) if logs else None)


def table(rows: list[dict]) -> str:
    lines = ['| case | base ns | new ns | new ns/item | new/base | IQR |', '|---|---:|---:|---:|---:|---|']
    for r in rows:
        name = r['case'] if gated(r['case']) else f"{r['case']} (not gated)"
        per_item = f"{r['new_ns'] / r['items']:.3f}" if r['items'] else ''
        lines.append(f"| {name} | {r['base_ns']:.0f} | {r['new_ns']:.0f} | {per_item} | {r['ratio']:.4f} "
                     f"| {r['q1']:.4f}-{r['q3']:.4f} |")
    return '\n'.join(lines)


def sh(trees: dict[str, Path], make_script) -> subprocess.CompletedProcess:
    """Run a shell script in the first tree. make_script gets each tree's path as the script
    sees it: in Docker, tree <name> is mounted at /<name>."""
    if MAC:
        paths = {name: str(tree.resolve()) for name, tree in trees.items()}
        return subprocess.run(['sh', '-ec', make_script(paths)], cwd=next(iter(paths.values())),
                              capture_output=True, text=True)
    paths = {name: f'/{name}' for name in trees}
    cmd = ['docker', 'run', '--rm', '--network', 'none', '--user', f'{os.getuid()}:{os.getgid()}']
    for name, tree in trees.items():
        cmd += ['-v', f'{tree.resolve()}:/{name}']
    cmd += ['-w', next(iter(paths.values())), IMAGE, 'sh', '-ec', make_script(paths)]
    return subprocess.run(cmd, capture_output=True, text=True)


def cpu_flags() -> set[str]:
    for line in Path('/proc/cpuinfo').read_text().splitlines():
        if line.startswith(('flags', 'Features')):  # x86, ARM
            return set(line.split(':', 1)[1].split())
    return set()


def flags(target: str) -> str:
    return TARGETS[target].get('mac_flags', '') if MAC else TARGETS[target]['flags']


def require_cpu(target: str) -> None:
    spec = TARGETS[target]
    if MAC:
        if 'mac' not in spec['test']:
            sys.exit(f'the Mac cannot run target {target}; use {" or ".join(spec["test"])}.')
        return
    missing = [f for f in spec['needs'] if f not in cpu_flags()]
    if missing:
        machines = sorted({*spec['test'], spec['time']} - {None, 'mac'})
        sys.exit(f'this CPU cannot run target {target}: no {" ".join(missing)}. Use {" or ".join(machines)}.')


def cpu_name() -> str:
    if MAC:
        return subprocess.run(['sysctl', '-n', 'machdep.cpu.brand_string'], capture_output=True, text=True).stdout.strip()
    lscpu = subprocess.run(['lscpu'], capture_output=True, text=True).stdout
    for line in lscpu.splitlines():
        if line.startswith('Model name:'):
            return line.split(':', 1)[1].strip()
    return platform.processor()


def unit_dir(unit: str) -> str:
    return 'frontend' if unit == 'frontend' else f'backend/{unit}'


def compile_(tree: Path, unit: str, source: str, target: str, sanitize: bool = False) -> tuple[Path, str | None]:
    """Compile <unit folder>/<source> in tree, from relative paths, so the same sources give
    byte-identical binaries in any tree. Returns the binary and, on a failure, the output."""
    out = f'build/{target}/{unit}-{Path(source).stem}{"-san" if sanitize else ""}'
    result = sh({'w': tree}, lambda _: f'mkdir -p build/{target}\n'
                f'{CXX} {options(sanitize)} {flags(target)} {unit_dir(unit)}/{source} -o {out}')
    return tree / out, (result.stdout + result.stderr if result.returncode else None)


def options(sanitize: bool) -> str:
    # -g1: line numbers for sanitizer reports, without GCC's variable tracking, which doubled the
    # slowest builds (factorials test: 190 s with -g, 96 s with -g1).
    return '-O1 -g1 -fsanitize=address,undefined -fno-sanitize-recover=all' if sanitize else '-O2'


def fingerprints(tree: Path, jobs: list[tuple[str, str, bool]], target: str) -> dict:
    """Hash of each job's preprocessed text in tree (-P: no line markers, so moving code does not
    change it); None where the file is missing or does not preprocess."""
    hasher = 'shasum -a 256' if MAC else 'sha256sum'
    steps = []
    for i, (unit, source, sanitize) in enumerate(jobs):
        flag = f'{options(sanitize)} {flags(target)}'
        preprocess = f'{CXX} {flag} -E -P {unit_dir(unit)}/{source} -o build/pp 2>/dev/null'
        steps.append(f'{preprocess} && echo "{i} $({hasher} < build/pp | cut -c1-64)" || echo "{i} -"')
    out = sh({'w': tree}, lambda _: 'mkdir -p build\n' + '\n'.join(steps)).stdout
    hashes = dict(line.split() for line in out.splitlines())
    return {job: None if hashes[str(i)] == '-' else hashes[str(i)] for i, job in enumerate(jobs)}


def build(tree: Path, unit: str, source: str, target: str, sanitize: bool = False) -> Path:
    binary, error = compile_(tree, unit, source, target, sanitize)
    if error is not None:
        sys.exit(f'build failed in {tree}:\n{error}')
    return binary


def build_slots() -> int:
    """Builds this machine has room for now: its idle cores, and 2 GiB of available memory each
    (a test build peaked at 0.94 GiB: factorials, ASan/UBSan, GCC 13). The VMs are shared."""
    slots = (os.cpu_count() or 1) - int(os.getloadavg()[0])
    if not MAC:
        meminfo = Path('/proc/meminfo').read_text()
        available_kib = int(next(line for line in meminfo.splitlines() if line.startswith('MemAvailable')).split()[1])
        slots = min(slots, available_kib >> 21)
    return max(1, slots)


def test(args) -> int:
    require_cpu(args.target)
    tree = Path('.')
    jobs = []
    for unit in args.kernel:
        sources = sorted(p.name for p in Path(unit_dir(unit)).glob('test*.cpp'))
        if not sources:
            print(f'FAIL {unit}: no test*.cpp in {unit_dir(unit)}')
            return 1
        modes = (False, True) if args.sanitize is None or unit in args.sanitize else (False,)
        jobs += [(unit, source, sanitize) for source in sources for sanitize in modes]
    same = set()
    if args.base:
        with ThreadPoolExecutor(2) as pool:
            new, base = [pool.submit(fingerprints, t, jobs, args.target) for t in (tree, Path(args.base))]
        same = {job for job in jobs if new.result()[job] is not None and new.result()[job] == base.result()[job]}
    # Runs stay one at a time, in order: some tests time their kernel (tune.hpp).
    with ThreadPoolExecutor(max(1, min(args.jobs, build_slots()))) as pool:
        builds = {job: pool.submit(compile_, tree, job[0], job[1], args.target, job[2])
                  for job in jobs if job not in same}
        for job in jobs:
            unit, source, sanitize = job
            label = f'{unit} {args.target} {source} {"ASan/UBSan" if sanitize else "-O2"}'
            if job in same:
                print(f'skip {label} (preprocessed text identical to base)')
                continue
            binary, error = builds[job].result()
            if error is None:
                result = sh({'w': tree}, lambda _: './' + shlex.quote(str(binary.relative_to(tree))))
                sys.stdout.write(result.stdout + result.stderr)
            if error is not None or result.returncode:
                if error is not None:
                    print(error)
                print(f'FAIL {label}{" (build)" if error is not None else ""}')
                pool.shutdown(wait=False, cancel_futures=True)
                return 1
            print(f'pass {label}')
    return 0


def bench(args) -> int:
    require_cpu(args.target)
    trees = {'new': Path('.')}
    if args.base:
        trees['base'] = Path(args.base)
    sources = sorted(p.name for p in Path(unit_dir(args.kernel)).glob('bench*.cpp'))
    with ThreadPoolExecutor() as pool:  # base and new at once
        builds = {name: [pool.submit(build, tree, args.kernel, source, args.target) for source in sources
                         if (tree / unit_dir(args.kernel) / source).exists()]
                  for name, tree in trees.items()}
    binaries = {name: [b.result() for b in bs] for name, bs in builds.items()}
    report = {'kernel': args.kernel, 'target': args.target, 'flags': flags(args.target),
              'cpu': cpu_name(), 'compiler': 'clang++' if MAC else IMAGE, 'rounds': args.rounds,
              'identical': False, 'cases': {}}
    digests = {tuple(hashlib.sha256(b.read_bytes()).hexdigest() for b in bs) for bs in binaries.values()}
    if args.base and len(digests) == 1:
        report['identical'] = True
        print('base and new bench binaries are byte-identical: nothing to time.')
    else:
        pin = '' if MAC else f'taskset -c {os.cpu_count() - 1} '  # core 0 takes most interrupts
        names = list(trees)

        def script(paths: dict[str, str]) -> str:
            lines = []
            for r in range(args.rounds):
                for name in (names if r % 2 == 0 else names[::-1]):
                    for b in binaries[name]:
                        binary = f'{paths[name]}/{b.relative_to(trees[name])}'
                        run = f'{pin}{binary} {shlex.quote(args.filter)}'
                        lines.append(f'echo "## {name} {r}"; {run}')
            return '\n'.join(lines)

        result = sh(trees, script)
        if result.returncode:
            sys.exit(f'bench failed:\n{result.stdout}{result.stderr}')
        current, round_, seen = None, None, set()
        for line in result.stdout.splitlines():
            if line.startswith('## '):
                current, round_ = line[3:].split()
            elif '\t' in line:
                case, ns, *rest = line.split('\t')  # older benches print no items
                if (current, round_, case) in seen:
                    sys.exit(f'bench case {case} appears twice in one round of {current}: '
                             'case names must be unique across the bench*.cpp files')
                seen.add((current, round_, case))
                entry = report['cases'].setdefault(case, {'items': 0})
                entry.setdefault(current, []).append(float(ns))
                if current == 'new' and rest:
                    entry['items'] = int(rest[0])
    print(f'{report["cpu"]}, target {args.target} ({report["flags"] or "no flags"}), {args.rounds} rounds')
    if args.base:
        rows, geomean = summarize(report['cases'])
        print(table(rows))
        if geomean is not None:
            print(f'\nGeomean over gated cases: {geomean:.4f}')
    else:
        for case, entry in report['cases'].items():
            ns = statistics.median(entry['new'])
            per_item = f'{ns / entry["items"]:12.3f} ns/item' if entry['items'] else ''
            print(f'{case:40} {ns:12.0f} ns {per_item}')
    if args.json:
        Path(args.json).write_text(json.dumps(report, indent=1))
    return 0


def check(_) -> int:
    failed = 0
    for generator in [*sorted(Path('backend').glob('*/gen_*.py')), Path('tools/tune.py')]:
        result = subprocess.run([sys.executable, str(generator), '--check'])
        print(f'{"pass" if result.returncode == 0 else "FAIL"} {generator} --check')
        failed |= result.returncode != 0
    return int(failed)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest='command', required=True)
    default = 'scalar' if MAC else 'avx2'
    p = sub.add_parser('test')
    p.add_argument('kernel', nargs='+')
    p.add_argument('--target', choices=TARGETS, default=default)
    p.add_argument('-j', '--jobs', type=int, default=os.cpu_count(), help='builds at a time, at most')
    p.add_argument('--base', help='root of a checkout that passed: skip a test whose preprocessed text is the same')
    p.add_argument('--sanitize', nargs='*', metavar='UNIT', help='only these units with ASan/UBSan (default: all)')
    p.set_defaults(run=test)
    p = sub.add_parser('bench')
    p.add_argument('kernel')
    p.add_argument('--target', choices=TARGETS, default=default)
    p.add_argument('--base', help='root of another checkout to compare against')
    p.add_argument('--rounds', type=int, default=11)
    p.add_argument('--filter', default='', help='regex the case names must contain')
    p.add_argument('--json', help='write the raw times here')
    p.set_defaults(run=bench)
    p = sub.add_parser('check')
    p.set_defaults(run=check)
    args = parser.parse_args()
    units = [] if args.command == 'check' else args.kernel if args.command == 'test' else [args.kernel]
    for unit in units:
        if not Path(unit_dir(unit)).is_dir():
            sys.exit(f'no unit {unit_dir(unit)}')
    timing_host = socket.gethostname().split('.')[0] in {t['time'] for t in TARGETS.values()}
    with bench_lock() if timing_host or args.command == 'bench' else contextlib.nullcontext():
        return args.run(args)


if __name__ == '__main__':
    sys.exit(main())
