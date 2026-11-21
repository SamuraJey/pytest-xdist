"""Compare complete xdist runs on a generated suite, independently of repository tests.

Exit 0 requires at least 25% lower median wall time on every requested pytest
major, and passing repository tests when --check-suite is given. Environment
creation and package installation are outside the timed region. Each pair uses
one environment, identical dependencies and alternating execution order.
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import venv


def run(command: list[str], *, cwd: Path, env: dict[str, str]) -> str:
    result = subprocess.run(
        command,
        check=False,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
    )
    if result.returncode:
        raise RuntimeError(f"{command!r} exited {result.returncode}\n{result.stdout}")
    return result.stdout


def environment(
    root: Path, baseline: Path, candidate: Path, major: int
) -> tuple[Path, dict[str, str]]:
    path = root / f"pytest-{major}"
    env = os.environ.copy()
    for key in (
        "PYTHONPATH",
        "PYTEST_ADDOPTS",
        "PYTEST_PLUGINS",
        "PYTEST_CURRENT_TEST",
    ):
        env.pop(key, None)
    env.update(
        PYTEST_DISABLE_PLUGIN_AUTOLOAD="1",
        PYTHONHASHSEED="0",
        SETUPTOOLS_SCM_PRETEND_VERSION="3.8.0",
        NO_COLOR="1",
    )
    python = path / ("Scripts/python.exe" if os.name == "nt" else "bin/python")
    uv = shutil.which("uv")
    if uv:
        run([uv, "venv", str(path)], cwd=root, env=env)
        installer = [uv, "pip", "install", "--python", str(python)]
    else:
        venv.EnvBuilder(with_pip=True).create(path)
        installer = [str(python), "-m", "pip", "install"]
    print(f"Preparing pytest {major} environment", flush=True)
    run(
        [
            *installer,
            f"pytest>={major},<{major + 1}",
            "execnet>=2.1",
            "filelock",
            "-e",
            str(baseline),
        ],
        cwd=root,
        env=env,
    )
    # Editable builds generate _version.py in each tree. Both imports are then
    # selected explicitly with PYTHONPATH, using exactly the same dependencies.
    run([*installer, "--no-deps", "-e", str(candidate)], cwd=root, env=env)
    return python, env


def source_env(env: dict[str, str], source: Path) -> dict[str, str]:
    return dict(env, PYTHONPATH=str(source / "src"))


def measure(
    python: Path, env: dict[str, str], suite: Path, workers: int, cases: int
) -> float:
    start = time.perf_counter()
    output = run(
        [
            str(python),
            "-m",
            "pytest",
            "-p",
            "xdist.plugin",
            "-q",
            "--color=no",
            "--dist=load",
            "-n",
            str(workers),
            str(suite),
        ],
        cwd=suite,
        env=env,
    )
    elapsed = time.perf_counter() - start
    if not re.search(rf"\b{cases} passed\b", output):
        raise RuntimeError(f"Expected {cases} passing tests:\n{output}")
    return elapsed


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--pytest-majors", type=int, nargs="+", default=[8, 9])
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--cases", type=int, default=20000)
    parser.add_argument("--warmups", type=int, default=1)
    parser.add_argument("--runs", type=int, default=7)
    parser.add_argument("--check-suite", action="store_true")
    args = parser.parse_args()
    if min(args.workers, args.cases, args.runs) < 1 or args.warmups < 0:
        parser.error("workers, cases and runs must be positive; warmups nonnegative")
    baseline, candidate = args.baseline.resolve(), args.candidate.resolve()
    for source in (baseline, candidate):
        if not (source / "src" / "xdist" / "plugin.py").is_file():
            parser.error(f"Not an xdist source tree: {source}")
    results = []
    success = True
    with tempfile.TemporaryDirectory(prefix="xdist-overhead-") as tmp:
        root = Path(tmp)
        suite = root / "suite"
        suite.mkdir()
        (suite / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        # Independent modules keep parsing costs modest without parametrization
        # or fixtures dominating the short test bodies we want to distribute.
        for first in range(0, args.cases, 100):
            (suite / f"test_batch_{first:06d}.py").write_text(
                "".join(
                    f"def test_{i:06d}():\n    assert {i} >= 0\n\n"
                    for i in range(first, min(first + 100, args.cases))
                ),
                encoding="utf-8",
            )
        for major in args.pytest_majors:
            python, env = environment(root, baseline, candidate, major)
            sources = {"baseline": baseline, "candidate": candidate}
            versions = {}
            for name, source in sources.items():
                versions[name] = run(
                    [
                        str(python),
                        "-c",
                        "import pytest, xdist; "
                        "print(pytest.__version__); print(xdist.__file__)",
                    ],
                    cwd=suite,
                    env=source_env(env, source),
                ).strip()
                if str(source / "src" / "xdist") not in versions[name]:
                    raise RuntimeError(f"Wrong source imported: {versions[name]}")
            print(json.dumps({"pytest_major": major, "imports": versions}), flush=True)
            samples: dict[str, list[float]] = {name: [] for name in sources}
            for index in range(args.warmups + args.runs):
                order = list(sources) if index % 2 == 0 else list(reversed(sources))
                for name in order:
                    elapsed = measure(
                        python,
                        source_env(env, sources[name]),
                        suite,
                        args.workers,
                        args.cases,
                    )
                    warmup = index < args.warmups
                    if not warmup:
                        samples[name].append(elapsed)
                    print(
                        json.dumps(
                            {
                                "pytest_major": major,
                                "source": name,
                                "warmup": warmup,
                                "seconds": elapsed,
                            }
                        ),
                        flush=True,
                    )
            medians = {
                name: statistics.median(values) for name, values in samples.items()
            }
            reduction = 1 - medians["candidate"] / medians["baseline"]
            suite_passed = None
            if args.check_suite:
                # Repo tests are correctness checks only, never timing samples.
                print(f"Checking repository tests on pytest {major}", flush=True)
                suite_env = source_env(env, candidate)
                suite_env.pop("PYTEST_DISABLE_PLUGIN_AUTOLOAD", None)
                output = run(
                    [str(python), "-m", "pytest", str(candidate / "testing")],
                    cwd=candidate,
                    env=suite_env,
                )
                print(output, flush=True)
                suite_passed = True
            result = {
                "pytest_major": major,
                "samples_seconds": samples,
                "medians_seconds": medians,
                "reduction": reduction,
                "suite_passed": suite_passed,
                "target_met": reduction >= 0.25,
            }
            results.append(result)
            success &= reduction >= 0.25
            print(json.dumps(result), flush=True)
    print(json.dumps({"results": results, "success": success}), flush=True)
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())
