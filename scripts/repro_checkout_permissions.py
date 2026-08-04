#!/usr/bin/env python3
"""Make a verified source checkout readable, never writable, by worker containers."""

from __future__ import annotations

import argparse
from collections.abc import Iterable
import os
import pathlib
import stat


class PermissionError(RuntimeError):
    """Raised when the source/control permission boundary is ambiguous."""


def secure_checkout(
    checkout: pathlib.Path,
    control_root: pathlib.Path,
    control_files: Iterable[pathlib.Path],
) -> None:
    if checkout.is_symlink() or not checkout.is_dir():
        raise PermissionError(f"checkout is not a regular directory: {checkout}")
    root = control_root.resolve(strict=True)
    source = checkout.resolve(strict=True)
    try:
        source.relative_to(root)
    except ValueError as exc:
        raise PermissionError("checkout must be contained by the control root") from exc
    if source == root:
        raise PermissionError("checkout cannot be the control root")

    controls = [path.resolve(strict=True) for path in control_files]
    for path in controls:
        if path.parent != root or path.is_symlink() or not path.is_file():
            raise PermissionError(f"control file is not a direct regular child of {root}: {path}")

    directories = [source]
    files: list[tuple[pathlib.Path, bool]] = []
    for current_root, names, filenames in os.walk(source, topdown=True, followlinks=False):
        current = pathlib.Path(current_root)
        for name in names:
            path = current / name
            if path.is_symlink():
                raise PermissionError(f"source checkout contains a symlink: {path}")
            if not path.is_dir():
                raise PermissionError(f"source checkout contains a non-directory entry: {path}")
            directories.append(path)
        for name in filenames:
            path = current / name
            if path.is_symlink():
                raise PermissionError(f"source checkout contains a symlink: {path}")
            if not path.is_file():
                raise PermissionError(f"source checkout contains a non-regular file: {path}")
            files.append((path, bool(path.stat().st_mode & stat.S_IXUSR)))

    for path, executable in files:
        path.chmod(0o555 if executable else 0o444)
    for directory in sorted(directories, key=lambda path: len(path.parts), reverse=True):
        directory.chmod(0o555)
    for path in controls:
        path.chmod(0o600)
    root.chmod(0o700)


def parse_args(argv: Iterable[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkout", type=pathlib.Path, required=True)
    parser.add_argument("--control-root", type=pathlib.Path, required=True)
    parser.add_argument("--control-file", type=pathlib.Path, action="append", required=True)
    return parser.parse_args(argv)


def main(argv: Iterable[str] | None = None) -> int:
    args = parse_args(argv)
    try:
        secure_checkout(args.checkout, args.control_root, args.control_file)
        return 0
    except (PermissionError, OSError) as exc:
        print(f"CHECKOUT PERMISSION HANDOFF REJECTED: {exc}", file=os.sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
