import pathlib

import pytest

from scripts import repro_checkout_permissions


def mode(path: pathlib.Path) -> int:
    return path.stat().st_mode & 0o777


def test_checkout_is_readable_but_controls_remain_inaccessible_to_container_uid(tmp_path):
    control = tmp_path / "control"
    checkout = control / "repo"
    script = checkout / "scripts" / "run.py"
    script.parent.mkdir(parents=True)
    script.write_text("print('ok')\n")
    executable = checkout / "tool"
    executable.write_text("#!/bin/sh\n")
    executable.chmod(0o700)
    secrets = []
    for name in ("worker-spec.json", "openpi.bundle", "source-verification.json", "launch-metadata.json"):
        path = control / name
        path.write_text("secret")
        secrets.append(path)

    repro_checkout_permissions.secure_checkout(checkout, control, secrets)

    assert mode(control) == 0o700
    assert all(mode(path) == 0o600 for path in secrets)
    assert mode(checkout) == 0o555
    assert mode(script.parent) == 0o555
    assert mode(script) == 0o444
    assert mode(executable) == 0o555
    assert mode(control) & 0o077 == 0  # UID 1000 cannot traverse unrelated /opt/pi05 controls.
    assert all(mode(path) & 0o077 == 0 for path in secrets)

    # Restore owner permissions so pytest can remove its temporary tree.
    for path in (checkout, script.parent):
        path.chmod(0o755)


def test_checkout_outside_control_root_or_with_symlink_fails_closed(tmp_path):
    control = tmp_path / "control"
    control.mkdir()
    secret = control / "worker-spec.json"
    secret.write_text("secret")
    outside = tmp_path / "outside"
    outside.mkdir()
    with pytest.raises(repro_checkout_permissions.PermissionError, match="contained"):
        repro_checkout_permissions.secure_checkout(outside, control, [secret])

    checkout = control / "repo"
    checkout.mkdir()
    (checkout / "link").symlink_to(secret)
    with pytest.raises(repro_checkout_permissions.PermissionError, match="symlink"):
        repro_checkout_permissions.secure_checkout(checkout, control, [secret])
