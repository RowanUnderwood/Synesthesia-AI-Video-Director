"""Command-line diagnostics for the application's GPU power-limit helper."""

from __future__ import annotations

import sys

from image_animator_extender.gpu_power import (
    DEFAULT_WATTS,
    apply_limits,
    current_state,
    helper_installed,
    resolve_roles,
    restore_defaults,
    role_bounds,
)


if __name__ == "__main__":
    print(current_state())
    print(f"helper installed: {helper_installed()}")
    for role, gpu in resolve_roles().items():
        low, high, stock = role_bounds(role)
        print(
            f"  {role}: nvidia-smi index {gpu['index']} ({gpu['name']}); "
            f"{low}-{high} W, stock {stock} W"
        )
    command = sys.argv[1].lower() if len(sys.argv) > 1 else ""
    if command == "apply":
        print(apply_limits(DEFAULT_WATTS)[1])
    elif command == "restore":
        print(restore_defaults()[1])
