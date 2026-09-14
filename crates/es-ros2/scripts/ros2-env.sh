#!/usr/bin/env bash
# scripts/ros2-env.sh <prefix> <cmd...> — run <cmd...> with a RoboStack ROS 2 environment at
# <prefix> activated, without micromamba/pixi at test time (docs/design/ros2-boundary.md
# section 8).
#
# Needs a real bash, not POSIX `/bin/sh`: RoboStack's own activation scripts (e.g.
# `ros-kilted-ros-workspace_activate.sh`) call bash's `source` builtin internally, which
# `/bin/sh` (dash on the oracle server) does not provide (`source: not found`) — this crate's
# W1a evidence found that gap (`docs/api-notes/ros2-cdr.md` "Live ROS 2 byte capture"), and W1b
# closes it by running everything below under bash instead of patching around `source`. The
# Rust harness (`tests/rmw_zenoh_interop.rs`) also invokes this file as `bash
# scripts/ros2-env.sh ...` explicitly, so it works even if the executable bit does not survive
# a checkout.
set -eu

if [ "$#" -lt 2 ]; then
    echo "usage: ros2-env.sh <prefix> <cmd...>" >&2
    exit 2
fi

prefix="$1"
shift

CONDA_PREFIX="$prefix"
export CONDA_PREFIX
PATH="$prefix/bin:$PATH"
export PATH

activate_dir="$prefix/etc/conda/activate.d"
if [ -d "$activate_dir" ]; then
    # Conda/RoboStack activation scripts are not written for `set -u`: they reference
    # variables such as $CONDA_BUILD that are normally just empty, not unset. Relax `-u`
    # (not `-e`: a real failure in one of them should still abort) for the sourcing only, so
    # this script's own logic stays strict everywhere else.
    set +u
    for script in "$activate_dir"/*.sh; do
        [ -e "$script" ] || continue
        # shellcheck disable=SC1090
        . "$script"
    done
    set -u
fi

exec "$@"
