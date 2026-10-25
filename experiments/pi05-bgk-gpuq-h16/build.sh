#!/usr/bin/env bash
# H=16 uses exactly the shared native backend from the H=20 experiment.
set -euo pipefail
experiment_root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
exec bash "$experiment_root/../pi05-bgk-gpuq/build.sh" "$@"
