#!/usr/bin/env bash
set -euo pipefail
source /opt/rh/gcc-toolset-14/enable
root=${1:?source checkout required}
audit=${2:?directory containing native_gio_concurrency.cpp required}
g++ -std=c++20 -O2 -I"$root/ramulator2/src" \
  "$audit/native_gio_concurrency.cpp" \
  "$root/ramulator2/src/ramulator/controller/impl/pim_controller.cpp" \
  "$root/ramulator2/src/ramulator/controller/impl/pim_lookahead.cpp" \
  "$root/ramulator2/src/ramulator/plugin/bus_checker.cpp" \
  -L"$root/ramulator2" -Wl,-rpath,"$root/ramulator2" -lramulator \
  -o "$audit/native_gio_concurrency"
"$audit/native_gio_concurrency" "$root/reference/spec/spec.json"
