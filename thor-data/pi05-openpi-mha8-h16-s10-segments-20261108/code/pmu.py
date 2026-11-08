"""Read the existing Thor uncore PMUs through Linux perf_event_open.

No kernel module, sysctl, power setting or permission is changed. These counters
are system-wide. Event discovery does not establish a GPU-only DRAM contract.
"""
from __future__ import annotations

import argparse
import ctypes
import fcntl
import json
import os
from pathlib import Path
import platform
import re
import struct
import time


ROOT = Path('/sys/bus/event_source/devices')
PERF_FORMAT = 1 | 2 | 8  # enabled time, running time, group values
IOC_ENABLE, IOC_DISABLE, IOC_RESET = 0x2400, 0x2401, 0x2403


def insert_field(registers, field, value):
    register, ranges = field.strip().split(':')
    if register not in ('config', 'config1', 'config2'):
        raise ValueError(f'Unsupported perf register: {register}')
    bits = []
    for token in ranges.split(','):
        ends = [int(n) for n in token.split('-')]
        bits.extend(range(ends[0], ends[-1] + 1))
    if value < 0 or value >= 1 << len(bits) or any(bit >= 64 for bit in bits):
        raise ValueError(f'Event value does not fit {field}: {value}')
    for index, bit in enumerate(bits):
        registers[register] |= ((value >> index) & 1) << bit


def event_spec(pmu, event, filters=None, root=ROOT):
    for name in (pmu, event, *(filters or {})):
        if not re.fullmatch(r'[A-Za-z0-9_]+', name):
            raise ValueError(f'Invalid sysfs name: {name}')
    directory = root / pmu
    values = {}
    for entry in (directory / 'events' / event).read_text().strip().split(','):
        key, value = entry.split('=')
        values[key] = int(value, 0)
    values.update(filters or {})
    registers = dict(config=0, config1=0, config2=0)
    for key, value in values.items():
        insert_field(registers, (directory / 'format' / key).read_text(), value)
    cpu = int((directory / 'cpumask').read_text().strip().split(',')[0].split('-')[0])
    return {'pmu': pmu, 'event': event, 'filters': filters or {},
            'type': int((directory / 'type').read_text()), 'cpu': cpu, **registers}


def attr_bytes(spec, leader=True):
    data = bytearray(128)  # PERF_ATTR_SIZE_VER7; fields beyond config2 remain zero
    struct.pack_into('<IIQQQQQ', data, 0, spec['type'], len(data), spec['config'],
                     0, 0, PERF_FORMAT, 5 if leader else 0)
    struct.pack_into('<QQ', data, 56, spec['config1'], spec['config2'])
    return data


def decode_group(raw, names):
    if len(raw) != 8 * (3 + len(names)):
        raise RuntimeError('Perf group was not scheduled or returned an incomplete read')
    values = struct.unpack('<' + 'Q' * (3 + len(names)), raw)
    if values[0] != len(names):
        raise RuntimeError('Perf group member count mismatch')
    return {'time_enabled_ns': values[1], 'time_running_ns': values[2],
            'values': dict(zip(names, values[3:]))}


class PerfGroup:
    def __init__(self, specs):
        self.specs = specs
        self.fds = []
        if not specs or len({(s['pmu'], s['cpu']) for s in specs}) != 1:
            raise ValueError('A perf group must use one PMU and CPU')
        if len({s['event'] for s in specs}) != len(specs):
            raise ValueError('Group event names must be unique')
        numbers = {'aarch64': 241, 'x86_64': 298}
        if platform.system() != 'Linux' or platform.machine() not in numbers:
            raise RuntimeError('Unsupported perf_event_open platform')
        libc = ctypes.CDLL(None, use_errno=True)
        libc.syscall.restype = ctypes.c_long
        try:
            for spec in specs:
                raw = attr_bytes(spec, leader=not self.fds)
                buffer = (ctypes.c_char * len(raw)).from_buffer(raw)
                fd = libc.syscall(ctypes.c_long(numbers[platform.machine()]), ctypes.byref(buffer),
                                  ctypes.c_int(-1), ctypes.c_int(spec['cpu']),
                                  ctypes.c_int(self.fds[0] if self.fds else -1), ctypes.c_ulong(8))
                if fd < 0:
                    code = ctypes.get_errno()
                    raise OSError(code, f"perf_event_open {spec['pmu']}/{spec['event']}: {os.strerror(code)}")
                self.fds.append(fd)
        except BaseException:
            self.close()
            raise

    def read(self):
        return decode_group(os.read(self.fds[0], 8 * (3 + len(self.fds))),
                            [s['event'] for s in self.specs])

    def start(self):
        fcntl.ioctl(self.fds[0], IOC_RESET, 1)
        self.before = self.read()
        fcntl.ioctl(self.fds[0], IOC_ENABLE, 1)

    def stop(self):
        fcntl.ioctl(self.fds[0], IOC_DISABLE, 1)
        after = self.read()
        enabled = after['time_enabled_ns'] - self.before['time_enabled_ns']
        running = after['time_running_ns'] - self.before['time_running_ns']
        return {**after, 'window_enabled_ns': enabled, 'window_running_ns': running,
                'running_fraction': running / enabled if enabled else 0.0}

    def close(self):
        for fd in reversed(self.fds):
            os.close(fd)
        self.fds = []

    def __enter__(self):
        return self

    def __exit__(self, *args):
        self.close()


def groups():
    ucf = 'nvidia_ucf_pmu_0'
    gpu = 'nvidia_ucf_gpu_pmu_0'
    return {
        'ucf_all': [event_spec(ucf, name) for name in ('local_dram_byte_rd', 'local_dram_byte_wr')],
        'ucf_noncpu': [event_spec(ucf, name, {'src_loc_noncpu': 1, 'dst_loc': 1})
                       for name in ('local_dram_byte_rd', 'local_dram_byte_wr')],
        'ucf_cpu': [event_spec(ucf, name, {'src_loc_cpu': 1, 'dst_loc': 1})
                   for name in ('local_dram_byte_rd', 'local_dram_byte_wr')],
        'gpu_ucf': [event_spec(gpu, name) for name in ('rd_bytes_loc_rem', 'wr_bytes_loc', 'wr_bytes_rem')],
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report = {'host': platform.node(), 'kernel': platform.release(), 'pid': os.getpid(),
              'perf_event_paranoid': Path('/proc/sys/kernel/perf_event_paranoid').read_text().strip(),
              'scope': 'Idle system-wide counting only; no GPU workload or model', 'groups': {}}
    for name, specs in groups().items():
        result = {'specs': specs}
        try:
            with PerfGroup(specs) as counters:
                counters.start()
                time.sleep(0.1)
                result.update(status='readable', idle=counters.stop())
        except OSError as error:
            result.update(status='unavailable', errno=error.errno, error=str(error))
        except RuntimeError as error:
            result.update(status='unavailable', error=str(error))
        report['groups'][name] = result
    with args.output.open('x') as handle:
        json.dump(report, handle, indent=2)
        handle.write('\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
