"""Byte-address traces for ordinary memory and coarse bank-parallel PIM requests."""

import hashlib
from pathlib import Path


def inspect_trace(
    path: Path, *, transaction_bytes: int, capacity_bytes: int, pim: bool = False
) -> dict:
    counts = dict.fromkeys(("LD", "ST", "PIM_MAC_AB", "PIM_WR_GB", "PIM_RD_RES", "PIM_MAC_HB", "PIM_LOCAL_GB", 'PIM_WR_BG', 'PIM_WR_Q', 'PIM_MAC_K', 'PIM_MAC_QK', 'PIM_RD_BG', 'PIM_LOCAL_WR', 'PIM_RD_Q'), 0)
    new_ops = ("PIM_SPILL", "PIM_ROPE_COEF", "PIM_ROPE_LOAD", "PIM_ROPE_EXEC", "PIM_ROPE_STORE", "PIM_Q_EXPORT", "PIM_Q_REMOTE", "PIM_Z_STORE", "PIM_Z_LOAD", "PIM_GEMV_RELEASE", "PIM_STAGE_BEGIN", "PIM_STAGE_SEAL", "PIM_STAGE_LOAD", "PIM_STAGE_FRAGMENT")
    counts.update(dict.fromkeys(new_ops, 0))
    controls = {"BARRIER": 0, "WAIT": 0}
    wait_cycles = 0
    staged_input_bytes = 0
    stage_rows_by_channel = {}
    stage_input_read_bursts = 0
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for number, raw in enumerate(stream, 1):
            digest.update(raw)
            fields = raw.split()
            if len(fields) not in (2, 3, 4):
                raise ValueError(
                    f"Trace line {number}: expected OP ADDRESS [BUFFER] or STAGE_BEGIN ADDRESS BYTES ROWS"
                )
            try:
                op, address_text = (part.decode("ascii") for part in fields[:2])
                address = int(address_text, 16 if address_text.lower().startswith("0x") else 10)
            except (UnicodeDecodeError, ValueError) as error:
                raise ValueError(f"Trace line {number}: invalid address") from error
            if op == "PIM_STAGE_BEGIN":
                if len(fields) != 4:
                    raise ValueError(f"Trace line {number}: STAGE_BEGIN needs ADDRESS BYTES ROWS")
                try:
                    stage_bytes, stage_rows = map(int, fields[2:])
                except ValueError as error:
                    raise ValueError(f"Trace line {number}: invalid stage geometry") from error
                if stage_rows not in (8, 16) or stage_bytes <= 0 or stage_bytes > 131072 or stage_bytes % (stage_rows*32):
                    raise ValueError(f"Trace line {number}: invalid stage geometry")
                staged_input_bytes += stage_bytes
                stage_rows_by_channel[(address >> 5) & 15] = stage_rows
            elif len(fields) == 4:
                raise ValueError(f"Trace line {number}: only STAGE_BEGIN accepts four fields")
            if len(fields) == 3 and (
                op not in ("PIM_WR_GB", "PIM_MAC_AB", "PIM_LOCAL_GB", "PIM_WR_BG", "PIM_WR_Q", "PIM_MAC_K", "PIM_MAC_QK", "PIM_RD_Q", "PIM_ROPE_LOAD", "PIM_ROPE_EXEC", "PIM_Q_REMOTE", "PIM_Z_LOAD", "PIM_GEMV_RELEASE", "PIM_STAGE_LOAD", "PIM_STAGE_FRAGMENT") or fields[2] not in (b"0", b"1")
            ):
                raise ValueError(f"Trace line {number}: BUFFER must be 0/1 on WRGB or MACAB")
            if op in ("PIM_LOCAL_GB", "PIM_WR_BG", "PIM_WR_Q", "PIM_MAC_K", "PIM_MAC_QK", "PIM_RD_Q", "PIM_ROPE_LOAD", "PIM_ROPE_EXEC", "PIM_Q_REMOTE", "PIM_Z_LOAD", "PIM_GEMV_RELEASE", "PIM_STAGE_LOAD", "PIM_STAGE_FRAGMENT") and len(fields) != 3:
                raise ValueError("PIM_LOCAL_GB requires BUFFER=0/1")
            if op in controls:
                if address < 0 or address > 2**60:
                    raise ValueError(f"Trace line {number}: invalid control value")
                controls[op] += 1
                if op == "WAIT":
                    wait_cycles += address
                continue
            if op not in counts:
                raise ValueError(
                    f"Trace line {number}: unsupported operation {op}; expected memory/PIM/control"
                )
            if op.startswith("PIM_") and not pim:
                raise ValueError(f"Trace line {number}: {op} requires pim: true")
            if address < 0 or address + transaction_bytes > capacity_bytes:
                raise ValueError(f"Trace line {number}: address exceeds configured capacity")
            if address % transaction_bytes:
                raise ValueError(
                    f"Trace line {number}: address must be {transaction_bytes}-byte aligned"
                )
            if op == "PIM_STAGE_FRAGMENT":
                if ((address >> 5) & 15) not in stage_rows_by_channel:
                    raise ValueError(f"Trace line {number}: STAGE_FRAGMENT requires preceding STAGE_BEGIN")
                stage_input_read_bursts += stage_rows_by_channel[(address >> 5) & 15]
            elif op == "PIM_STAGE_LOAD":
                stage_input_read_bursts += 1
            counts[op] += 1
    if not sum(counts.values()) and not sum(controls.values()):
        raise ValueError("Empty trace")
    return dict(
        **{name.lower(): counts[name] for name in new_ops if counts[name]},
        **({"stage_input_declared_bytes": staged_input_bytes, "stage_input_read_bursts": stage_input_read_bursts} if staged_input_bytes else {}),
        reads=counts["LD"],
        writes=counts["ST"],
        pim_mac_ab=counts["PIM_MAC_AB"],
        pim_mac_hb=counts["PIM_MAC_HB"],
        pim_wr_gb=counts["PIM_WR_GB"],
        pim_rd_res=counts["PIM_RD_RES"],
        pim_local_gb=counts["PIM_LOCAL_GB"],
        pim_wr_bg=counts["PIM_WR_BG"],
        pim_wr_q=counts["PIM_WR_Q"],
        pim_mac_k=counts["PIM_MAC_K"],
        pim_mac_qk=counts["PIM_MAC_QK"],
        pim_rd_bg=counts["PIM_RD_BG"],
        pim_local_wr=counts["PIM_LOCAL_WR"],
        pim_rd_q=counts["PIM_RD_Q"],
        barriers=controls["BARRIER"],
        waits=controls["WAIT"],
        wait_cycles=wait_cycles,
        records=sum(counts.values()) + sum(controls.values()),
        total=sum(counts.values()),
        sha256=digest.hexdigest(),
    )


def generate_trace(path: str | Path, *, requests: int = 128, pattern: str = "mixed") -> Path:
    if type(requests) is not int or requests <= 0 or pattern not in ("read", "write", "mixed"):
        raise ValueError("Positive requests and read/write/mixed pattern are required")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x") as stream:
        for index in range(requests):
            write = pattern == "write" or (pattern == "mixed" and index % 5 == 0)
            stream.write(f"{'ST' if write else 'LD'} 0x{index * 32:x}\n")
    return path
