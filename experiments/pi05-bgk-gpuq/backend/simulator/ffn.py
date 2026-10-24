"""Static output-major FFN layout and completion-ordered all-bank PIM traces.

No numerical tensors are allocated. The manifest defines arithmetic and buffers;
the trace drives DRAM/PIM timing, not the arithmetic values themselves.
"""

import csv
import hashlib
import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

import yaml

from .config import MemoryConfig

TX_BYTES = 32
ROW_BYTES = 2048
BANKS_PER_RANK = 16
WEIGHT_BYTES = 2
RESULT_BYTES = 4


@dataclass(frozen=True)
class FFNConfig:
    packages: int = 8
    channels_per_package: int = 2
    ranks: int = 1
    horizon: int = 50
    hidden_size: int = 1024
    intermediate_size: int = 4096
    model: str = "lpddr5x-8533"
    density_gbit: int = 16
    refresh: bool = True
    reuse_tokens: int = 1
    macab_tokens_per_command: int = 1
    macab_issue_interval_ck: int | None = None
    input_buffer_bytes: int | None = None
    accumulator_buffer_bytes_per_bank: int | None = None
    nonlinear_cycles_per_token: int | None = None
    frontend_ratio: int = 32
    max_cycles: int = 10_000_000

    def __post_init__(self):
        for name in (
            "packages", "channels_per_package", "ranks", "horizon", "hidden_size",
            "intermediate_size", "reuse_tokens", "macab_tokens_per_command",
            "frontend_ratio", "max_cycles",
        ):
            if type(getattr(self, name)) is not int or getattr(self, name) <= 0:
                raise ValueError(f"{name} must be a positive integer")
        self.memory_config()  # Check total channels, rank count, rate and density.
        if self.macab_tokens_per_command > 1 and (
            self.macab_tokens_per_command != self.reuse_tokens
        ):
            raise ValueError(
                "macab_tokens_per_command must be 1 or equal reuse_tokens"
            )
        if self.macab_issue_interval_ck is not None and (
            type(self.macab_issue_interval_ck) is not int
            or self.macab_issue_interval_ck <= 0
        ):
            raise ValueError("macab_issue_interval_ck must be a positive integer or None")
        for dim in (self.hidden_size, self.intermediate_size):
            if dim % self.total_banks:
                raise ValueError("Both output dimensions must be divisible by total PIM banks")
            if dim % (TX_BYTES // WEIGHT_BYTES):
                raise ValueError("Reduction dimensions must be multiples of 16 BF16 elements")
        tile_size = min(self.reuse_tokens, self.horizon)
        for name, required in (
            ("input_buffer_bytes", max(self.hidden_size, self.intermediate_size) * 2 * tile_size),
            ("accumulator_buffer_bytes_per_bank", RESULT_BYTES * tile_size),
        ):
            if getattr(self, name) is None:
                object.__setattr__(self, name, required)
            if type(getattr(self, name)) is not int or getattr(self, name) < required:
                raise ValueError(f"{name} must be an integer >= {required} for the input tile")
        if self.nonlinear_cycles_per_token is not None and (
            type(self.nonlinear_cycles_per_token) is not int or self.nonlinear_cycles_per_token < 0
        ):
            raise ValueError("nonlinear_cycles_per_token must be a nonnegative integer or None")

    @property
    def channels(self):
        return self.packages * self.channels_per_package

    @property
    def total_banks(self):
        return self.channels * self.ranks * BANKS_PER_RANK

    def memory_config(self):
        timing_overrides = {}
        if self.macab_issue_interval_ck is not None:
            timing_overrides["nCCDAB"] = self.macab_issue_interval_ck
        return MemoryConfig(
            model=self.model, channels=self.channels, ranks=self.ranks,
            density_gbit=self.density_gbit, refresh=self.refresh, pim=True,
            frontend_ratio=self.frontend_ratio, max_cycles=self.max_cycles,
            timing_overrides=timing_overrides,
        )


def encode_address(cfg: FFNConfig, channel: int, rank: int, bg: int, bank: int,
                   bank_byte_offset: int) -> int:
    """Inverse of CacheLineInterleave + RoBaRaCoCh (BG bits below bank bits)."""
    if not (0 <= channel < cfg.channels and 0 <= rank < cfg.ranks
            and 0 <= bg < 4 and 0 <= bank < 4 and bank_byte_offset >= 0):
        raise ValueError("Invalid physical coordinate")
    row, col_byte = divmod(bank_byte_offset, ROW_BYTES)
    intra = (((row * 4 + bank) * 4 + bg) * cfg.ranks + rank) * ROW_BYTES + col_byte
    tx, byte = divmod(intra, TX_BYTES)
    return (tx * cfg.channels + channel) * TX_BYTES + byte


def decode_address(cfg: FFNConfig, address: int) -> dict:
    if address < 0:
        raise ValueError("Negative address")
    tx, byte = divmod(address, TX_BYTES)
    tx, channel = divmod(tx, cfg.channels)
    upper, col = divmod(tx, ROW_BYTES // TX_BYTES)
    upper, rank = divmod(upper, cfg.ranks)
    upper, bg = divmod(upper, 4)
    row, bank = divmod(upper, 4)
    return dict(channel=channel, rank=rank, bankgroup=bg, bank=bank,
                row=row, column_bytes=col * TX_BYTES + byte)


def build_layout(cfg: FFNConfig) -> dict:
    """Independent of H: consecutive N ranges per bank, full K vectors local."""
    matrices = []
    base_row = 0
    capacity_per_bank = cfg.density_gbit * 1024**3 // 8 // BANKS_PER_RANK
    for name, k, n in (
        ("gate", cfg.hidden_size, cfg.intermediate_size),
        ("up", cfg.hidden_size, cfg.intermediate_size),
        ("down", cfg.intermediate_size, cfg.hidden_size),
    ):
        per_bank = n // cfg.total_banks
        bank_bytes = per_bank * k * WEIGHT_BYTES
        rows = (bank_bytes + ROW_BYTES - 1) // ROW_BYTES
        matrices.append(dict(
            name=name, K=k, N=n, outputs_per_bank=per_bank,
            weights_per_bank=per_bank * k, weight_bytes_per_bank=bank_bytes,
            total_weight_bytes=k * n * WEIGHT_BYTES,
            base_row=base_row, rows_per_bank=rows, vector_bytes=k * WEIGHT_BYTES,
            mac_requests_per_token=(k * n // (BANKS_PER_RANK * (TX_BYTES // WEIGHT_BYTES))),
        ))
        base_row += rows
    if base_row * ROW_BYTES > capacity_per_bank:
        raise ValueError("Static FFN weights exceed configured bank capacity")
    layout = dict(
        schema_version=1, order="output-major, contiguous K within the same bank",
        output_partition="contiguous equal N ranges in global-bank order",
        global_bank_order="package, channel-within-package, rank, bankgroup, bank",
        packages=cfg.packages, channels_per_package=cfg.channels_per_package,
        total_channels=cfg.channels, ranks_per_channel=cfg.ranks,
        bankgroups_per_rank=4, banks_per_group=4, total_banks=cfg.total_banks,
        channel_width_bits=16, density_gbit_per_channel_rank=cfg.density_gbit,
        row_bytes=ROW_BYTES, transaction_bytes=TX_BYTES, weight_dtype="BF16",
        matrices=matrices, reserved_rows_per_bank=base_row,
        address_mapping="synthetic RoBaRaCoCh; not Thor physical address hash",
    )
    layout["sha256"] = hashlib.sha256(
        json.dumps(layout, sort_keys=True).encode()
    ).hexdigest()
    return layout


def weight_location(cfg: FFNConfig, matrix: dict, output: int, reduction: int = 0) -> dict:
    if not (0 <= output < matrix["N"] and 0 <= reduction < matrix["K"]):
        raise ValueError("Weight index out of range")
    global_bank, local_output = divmod(output, matrix["outputs_per_bank"])
    channel_rank, local_bank = divmod(global_bank, BANKS_PER_RANK)
    channel, rank = divmod(channel_rank, cfg.ranks)
    bg, bank = divmod(local_bank, 4)
    offset = (matrix["base_row"] * ROW_BYTES
              + (local_output * matrix["K"] + reduction) * WEIGHT_BYTES)
    return dict(
        output=output, global_bank=global_bank, package=channel // cfg.channels_per_package,
        channel_in_package=channel % cfg.channels_per_package, channel=channel, rank=rank,
        bankgroup=bg, bank=bank, bank_byte_offset=offset,
        row=offset // ROW_BYTES, column_bytes=offset % ROW_BYTES,
        address=encode_address(cfg, channel, rank, bg, bank, offset),
    )


def generate_ffn(cfg: FFNConfig, output_dir: str | Path) -> dict:
    layout = build_layout(cfg)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / "layout.json").write_text(json.dumps(layout, indent=2) + "\n")
    (output / "memory.yaml").write_text(
        yaml.safe_dump(asdict(cfg.memory_config()), sort_keys=False)
    )
    with (output / "weights.csv").open("w") as stream:
        writer = None
        for matrix in layout["matrices"]:
            for n in range(matrix["N"]):
                row = dict(matrix=matrix["name"], **weight_location(cfg, matrix, n),
                           vector_bytes=matrix["vector_bytes"])
                if writer is None:
                    writer = csv.DictWriter(stream, fieldnames=list(row))
                    writer.writeheader()
                writer.writerow(row)
    counts = Counter()
    markers = []
    record = 0
    logical_macs_emitted = 0
    digest = hashlib.sha256()
    with (output / "ffn.trace").open("w") as stream, (
        output / "compute_schedule.csv"
    ).open("w", newline="") as schedule_stream:
        schedule = csv.DictWriter(schedule_stream, fieldnames=[
            "first_trace_line", "last_trace_line", "phase", "tile_start", "tile_size",
            "token", "input_slot", "accumulator_slot", "local_output", "bank_row",
            "bank_byte_offset", "reduction_start", "elements", "input_byte_offset",
        ])
        schedule.writeheader()

        def emit(op, value):
            nonlocal record
            line = f"{op} 0x{value:x}\n"
            stream.write(line)
            digest.update(line.encode())
            counts[op] += 1
            record += 1

        def barrier(h, phase, boundary):
            marker = len(markers)
            markers.append(dict(
                id=marker, token=h, tile_size=tile_size, phase=phase,
                boundary=boundary, line=record + 1,
            ))
            emit("BARRIER", marker)

        def for_ranks(op, offset):
            for rank in range(cfg.ranks):
                for channel in range(cfg.channels):
                    emit(op, encode_address(cfg, channel, rank, 0, 0, offset))

        def load_vector(h, phase, elements):
            barrier(h, phase, "start")
            for offset in range(0, tile_size * elements * WEIGHT_BYTES, TX_BYTES):
                for_ranks("PIM_WR_GB", offset)
            barrier(h, phase, "end")

        def projection(h, matrix):
            nonlocal logical_macs_emitted
            phase = matrix["name"]
            barrier(h, phase, "start")
            for slot in range(matrix["outputs_per_bank"]):
                offset = matrix["base_row"] * ROW_BYTES + slot * matrix["vector_bytes"]
                k_byte = 0
                while k_byte < matrix["vector_bytes"]:
                    segment_bytes = min(
                        ROW_BYTES - (offset + k_byte) % ROW_BYTES,
                        matrix["vector_bytes"] - k_byte,
                    )
                    # Default: every token issues every column MACAB. Experimental
                    # fused mode: one enhanced MACAB applies the read K block to
                    # every real token/accumulator context in this tile.
                    input_slots = (
                        range(tile_size)
                        if cfg.macab_tokens_per_command == 1
                        else range(1)
                    )
                    for input_slot in input_slots:
                        command_tokens = (
                            1 if cfg.macab_tokens_per_command == 1 else tile_size
                        )
                        first_line = record + 1
                        for byte in range(k_byte, k_byte + segment_bytes, TX_BYTES):
                            for_ranks("PIM_MAC_AB", offset + byte)
                        trace_commands = record - first_line + 1
                        logical_macs_emitted += (
                            trace_commands * BANKS_PER_RANK
                            * (TX_BYTES // WEIGHT_BYTES) * command_tokens
                        )
                        schedule.writerow(dict(
                            first_trace_line=first_line, last_trace_line=record,
                            phase=phase, tile_start=h, tile_size=tile_size,
                            token=h + input_slot, input_slot=input_slot,
                            accumulator_slot=input_slot, local_output=slot,
                            bank_row=(offset + k_byte) // ROW_BYTES,
                            bank_byte_offset=offset + k_byte,
                            reduction_start=k_byte // WEIGHT_BYTES,
                            elements=segment_bytes // WEIGHT_BYTES,
                            input_byte_offset=input_slot * matrix["vector_bytes"] + k_byte,
                        ))
                    k_byte += segment_bytes
                # R independent accumulators persist across row changes and refresh.
                # Export the completed tile before reusing them for the next output.
                barrier(h, phase, "mac_done")
                for result_byte in range(0, tile_size * BANKS_PER_RANK * RESULT_BYTES, TX_BYTES):
                    for_ranks("PIM_RD_RES", result_byte)
                barrier(h, phase, "result_done")
            barrier(h, phase, "end")

        for h in range(0, cfg.horizon, cfg.reuse_tokens):
            tile_size = min(cfg.reuse_tokens, cfg.horizon - h)
            load_vector(h, "input_x", cfg.hidden_size)
            projection(h, layout["matrices"][0])
            # Reuse the same X in each rank input SRAM for Up.
            projection(h, layout["matrices"][1])
            barrier(h, "nonlinear", "start")
            emit("WAIT", (cfg.nonlinear_cycles_per_token or 0) * tile_size)
            barrier(h, "nonlinear", "end")
            load_vector(h, "input_z", cfg.intermediate_size)
            projection(h, layout["matrices"][2])

    expected_macs = cfg.horizon * sum(m["K"] * m["N"] for m in layout["matrices"])
    assert logical_macs_emitted == expected_macs
    manifest = dict(
        schema_version=2, config=asdict(cfg), layout_sha256=layout["sha256"],
        trace_sha256=digest.hexdigest(), command_counts=dict(counts), markers=markers,
        logical_mac_operations=expected_macs, gemm_flops=2 * expected_macs,
        enhanced_macab=dict(
            enabled=cfg.macab_tokens_per_command > 1,
            tokens_per_full_command=cfg.macab_tokens_per_command,
            tail_commands_use_real_tile_size=True,
            issue_interval_ck=cfg.macab_issue_interval_ck,
            semantics=(
                "one K-block read applied to every real token context in the tile"
                if cfg.macab_tokens_per_command > 1 else
                "legacy one token context per MACAB"
            ),
        ),
        unique_weight_bytes=sum(m["total_weight_bytes"] for m in layout["matrices"]),
        row_reuse=dict(
            kind="open DRAM row reuse across input tokens; no weight SRAM",
            requested_tokens=cfg.reuse_tokens,
            effective_max_tokens=min(cfg.reuse_tokens, cfg.horizon),
            weight_scans=(cfg.horizon + cfg.reuse_tokens - 1) // cfg.reuse_tokens,
            ideal_actab_no_refresh=(
                (cfg.horizon + cfg.reuse_tokens - 1) // cfg.reuse_tokens
                * sum(m["rows_per_bank"] for m in layout["matrices"])
                * cfg.channels * cfg.ranks
            ),
            operand_schedule="compute_schedule.csv; token/input/accumulator selection sideband",
            result_export="batch tile results per output; unchanged bytes, fewer completion fences",
        ),
        buffers=dict(
            input_bytes_per_channel_rank=cfg.input_buffer_bytes,
            accumulator_bytes_per_bank=cfg.accumulator_buffer_bytes_per_bank,
            additional_weight_sram_bytes_per_bank=0,
            input_bytes_total=cfg.input_buffer_bytes * cfg.channels * cfg.ranks,
            accumulator_bytes_total=cfg.accumulator_buffer_bytes_per_bank * cfg.total_banks,
            shared_gate_up_z_allocated_bytes=(
                min(cfg.reuse_tokens, cfg.horizon) * cfg.intermediate_size
                * (2 * RESULT_BYTES + WEIGHT_BYTES)
            ),
        ),
        traffic_bytes=dict(
            weight_internal=counts["PIM_MAC_AB"] * BANKS_PER_RANK * TX_BYTES,
            input_broadcast=counts["PIM_WR_GB"] * TX_BYTES,
            result_read=counts["PIM_RD_RES"] * TX_BYTES,
        ),
        nonlinear=dict(
            operation="Z=BF16(GELU(G)*U), G/U and bank accumulators FP32",
            location="shared logic outside banks",
            included=cfg.nonlinear_cycles_per_token is not None,
            cycles_per_token=cfg.nonlinear_cycles_per_token,
            total_cycles=cfg.horizon * (cfg.nonlinear_cycles_per_token or 0),
        ),
        assumptions=[
            "One FFN layer; tiles and projections serialized, tokens within each open row tiled",
            "Weights preloaded; no weight upload or initialization time",
            "Complete BF16 input tile SRAM per channel/rank, broadcast locally to 16 banks",
            "X loaded once for Gate/Up; Z loaded once for Down; no cross-rank free broadcast",
            "One FP32 accumulator per tile token per bank; persists across row changes and refresh",
            "MACAB reads 16 BF16 weights/bank; SIMD throughput assumed, not silicon calibrated",
            (
                "Enhanced MACAB reuses one K-block read across every real tile token"
                if cfg.macab_tokens_per_command > 1 else
                "Each token re-reads every weight column; MACAB count and timing remain unchanged"
            ),
            "Token/accumulator context in compute_schedule.csv; no extra selection cost modeled",
            "All-bank commands share rank row/column; no split-K or cross-bank sum reduction",
            "32B WRGB/RDRES transfers use DQ; SRAM access modeled with CWL/CL+burst proxies",
            "Shared logic holds gathered G/U and Z; no DRAM spill; result Y exported as FP32",
            "No extra package/interconnect/host-memory transfer or control-reset cost modeled",
            "No residual, AdaRMSNorm, numerical validation, or energy model",
            "Rank count and density are simulation parameters, not a claim of Thor BOM",
        ],
    )
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def run_ffn(cfg: FFNConfig, output_dir: str | Path) -> dict:
    from .backend import run_trace

    output = Path(output_dir).resolve()
    manifest = generate_ffn(cfg, output)
    timing = run_trace(cfg.memory_config(), output / "ffn.trace", output / "simulation")
    for trace_op, command in (("PIM_MAC_AB", "MACAB"), ("PIM_WR_GB", "WRGB"),
                              ("PIM_RD_RES", "RDRES")):
        if timing["issued_commands"][command] != manifest["command_counts"][trace_op]:
            raise RuntimeError(f"FFN command count mismatch: {command}")
    if timing["requests"]["sha256"] != manifest["trace_sha256"]:
        raise RuntimeError("Simulated trace differs from generated trace")
    events = {}
    with (output / "simulation/barriers.csv").open() as stream:
        for event in csv.DictReader(stream):
            events[int(event["marker"])] = int(event["frontend_tick"])
    if len(events) != len(manifest["markers"]):
        raise RuntimeError("FFN did not complete every dependency barrier")
    phases = Counter()
    starts = {}
    for marker in manifest["markers"]:
        key = (marker["token"], marker["phase"])
        if marker["boundary"] == "start":
            starts[key] = events[marker["id"]]
        elif marker["boundary"] == "end":
            phases[marker["phase"]] += events[marker["id"]] - starts.pop(key)
    phase_ns = {p: ticks * timing["tCK_ns"] / cfg.frontend_ratio for p, ticks in phases.items()}
    result = dict(
        schema_version=2, config=asdict(cfg), layout_sha256=manifest["layout_sha256"],
        memory_cycles=timing["memory_cycles"], latency_ns=timing["latency_ns"],
        latency_scope=("FFN modeled latency including configured shared-logic delay"
                       if cfg.nonlinear_cycles_per_token is not None else
                       "memory/PIM only; excludes GELU-product compute latency"),
        nonlinear_compute_included=manifest["nonlinear"]["included"],
        nonlinear_delay_ns=manifest["nonlinear"]["total_cycles"] * timing["tCK_ns"],
        phase_latency_ns=phase_ns,
        phase_timing_basis="completion barriers, frontend tick resolution; overlaps forbidden",
        logical_mac_operations=manifest["logical_mac_operations"],
        enhanced_macab=manifest["enhanced_macab"],
        row_reuse=manifest["row_reuse"], buffers=manifest["buffers"],
        pim_issue_interval_ck=timing["pim_model"]["issue_interval_ck"],
        row_activation_bytes=(timing["issued_commands"]["ACTAB"] * BANKS_PER_RANK * ROW_BYTES),
        issued_commands=timing["issued_commands"], traffic_bytes=timing["traffic_bytes"],
        provenance=timing["provenance"], energy_pj=None,
    )
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
