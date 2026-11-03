"""Ramulator adapter: configuration/trace in, unit-explicit results out."""

import csv
import json
import shutil
import subprocess
from dataclasses import asdict
from pathlib import Path

from .config import MODELS, MemoryConfig
from .trace import inspect_trace


def run_trace(config: MemoryConfig, trace_path: str | Path, output_dir: str | Path) -> dict:
    import ramulator

    model_name, preset = MODELS[config.model]
    device_name = f"{model_name}PIM" if config.pim else model_name
    if not hasattr(ramulator.dram, device_name):
        raise ValueError(f"{device_name} is not available in this branch/build")
    model_class = getattr(ramulator.dram, device_name)

    def make_model():
        return model_class(
            org_preset=f"{model_name}_{config.density_gbit}Gb_x16",
            timing_preset=preset,
            rank=config.ranks,
            **({"pim_data_path": config.pim_data_path, "pim_rank_controller": config.pim_rank_controller} if config.pim else {}),
            **config.timing_overrides,
        )

    model = make_model()
    model.to_config()  # Validate before creating output.
    organization, timings = model.resolve()
    if timings["nRC"] < timings["nRAS"] + timings["nRP"]:
        raise ValueError("nRC must cover nRAS+nRP")
    transaction_bytes = model.internal_prefetch_size * organization["channel_width"] // 8
    capacity = config.channels * config.ranks * config.density_gbit * 1024**3 // 8
    source = Path(trace_path).resolve(strict=True)
    output = Path(output_dir).resolve()
    output.mkdir(parents=True, exist_ok=False)
    snapshot = output / "input.trace"
    shutil.copyfile(source, snapshot)
    requests = inspect_trace(
        snapshot, transaction_bytes=transaction_bytes, capacity_bytes=capacity, pim=config.pim
    )
    if (requests["pim_wr_gb"] or requests["pim_rd_res"]) and config.wck_mode != "always_on":
        raise ValueError("PIM buffer transfers require pre-synchronized always_on WCK")
    if config.pim_input_broadcast and requests["pim_mac_hb"]:
        raise ValueError("pim_input_broadcast supports all-bank traces only")
    if config.pim_mac_scope == "channel" and requests["pim_mac_hb"]:
        raise ValueError("pim_mac_scope=channel supports all-bank traces only")
    if config.pim_data_path == "attacc" and requests["pim_mac_hb"]:
        raise ValueError("AttAcc data path supports all-bank traces only")
    controller_class = ramulator.controller.LPDDR5PIM if config.pim else ramulator.controller.LPDDR5
    controllers = []
    for channel in range(config.channels):
        controllers.append(
            controller_class(
                dram=make_model(),
                scheduler=ramulator.scheduler.FRFCFSRowHit(),
                refresh_manager=(
                    ramulator.refresh_manager.AllBank()
                    if config.refresh
                    else ramulator.refresh_manager.NoRefresh()
                ),
                row_policy=ramulator.row_policy.Open(),
                addr_mapper=ramulator.addr_mapper.RoBaRaCoCh(),
                wck_sync_mode=config.wck_mode,
                **({"pim_input_broadcast": config.pim_input_broadcast,
                    "pim_mac_scope": config.pim_mac_scope,
                    "pim_data_path": config.pim_data_path,
                    "pim_local_transfer_cycles": config.pim_local_transfer_cycles,
                    "pim_unified_gemv": config.pim_unified_gemv,
                    "pim_rank_staging": config.pim_rank_staging,
                    "pim_staging_banks": config.pim_staging_banks,
                    "pim_rank_controller": config.pim_rank_controller,
                    "pim_rank_descriptors": config.pim_rank_descriptors,
                    "rank_descriptor_events_path": str(output / f"rank_descriptors.csv.ch{channel}") if config.pim_rank_descriptors else "",
                    "rank_ca_events_path": str(output / f"rank_ca_events.csv.ch{channel}") if config.pim_rank_controller else "",
                    "staging_events_path": str(output / f"staging_events.csv.ch{channel}") if config.pim_rank_staging else "",
                    "gemv_port_cycles": config.gemv_port_cycles,
                    "unified_gemv_events_path": str(output / f"unified_gemv.csv.ch{channel}") if config.pim_unified_gemv else "",
                    "pim_token_pv": config.pim_token_pv,
                    "pim_qkv_fusion": config.pim_qkv_fusion,
                    "pim_whole_srf": config.pim_whole_srf,
                    "v_cache_preinstalled": config.v_cache_preinstalled,
                    "v_cache_events_path": str(output / f"v_cache.csv.ch{channel}") if config.pim_token_pv else "",
                    "pim_rank_rope": config.pim_rank_rope,
                    "pim_horizon": config.pim_horizon,
                    "pim_ffn_dram": config.pim_ffn_dram,
                    "ffn_z_preinstalled": config.ffn_z_preinstalled,
                    "ffn_dram_events_path": str(output / f"ffn_dram_events.csv.ch{channel}") if config.pim_ffn_dram else "",
                    "rope_ii_cycles": config.rope_ii_cycles,
                    "rope_latency_cycles": config.rope_latency_cycles,
                    "pim_cast_cycles": config.pim_cast_cycles,
                    "rank_reduce_cycles": config.rank_reduce_cycles,
                    "local_input_trace_path": str(output / f"local_input.csv.ch{channel}"),
                    "bgk_events_path": str(output / f"bgk_events.csv.ch{channel}")} if config.pim else {}),
                controller_plugins=[
                    ramulator.controller_plugin.CommandCounter(
                        commands_to_count=model.commands,
                        path=str(output / f"commands.ch{channel}.csv"),
                    ),
                    ramulator.controller_plugin.CmdTraceRecorder(path=str(output / "timeline.csv")),
                ],
            )
        )
    memory = ramulator.memory_system.GenericDRAM(
        clock_ratio=config.memory_ratio,
        controllers=controllers,
        channel_mapper=ramulator.channel_mapper.CacheLineInterleave(),
        channel_independent_dispatch=config.channel_independent_dispatch,
        channel_queue_capacity=config.channel_queue_capacity,
        ingress_events_path=str(output / "channel_ingress.csv") if config.channel_independent_dispatch else "",
    )
    frontend = ramulator.frontend.LoadStoreTrace(
        clock_ratio=config.frontend_ratio,
        path=str(snapshot),
        wait_for_completion=True,
        write_tail_cycles=timings["nCWL"] + timings["nBL_max"] + timings["nWR"],
        max_frontend_ticks=(config.max_cycles * config.frontend_ratio // config.memory_ratio + 1),
        events_path=str(output / "barriers.csv"),
        channel_independent_dispatch=config.channel_independent_dispatch,
        dispatch_events_path=str(output / "frontend_dispatch.csv") if config.channel_independent_dispatch else "",
    )
    sim = ramulator.Simulation(frontend, memory)
    sim.run()
    sim.finalize()
    stats = sim.stats
    if stats["frontend"]["requests_completed"] != requests["total"]:
        raise RuntimeError("Simulation did not complete every accepted request")
    counters = {name: 0 for name in model.commands}
    for channel in range(config.channels):
        with (output / f"commands.ch{channel}.csv").open() as stream:
            for name, count in csv.reader(stream):
                counters[name] += int(count)
    per_channel = stats["memory_system"]["controller"]
    if isinstance(per_channel, dict):
        per_channel = [per_channel]
    cycles = max(c["cycles"] for c in per_channel)
    repo = Path(__file__).resolve().parents[1]
    revision = subprocess.run(
        ["git", "rev-parse", "HEAD"], cwd=repo, text=True, capture_output=True, check=False
    )
    dirty = subprocess.run(
        ["git", "status", "--porcelain"], cwd=repo, text=True, capture_output=True, check=False
    )
    result = {
        "schema_version": 9,
        "bgk_model": {
            "enabled": any(requests[x] for x in ("pim_wr_bg", "pim_wr_q", "pim_mac_k", "pim_mac_qk", "pim_rd_bg", "pim_local_wr", "pim_rd_q")),
            "K_service_CK": timings.get("nCCDK"), "QK_service_CK": timings.get("nCCDQK"),
            "local_write_CK_per_32B": timings.get("nLOCALWR"),
            "K_valid_tokens_per_BG": "1 or 2, encoded by emitted BG input bursts; channel MAC service uses maximum 2",
            "Q_DRAM_refill": {
                "request": "PIM_RD_Q ADDRESS BUFFER",
                "source": "rank0/BG0, four bank-index sources per channel",
                "staging_bytes_per_channel": 32,
                "scheduling": "conservative serial RD return then WRGB relay; one composite in flight",
                "read_completion": "native ordinary RD with ACT/PRE and CL+BL return",
                "relay_completion": "charged shared-path WRGB + nBL_min; same bank index in every BG/rank",
                "callback": "after relay completion only; no GPU round trip",
                "fanout_extra_latency_CK": 0,
            },
            "Q_multicast": "same bank-index in every BG/rank; independent bank-local double-buffer ownership",
            "input_ready": "WRGB issue + nBL_min; consumers wait for every target bank and every queued same-slot fill",
            "MAC_completion": "new MACK/MACQK complete at issue + nCCDK/nCCDQK, native array cadence applies independently",
            "BG_reduction": "implicit lockstep four-bank reduction sustains MAC; RD_BG waits for BG compute readiness; final-fragment dependency in trace",
            "local_write": "independent bank-local 32B ports; native row state and ACT/PRE/recovery; completion includes nLOCALWR+nWR",
            "local_write_no_CA": True, "local_write_no_external_DQ": True,
            "RoPE_conversion_packing": None,
        },
        "config": asdict(config),
        "capacity_bytes": capacity,
        "transaction_bytes": transaction_bytes,
        "requests": requests,
        "completed_requests": stats["frontend"]["requests_completed"],
        "memory_cycles": cycles,
        "tCK_ns": timings["tCK_ps"] / 1000,
        "latency_ns": cycles * timings["tCK_ps"] / 1000,
        "completion_policy": (
            "read callbacks; write issue callbacks plus CWL+BLmax+WR recovery; "
            "PIM MAC callbacks at issue; annotated double-buffer SRAM ownership lasts nCCDAB; "
            + ("AttAcc WRGB completes after nBL_min, result transfers retire at issue; " if config.pim_data_path == "attacc"
               else "PIM buffer-transfer completion callbacks; ")
            + "half-bank compute overlaps disjoint-bank reads; "
            "global drain at other normal/PIM mode transitions"
            "; explicit BARRIER completion fences and WAIT memory-cycle delays"
        ),
        "issued_commands": counters,
        "traffic_bytes": {
            "requested_read": requests["reads"] * transaction_bytes,
            "requested_write": requests["writes"] * transaction_bytes,
            "dram_read": (counters["RD"] + counters["RDA"]) * transaction_bytes,
            "dram_write": (counters["WR"] + counters["WRA"]) * transaction_bytes,
            "pim_input_write": (counters.get("WRGB", 0) - requests["pim_rd_q"]) * transaction_bytes,
            "q_dram_read": requests["pim_rd_q"] * transaction_bytes,
            "q_internal_relay": requests["pim_rd_q"] * transaction_bytes,
            "pim_movement_write_total": counters.get("WRGB", 0) * transaction_bytes,
            "pim_input_delivered": requests["pim_wr_gb"] * transaction_bytes
            * (config.ranks if config.pim_input_broadcast else 1),
            "bgk_input_delivered_to_bank_buffers": (requests["pim_wr_bg"] * organization["bank"]
                + (requests["pim_wr_q"] + requests["pim_rd_q"]) * config.ranks * organization["bankgroup"]) * transaction_bytes,
            "pim_result_read": counters.get("RDRES", 0) * transaction_bytes,
            "pim_local_input_write": requests["pim_local_gb"] * transaction_bytes,
            "pim_bank_local_write": requests["pim_local_wr"] * transaction_bytes,
            "bg_dispatched_input": requests["pim_wr_bg"] * transaction_bytes,
            "bank_index_multicast_input": requests["pim_wr_q"] * transaction_bytes,
            "bg_complete_result": requests["pim_rd_bg"] * transaction_bytes,
            "bgk_internal_array_read": (counters.get("MACK",0) + counters.get("MACQK",0))
                * config.ranks * organization["bankgroup"] * organization["bank"] * transaction_bytes,
            "pim_internal": (counters.get("MACAB", 0) * 2
                             * (config.ranks if config.pim_mac_scope == "channel" else 1)
                             + counters.get("MACHB", 0))
            * transaction_bytes
            * organization["bankgroup"]
            * organization["bank"] // 2,
        },
        "local_input_model": {
            "request": "PIM_LOCAL_GB", "transaction_bytes": 32,
            "cycles_per_burst": config.pim_local_transfer_cycles,
            "ports_per_rank": 1, "scope": "selected rank only; independent rank-local ports",
            "completion": "issue + cycles_per_burst; MAC waits for selected buffer readiness",
            "external_dq_bytes": 0,
            "source": "already-ready Z in rank-local SRAM; gather/GELU/product/cast not computed",
            "assumption": "declared configurable local port, not an LPDDR specification or measured NL unit",
        },
        "pim_model": {
            "enabled": config.pim,
            "scope": f"MACAB addressed {config.pim_mac_scope}; RDRES per rank; "
                     "WRGB targets depend on input_broadcast",
            "banks_per_command": organization["bankgroup"] * organization["bank"]
                                 * (config.ranks if config.pim_mac_scope == "channel" else 1),
            "mac_scope": config.pim_mac_scope,
            "data_path": config.pim_data_path,
            "transfer_scheduling": (
                "AttAcc-style shared WRGB/RDRES movement path: BL/CCDS/CCDL; "
                "independent MAC path; trace/barriers supply result dependencies; "
                "WRGB waits nBL_min (no CWL), RDRES retires at issue; ordinary RD retains CL+BL; "
                "no additional broadcast-network latency or RDRES-to-MAC lock"
                if config.pim_data_path == "attacc" else
                "LPDDR external DQ transfers with CL/CWL completion and rank turnaround"
            ),
            "mac_timing_basis": "explicit local-array ACT/PRE/refresh/column constraints; "
                                "no inherited external RD DQ/CAS/rank-turnaround constraints",
            "channel_mac": {
                "enabled": config.pim_mac_scope == "channel",
                "trace": "one rank-0 MACAB with explicit BUFFER; command Rank=-1 targets all ranks",
                "activation": "ACTAB/PREab prerequisites cover every target bank and rank",
                "readiness": "actual MAC waits for every target input; ACT/PRE may prepare earlier",
                "assumptions": "one shared CA command fans out to all ranks with no extra delay; "
                               "local PEs/SRAM/accumulators operate concurrently",
            },
            "half_bank": {
                "request": "PIM_MAC_HB",
                "command": "MACHB",
                "scope": "addressed half of bank groups in the addressed channel/rank",
                "banks_per_command": organization["bankgroup"] * organization["bank"] // 2,
                "gpu_read_overlap": "ordinary LD to the complementary half; unchanged read timings",
                "transfer_latency_scale": 1.0,
            },
            "issue_interval_ck": timings.get("nCCDAB"),
            "issue_interval_basis": "nCCDL (same-BG column interval), unless explicitly overridden",
            "mac_completion": "retire at issue, as in AttAcc; no independent completion parameter",
            "double_buffer": "optional BUFFER=0/1 on WRGB/MACAB; per-rank input-slot ownership; "
                             "independent writes may bypass blocked MACs; "
                             + ("result dependencies supplied by trace/barriers" if config.pim_data_path == "attacc"
                                else "RDRES waits for final SRAM use"),
            "pe_clock_model": "external; PE throughput assumed sufficient for the configured MAC interval",
            "numerical_compute": False,
            "input_buffer_write_latency_ck": (timings.get("nBL_min") if config.pim_data_path == "attacc"
                                               else timings.get("nPIMWrLatency")),
            "result_buffer_read_latency_ck": timings.get("nPIMRdLatency"),
            "buffer_transfer_basis": "32B DQ burst; SRAM, no array ACT/PRE/write recovery",
            "input_broadcast": {
                "enabled": config.pim_input_broadcast,
                "scope": "all ranks in addressed channel" if config.pim_input_broadcast
                         else "addressed rank in addressed channel",
                "ranks_per_write": config.ranks if config.pim_input_broadcast else 1,
                "trace_rule": "WRGB uses rank-0 address; WRGB and MACAB require BUFFER=0/1"
                              if config.pim_input_broadcast else "original per-rank WRGB stream",
                "extra_fanout_latency_ck": 0,
                "completion": ("issue + nBL_min makes every target buffer ready (AttAcc)"
                               if config.pim_data_path == "attacc"
                               else "one CWL+BLmin completion updates every target buffer"),
                "ownership": "every target slot must be free before issuing a broadcast",
                "dq_accounting": "one 32B burst per WRGB; delivered bytes counted separately",
                "limitations": "assumed internal fanout and sufficient SRAM ports; no broadcast "
                               "network/energy model; no cross-channel or result broadcast",
            },
        },
        "energy_pj": None,
        "resolved_timings_ck": timings,
        "provenance": {
            "commit": revision.stdout.strip() if revision.returncode == 0 else None,
            "dirty": bool(dirty.stdout) if dirty.returncode == 0 else None,
        },
        "raw_stats": stats,
    }
    if config.pim_rank_rope:
        result["schema_version"] = 10
        result["rank_rope_model"] = {
            "active_physical_rank": 1, "gpu_reserved_rank": 0,
            "projection_batch": config.pim_horizon // 2, "qk_batch": config.pim_horizon,
            "projection_mac_completion_ck": timings["nCCDAB"],
            "rope_pairs_per_request": 8, "rope_ii_cycles": config.rope_ii_cycles,
            "rope_latency_cycles": config.rope_latency_cycles,
            "rope_request_completion_ck": 7 * config.rope_ii_cycles + config.rope_latency_cycles,
            "cast_cycles_per_32B": config.pim_cast_cycles,
            "rank_reduce_cycles": config.rank_reduce_cycles,
            "input_buffers_bytes_per_bank": 64, "coefficient_buffer_bytes_per_bank": 32,
            "output_fifo_bytes_per_bank": 64, "query_buffer_bytes_per_bank": 32 * config.pim_horizon,
            "local_read_completion": "LOCALRD issue + nCL + nBL_min; native array timing, no external DQ",
            "local_write_completion": "LOCALWR issue + nLOCALWR + nWR; native array timing, no external DQ",
            "query_export": "native RD and local buffer tap; callback after actual read return",
            "query_remote": "peer channel ADDRESS xor 32 export must complete before WRGB to one bank buffer",
            "query_ownership": "global_query=2*local_token+channel_parity; paired channels own alternate tokens",
            "numerical_compute": False,
            "latency_basis": "configured research-unit timing; not synthesized or measured hardware",
        }
        result["pim_model"]["mac_completion"] = "projection issue+nCCDAB; QK issue+nCCDQK; rank reduction then result export"
        result["bgk_model"]["BG_reduction"] = "rank1 sixteen-bank reduction; explicit configurable latency per 16-query K-vector"
        if config.pim_horizon != 16:
            result["bgk_model"]["BG_reduction"] = f"rank1 sixteen-bank reduction; explicit configurable latency per {config.pim_horizon}-query K-vector"
        result["bgk_model"]["RoPE_conversion_packing"] = result["rank_rope_model"]
        result["traffic_bytes"].update({
            "bgk_internal_array_read": counters.get("MACQK", 0) * 16 * transaction_bytes,
            "local_array_read": counters.get("LOCALRD", 0) * transaction_bytes,
            "pim_bank_local_write": counters.get("LOCALWR", 0) * transaction_bytes,
            "q_export_external_read": requests.get("pim_q_export", 0) * transaction_bytes,
            "q_remote_external_write": requests.get("pim_q_remote", 0) * transaction_bytes,
            "rank_complete_result": requests["pim_rd_bg"] * transaction_bytes,
            "external_dq_total": (counters.get("RD", 0) + counters.get("RDA", 0) +
                counters.get("WR", 0) + counters.get("WRA", 0) + counters.get("WRGB", 0) +
                counters.get("RDRES", 0)) * transaction_bytes,
        })
    else:
        if not config.pim_ffn_dram and not config.pim_rank_staging:
            result["issued_commands"].pop("LOCALRD", None)
        for key in ("pim_rank_rope", "rope_ii_cycles", "rope_latency_cycles", "pim_cast_cycles", "rank_reduce_cycles"):
            result["config"].pop(key, None)
    if config.pim_ffn_dram:
        result["schema_version"] = 11
        result["ffn_dram_model"] = {
            "active_physical_rank": 1, "z_preinstalled": config.ffn_z_preinstalled,
            "z_base_bytes_per_bank": 360448, "z_payload_bytes_per_bank": 32 * config.pim_horizon,
            "z_reserved_bytes_per_bank": 2048, "rank_resident_z_sram_bytes": 0,
            "rank_nonlinear_input_staging_bytes": 128, "rank_nonlinear_output_staging_bytes": 32,
            "bank_staging_bytes": 32, "input_double_buffer_bytes_per_rank": 64 * config.pim_horizon,
            "rank_port_cycles_per_32B": config.pim_local_transfer_cycles,
            "store": "128B G/U gather via rank port, excluded nonlinear value boundary,32B return, native LOCALWR+nWR",
            "store_overlap": "Rank materialization staging released at RETURN completion; independent32B bank staging remains until LOCALWR payload consumed",
            "load": "native LOCALRD+nCL+nBL_min,32B rank-port gather/transpose; all16 matching bank columns required before MAC",
            "mac_completion": "Legacy issue retirement with explicit workload WAIT16; buffer remains owned for nCCDAB",
            "nonlinear_latency_ck": None, "numerical_compute": False,
            "latency_basis": "Configured research timing model; not synthesized/measured nonlinear hardware",
        }
        result["traffic_bytes"].update({
            "ffn_z_array_write": counters.get("LOCALWR", 0) * transaction_bytes,
            "ffn_z_array_read": counters.get("LOCALRD", 0) * transaction_bytes,
            "ffn_gu_local_gather": counters.get("LOCALWR", 0) * 128,
            "ffn_z_local_return": counters.get("LOCALWR", 0) * transaction_bytes,
            "ffn_z_local_load": counters.get("LOCALRD", 0) * transaction_bytes,
            "external_dq_total": sum(counters.get(op, 0) for op in ("RD", "RDA", "WR", "WRA", "WRGB", "RDRES")) * transaction_bytes,
        })
        if config.pim_horizon != 16:
            blocks = (config.pim_horizon + 15) // 16
            result["ffn_dram_model"].update({
                "z_token_blocks": blocks,
                "z_physical_payload_bytes_per_bank": 512 * blocks,
                "store": "Per source column, gather valid FP32 G/U in whole32B rank-port bursts; excluded nonlinear boundary;32B padded return and native LOCALWR+nWR",
                "load": "Native LOCALRD+nCL+nBL_min and32B rank-port transpose for all16 banks and token blocks; discard tail padding before compact H-query input",
                "mac_completion": "Legacy issue retirement with explicit workload WAIT H; buffer remains owned for nCCDAB",
            })
            def sum_stat(value, key):
                if isinstance(value, dict):
                    return sum(int(item) if name == key else sum_stat(item, key) for name, item in value.items())
                if isinstance(value, list):
                    return sum(sum_stat(item, key) for item in value)
                return 0
            result["traffic_bytes"]["ffn_gu_local_gather"] = sum_stat(stats, "ffn_gu_gather_bytes")
    else:
        result["config"].pop("pim_ffn_dram", None)
        result["config"].pop("ffn_z_preinstalled", None)
    if config.pim_horizon == 16:
        result["config"].pop("pim_horizon", None)  # Preserve frozen H16 result schema.
    if config.pim_unified_gemv:
        result["schema_version"] = 13
        result["config"].update(pim_horizon=config.pim_horizon,
                                pim_ffn_dram=config.pim_ffn_dram,
                                ffn_z_preinstalled=config.ffn_z_preinstalled)
        result["local_input_model"]["enabled"] = bool(requests["pim_local_gb"])
        result["pim_model"].update({
            "transfer_scheduling": "AttAcc shared WRGB/RDRES movement path with BL/CCDS/CCDL issue constraints; "
                "ordinary RD/WR retain CL/CWL; WRGB and RDRES complete after nBL_min; "
                "WRGB fanout has no added latency; RDRES cannot overtake queued MACs and waits for bank compute completion",
            "mac_completion": "Native completion: eight CK for Q/K projection and token-sharded V at [131072,163840), "
                "sixteen CK for legacy V, PV, Gate/Up and GPU-fed Down; "
                "the bank PE is shared by both input slots",
            "double_buffer": "One physical pair of 512B GEMV slots per bank; inactive-slot GPU writes may overlap MAC; "
                "full token/lane coverage and completed consumption required before use/overwrite; explicit release",
            "result_buffer_read_latency_ck": max(timings.get("nPIMRdLatency", 0), timings["nBL_min"]),
            "bank_input_multicast": {"enabled": True, "targets_per_addressed_rank": 16,
                "extra_fanout_latency_ck": 0,
                "completion": "All selected bank input rows become ready at WRGB issue+nBL_min"},
        })
        result["unified_gemv_model"] = {
            "bank_slots": 2, "slot_bytes": 512, "input_bytes_per_bank": 1024,
            "rank_full_input_sram_bytes": 0, "rank_ingress_staging_bytes": 32,
            "rank_port_cycles_per_32B": config.gemv_port_cycles,
            "multicast_targets": 16,
            "external_multicast_extra_cycles": 0,
            "external_multicast": "One external32B WRGB burst writes the same token row in all16 bank GEMV slots at DQ completion; "
                "no extra internal-port service or separate source transpose",
            "port_sharing": "Bank-local/remote Q fills retain one configured rank port; "
                "ordinary GPU WRGB multicast does not reserve that port. Legacy DRAM-Z movements use it only when pim_ffn_dram is enabled",
            "z_transpose": ("Legacy DRAM-Z: source-bank column scatter into token rows, charged to rank port"
                            if config.pim_ffn_dram else None),
            "ffn_dataflow": ("Legacy DRAM-Z path enabled" if config.pim_ffn_dram else
                "Gate/Up FP32 outputs return to GPU; GPU GeLU/elementwise multiply/BF16 cast is an excluded value-ready boundary; "
                "GPU sends full4096-feature Z to each rank, whose banks multiply resident Down weights"),
            "down_layout": ("Legacy reduction-dimension partition with16 rank partials" if config.pim_ffn_dram else
                "Output partition:64 complete output features per rank,4 per bank;256 input fragments and4 MAC columns per fragment; "
                "no PIM Z DRAM allocation and no cross-rank partial-result reduction"),
            "completion": "Native MAC completion covers eight Q/K/token-sharded-V rows or sixteen legacy-V/PV/Gate/Up/Down rows; slot release is explicit",
            "token_sharded_v_geometry": {"weight_base_bytes": 131072, "weight_end_bytes": 163840,
                "input_rows_per_rank": 8, "mac_columns_per_fragment": 16,
                "mac_service_ck": 8, "accumulator_bytes_per_bank": 512,
                "selection": "Recognized by the MAC weight address; legacy V region remains supported"},
            "numerical_compute": False,
            "assumption": "Ordinary external WRGB broadcasts to16 banks without extra fanout latency, as specified; "
                "configured bank-local Q port service remains an architecture parameter, not synthesized hardware",
        }
        if "ffn_dram_model" in result:
            result["ffn_dram_model"]["input_double_buffer_bytes_per_rank"] = 0
            result["ffn_dram_model"]["mac_completion"] = "Native MAC completion; unified per-bank slots"
    else:
        result["config"].pop("pim_unified_gemv", None)
        result["config"].pop("gemv_port_cycles", None)
    if config.pim_rank_staging:
        result["schema_version"] = 14
        result["bgk_model"].update(local_write_no_CA=False,
            local_write="Bank-local array access with native row/recovery constraints and shared rank CA")
        result["traffic_bytes"].update(
            gpu_input_staging_write=requests.get("stage_input_declared_bytes", 0),
            input_staging_array_read=requests.get("stage_input_read_bursts", requests.get("pim_stage_load", 0))*transaction_bytes,
            input_srf_multicast_source=requests.get("stage_input_read_bursts", requests.get("pim_stage_load", 0))*transaction_bytes,
            input_srf_multicast_destination=requests.get("stage_input_read_bursts", requests.get("pim_stage_load", 0))*transaction_bytes*16,
            local_array_read=counters.get("LOCALRD", 0)*transaction_bytes)
        result["completion_policy"] = (
            "Input ST completes at native WR issue+CWL+BLmax+WR; SEAL waits for all declared columns; "
            "staged local reads and SRF multicast complete within nCL+nBL_min with rank CA arbitration; "
            "native per-bank GEMV fill/consume/release and global BARRIER completion fences")
        result["pim_model"].update(
            input_buffer_write_latency_ck=timings["nCL"]+timings["nBL_min"],
            buffer_transfer_basis="GPU-to-bank0 ordinary WR; bank0-to-SRF and rank SRF multicast via internal LOCALRD")
        result["pim_model"]["transfer_scheduling"] = (
            "GPU broadcast inputs first use ordinary WR to dedicated bank0 DRAM; all writes complete before local loads. "
            "Rank controller expands local reads and SRF multicast; all local read/broadcast/compute commands share rank CA.")
        result["pim_model"]["double_buffer"] = (
            "One physical pair of512B GEMV slots per bank; rank-local staging reads may overlap MAC on the other slot, "
            "subject to the shared rank command bus and source-bank row hazards.")
        result["pim_model"]["bank_input_multicast"].update(
            completion="After source-bank DRAM column read; bank-to-SRF and rank multicast latency included in nCL+nBL_min",
            source="Dedicated rank1 BG0 bank0 DRAM staging")
        result["unified_gemv_model"].update(
            external_multicast="Entire operator input written once per rank to bank0 DRAM before any SRF fill; no direct GPU-to-SRF multicast",
            input_staging_bank=0, input_staging_base_bytes=524288, input_staging_capacity_bytes=131072,
            input_staging_location="DRAM, separate from the unchanged1KiB per-bank GEMV SRAM",
            port_sharing="Staging local read, SRF broadcast and compute arbitrate on one rank command bus",
            down_layout="Output partition:64 complete output features per rank,4 per bank; complete128KiB GPU Z staged in bank0 DRAM",
            assumption="Bank-to-SRF and intra-rank SRF multicast data latency included in DRAM column-read completion; distinct command arbitration still charged")
        result["rank_controller_model"] = dict(enabled=config.pim_rank_controller,
            local_read_and_broadcast_extra_data_latency_ck=0,
            input_array_read_completion_ck=timings["nCL"]+timings["nBL_min"],
            shared_rank_commands=["single-bank array commands", "SRF multicast", "PIM compute"])
    if config.pim_staging_banks == 2:
        result["pim_model"]["buffer_transfer_basis"] = (
            "GPU ordinary WR to two source banks; source-bank-to-SRF and rank SRF multicast via internal LOCALRD")
        result["pim_model"]["transfer_scheduling"] = (
            "GPU broadcast inputs first use ordinary WR to rank1 BG0/B0 and BG1/B0 DRAM; "
            "all writes complete before local loads. Fragments alternate source banks; "
            "local read/broadcast/compute commands share rank CA.")
        result["pim_model"]["bank_input_multicast"]["source"] = "Dedicated rank1 BG0/B0 and BG1/B0 DRAM staging"
        result["unified_gemv_model"].pop("input_staging_bank", None)
        result["unified_gemv_model"].update(
            external_multicast="Entire operator input written once per rank across two source banks before any SRF fill; no direct GPU-to-SRF multicast",
            input_staging_banks=[0, 4], input_staging_capacity_bytes_per_bank=65536,
            input_staging_layout="fragment f maps to BG=f%2, bank0, physical offset=524288+(floor(f/2)*rows+token)*32",
            down_layout="Output partition:64 complete output features per rank,4 per bank; complete128KiB GPU Z staged across two64KiB source-bank regions")
    else:
        # Preserve result metadata of the single-bank and direct-input baselines.
        result["config"].pop("pim_staging_banks", None)
    if config.pim_rank_descriptors or config.channel_independent_dispatch:
        result["rank_dispatch_model"] = dict(
            rank_descriptors=config.pim_rank_descriptors,
            channel_independent_dispatch=config.channel_independent_dispatch,
            channel_queue_capacity=config.channel_queue_capacity if config.channel_independent_dispatch else None,
            descriptor_children=16, descriptor_completion="All sixteen bank operations complete",
            requests_total_basis="Trace requests including one parent per rank descriptor; child operations counted separately",
            GEMV_compute_instructions="Unchanged original PIM_MAC_AB and PIM_MAC_QK")
    if not config.pim_rank_descriptors:
        result["config"].pop("pim_rank_descriptors", None)
    if not config.channel_independent_dispatch:
        result["config"].pop("channel_independent_dispatch", None)
        result["config"].pop("channel_queue_capacity", None)
    if not config.pim_rank_staging:
        result["config"].pop("pim_rank_staging", None)
    if not config.pim_rank_controller:
        result["config"].pop("pim_rank_controller", None)
    if config.pim_whole_srf:
        result["unified_gemv_model"].update(bank_slots="dynamic logical views", slot_bytes="256 or 512",
            input_bytes_per_bank=1024, shared_fp32_output_bytes_per_bank=2048,
            view_partition={"eight_rows": {"views": 4, "bytes": 256}, "sixteen_rows": {"views": 2, "bytes": 512}},
            output_capacity_enforced=True, whole_batch_consumption_enforced=True)
        result["pim_model"]["double_buffer"] = "One 1KiB physical SRF; fill all valid 256B/512B views before compute; consume the complete batch before any overwrite"
    if config.pim_token_pv:
        # V local conversion also uses this unit without the RoPE pipeline.
        result["config"]["pim_cast_cycles"] = config.pim_cast_cycles
        def v_stat(value, key):
            if isinstance(value, dict):
                return (value.get(key, 0) if isinstance(value.get(key, 0), (int, float)) else 0) + sum(v_stat(v, key) for k, v in value.items() if k != key)
            if isinstance(value, list):
                return sum(v_stat(v, key) for v in value)
            return 0
        result["v_cache_model"] = dict(cache_base_bytes=163840, cache_end_bytes=180224,
            cache_order="[local_KV_fragment31][feature16][token_lane16] BF16",
            valid_prefix_tokens_per_rank=484, valid_suffix_tokens_per_rank=8, padded_tokens_per_rank=496,
            preinstalled=config.v_cache_preinstalled, fused_qkv=config.pim_qkv_fusion,
            prefix_precondition="Existing prefix resident in token-parity layout",
            suffix_update="Per bank/feature: LOCALRD32 -> gather8 FP32 (32B) -> cast8 BF16 (16B) -> LOCALWR32 RMW",
            preserved_mask=15, updated_mask=4080, zero_mask=61440,
            bank_rmw_staging_bytes=32, gather_cycles=config.gemv_port_cycles, cast_cycles=config.pim_cast_cycles,
            partial_output_elements_per_rank=16*256, cross_rank_partial_sum="GPU; arithmetic latency excluded",
            qkv_weight_base_bytes=393216 if config.pim_qkv_fusion else None,
            qkv_columns=48 if config.pim_qkv_fusion else None, numerical_compute=False)
        result["traffic_bytes"].update(v_cache_rmw_array_read=v_stat(stats, "v_cache_array_reads")*32,
            v_cache_array_write=v_stat(stats, "v_cache_array_writes")*32,
            v_cache_fp32_gather=v_stat(stats, "v_cache_gather_bytes"),
            v_cache_bf16_cast_output=v_stat(stats, "v_cache_cast_bytes"))
    for field in ("pim_token_pv", "pim_qkv_fusion", "pim_whole_srf", "v_cache_preinstalled"):
        if not getattr(config, field):
            result["config"].pop(field, None)
    (output / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    return result
