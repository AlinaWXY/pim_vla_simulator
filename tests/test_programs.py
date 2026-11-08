"""Tensor coverage, real traffic and bounded storage tests (no timing oracle)."""
import copy
import json
import unittest
from programs.gen_pim_program import VARIANTS, EXTERNAL, generate, walk, validate_ops
from workloads import layout as L


def audit_transient_array_producers(program):
    """Independent address coverage audit; initialized weights/prefix are excluded."""
    producers = {}
    for stage in program["stages"]:
        name = stage["name"]
        for op in walk(stage["channels"][0]):
            addr = op["addr"]
            offset = addr[4] * 2048 + addr[5] * 32
            if op["op"] in ("WR", "LOCALWR"):
                producers[tuple(addr)] = (name, op["op"], op.get("source", ""))
            elif op["op"] in ("RD", "LOCALRD", "MACAB", "MACQK"):
                expected = None
                if L.STAGING_BASE <= offset < L.STAGING_BASE + 65536:
                    expected = "pv" if name == "o" else name
                elif L.Q_IMPORT_BASE <= offset < L.Q_IMPORT_BASE + 2048:
                    expected = "q_exchange_import"
                elif L.SCORE_BASE <= offset < L.SCORE_BASE + 16384:
                    expected = "qkt"
                elif L.Q_BASE <= offset < L.K_SUFFIX_BASE + 256:
                    if name in ("rope", "rope_export"):
                        expected = "qkv"
                    else:
                        expected = "rope_import" if program["rope"] == "gpu" else "rope"
                elif op["op"] == "MACAB" and L.V_BASE + 15360 <= offset < L.V_BASE + 15872:
                    expected = "qkv"
                if expected is not None:
                    banks = range(16) if addr[2] == -1 else (addr[2] * 4 + addr[3],)
                    for b in banks:
                        concrete = tuple(addr[:2] + [b // 4, b % 4] + addr[4:])
                        actual = producers.get(concrete)
                        if actual is None or actual[0] != expected:
                            raise ValueError(f"{name}: {op['op']} reads {concrete} without {expected} producer")
                        if name == "o" and L.STAGING_BASE <= offset < L.STAGING_BASE + 65536:
                            if actual != ("pv", "LOCALWR", "pv_bf16"):
                                raise ValueError("O input is stale P data instead of locally written PV output")


class ProgramTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.programs = {v: generate(v, 1) for v in VARIANTS}

    def stage(self, variant, name):
        return next(s for s in self.programs[variant]["stages"] if s["name"] == name)["channels"][0]

    def test_tensor_partition_is_bijective(self):
        self.assertEqual({L.feature_from_lane(b, l) for b in range(16) for l in range(16)}, set(range(256)))
        for f in range(256):
            self.assertEqual(L.feature_from_lane(*L.feature_owner(f)), f)
        owners = {L.token_owner(h, t) for h in range(8) for t in range(992)}
        self.assertEqual(owners, {(c, t) for c in range(16) for t in range(496)})
        for head in range(8):
            shard_a = {L.feature_shard_from_lane(2 * head, b, x) for b in range(16) for x in range(8)}
            shard_b = {L.feature_shard_from_lane(2 * head + 1, b, x) for b in range(16) for x in range(8)}
            self.assertFalse(shard_a & shard_b)
            self.assertEqual(shard_a | shard_b, set(range(256)))
            self.assertEqual(shard_a, set(range(64)) | set(range(128, 192)))

    def test_operator_scalar_mac_coverage_matches_tensor_shapes(self):
        for variant, p in self.programs.items():
            shapes = p["tensor_contract"]["operator_shapes"]
            for name, shape in shapes.items():
                ops = self.stage(variant, name)
                scalar_macs = sum(o["rows"] * 16 * o.get("lanes", 16)
                                  for o in walk(ops) if o["op"] in ("MACAB", "MACQK")) * 16
                self.assertEqual(scalar_macs, shape["scalar_macs_all_channels"], (variant, name))
            # Token KV parity repeats O projection before reducing its two partials.
            self.assertEqual(shapes["o"]["scalar_macs_all_channels"],
                             16 * 2048 * 1024 * (2 if p["layout"] == "token" else 1))

    def test_feature_q_and_k_cover_both_packed_halves(self):
        q = [o for o in self.stage("feature_gpu_before", "q_exchange_export") if o["op"] == "LOCALRD"]
        self.assertEqual({(o["dest_slot"], o["source_lane_offset"]) for o in q},
                         {(i, 8 * (i % 2)) for i in range(16)})
        k = [o for o in self.stage("feature_gpu_before", "qkt") if o["op"] == "MACQK"]
        self.assertEqual({(o["kv_token"], o["weight_lane_offset"]) for o in k},
                         {(i, 8 * (i % 2)) for i in range(992)})

    def test_eighteen_physical_layers_are_disjoint_and_in_bounds(self):
        contract = L.contract("token")
        self.assertEqual(contract["layer_stride_bytes_per_bank"], 1024 * 1024)
        regions = sorted(contract["regions_per_bank"].values())
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(regions, regions[1:])))
        all_regions = sorted((start + layer * L.LAYER_STRIDE_BYTES, end + layer * L.LAYER_STRIDE_BYTES)
                             for layer in range(18) for start, end in regions)
        self.assertTrue(all(a[1] <= b[0] for a, b in zip(all_regions, all_regions[1:])))
        self.assertLessEqual(all_regions[-1][1], 64 * 1024 * 1024)  # 8Gb rank / sixteen banks
        for layer in range(18):
            a = L.layer_address([0, 1, 0, 0, 511, 63], layer)
            self.assertEqual(a[4], layer * 512 + 511)
            self.assertLess(a[4], 32768)

    def test_transient_arrays_have_complete_current_producers(self):
        for p in self.programs.values():
            audit_transient_array_producers(p)
        broken = copy.deepcopy(self.programs["token_local_after"])
        # Same number of writes and bytes, but a duplicated row leaves a hole.
        broken["stages"][0]["channels"][0][0]["addr"][4] += 1
        with self.assertRaisesRegex(ValueError, "without qkv producer"):
            audit_transient_array_producers(broken)
        broken = copy.deepcopy(self.programs["token_local_after"])
        pv = next(s for s in broken["stages"] if s["name"] == "pv")
        missing = next(o for o in walk(pv["channels"][0]) if o.get("source") == "pv_bf16")
        missing["op"] = "BARRIER"
        with self.assertRaisesRegex(ValueError, "stale P data"):
            audit_transient_array_producers(broken)

    def test_dq_conservation_and_historical_tensor_contract(self):
        # Independent established byte totals; timing is deliberately not reused.
        kib = (6096, 5352, 5160, 4968, 4776)
        for (variant, p), expected in zip(self.programs.items(), kib):
            calculated = sum(op["op"] in EXTERNAL for s in p["stages"] for op in walk(s["channels"][0])) * 32
            self.assertEqual(calculated, expected * 1024 // 16, variant)
            self.assertEqual(p["expected_dq_bytes"], calculated)
            self.assertEqual(sum(s["expected_dq_bytes"] for s in p["stages"]), calculated)

    def test_rope_and_layout_are_single_variable_traffic_differences(self):
        b = {v: p["expected_dq_bytes"] for v, p in self.programs.items()}
        self.assertEqual((b["feature_gpu_before"] - b["token_gpu_before"]) * 16, 744 * 1024)
        self.assertEqual((b["token_gpu_before"] - b["token_local_before"]) * 16, 192 * 1024)
        self.assertEqual((b["token_gpu_after"] - b["token_local_after"]) * 16, 192 * 1024)

    def test_streaming_has_no_descriptor_or_staging(self):
        for v in list(VARIANTS)[:3]:
            for s in self.programs[v]["stages"]:
                ops = list(walk(s["channels"][0]))
                self.assertFalse(any(o["op"] in ("MAC_RUN", "ROPE_RUN", "BCAST") for o in ops))
                self.assertFalse(any(o["op"] == "WR" and o.get("epoch", "").endswith("_input") for o in ops))
        self.assertEqual(sum(o["op"] == "WRGB" for o in self.stage("token_local_before", "o")), 512)

    def test_fused_has_six_mac_descriptors_and_optional_rope(self):
        for v in list(VARIANTS)[3:]:
            ops = [o for s in self.programs[v]["stages"] for o in s["channels"][0]]
            self.assertEqual(sum(o["op"] == "MAC_RUN" for o in ops), 6)
            self.assertEqual(sum(o["op"] == "ROPE_RUN" for o in ops), int("local" in v))
            self.assertFalse(any(o["op"] == "WRGB" for o in ops))

    def test_rank_rope_has_one_shared_unit_and_exact_work(self):
        for variant in ("token_local_before", "token_local_after"):
            ops = list(walk(self.stage(variant, "rope")))
            self.assertEqual(sum(o["op"] == "LOCALRD" for o in ops), 384)
            self.assertEqual(sum(o["op"] == "LOCALWR" for o in ops), 256)
            execs = [o for o in ops if o["op"] == "ROPE_EXEC"]
            self.assertEqual(sum(o["pairs"] for o in execs), 2048)
            self.assertEqual({o["shared_unit"] for o in execs}, {"rank_rope"})
            for previous, current in zip(execs, execs[1:]):
                self.assertIn(previous["id"], current["deps"])

    def test_score_leaves_reduction_immediately_without_tile_cache(self):
        for variant in VARIANTS:
            ops = self.stage(variant, "qkt")
            if variant.endswith("after"):
                ops = ops[0]["children"]
            macs = [o for o in ops if o["op"] == "MACQK"]
            self.assertEqual(len(macs), 992 if variant.startswith("feature") else 496)
            for mac in macs:
                red, p0, p1 = ops[mac["id"] + 1:mac["id"] + 4]
                self.assertEqual(red["op"], "REDUCE")
                self.assertFalse(red["cache_result"])
                self.assertEqual([p0["op"], p1["op"]], ["LOCALWR"] * 2 if variant.endswith("after") else ["RDRES"] * 2)
                self.assertEqual(p0["deps"], [red["id"]])

    def test_output_pool_allocation_and_two_o_tile_lifetimes(self):
        for variant, p in self.programs.items():
            for s in p["stages"]:
                allocated = {"qkv": 1536, "pv": 512 if variant.startswith("feature") else 1024,
                             "o": 2048, "gate_up": 2048, "down": 256}.get(s["name"], 2048)
                for o in walk(s["channels"][0]):
                    if o["op"] == "MACAB":
                        self.assertEqual(o["output_bytes_per_bank"], o["rows"] * o["output_columns"] * 4)
                        self.assertLessEqual(o["output_bytes_per_bank"], 2048)
                    for r in o.get("source_regions", []):
                        self.assertLessEqual(r["byte_offset"] + r["bytes"], allocated)
            ops = self.stage(variant, "o")
            if variant.endswith("after"):
                ops = ops[0]["children"]
            first_tile1 = next(o for o in ops if o.get("output_tile") == 1 and o["op"] == "MACAB")
            inputs = [ops[i] for i in first_tile1["deps"]]
            if variant.endswith("after"):
                inputs = [ops[o["deps"][0]] for o in inputs]  # BCAST -> LOCALRD
            barriers = {d for o in inputs for d in o["deps"]}
            self.assertEqual(len(barriers), 1)
            barrier = ops[barriers.pop()]
            self.assertEqual(barrier["op"], "BARRIER")
            self.assertEqual(len(barrier["deps"]), 512)
            self.assertTrue(all(ops[i]["op"] == "RDRES" and ops[i]["output_tile"] == 0 for i in barrier["deps"]))

    def test_q_exchange_is_addressed_memory_not_masked_broadcast(self):
        for variant in ("token_gpu_before", "token_local_before"):
            ops = self.stage(variant, "q_exchange_import")
            writes = [o for o in ops if o["op"] == "WR"]
            self.assertEqual(len(writes), 128)
            self.assertEqual({o["source_channel"] for o in writes}, {1})
            self.assertFalse(any(o["op"] == "WRGB" for o in ops))
            for wr in writes:
                rd = ops[wr["id"] + 1]
                self.assertEqual(rd["op"], "LOCALRD")
                self.assertEqual(rd["deps"], [wr["id"]])
                self.assertEqual(rd["addr"], wr["addr"])

    def test_generator_rejects_capacity_or_dependency_corruption(self):
        ops = copy.deepcopy(self.stage("token_gpu_before", "qkv"))
        ops[0]["slot"] = 32
        with self.assertRaisesRegex(ValueError, "SRF"):
            validate_ops(ops, 0)
        ops[0]["slot"] = 0
        ops[0]["deps"] = [1]
        with self.assertRaisesRegex(ValueError, "earlier"):
            validate_ops(ops, 0)

    def test_program_is_deterministic_and_has_no_timing_fields(self):
        p = self.programs["token_local_after"]
        self.assertEqual(json.dumps(p), json.dumps(generate("token_local_after", 1)))
        forbidden = {"dq_latency", "latency_ck", "nCL", "nCWL", "ready_at", "service_ck"}
        for s in p["stages"]:
            for o in walk(s["channels"][0]):
                self.assertFalse(forbidden.intersection(o))


if __name__ == "__main__":
    unittest.main()
