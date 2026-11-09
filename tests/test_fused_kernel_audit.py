"""Independent saved-timeline checks for coarse fused execution results."""
import copy
import unittest

from tools.audit_fused_kernels import MODEL, audit_timeline


def interval(start, end, **extra):
    return dict(start_ck=start, end_ck=end, start_us=start/1000, end_us=end/1000, **extra)


def fixture():
    phases = []
    for sid, spans in ((0, (("upload", 0, 10), ("compute", 10, 30), ("join", 30, 35), ("download", 35, 40))),
                       (1, (("transfer", 50, 60),)),
                       (2, (("upload", 60, 70), ("compute", 70, 90), ("join", 90, 95), ("download", 95, 100)))):
        for name, start, end in spans:
            ready = [end]*16 if name == "join" else [max(start, end-1)]*15 + [end]
            phases.append(interval(start, end, phase=name, stage_id=sid, channel_ready_ck=ready))
    stages = [dict(stage_id=i, name=name, step=0, layer=0, start_us=a/1000,
                   end_us=b/1000, duration_us=(b-a)/1000)
              for i, name, a, b in ((0, "first", 0, 40), (1, "handoff", 50, 60), (2, "second", 60, 100))]
    durations = [25, 5, 20, 7]
    units = [dict(ordinal=i, name=f"kernel-{i}", kind="kernel", duration_cycles=d, us=d/1000)
             for i, d in enumerate(durations)]
    kernels = [interval(start, start+unit["duration_cycles"], **unit, duration_us=unit["us"])
               for start, unit in zip((10, 70, 75, 110), units)]
    ae_gpu = [interval(40, 50, name="nonlinear", step=0, layer=0), interval(100, 110, name="final", step=0, layer=0)]
    schedule = dict(events=[dict(kind="pim", stage="first", step=0, layer=0),
                            dict(kind="gpu", name="nonlinear", us=.01, step=0, layer=0),
                            dict(kind="pim", stage="handoff", step=0, layer=0),
                            dict(kind="pim", stage="second", step=0, layer=0),
                            dict(kind="gpu", name="final", us=.01, step=0, layer=0)])
    program = dict(execution_model=MODEL, stages=[dict(name=name, channels=copy.deepcopy([ops]*16)) for name, ops in
                   (("first", [dict(op="MAC_RUN", children=[dict(op="LOCALRD")])]),
                    ("handoff", [dict(op="WR")]), ("second", [dict(op="ROPE_RUN", children=[dict(op="LOCALRD")])]))])
    ledger = dict(kind="calibrated_vlm_durations", clock_ps=1000, total_cycles=sum(durations), units=units)
    raw = dict(execution_model=MODEL, clock_ps=1000, stages=stages, phases=phases, kernels=kernels, ae_gpu=ae_gpu,
               metrics=dict(total_us=.117, AE_elapsed_us=.110, VLM_remaining_us=.007, AE_VLM_join_us=.010,
                            AE_GPU_wait_us=0, serial_epoch_reset=False, vlm_calibrated_sum_us=.057))
    return raw, schedule, ledger, program


class FusedKernelAuditTests(unittest.TestCase):
    def test_valid_compute_join_transfer_and_tail(self):
        result = audit_timeline(*fixture())
        self.assertEqual(result["total_cycles"], 117)
        self.assertEqual(result["AE_cycles"], 110)
        self.assertEqual(result["join_cycles"], 10)
        self.assertEqual(result["native_PIM_cycles"], 80)
        self.assertEqual(result["AE_GPU_cycles"], 20)

    def test_rejects_fractional_or_changed_calibrated_duration(self):
        for duration in (25.0, 24):
            raw, schedule, ledger, program = fixture()
            ledger["units"][0]["duration_cycles"] = duration
            with self.assertRaisesRegex(ValueError, "invalid cycle|duration changed"):
                audit_timeline(raw, schedule, ledger, program)

    def test_rejects_native_kernel_preemption_or_extension(self):
        raw, schedule, ledger, program = fixture()
        raw["kernels"][0].update(end_ck=36, end_us=.036)
        with self.assertRaisesRegex(ValueError, "duration changed or kernel was preempted"):
            audit_timeline(raw, schedule, ledger, program)

    def test_rejects_upload_overlap(self):
        raw, schedule, ledger, program = fixture()
        raw["kernels"][0].update(start_ck=5, start_us=.005, end_ck=30, end_us=.030)
        with self.assertRaisesRegex(ValueError, "overlaps exclusive"):
            audit_timeline(raw, schedule, ledger, program)

    def test_rejects_start_at_compute_end(self):
        raw, schedule, ledger, program = fixture()
        raw["kernels"][0].update(start_ck=30, start_us=.030, end_ck=55, end_us=.055)
        with self.assertRaisesRegex(ValueError, "overlaps exclusive|eligible"):
            audit_timeline(raw, schedule, ledger, program)

    def test_rejects_crossing_last_channel_barrier(self):
        raw, schedule, ledger, program = fixture()
        raw["phases"][0]["channel_ready_ck"][15] = 11
        with self.assertRaisesRegex(ValueError, "channel barrier early"):
            audit_timeline(raw, schedule, ledger, program)

    def test_rejects_missing_channel_barrier_record(self):
        raw, schedule, ledger, program = fixture()
        raw["phases"][1]["channel_ready_ck"].pop()
        with self.assertRaisesRegex(ValueError, "channel barrier early"):
            audit_timeline(raw, schedule, ledger, program)

    def test_rejects_phase_reordering(self):
        raw, schedule, ledger, program = fixture()
        raw["phases"][0]["phase"] = "download"
        with self.assertRaisesRegex(ValueError, "phase sequence"):
            audit_timeline(raw, schedule, ledger, program)

    def test_rejects_wrong_join_sum(self):
        raw, schedule, ledger, program = fixture()
        raw["metrics"]["AE_VLM_join_us"] = .009
        with self.assertRaisesRegex(ValueError, "join accounting"):
            audit_timeline(raw, schedule, ledger, program)

    def test_rejects_tail_double_counting(self):
        raw, schedule, ledger, program = fixture()
        raw["metrics"]["VLM_remaining_us"] = .017
        with self.assertRaisesRegex(ValueError, "completion/tail identity"):
            audit_timeline(raw, schedule, ledger, program)

    def test_rejects_AE_GPU_duration_change(self):
        raw, schedule, ledger, program = fixture()
        schedule["events"][1]["us"] = .011
        with self.assertRaisesRegex(ValueError, "AE measured GPU duration changed"):
            audit_timeline(raw, schedule, ledger, program)

    def test_rejects_unexplained_idle_inside_compute_window(self):
        raw, schedule, ledger, program = fixture()
        raw["kernels"][1].update(start_ck=71, start_us=.071, end_ck=76, end_us=.076)
        raw["kernels"][2].update(start_ck=76, start_us=.076, end_ck=96, end_us=.096)
        # Keep the new kernel end outside export to isolate the launch policy.
        raw["kernels"][2]["duration_cycles"] = 19
        raw["kernels"][2]["duration_us"] = .019
        raw["kernels"][2].update(end_ck=95, end_us=.095)
        ledger["units"][2].update(duration_cycles=19, us=.019)
        ledger["total_cycles"] = 56
        raw["metrics"]["vlm_calibrated_sum_us"] = .056
        with self.assertRaisesRegex(ValueError, "eligible compute/tail opportunity"):
            audit_timeline(raw, schedule, ledger, program)


if __name__ == "__main__":
    unittest.main()
