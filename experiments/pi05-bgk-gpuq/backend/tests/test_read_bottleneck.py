"""Independent small timelines check accounting boundaries and demand visibility."""

import pytest

from simulator.read_bottleneck import analyze_channel

TIMING = {"nBL_min": 2, "nCL": 17, "nCCDS": 2, "nCCDL": 4,
          "nRCD": 15, "tCK_ps": 1250}


def event(clock, command="RD", bg=0, row=0):
    return {"clock": str(clock), "command": command, "Rank": "0", "BankGroup": str(bg),
            "Bank": "0", "Row": str(row)}


def test_gap_accounting_and_admission_visibility():
    # Third read arrives at controller CK8, so it is visible from CK9.
    # The empty demand at CK8 and the same-BG spacing overlap: neither is
    # an exclusive causal attribution of the two-cycle RD gap.
    events = [event(4), event(6, bg=1), event(10, bg=1)]
    done = [{"arrive": "0", "depart": "23"}, {"arrive": "1", "depart": "25"},
            {"arrive": "8", "depart": "29"}]
    result = analyze_channel(events, done, TIMING, 32)
    assert result["window_ck"] == 25
    assert (result["dq_busy_ck"], result["rd_gap_ck"], result["boundary_ck"]) == (6, 2, 17)
    assert result["observed_order_column_gap_ck"] == 2
    assert result["no_admitted_read_gap_ck"] == 1
    assert result["admission_to_issue_mean_ck"] == pytest.approx(11 / 3)
    assert result["rd_interval_histogram"] == {2: 1, 4: 1}


def test_changed_row_gap_and_rcd_boundary():
    events = [event(2, "ACT2"), event(17), event(20, "PREpb", row=1),
              event(34, "ACT1", row=1), event(35, "ACT2", row=1), event(50, row=1)]
    done = [{"arrive": "0", "depart": "36"}, {"arrive": "1", "depart": "69"}]
    result = analyze_channel(events, done, TIMING, 32)
    assert result["gaps_ending_on_changed_row_ck"] == 31
    assert result["no_admitted_read_gap_ck"] == 0
    gap = result["largest_gaps"][0]
    assert gap["next_bank_pre_ck"] == 20
    assert gap["next_bank_rcd_ready_ck"] == 50
    assert gap["next_rd_at_rcd_boundary"]


def test_single_read_boundary_and_pim_tail():
    done = [{"arrive": "0", "depart": "36"}]
    result = analyze_channel([event(17), event(10000, "MACHB")], done, TIMING, 32)
    assert result["window_ck"] == 19
    assert result["rd_gap_ck"] == 0
    assert result["min_admitted_unissued_reads_in_gaps"] is None
    assert analyze_channel([event(10000, "MACHB")], [], TIMING, 32) is None


@pytest.mark.parametrize("events,done", [
    ([event(4)], []),
    ([], [{"arrive": "0", "depart": "23"}]),
    ([event(4)], [{"arrive": "0", "depart": "24"}]),
    ([event(4), event(5, bg=1)],
     [{"arrive": "0", "depart": "23"}, {"arrive": "1", "depart": "24"}]),
    ([event(4), event(6)],
     [{"arrive": "0", "depart": "23"}, {"arrive": "1", "depart": "25"}]),
    ([event(4, "WR")], []),
])
def test_reject_inconsistent_or_unsupported_traces(events, done):
    with pytest.raises(ValueError):
        analyze_channel(events, done, TIMING, 32)
