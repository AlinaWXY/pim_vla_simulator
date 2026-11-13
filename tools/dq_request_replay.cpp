// Replay integer 32 B requests on the calibrated DQ-only timeline. Runs of
// equal-direction bursts are batched arithmetically without splitting a burst.
#include "../third_party/nlohmann_json/json.hpp"
#include <algorithm>
#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <stdexcept>
#include <string>
#include <vector>

namespace {
using Json = nlohmann::json;
using Clock = int64_t;
using Count = int64_t;
using Counts = std::array<Count, 2>;
constexpr int channels = 16;
constexpr Clock burst_ck = 2;
constexpr Clock unbounded = std::numeric_limits<Clock>::max() / 4;
enum Direction { Read, Write, Open, Unset, Conservative };
using Edges = std::array<std::array<int, 3>, 3>;

void check(bool ok, const char* message) {
  if (!ok) throw std::runtime_error(message);
}

Clock round_up(double time) {
  return static_cast<Clock>(std::ceil(time - 1e-7));
}

struct Unit {
  Counts total;
  double nominal;
};

struct Gap {
  Clock start, end, open, close;
  int left, right;
};

struct Stage {
  std::string name;
  Clock duration;
  bool pim;
  std::array<std::vector<Gap>, channels> gaps;
};

struct Channel {
  Counts original{}, remaining{};
  double interval = 0, ready = 0;
  Clock burst_end = 0;
  int previous = Unset;
};

struct Progress {
  std::array<Channel, channels> lane{};
  size_t ordinal = 0;
  Clock start = 0;
  bool active = false;
};

struct Slot {
  int hint = Unset, retries = 0, first = Unset, last = Unset, assumed = Unset;
};

struct Pass {
  Progress progress;
  Counts issued{};
  Clock active_ck = 0, stop = 0;
  std::vector<Clock> completed_durations;
};

void launch(Progress& progress, const Unit& unit, Clock start) {
  progress.start = start;
  progress.active = true;
  for (int c = 0; c < channels; ++c) {
    auto& lane = progress.lane[c];
    lane = {};
    for (int d = 0; d < 2; ++d)
      lane.original[d] = lane.remaining[d] = (unit.total[d] + channels - 1 - c) / channels;
    const auto count = lane.original[Read] + lane.original[Write];
    lane.interval = count ? std::max(double(burst_ck), unit.nominal / count) : 0;
    lane.ready = start;
    lane.burst_end = start;
  }
}

Count sendable(const Channel& lane, int direction) {
  if (direction == Read || !lane.original[Read]) return lane.remaining[direction];
  const auto reserved = (lane.remaining[Read] * lane.original[Write] + lane.original[Read] - 1)
                        / lane.original[Read];
  return std::max<Count>(0, lane.remaining[Write] - reserved);
}

int choose(const Channel& lane) {
  if (!sendable(lane, Read)) return Write;
  if (!sendable(lane, Write)) return Read;
  return lane.previous == Write ? Write : Read;
}

using Slots = std::array<std::vector<Slot>, channels>;

Pass run_window(const Stage& stage, Clock start, const std::vector<Unit>& units,
                Progress progress, Slots& slots, const Edges& edges, bool admit,
                bool record_durations) {
  const auto end = start + stage.duration;
  Pass out;
  out.stop = start;
  std::array<size_t, channels> indices{};
  Clock cursor = start;
  while (cursor < end && progress.ordinal < units.size() && (admit || progress.active)) {
    const auto& unit = units[progress.ordinal];
    if (!progress.active) launch(progress, unit, cursor);
    auto finish = progress.start + round_up(unit.nominal);
    bool all_done = true;
    for (int c = 0; c < channels; ++c) {
      auto& lane = progress.lane[c];
      auto& index = indices[c];
      const auto& gaps = stage.gaps[c];
      while (lane.remaining[Read] + lane.remaining[Write] > 0 && index < gaps.size()) {
        const auto& gap = gaps[index];
        auto& slot = slots[c][index];
        int direction = choose(lane);
        const bool unused = slot.first == Unset;
        const auto earliest = std::max({cursor, start + gap.open, lane.burst_end,
                                        unused ? start + gap.start + edges[gap.left][direction] : start});
        const auto assumed = unused ? (slot.hint == Unset ? direction : slot.hint) : slot.assumed;
        const auto guard = assumed == Conservative
            ? std::max(edges[Read][gap.right], edges[Write][gap.right])
            : edges[assumed][gap.right];
        const auto last_start = start + std::min(gap.close, gap.end - guard) - burst_ck;
        // Profile pacing carries the solo cost of VLM-to-VLM turnarounds.
        // Only edges against the native AE trace add explicit guards here.
        while (sendable(lane, direction)) {
          const double origin = std::max(lane.ready, double(earliest));
          if (round_up(origin) > last_start) break;
          const auto capacity = static_cast<Count>(std::floor((last_start - origin + 1e-7) / lane.interval)) + 1;
          const auto count = std::min(capacity, sendable(lane, direction));
          check(count > 0, "nonpositive DQ run length");
          lane.burst_end = round_up(origin + (count - 1) * lane.interval) + burst_ck;
          lane.ready = origin + count * lane.interval;
          lane.remaining[direction] -= count;
          out.issued[direction] += count;
          lane.previous = direction;
          if (slot.first == Unset) {
            slot.first = direction;
            slot.assumed = assumed;
          }
          slot.last = direction;
          direction = 1 - direction;
        }
        if (lane.remaining[Read] + lane.remaining[Write] > 0) ++index;
      }
      if (lane.remaining[Read] + lane.remaining[Write] > 0) all_done = false;
      finish = std::max(finish, lane.burst_end);
    }
    if (!all_done || finish > end) {
      out.active_ck += end - cursor;
      out.stop = end;
      break;
    }
    out.active_ck += finish - cursor;
    out.stop = cursor = finish;
    if (record_durations) out.completed_durations.push_back(finish - progress.start);
    ++progress.ordinal;
    progress.active = false;
  }
  out.progress = progress;
  return out;
}

Pass resolve(const Stage& stage, Clock start, const std::vector<Unit>& units,
             const Progress& progress, const Edges& edges, bool admit,
             bool record_durations = false) {
  Slots slots;
  for (int c = 0; c < channels; ++c) slots[c].resize(stage.gaps[c].size());
  for (;;) {
    auto pass = run_window(stage, start, units, progress, slots, edges, admit, record_durations);
    bool retry = false;
    for (int c = 0; c < channels; ++c) {
      for (size_t i = 0; i < slots[c].size(); ++i) {
        auto& slot = slots[c][i];
        const auto right = stage.gaps[c][i].right;
        if (slot.last != Unset && slot.assumed != Conservative
            && edges[slot.last][right] != edges[slot.assumed][right]) {
          slot.hint = slot.retries++ < 2 ? slot.last : Conservative;
          retry = true;
        }
        slot.first = slot.last = slot.assumed = Unset;
      }
    }
    if (!retry) return pass;
  }
}

Stage open_dq() {
  Stage stage{"open", unbounded, false, {}};
  for (auto& gaps : stage.gaps) gaps.push_back({0, unbounded, 0, unbounded, Open, Open});
  return stage;
}

void self_test(const Edges& edges) {
  Stage pim{"pim", 13, true, {}};
  for (auto& gaps : pim.gaps) gaps.push_back({0, 13, 0, 13, Write, Write});
  const std::vector<Unit> units{{{32, 0}, 4}, {{16, 0}, 2}};
  auto first = resolve(pim, 0, units, {}, edges, true);
  check(first.issued == Counts{} && first.progress.active && first.progress.ordinal == 0,
        "DQ turnaround admitted a burst into a short gap");
  const auto open = open_dq();
  auto join = resolve(open, 13, units, first.progress, edges, false);
  check(join.stop == 17 && join.issued == Counts{32, 0} && join.progress.ordinal == 1
        && !join.progress.active, "GPU handoff ran beyond the active unit");
  auto tail = resolve(open, 17, units, join.progress, edges, true);
  check(tail.stop == 19 && tail.issued == Counts{16, 0}, "tail accounting differs");

  pim.duration = 30;
  for (auto& gaps : pim.gaps) gaps = {{0, 30, 0, 30, Write, Write}};
  const auto fit = resolve(pim, 0, units, {}, edges, true);
  check(fit.issued == Counts{48, 0} && fit.stop == 17 && fit.progress.ordinal == 2,
        "unexpected request placement in a guarded gap");

  pim.duration = 2;
  for (auto& gaps : pim.gaps) gaps = {{0, 2, 0, 2, Open, Open}};
  const std::vector<Unit> slow{{{16, 0}, 8}, {{16, 0}, 2}};
  const auto floor = resolve(pim, 0, slow, {}, edges, true);
  check(floor.issued == Counts{16, 0} && floor.progress.active, "kernel floor retired too early");
  check(resolve(open, 2, slow, floor.progress, edges, false).stop == 8,
        "GPU handoff ignored the profile floor");

  pim.duration = 19;
  for (auto& gaps : pim.gaps) gaps = {{0, 13, 0, 13, Open, Write}};
  const auto blocked = resolve(pim, 0, units, {}, edges, true);
  check(blocked.progress.ordinal == 2 && blocked.stop == 6,
        "sequential kernels in one gap did not complete in order");
}

Json replay(const std::vector<Stage>& stages, const std::vector<Unit>& units,
            const Edges& edges, int repeats, bool serial) {
  const auto open = open_dq();
  const auto solo = resolve(open, 0, units, {}, edges, true, true);
  check(solo.progress.ordinal == units.size(), "VLM solo did not finish");
  double profile = 0;
  Count expected_reads = 0, expected_writes = 0;
  for (const auto& unit : units) {
    profile += unit.nominal;
    expected_reads += unit.total[Read];
    expected_writes += unit.total[Write];
  }
  check(solo.issued == Counts{expected_reads, expected_writes}, "VLM solo lost requests");
  Progress progress;
  Clock now = 0, ae_solo = 0, join_ck = 0, overlap_ck = 0, vlm_end = -1;
  Counts total{}, pim_counts{}, join_counts{};
  std::vector<Counts> by_stage(stages.size());
  std::vector<Clock> overlap_by_stage(stages.size());
  auto account = [&](const Pass& pass, Counts& bucket) {
    progress = pass.progress;
    for (int d = 0; d < 2; ++d) {
      total[d] += pass.issued[d];
      bucket[d] += pass.issued[d];
    }
    if (progress.ordinal == units.size() && vlm_end < 0) vlm_end = pass.stop;
  };
  for (int repeat = 0; repeat < repeats; ++repeat) {
    for (size_t i = 0; i < stages.size(); ++i) {
      const auto& stage = stages[i];
      if (!serial && stage.pim && progress.ordinal < units.size()) {
        const auto pass = resolve(stage, now, units, progress, edges, true);
        account(pass, by_stage[i]);
        for (int d = 0; d < 2; ++d) pim_counts[d] += pass.issued[d];
        overlap_by_stage[i] += pass.active_ck;
        overlap_ck += pass.active_ck;
      } else if (!stage.pim && progress.active) {
        const auto pass = resolve(open, now, units, progress, edges, false);
        check(!pass.progress.active && pass.progress.ordinal == progress.ordinal + 1,
              "AE GPU handoff must finish only the current VLM unit");
        account(pass, join_counts);
        join_ck += pass.stop - now;
        now = pass.stop;
      }
      now += stage.duration;
      ae_solo += stage.duration;
    }
  }
  const auto ae_end = now;
  const auto at_ae_end = progress.ordinal;
  Counts tail_counts{};
  if (progress.ordinal < units.size()) {
    const auto tail = resolve(open, now, units, progress, edges, true);
    account(tail, tail_counts);
    now = tail.stop;
  }
  check(total == solo.issued && progress.ordinal == units.size(), "concurrent replay lost requests");
  check(!serial || (pim_counts == Counts{} && now == ae_solo + solo.stop),
        "serial replay differs from independent solos");
  check(ae_end == ae_solo + join_ck && now == std::max(ae_end, vlm_end),
        "E2E does not reconcile with AE and VLM endpoints");
  auto rows = Json::array();
  for (size_t i = 0; i < stages.size(); ++i)
    if (stages[i].pim) rows.push_back({{"name", stages[i].name}, {"event_index", i},
                                     {"requests", by_stage[i]}, {"overlap_ck", overlap_by_stage[i]}});
  return {{"total_ck", now}, {"ae_ck", ae_end}, {"ae_solo_ck", ae_solo}, {"join_ck", join_ck},
          {"tail_ck", now - ae_end}, {"vlm_end_ck", vlm_end}, {"overlap_ck", overlap_ck},
          {"profile_ck", profile}, {"vlm_solo_ck", solo.stop}, {"units_at_ae_end", at_ae_end},
          {"pim_requests", pim_counts}, {"join_requests", join_counts}, {"tail_requests", tail_counts},
          {"total_requests", total}, {"stages", std::move(rows)}};
}
} // namespace

int main(int argc, char** argv) {
  try {
    check(argc == 4, "usage: dq_request_replay input repeats serial");
    std::ifstream input(argv[1]);
    input.exceptions(std::ios::failbit | std::ios::badbit);
    const int repeats = std::stoi(argv[2]);
    const bool serial = std::stoi(argv[3]);
    check(repeats > 0, "repeats must be positive");
    Edges edges{};
    for (auto& row : edges) for (auto& cost : row) input >> cost;
    size_t unit_count, stage_count;
    input >> unit_count >> stage_count;
    check(unit_count > 0 && stage_count > 0, "replay input is empty");
    std::vector<Unit> units(unit_count);
    for (auto& unit : units) {
      input >> unit.total[Read] >> unit.total[Write] >> unit.nominal;
      check(unit.total[Read] >= 0 && unit.total[Write] >= 0 && unit.nominal > 0,
            "invalid VLM work unit");
    }
    std::vector<Stage> stages(stage_count);
    for (auto& stage : stages) {
      input >> std::quoted(stage.name) >> stage.duration >> stage.pim;
      check(stage.duration >= 0, "negative AE stage length");
      if (!stage.pim) continue;
      for (auto& gaps : stage.gaps) {
        size_t size;
        input >> size;
        gaps.resize(size);
        for (auto& g : gaps) {
          input >> g.start >> g.end >> g.open >> g.close >> g.left >> g.right;
          check(g.start <= g.open && g.open < g.close && g.close <= g.end
                && g.open >= 0 && g.close <= stage.duration
                && g.left >= Read && g.left <= Open && g.right >= Read && g.right <= Open,
                "invalid DQ window");
        }
      }
    }
    self_test(edges);
    std::cout << replay(stages, units, edges, repeats, serial).dump() << '\n';
  } catch (const std::exception& error) {
    std::cerr << "dq_request_replay: " << error.what() << '\n';
    return 1;
  }
}
