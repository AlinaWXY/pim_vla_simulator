# Published whole-horizon sweep

Start with [REPORT.md](REPORT.md) for timing, complete external-command-free
phases, layout and buffer capacities. [comparison.csv](comparison.csv) contains
the same numeric comparison; [summary.json](summary.json) retains stage and layout
details for all four horizons.

`horizon-sweep-1876849.tar.gz` is the complete native result archive, expanding to
`sweep-1876849/`. It includes24 audited case runs, per-channel timelines, source
traces and completion events,96 unit/native checks, exact H16 timing regressions,
source snapshots, Slurm logs and independent channel-window verification.

Verify the downloaded files against [SHA256.json](SHA256.json). After extraction,
`artifact-sha256.json` verifies all2959 individual archived files. The exact runtime
sources are preserved in `runtime-source.tar.gz`; see artifact-notes.json for the
small publication-only provenance improvement to the runner.

The original H16 implementation and results remain frozen under tag
`xinyao_1027_h16`, commit `41199cee64c40c55cafb330ed838d7e5c13fc0b8`.

The zero-transfer-phase statistic excludes every gap inside input/output phases.
It has not deducted boundary guards or proven concurrent GPU service. Full AE
end-to-end time, nonlinear/GPU arithmetic and background VLM contention remain
unmeasured.
