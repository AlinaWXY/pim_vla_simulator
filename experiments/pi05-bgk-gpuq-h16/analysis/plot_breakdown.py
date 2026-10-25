"""Render a standalone scientific figure from the verified additive table."""
import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("breakdown_json", type=Path)
    args = parser.parse_args()
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    data = json.loads(args.breakdown_json.read_text(encoding="utf-8"))
    rows = data["primary_rows"]
    groups = [
        ("Input only", ("gpu_input_only_us","local_input_only_us"), "#4d83b8"),
        ("Q DRAM read", ("q_dram_read_only_us",), "#496585"),
        ("Q local rebroadcast", ("q_broadcast_only_us",), "#6a9db9"),
        ("Array service + column guard", ("mac_array_only_us","array_column_only_us"), "#327b68"),
        ("Overlap", ("overlap_us",), "#88b5a2"),
        ("Local writeback", ("local_write_only_us",), "#e4a84f"),
        ("DRAM read / write", ("dram_read_only_us","dram_write_only_us"), "#af783f"),
        ("Output to GPU boundary", ("gpu_output_only_us",), "#9e6b9d"),
        ("Local gather", ("local_gather_only_us","bg_reduce_only_us"), "#d296aa"),
        ("Other / boundary", ("other_us","boundary_us"), "#a8adb5"),
    ]
    labels = {"qkv":"Fused QKV + cache I/O", "k":"K projection", "v":"V projection", "qk":"QK^T", "pv":"PV",
              "gate_up":"FFN Gate + Up", "down_tiled":"FFN Down"}
    plt.rcParams.update({"font.family":"DejaVu Sans","font.size":10,"svg.fonttype":"none"})
    fig, ax = plt.subplots(figsize=(11,5.8), constrained_layout=True)
    left = [0.0] * len(rows)
    for label, columns, color in groups:
        values = [sum(row.get(c,0.0) for c in columns) for row in rows]
        ax.barh(range(len(rows)), values, left=left, color=color, label=label, height=0.58)
        left = [a+b for a,b in zip(left,values)]
    for i,(row,total) in enumerate(zip(rows,left)):
        assert abs(total-row["total_us"]) < 1e-8
        ax.text(total+0.25,i,f"{total:.3f}",va="center",fontsize=10)
    ax.set_yticks(range(len(rows)),[labels.get(r["operator"],r["operator"]) for r in rows])
    ax.invert_yaxis()
    ax.set_xlabel("Native modeled service per layer and denoise step (microseconds)")
    ax.set_xlim(0,max(left)*1.14)
    ax.xaxis.grid(True,alpha=.2)
    ax.set_axisbelow(True)
    ax.spines[["top","right","left"]].set_visible(False)
    ax.tick_params(axis="y",length=0)
    ax.set_title("Pi0.5 H=16, 8 heads: shared-X QKV, GPU RoPE, DRAM Q relay",loc="left",pad=28,weight="bold")
    ax.text(0,1.025,"16 LPDDR channels | 16 action tokens | 984 valid KV tokens | refresh enabled",
            transform=ax.transAxes,color="#525962",fontsize=10)
    ax.legend(loc="upper center",bbox_to_anchor=(.5,-.17),ncol=3,frameon=False,fontsize=9)
    fig.text(.01,-.035,
        f"Measured PIM subset x18 layers x10 steps: {data['measured_pim_subset_ms_18layers_10steps']:.3f} ms. "
        "GPU RoPE/repack, O, softmax, nonlinear and reduction compute remain unmeasured.\n"
        "Exclusive colors sum to wall time. Array reads are included with MAC service. This is not complete AE latency.",
        fontsize=9,color="#525962")
    base = args.breakdown_json.parent / "linear-latency-breakdown"
    for suffix in ("png","svg"):
        fig.savefig(base.with_suffix("."+suffix),dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
