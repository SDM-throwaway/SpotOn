# USE: python graph_gen.py --input comparison_summary.json --output defense_performance.png --metric asr

import os
import json
import argparse
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

# IEEE figure defaults
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 9,
    "axes.labelsize": 9,
    "axes.titlesize": 10,
    "xtick.labelsize": 8,
    "ytick.labelsize": 8,
    "legend.fontsize": 8,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.linewidth": 0.8,
    "xtick.major.width": 0.8,
    "ytick.major.width": 0.8,
    "lines.linewidth": 0.8,
    "pdf.fonttype": 42,  # TrueType: text is selectable in PDF/LaTeX
    "ps.fonttype": 42,
})

# IEEE double-column figure width: 7.16 in
_FIG_WIDTH = 7.16
_FIG_HEIGHT = 4.0

_HATCHES = ["", "///", "xxx"]
# ColorBrewer sequential blues — safe for deuteranopia/protanopia
_COLORS = ["#9ecae1", "#3182bd", "#08519c"]


def load_data(json_path):
    if not os.path.exists(json_path):
        raise FileNotFoundError(f"Could not find JSON file at: {json_path}")
    with open(json_path, "r") as f:
        return json.load(f)


def parse_metrics(data, metric_key):
    defenses = data.get("defenses", {})

    def get_val(name):
        stat_str = defenses.get(name, {}).get(metric_key, "0.00%")
        return float(stat_str.replace("%", ""))

    baseline_val = get_val("none")
    categories = ["Delimiting", "Datamarking", "Encoding"]
    standard_vals = [get_val("delimiting"), get_val("datamarking"), get_val("encoding")]
    randomized_vals = [
        get_val("randomized_delimiting"),
        get_val("randomized_datamarking"),
        get_val("randomized_encoding"),
    ]
    return baseline_val, categories, standard_vals, randomized_vals


def draw_png_chart(
    baseline_val, categories, standard_vals, randomized_vals, metadata, metric_name, output_path
):
    width = 0.30
    x_baseline = np.array([0])
    x_defenses = np.array([1.5, 3.0, 4.5])

    fig, ax = plt.subplots(figsize=(_FIG_WIDTH, _FIG_HEIGHT))
    fig.patch.set_facecolor("white")
    ax.set_facecolor("white")

    # Baseline bar
    ax.bar(
        x_baseline, baseline_val, width,
        label="Baseline (no defense)",
        color=_COLORS[0], hatch=_HATCHES[0],
        edgecolor="black", linewidth=0.6, zorder=3,
    )
    ax.axhline(y=baseline_val, color="black", linestyle="--", linewidth=0.6, alpha=0.5, zorder=1)

    # Standard and randomized bars
    ax.bar(
        x_defenses - width / 2, standard_vals, width,
        label="Non-randomized",
        color=_COLORS[1], hatch=_HATCHES[1],
        edgecolor="black", linewidth=0.6, zorder=3,
    )
    ax.bar(
        x_defenses + width / 2, randomized_vals, width,
        label="Randomized",
        color=_COLORS[2], hatch=_HATCHES[2],
        edgecolor="black", linewidth=0.6, zorder=3,
    )

    ax.set_ylabel(f"{metric_name} (%)")
    model_name = metadata.get("target_model", "Unknown Model")
    dataset_name = metadata.get("dataset", "Unknown Dataset").capitalize()
    ax.set_title(
        f"Spotlighting Defense Evaluation - {metric_name}\n"
        f"Model: {model_name} | Dataset: {dataset_name}",
        pad=8,
    )

    all_x = list(x_baseline) + list(x_defenses)
    all_labels = ["Baseline\n(No Defense)"] + categories
    ax.set_xticks(all_x)
    ax.set_xticklabels(all_labels)

    ax.legend(frameon=True, facecolor="white", edgecolor="black", loc="upper right")

    max_val = max([baseline_val] + standard_vals + randomized_vals)
    ax.set_ylim(0, min(100, max_val + max_val * 0.20 + 5))

    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.yaxis.grid(True, linestyle="--", linewidth=0.5, alpha=0.4, color="gray", zorder=0)
    ax.set_axisbelow(True)

    def _autolabel(rects):
        for rect in rects:
            h = rect.get_height()
            if np.isnan(h):
                continue
            ax.annotate(
                f"{h:.1f}%",
                xy=(rect.get_x() + rect.get_width() / 2, h),
                xytext=(0, 3),
                textcoords="offset points",
                ha="center", va="bottom", fontsize=7,
            )

    for group in ax.containers:
        _autolabel(group)

    fig.tight_layout()
    plt.savefig(output_path, bbox_inches="tight")
    print(f"[✓] Chart saved to: {output_path}")
    plt.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate IEEE-style comparison chart for BIPIA.")
    parser.add_argument("--input", type=str, default="comparison_summary.json")
    parser.add_argument("--metric", type=str, choices=["asr", "block_rate"], default="asr")
    parser.add_argument("--output", type=str, default="defense_comparison.pdf")
    args = parser.parse_args()

    if not os.path.exists(args.input):
        print(f"File '{args.input}' not found. Generating dummy test JSON...")
        sample_data = {
            "dataset": "email",
            "total_samples_tested": 1125,
            "target_model": "deepseek-v4-flash",
            "defenses": {
                "none":                    {"blocks": 952,  "success_rate": "84.62%", "attack_success_rate": "15.38%"},
                "delimiting":              {"blocks": 1119, "success_rate": "99.47%", "attack_success_rate": "0.53%"},
                "randomized_delimiting":   {"blocks": 1116, "success_rate": "99.20%", "attack_success_rate": "0.80%"},
                "encoding":                {"blocks": 1122, "success_rate": "99.73%", "attack_success_rate": "0.27%"},
                "randomized_encoding":     {"blocks": 1124, "success_rate": "99.91%", "attack_success_rate": "0.09%"},
                "datamarking":             {"blocks": 1032, "success_rate": "91.73%", "attack_success_rate": "8.27%"},
                "randomized_datamarking":  {"blocks": 1037, "success_rate": "92.18%", "attack_success_rate": "7.82%"},
            },
        }
        with open(args.input, "w") as f:
            json.dump(sample_data, f, indent=4)

    if args.metric == "asr":
        metric_key = "attack_success_rate"
        metric_label = "Attack Success Rate (lower is better)"
    else:
        metric_key = "success_rate"
        metric_label = "Defense Block Rate"

    try:
        raw_data = load_data(args.input)
        baseline_val, categories, standard_vals, randomized_vals = parse_metrics(raw_data, metric_key)
        draw_png_chart(
            baseline_val, categories, standard_vals, randomized_vals,
            raw_data, metric_label, args.output,
        )
    except Exception as e:
        print(f"Failed to generate PNG chart: {e}")
