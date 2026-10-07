"""Aggregate <tag>_threshold_sweep.csv and <tag>_latency.csv of several detectors
into results/model_comparison.csv and a side-by-side figure."""
from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SUM_COLUMNS = ["gt_cars", "gt_cars_moderate", "pred_cars", "pred_cars_in_fov",
               "pred_cars_outside_fov", "matched_gt_cars", "matched_gt_moderate", "unmatched_pred_in_fov"]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--tags", nargs="+", default=["pointpillars", "second"])
    ap.add_argument("--results", type=Path, default=Path("results"))
    args = ap.parse_args()
    tables = []
    for tag in args.tags:
        sweep = pd.read_csv(args.results / f"{tag}_threshold_sweep.csv", dtype={"frame_id": str})
        table = sweep.groupby("score_thr")[SUM_COLUMNS].sum().reset_index()
        table["center_recall"] = table.matched_gt_cars / table.gt_cars
        table["center_recall_moderate"] = table.matched_gt_moderate / table.gt_cars_moderate
        latency = pd.read_csv(args.results / f"{tag}_latency.csv")
        for column, name in (("latency_file_ms", "file"), ("latency_in_memory_ms", "in_memory")):
            table[f"latency_{name}_p50_ms"] = np.percentile(latency[column], 50)
            table[f"latency_{name}_p95_ms"] = np.percentile(latency[column], 95)
        table.insert(0, "model", tag)
        tables.append(table)
    result = pd.concat(tables, ignore_index=True)
    result.to_csv(args.results / "model_comparison.csv", index=False, float_format="%.4f")
    print(result[["model", "score_thr", "pred_cars_in_fov", "unmatched_pred_in_fov", "matched_gt_cars",
                  "center_recall", "center_recall_moderate", "latency_in_memory_p50_ms"]].to_string(index=False))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for tag, table in zip(args.tags, tables):
        axes[0].plot(table.score_thr, table.center_recall, "o-", label=f"{tag}: recall (all GT)")
        axes[0].plot(table.score_thr, table.center_recall_moderate, "^--", label=f"{tag}: recall (moderate)")
        axes[1].plot(table.score_thr, table.unmatched_pred_in_fov, "o-", label=f"{tag}: unmatched in FOV")
    axes[0].set(xlabel="Score threshold", ylabel="Center recall (2 m)", ylim=(0, 1.05), title="Recall")
    axes[1].set(xlabel="Score threshold", ylabel="Boxes", title="Unmatched predictions inside camera FOV")
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(args.results / "figures" / "model_comparison.png", dpi=150)
    plt.close(fig)


if __name__ == "__main__":
    main()
