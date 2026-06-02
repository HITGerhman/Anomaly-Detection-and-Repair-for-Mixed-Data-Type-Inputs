from __future__ import annotations

import argparse
import json
import textwrap
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
from matplotlib.patches import FancyBboxPatch, Rectangle


REPO_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT_DIR = Path(__file__).resolve().parent / "slide10_figures"

DATASET_ORDER = ["stroke", "orders_transactions", "user_device_logs"]
DATASET_LABELS = {
    "stroke": "stroke",
    "orders_transactions": "orders_transactions",
    "user_device_logs": "user_device_logs",
    "orders_transactions_1m_labeled": "1M labeled",
    "orders_transactions_10m_labeled": "10M labeled",
}

C = {
    "ink": "#17202A",
    "muted": "#52606D",
    "pale": "#F4F7FA",
    "line": "#D7DEE8",
    "blue": "#2458A6",
    "teal": "#168A78",
    "amber": "#B36B00",
    "amber_fill": "#FFF4E5",
    "red": "#B42318",
    "white": "#FFFFFF",
    "finding_fill": "#EEF5FF",
}


def fmt3(value: object) -> str:
    return f"{float(value):.3f}"


def require_path(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(f"Required input is missing: {path}")
    return path


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10.5,
            "figure.dpi": 120,
            "savefig.dpi": 300,
            "axes.linewidth": 0,
        }
    )


def make_panel(title: str, subtitle: str | None = None, size: tuple[float, float] = (6.4, 3.55)):
    fig = plt.figure(figsize=size)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=C["white"], edgecolor="none"))
    ax.add_patch(
        FancyBboxPatch(
            (0.018, 0.018),
            0.964,
            0.964,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            facecolor=C["white"],
            edgecolor=C["line"],
            linewidth=1.15,
        )
    )
    ax.add_patch(Rectangle((0.018, 0.952), 0.964, 0.018, facecolor=C["teal"], edgecolor="none"))
    ax.text(0.06, 0.89, title, ha="left", va="center", color=C["ink"], fontsize=18, fontweight="bold")
    if subtitle:
        ax.text(0.06, 0.815, subtitle, ha="left", va="center", color=C["blue"], fontsize=10.5, fontweight="bold")
    return fig, ax


def save(fig, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, facecolor=C["white"])
    plt.close(fig)


def draw_table(
    ax,
    headers: list[str],
    rows: list[list[str]],
    *,
    x: float,
    y_top: float,
    width: float,
    row_h: float,
    col_weights: list[float],
    aligns: list[str] | None = None,
    font_size: float = 10.2,
    header_size: float = 10.3,
) -> None:
    total_weight = sum(col_weights)
    col_widths = [width * weight / total_weight for weight in col_weights]
    aligns = aligns or ["left"] + ["right"] * (len(headers) - 1)
    table_rows = [headers] + rows

    for row_idx, row in enumerate(table_rows):
        y = y_top - (row_idx + 1) * row_h
        left = x
        is_header = row_idx == 0
        for col_idx, raw_cell in enumerate(row):
            cell = str(raw_cell)
            col_w = col_widths[col_idx]
            ax.add_patch(
                Rectangle(
                    (left, y),
                    col_w,
                    row_h,
                    facecolor=C["pale"] if is_header else C["white"],
                    edgecolor=C["line"],
                    linewidth=0.8,
                )
            )
            align = aligns[col_idx]
            if align == "right":
                tx = left + col_w - 0.014
            elif align == "center":
                tx = left + col_w / 2
            else:
                tx = left + 0.014
            ax.text(
                tx,
                y + row_h / 2,
                cell,
                ha=align,
                va="center",
                color=C["ink"],
                fontsize=header_size if is_header else font_size,
                fontweight="bold" if is_header else "normal",
                linespacing=1.15,
            )
            left += col_w


def draw_note(
    ax,
    text: str,
    *,
    x: float,
    y: float,
    width: float,
    height: float,
    fill: str,
    edge: str,
    color: str,
    fontsize: float = 10.0,
    bold_prefix: str | None = None,
    wrap_width: int = 72,
) -> None:
    ax.add_patch(
        FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.006,rounding_size=0.014",
            facecolor=fill,
            edgecolor=edge,
            linewidth=0.9,
        )
    )
    if bold_prefix and text.startswith(bold_prefix):
        prefix_w = min(0.19, width * 0.32)
        ax.text(
            x + 0.018,
            y + height / 2,
            bold_prefix,
            ha="left",
            va="center",
            color=color,
            fontsize=fontsize,
            fontweight="bold",
        )
        ax.text(
            x + 0.018 + prefix_w,
            y + height / 2,
            text[len(bold_prefix) :].strip(),
            ha="left",
            va="center",
            color=color,
            fontsize=fontsize,
        )
    else:
        wrapped = textwrap.fill(text, width=wrap_width)
        ax.text(
            x + 0.018,
            y + height / 2,
            wrapped,
            ha="left",
            va="center",
            color=color,
            fontsize=fontsize,
            linespacing=1.18,
        )


def load_controlled_accuracy(repo_root: Path) -> tuple[list[list[str]], str]:
    path = require_path(repo_root / "artifacts" / "experiments" / "cross_dataset" / "summary_detection_metrics.csv")
    df = pd.read_csv(path)
    overall = df[df["issue_type"] == "Overall"].set_index("dataset")
    numeric = df[df["issue_type"] == "numeric_outlier"].set_index("dataset")
    rows = []
    for dataset in DATASET_ORDER:
        item = overall.loc[dataset]
        rows.append([DATASET_LABELS[dataset], fmt3(item["precision"]), fmt3(item["recall"]), fmt3(item["f1"])])
    stroke_fp = int(numeric.loc["stroke", "fp"])
    orders_fp = int(numeric.loc["orders_transactions", "fp"])
    note = f"Numeric-outlier FPs are the main lower-precision source: stroke={stroke_fp}; orders_tx={orders_fp}."
    return rows, note


def load_repair_effect(repo_root: Path) -> list[list[str]]:
    path = require_path(repo_root / "artifacts" / "experiments" / "cross_dataset" / "summary_repair_metrics.csv")
    df = pd.read_csv(path)
    overall = df[df["issue_type"] == "Overall"].set_index("dataset")
    rows = []
    for dataset in DATASET_ORDER:
        item = overall.loc[dataset]
        rows.append([DATASET_LABELS[dataset], fmt3(item["exact_rate"]), fmt3(item["improved_or_exact_rate"])])
    return rows


def load_large_scale_stability(repo_root: Path) -> list[list[str]]:
    path = require_path(repo_root / "outputs" / "stability_reprobe_20260517" / "reprobe_summary.json")
    data = json.loads(path.read_text(encoding="utf-8"))
    runs = {run["run"]: run for run in data["auto_runs"]}
    required = ["medium_auto_500k", "auto_1m_streaming", "auto_10m_streaming_incremental_retry"]
    missing = [name for name in required if name not in runs]
    if missing:
        raise KeyError(f"Missing expected auto runs in {path}: {missing}")

    ten_m = runs["auto_10m_streaming_incremental_retry"]
    ten_m_seconds = float(ten_m["timings_ms"]["total_duration_ms"]) / 1000.0
    ten_m_gb = float(ten_m["output_size_bytes"]) / 1_000_000_000.0
    return [
        ["500k", "completed", "scan / preview / write / validation"],
        ["1M", "completed", "streaming write + rollback metadata"],
        ["10M", f"{ten_m_seconds:.3f} s\n{ten_m_gb:.2f} GB output", "affected-column validation"],
    ]


def load_labeled_scale_validation(repo_root: Path) -> list[list[str]]:
    path = require_path(
        repo_root
        / "artifacts"
        / "experiments"
        / "large_labeled_validation"
        / "summary_detection_metrics.csv"
    )
    df = pd.read_csv(path)
    overall = df[df["issue_type"] == "Overall"].set_index("dataset")
    rows = []
    for dataset in ["orders_transactions_1m_labeled", "orders_transactions_10m_labeled"]:
        item = overall.loc[dataset]
        rows.append([DATASET_LABELS[dataset], f"{int(item['gt'])}", f"{int(item['tp'])}", fmt3(item["recall"])])
    return rows


def render_controlled_accuracy(repo_root: Path, output_dir: Path) -> None:
    rows, note = load_controlled_accuracy(repo_root)
    fig, ax = make_panel("Controlled Accuracy Baseline", "Controlled Datasets")
    draw_table(
        ax,
        ["Dataset", "Precision", "Recall", "F1"],
        rows,
        x=0.06,
        y_top=0.765,
        width=0.88,
        row_h=0.105,
        col_weights=[2.25, 1.05, 0.95, 0.8],
        aligns=["left", "right", "right", "right"],
    )
    draw_note(
        ax,
        note,
        x=0.06,
        y=0.105,
        width=0.88,
        height=0.12,
        fill=C["pale"],
        edge=C["line"],
        color=C["muted"],
    )
    save(fig, output_dir / "controlled_accuracy_baseline.png")


def render_repair_effect(repo_root: Path, output_dir: Path) -> None:
    rows = load_repair_effect(repo_root)
    fig, ax = make_panel("Repair Effect", "Repair quality on repairable injected cells")
    draw_table(
        ax,
        ["Dataset", "Exact Rate", "Improved-or-Exact"],
        rows,
        x=0.06,
        y_top=0.765,
        width=0.88,
        row_h=0.105,
        col_weights=[2.35, 1.05, 1.45],
        aligns=["left", "right", "right"],
    )
    draw_note(
        ax,
        "Repair improves part of the corrupted data, but does not guarantee true-value restoration.",
        x=0.06,
        y=0.105,
        width=0.88,
        height=0.12,
        fill=C["pale"],
        edge=C["line"],
        color=C["muted"],
    )
    save(fig, output_dir / "repair_effect.png")


def render_large_scale_stability(repo_root: Path, output_dir: Path) -> None:
    rows = load_large_scale_stability(repo_root)
    fig, ax = make_panel("Large-scale Stability", "Real AppShell workflow execution")
    draw_table(
        ax,
        ["Scale", "Result", "Evidence"],
        rows,
        x=0.06,
        y_top=0.765,
        width=0.88,
        row_h=0.118,
        col_weights=[0.85, 1.45, 2.25],
        aligns=["left", "left", "left"],
        font_size=9.6,
    )
    draw_note(
        ax,
        "This verifies workflow execution at scale, not ground-truth repair accuracy.",
        x=0.06,
        y=0.105,
        width=0.88,
        height=0.12,
        fill=C["finding_fill"],
        edge="#BFD4F2",
        color=C["blue"],
    )
    save(fig, output_dir / "large_scale_stability.png")


def render_labeled_scale_validation(repo_root: Path, output_dir: Path) -> None:
    rows = load_labeled_scale_validation(repo_root)
    fig, ax = make_panel("Labeled Scale Validation", "Supplementary ground-truth scale check")
    draw_table(
        ax,
        ["Dataset", "Injected", "Recalled", "Recall"],
        rows,
        x=0.06,
        y_top=0.765,
        width=0.88,
        row_h=0.118,
        col_weights=[1.75, 1.05, 1.05, 0.95],
        aligns=["left", "right", "right", "right"],
    )
    draw_note(
        ax,
        "Limitation: numeric outlier thresholds caused many false positives.",
        x=0.06,
        y=0.105,
        width=0.88,
        height=0.13,
        fill=C["amber_fill"],
        edge="#E5A33B",
        color=C["amber"],
        fontsize=10.3,
        bold_prefix="Limitation:",
    )
    save(fig, output_dir / "labeled_scale_validation.png")


def render_key_finding(output_dir: Path) -> None:
    fig = plt.figure(figsize=(12.8, 1.35))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), 1, 1, facecolor=C["white"], edgecolor="none"))
    ax.add_patch(
        FancyBboxPatch(
            (0.02, 0.13),
            0.96,
            0.74,
            boxstyle="round,pad=0.006,rounding_size=0.018",
            facecolor=C["finding_fill"],
            edgecolor="#BFD4F2",
            linewidth=1.2,
        )
    )
    ax.add_patch(Rectangle((0.02, 0.13), 0.012, 0.74, facecolor=C["blue"], edgecolor="none"))
    ax.text(0.055, 0.5, "Key Finding:", ha="left", va="center", color=C["blue"], fontsize=19, fontweight="bold")
    ax.text(
        0.205,
        0.5,
        "The system shows strong recall and complete workflow execution,\n"
        "but numeric outlier detection needs domain-specific threshold tuning.",
        ha="left",
        va="center",
        color=C["ink"],
        fontsize=16.5,
        linespacing=1.22,
    )
    save(fig, output_dir / "key_finding_bar.png")


def generate(repo_root: Path, output_dir: Path) -> list[Path]:
    configure_matplotlib()
    output_dir.mkdir(parents=True, exist_ok=True)
    render_controlled_accuracy(repo_root, output_dir)
    render_repair_effect(repo_root, output_dir)
    render_large_scale_stability(repo_root, output_dir)
    render_labeled_scale_validation(repo_root, output_dir)
    render_key_finding(output_dir)
    return [
        output_dir / "controlled_accuracy_baseline.png",
        output_dir / "repair_effect.png",
        output_dir / "large_scale_stability.png",
        output_dir / "labeled_scale_validation.png",
        output_dir / "key_finding_bar.png",
    ]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate slide 10 experiment-result figure assets.")
    parser.add_argument("--repo-root", type=Path, default=REPO_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    paths = generate(args.repo_root.resolve(), args.output_dir.resolve())
    for path in paths:
        print(path)


if __name__ == "__main__":
    main()
