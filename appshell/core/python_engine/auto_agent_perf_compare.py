from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return data if isinstance(data, dict) else {}


def number(value: Any) -> float | None:
    if isinstance(value, bool) or value is None:
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def pct_drop(before: Any, after: Any) -> float | None:
    before_num = number(before)
    after_num = number(after)
    if before_num is None or after_num is None or before_num == 0:
        return None
    return (before_num - after_num) / before_num


def format_num(value: Any) -> str:
    numeric = number(value)
    if numeric is None:
        return "missing"
    if numeric.is_integer():
        return str(int(numeric))
    return f"{numeric:.4f}".rstrip("0").rstrip(".")


def format_pct(value: float | None) -> str:
    if value is None:
        return "missing"
    return f"{value * 100:.2f}%"


def top_stages(timings: dict[str, Any]) -> list[dict[str, Any]]:
    stages = timings.get("top_slowest_stages")
    if not isinstance(stages, list):
        return []
    return [item for item in stages if isinstance(item, dict)]


def metric_row(name: str, before: dict[str, Any], after: dict[str, Any]) -> str:
    before_value = before.get(name)
    after_value = after.get(name)
    return f"| `{name}` | {format_num(before_value)} | {format_num(after_value)} | {format_pct(pct_drop(before_value, after_value))} |"


def build_markdown(before_dir: Path, after_dir: Path) -> str:
    before_summary = load_json(before_dir / "summary.json")
    after_summary = load_json(after_dir / "summary.json")
    before_timings = load_json(before_dir / "timings_summary.json")
    after_timings = load_json(after_dir / "timings_summary.json")

    before_p95 = before_summary.get("p95_total_ms")
    after_p95 = after_summary.get("p95_total_ms")
    p95_drop = pct_drop(before_p95, after_p95)
    before_avg = before_summary.get("avg_total_ms")
    after_avg = after_summary.get("avg_total_ms")
    avg_drop = pct_drop(before_avg, after_avg)

    acceptance = "pass" if p95_drop is not None and p95_drop >= 0.20 else "needs_explanation"
    stability_metrics = ["success_rate", "accepted_rate", "rollback_manifest_created_rate"]
    stability_ok = all(
        (number(after_summary.get(name)) is not None and number(before_summary.get(name)) is not None and number(after_summary.get(name)) >= number(before_summary.get(name)))
        for name in stability_metrics
    )

    lines = [
        "# Auto Agent Performance Before/After",
        "",
        f"- before_dir: `{before_dir}`",
        f"- after_dir: `{after_dir}`",
        f"- avg_total_ms_drop: `{format_pct(avg_drop)}`",
        f"- p95_total_ms_drop: `{format_pct(p95_drop)}`",
        f"- p95_acceptance: `{acceptance}`",
        f"- safety_rates_not_decreased: `{stability_ok}`",
        "",
        "## Summary Metrics",
        "",
        "| metric | before | after | drop |",
        "|---|---:|---:|---:|",
        metric_row("avg_total_ms", before_summary, after_summary),
        metric_row("p95_total_ms", before_summary, after_summary),
        metric_row("success_rate", before_summary, after_summary),
        metric_row("accepted_rate", before_summary, after_summary),
        metric_row("rollback_manifest_created_rate", before_summary, after_summary),
        metric_row("fallback_rate", before_summary, after_summary),
        "",
        "## Top Slowest Stages",
        "",
        "| rank | before stage | before avg ms | after stage | after avg ms |",
        "|---:|---|---:|---|---:|",
    ]
    before_top = top_stages(before_timings)
    after_top = top_stages(after_timings)
    for idx in range(3):
        before_item = before_top[idx] if idx < len(before_top) else {}
        after_item = after_top[idx] if idx < len(after_top) else {}
        lines.append(
            "| {rank} | `{before_stage}` | {before_avg} | `{after_stage}` | {after_avg} |".format(
                rank=idx + 1,
                before_stage=before_item.get("stage", "missing"),
                before_avg=format_num(before_item.get("avg_ms")),
                after_stage=after_item.get("stage", "missing"),
                after_avg=format_num(after_item.get("avg_ms")),
            )
        )

    remaining = after_timings.get("dominant_area")
    if acceptance == "needs_explanation":
        lines.extend(
            [
                "",
                "## Acceptance Note",
                "",
                f"`p95_total_ms` did not drop by 20%. Remaining dominant area: `{remaining or 'missing'}`.",
            ]
        )
    return "\n".join(lines) + "\n"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compare Auto Agent live benchmark before/after summaries.")
    parser.add_argument("--before-dir", type=Path, required=True)
    parser.add_argument("--after-dir", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(build_markdown(args.before_dir.resolve(), args.after_dir.resolve()), encoding="utf-8")
    print(f"Wrote performance comparison to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
