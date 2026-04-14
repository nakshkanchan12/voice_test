from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean, median
from typing import Any

LOWER_IS_BETTER_METRICS: tuple[str, ...] = (
    "speech_to_first_audio_p50_ms",
    "speech_to_first_audio_p95_ms",
    "speech_to_done_p50_ms",
    "speech_to_done_p95_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
    "mean_asr_queue_wait_ms",
    "mean_tts_queue_wait_ms",
)

HIGHER_IS_BETTER_METRICS: tuple[str, ...] = (
    "lt_700ms_first_audio_hit_rate_pct",
)

ALL_METRICS: tuple[str, ...] = LOWER_IS_BETTER_METRICS + HIGHER_IS_BETTER_METRICS


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _pct_improvement(
    baseline_http: float | None,
    candidate_webrtc: float | None,
    *,
    higher_is_better: bool,
) -> float | None:
    if baseline_http is None or candidate_webrtc is None:
        return None

    if baseline_http == 0:
        if candidate_webrtc == 0:
            return 0.0
        return None

    if higher_is_better:
        return round(((candidate_webrtc - baseline_http) / abs(baseline_http)) * 100.0, 2)

    return round(((baseline_http - candidate_webrtc) / abs(baseline_http)) * 100.0, 2)


def _load_artifact(path: str) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def build_comparison_report(pairs: list[dict[str, Any]]) -> dict[str, Any]:
    pair_reports: list[dict[str, Any]] = []

    for pair in pairs:
        label = str(pair.get("label", "")).strip() or "pair"
        http_artifact = str(pair["http_artifact"])
        webrtc_artifact = str(pair["webrtc_artifact"])

        http_payload = _load_artifact(http_artifact)
        webrtc_payload = _load_artifact(webrtc_artifact)

        http_summary = http_payload.get("summary", {})
        webrtc_summary = webrtc_payload.get("summary", {})

        metrics: dict[str, dict[str, float | None]] = {}
        for metric in ALL_METRICS:
            http_value = _to_float(http_summary.get(metric))
            webrtc_value = _to_float(webrtc_summary.get(metric))
            improvement_pct = _pct_improvement(
                http_value,
                webrtc_value,
                higher_is_better=metric in HIGHER_IS_BETTER_METRICS,
            )

            delta = None
            if http_value is not None and webrtc_value is not None:
                delta = round(webrtc_value - http_value, 2)

            metrics[metric] = {
                "http": http_value,
                "webrtc": webrtc_value,
                "delta_webrtc_minus_http": delta,
                "improvement_pct": improvement_pct,
            }

        pair_reports.append(
            {
                "label": label,
                "http_artifact": http_artifact,
                "webrtc_artifact": webrtc_artifact,
                "pair_metadata": pair.get("metadata", {}),
                "http_transport_config": http_payload.get("transport_config", {}),
                "webrtc_transport_config": webrtc_payload.get("transport_config", {}),
                "metrics": metrics,
            }
        )

    aggregate_mean: dict[str, float | None] = {}
    aggregate_median: dict[str, float | None] = {}

    for metric in ALL_METRICS:
        values = [
            pair_report["metrics"][metric]["improvement_pct"]
            for pair_report in pair_reports
            if pair_report["metrics"][metric]["improvement_pct"] is not None
        ]

        if not values:
            aggregate_mean[metric] = None
            aggregate_median[metric] = None
            continue

        aggregate_mean[metric] = round(mean(values), 2)
        aggregate_median[metric] = round(median(values), 2)

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "transport_ab_comparison",
        "pair_count": len(pair_reports),
        "lower_is_better_metrics": list(LOWER_IS_BETTER_METRICS),
        "higher_is_better_metrics": list(HIGHER_IS_BETTER_METRICS),
        "aggregate_improvement_pct_mean": aggregate_mean,
        "aggregate_improvement_pct_median": aggregate_median,
        "pairs": pair_reports,
    }


def render_markdown(report: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# Transport A/B Comparison Report")
    lines.append("")
    lines.append(f"- Generated at: {report['generated_at']}")
    lines.append(f"- Pairs analyzed: {report['pair_count']}")
    lines.append("")
    lines.append("## Aggregate Improvement (%): mean")
    lines.append("")
    lines.append("| Metric | Improvement % |")
    lines.append("|---|---:|")
    for metric in ALL_METRICS:
        value = report["aggregate_improvement_pct_mean"].get(metric)
        rendered = "n/a" if value is None else f"{value:.2f}%"
        lines.append(f"| {metric} | {rendered} |")

    lines.append("")
    lines.append("## Pair Details")
    lines.append("")

    for pair in report["pairs"]:
        lines.append(f"### {pair['label']}")
        lines.append("")
        lines.append(f"- HTTP artifact: {pair['http_artifact']}")
        lines.append(f"- WebRTC artifact: {pair['webrtc_artifact']}")
        lines.append("")
        lines.append("| Metric | HTTP | WebRTC | Delta (W-H) | Improvement % |")
        lines.append("|---|---:|---:|---:|---:|")

        for metric in ALL_METRICS:
            metric_row = pair["metrics"][metric]
            http_value = metric_row["http"]
            webrtc_value = metric_row["webrtc"]
            delta = metric_row["delta_webrtc_minus_http"]
            improvement = metric_row["improvement_pct"]

            def _render(v: float | None) -> str:
                return "n/a" if v is None else f"{v:.2f}"

            rendered_improvement = "n/a" if improvement is None else f"{improvement:.2f}%"
            lines.append(
                f"| {metric} | {_render(http_value)} | {_render(webrtc_value)} | {_render(delta)} | {rendered_improvement} |"
            )

        lines.append("")

    return "\n".join(lines)


def _save_report(report: dict[str, Any], markdown: str, output_json: str, output_md: str) -> None:
    json_path = Path(output_json)
    md_path = Path(output_md)

    json_path.parent.mkdir(parents=True, exist_ok=True)
    md_path.parent.mkdir(parents=True, exist_ok=True)

    json_path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    md_path.write_text(markdown, encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Compute % improvements from paired HTTP/WebRTC artifacts.")
    parser.add_argument("--manifest", type=str, required=True, help="Path to pair manifest JSON.")
    parser.add_argument("--output-json", type=str, required=True)
    parser.add_argument("--output-md", type=str, required=True)
    return parser.parse_args()


def _load_manifest(path: str) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    pairs = payload.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("Manifest must contain a list under 'pairs'.")
    return [pair for pair in pairs if isinstance(pair, dict)]


if __name__ == "__main__":
    args = parse_args()
    pairs = _load_manifest(args.manifest)
    report = build_comparison_report(pairs)
    markdown = render_markdown(report)
    _save_report(report, markdown, args.output_json, args.output_md)

    print(f"pair_count={report['pair_count']}")
    print(f"report_json={args.output_json}")
    print(f"report_md={args.output_md}")
