from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def _parse_thresholds(value: str) -> list[float]:
    parts = [token.strip() for token in value.split(",") if token.strip()]
    if not parts:
        raise argparse.ArgumentTypeError("At least one threshold is required.")

    thresholds: list[float] = []
    for part in parts:
        try:
            thresholds.append(float(part))
        except Exception as exc:
            raise argparse.ArgumentTypeError(f"Invalid threshold: {part}") from exc

    return sorted(set(thresholds))


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    try:
        return float(value)
    except Exception:
        return None


def _stats(rows: list[dict[str, Any]], thresholds: list[float], latency_field: str) -> dict[str, Any]:
    total = len(rows)
    out: dict[str, Any] = {"total_calls": total}

    for threshold in thresholds:
        passed = 0
        for row in rows:
            latency = _to_float(row.get(latency_field))
            if latency is not None and latency < threshold:
                passed += 1

        pass_rate = 0.0 if total == 0 else round((passed / total) * 100.0, 2)
        key = f"lt_{int(threshold) if threshold.is_integer() else threshold}_ms"
        out[key] = {
            "pass_count": passed,
            "pass_rate_pct": pass_rate,
        }

    return out


def _build_rows(manifest: dict[str, Any], latency_field: str, thresholds: list[float]) -> list[dict[str, Any]]:
    pairs = manifest.get("pairs", [])
    if not isinstance(pairs, list):
        raise ValueError("Manifest 'pairs' must be a list.")

    rows: list[dict[str, Any]] = []

    for pair in pairs:
        if not isinstance(pair, dict):
            continue

        label = str(pair.get("label", ""))
        metadata = pair.get("metadata", {})
        if not isinstance(metadata, dict):
            metadata = {}

        artifacts = [
            ("http", pair.get("http_artifact")),
            ("webrtc", pair.get("webrtc_artifact")),
        ]

        for transport_from_pair, artifact_path in artifacts:
            if not isinstance(artifact_path, str) or not artifact_path.strip():
                continue

            artifact = json.loads(Path(artifact_path).read_text(encoding="utf-8"))
            transport = str(artifact.get("transport", transport_from_pair))
            calls = artifact.get("calls", [])
            if not isinstance(calls, list):
                continue

            for index, call in enumerate(calls):
                if not isinstance(call, dict):
                    continue

                latency = _to_float(call.get(latency_field))

                row: dict[str, Any] = {
                    "label": label,
                    "phase": metadata.get("phase", ""),
                    "transport": transport,
                    "calls_target": metadata.get("calls", ""),
                    "concurrency": metadata.get("concurrency", ""),
                    "repeat": metadata.get("repeat", ""),
                    "webrtc_timeout_s": metadata.get("webrtc_timeout_s", ""),
                    "webrtc_reuse_session": metadata.get("webrtc_reuse_session", ""),
                    "artifact": artifact_path,
                    "call_index": index,
                    "call_id": call.get("call_id", ""),
                    "turn_id": call.get("turn_id", ""),
                    "source_id": call.get("source_id", ""),
                    "speech_to_first_text_ms": call.get("speech_to_first_text_ms", ""),
                    "speech_to_first_audio_ms": call.get("speech_to_first_audio_ms", ""),
                    "mic_to_first_audio_ms": call.get("mic_to_first_audio_ms", ""),
                    "mic_to_first_audio_lt_700": call.get("mic_to_first_audio_lt_700", ""),
                    "mic_to_first_audio_lt_800": call.get("mic_to_first_audio_lt_800", ""),
                    "speech_to_done_ms": call.get("speech_to_done_ms", ""),
                    "e2e_ms": call.get("e2e_ms", ""),
                    "asr_queue_wait_ms": call.get("asr_queue_wait_ms", ""),
                    "tts_queue_wait_ms": call.get("tts_queue_wait_ms", ""),
                    "pipeline_latency_field": latency_field,
                    "pipeline_latency_ms": latency,
                }

                for threshold in thresholds:
                    suffix = int(threshold) if threshold.is_integer() else threshold
                    key = f"pipeline_latency_lt_{suffix}_ms"
                    row[key] = bool(latency is not None and latency < threshold)

                rows.append(row)

    return rows


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return

    fieldnames = list(rows[0].keys())
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export per-call latency rows from paired matrix artifacts and evaluate per-call targets."
        )
    )
    parser.add_argument(
        "--manifest",
        type=str,
        default="results/transport_ab_matrix_20260413_full/paired_manifest.json",
    )
    parser.add_argument(
        "--latency-field",
        type=str,
        default="speech_to_first_audio_ms",
        choices=[
            "speech_to_first_audio_ms",
            "mic_to_first_audio_ms",
            "speech_to_done_ms",
            "e2e_ms",
            "speech_to_first_text_ms",
        ],
    )
    parser.add_argument("--thresholds-ms", type=str, default="700,800")
    parser.add_argument(
        "--output-csv",
        type=str,
        default="results/transport_ab_matrix_20260413_full/per_call_latency_all.csv",
    )
    parser.add_argument(
        "--output-violations-csv",
        type=str,
        default="results/transport_ab_matrix_20260413_full/per_call_latency_violations.csv",
    )
    parser.add_argument(
        "--output-summary-json",
        type=str,
        default="results/transport_ab_matrix_20260413_full/per_call_latency_summary.json",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    thresholds = _parse_thresholds(args.thresholds_ms)

    manifest_path = Path(args.manifest)
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    rows = _build_rows(manifest, args.latency_field, thresholds)

    max_threshold = max(thresholds)
    violations: list[dict[str, Any]] = []
    for row in rows:
        latency = _to_float(row.get("pipeline_latency_ms"))
        if latency is None or latency >= max_threshold:
            violations.append(row)

    output_csv = Path(args.output_csv)
    output_violations_csv = Path(args.output_violations_csv)
    output_summary_json = Path(args.output_summary_json)

    _write_csv(output_csv, rows)
    _write_csv(output_violations_csv, violations)

    by_transport: dict[str, list[dict[str, Any]]] = {}
    by_phase: dict[str, list[dict[str, Any]]] = {}

    for row in rows:
        transport = str(row.get("transport", ""))
        phase = str(row.get("phase", ""))
        by_transport.setdefault(transport, []).append(row)
        by_phase.setdefault(phase, []).append(row)

    summary = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "manifest": str(manifest_path),
        "latency_field": args.latency_field,
        "thresholds_ms": thresholds,
        "total_rows": len(rows),
        "violations_threshold_ms": max_threshold,
        "violations_count": len(violations),
        "overall": _stats(rows, thresholds, args.latency_field),
        "by_transport": {
            key: _stats(value, thresholds, args.latency_field)
            for key, value in sorted(by_transport.items())
        },
        "by_phase": {
            key: _stats(value, thresholds, args.latency_field)
            for key, value in sorted(by_phase.items())
        },
        "outputs": {
            "all_rows_csv": str(output_csv),
            "violations_csv": str(output_violations_csv),
            "summary_json": str(output_summary_json),
        },
    }

    output_summary_json.parent.mkdir(parents=True, exist_ok=True)
    output_summary_json.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(f"rows_exported={len(rows)}")
    print(f"violations_count={len(violations)}")
    print(f"all_rows_csv={output_csv}")
    print(f"violations_csv={output_violations_csv}")
    print(f"summary_json={output_summary_json}")


if __name__ == "__main__":
    main()
