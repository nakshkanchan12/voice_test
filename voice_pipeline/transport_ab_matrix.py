from __future__ import annotations

import argparse
import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from statistics import mean
from typing import Any

from .live_benchmark import run_live_benchmark
from .transport_compare_report import ALL_METRICS, build_comparison_report, render_markdown

PROFILE_METRICS: tuple[str, ...] = (
    "speech_to_first_audio_p50_ms",
    "speech_to_done_p50_ms",
    "e2e_p50_ms",
    "e2e_p95_ms",
)


def _parse_int_list(value: str) -> list[int]:
    parsed = [int(token.strip()) for token in value.split(",") if token.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one integer value.")
    return parsed


def _parse_float_list(value: str) -> list[float]:
    parsed = [float(token.strip()) for token in value.split(",") if token.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one float value.")
    return parsed


def _parse_bool(value: str) -> bool:
    token = value.strip().lower()
    if token in {"1", "true", "yes", "y", "on"}:
        return True
    if token in {"0", "false", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _parse_bool_list(value: str) -> list[bool]:
    parsed = [_parse_bool(token.strip()) for token in value.split(",") if token.strip()]
    if not parsed:
        raise argparse.ArgumentTypeError("Expected at least one boolean value.")
    return parsed


def _save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _build_live_args(
    args: argparse.Namespace,
    *,
    calls: int,
    concurrency: int,
    transport: str,
    webrtc_timeout_s: float,
    webrtc_reuse_session: bool,
) -> argparse.Namespace:
    return argparse.Namespace(
        calls=calls,
        concurrency=concurrency,
        sample_pool=args.sample_pool,
        dataset_source=args.dataset_source,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        asr_url=args.asr_url,
        llm_base_url=args.llm_base_url,
        llm_model=args.llm_model,
        llm_api_key=args.llm_api_key,
        system_prompt=args.system_prompt,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        tts_url=args.tts_url,
        transport=transport,
        asr_webrtc_offer_url=args.asr_webrtc_offer_url,
        llm_webrtc_offer_url=args.llm_webrtc_offer_url,
        tts_webrtc_offer_url=args.tts_webrtc_offer_url,
        webrtc_timeout_s=webrtc_timeout_s,
        webrtc_reuse_session=webrtc_reuse_session,
        output="",
    )


async def _run_one(
    args: argparse.Namespace,
    *,
    calls: int,
    concurrency: int,
    transport: str,
    webrtc_timeout_s: float,
    webrtc_reuse_session: bool,
    artifact_path: Path,
) -> dict[str, Any]:
    bench_args = _build_live_args(
        args,
        calls=calls,
        concurrency=concurrency,
        transport=transport,
        webrtc_timeout_s=webrtc_timeout_s,
        webrtc_reuse_session=webrtc_reuse_session,
    )
    report = await run_live_benchmark(bench_args)
    _save_json(artifact_path, report)
    return report


def _build_profile_summary(profile_runs: list[dict[str, Any]]) -> dict[str, Any]:
    grouped: dict[tuple[float, bool], list[dict[str, Any]]] = {}
    for run in profile_runs:
        key = (float(run["webrtc_timeout_s"]), bool(run["webrtc_reuse_session"]))
        grouped.setdefault(key, []).append(run)

    configs: list[dict[str, Any]] = []
    for (timeout_s, reuse_session), runs in grouped.items():
        row: dict[str, Any] = {
            "webrtc_timeout_s": timeout_s,
            "webrtc_reuse_session": reuse_session,
            "runs": len(runs),
        }

        for metric in PROFILE_METRICS:
            values = []
            for run in runs:
                value = run["summary"].get(metric)
                if value is not None:
                    values.append(float(value))

            row[f"{metric}_mean"] = round(mean(values), 2) if values else None

        configs.append(row)

    best = None
    ranked = [cfg for cfg in configs if cfg.get("e2e_p50_ms_mean") is not None]
    if ranked:
        best = sorted(ranked, key=lambda x: float(x["e2e_p50_ms_mean"]))[0]

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "profile_run_count": len(profile_runs),
        "metrics": list(PROFILE_METRICS),
        "configs": sorted(configs, key=lambda x: (x["webrtc_timeout_s"], x["webrtc_reuse_session"])),
        "best_config_by_e2e_p50": best,
    }


def _render_profile_markdown(summary: dict[str, Any]) -> str:
    lines: list[str] = []
    lines.append("# WebRTC Timeout/Reuse Profile")
    lines.append("")
    lines.append(f"- Generated at: {summary['generated_at']}")
    lines.append(f"- Profile runs: {summary['profile_run_count']}")
    lines.append("")

    best = summary.get("best_config_by_e2e_p50")
    if best is not None:
        lines.append("## Best Config (lowest e2e_p50)")
        lines.append("")
        lines.append(f"- webrtc_timeout_s: {best['webrtc_timeout_s']}")
        lines.append(f"- webrtc_reuse_session: {best['webrtc_reuse_session']}")
        lines.append(f"- e2e_p50_ms_mean: {best['e2e_p50_ms_mean']}")
        lines.append("")

    lines.append("## Config Means")
    lines.append("")
    lines.append("| timeout_s | reuse_session | runs | speech_to_first_audio_p50_mean | speech_to_done_p50_mean | e2e_p50_mean | e2e_p95_mean |")
    lines.append("|---:|:---:|---:|---:|---:|---:|---:|")

    for cfg in summary["configs"]:
        lines.append(
            "| "
            f"{cfg['webrtc_timeout_s']:.1f} | "
            f"{cfg['webrtc_reuse_session']} | "
            f"{cfg['runs']} | "
            f"{cfg.get('speech_to_first_audio_p50_ms_mean', 'n/a')} | "
            f"{cfg.get('speech_to_done_p50_ms_mean', 'n/a')} | "
            f"{cfg.get('e2e_p50_ms_mean', 'n/a')} | "
            f"{cfg.get('e2e_p95_ms_mean', 'n/a')} |"
        )

    lines.append("")
    return "\n".join(lines)


async def run_matrix(args: argparse.Namespace) -> dict[str, Any]:
    calls_matrix = _parse_int_list(args.calls_matrix)
    concurrency_matrix = _parse_int_list(args.concurrency_matrix)
    profile_timeouts = _parse_float_list(args.profile_timeouts)
    profile_reuse_options = _parse_bool_list(args.profile_reuse_options)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    pairs: list[dict[str, Any]] = []
    matrix_runs: list[dict[str, Any]] = []
    profile_runs: list[dict[str, Any]] = []

    for calls in calls_matrix:
        for concurrency in concurrency_matrix:
            for repeat in range(1, args.matrix_repeats + 1):
                base_label = f"matrix_calls{calls}_conc{concurrency}_rep{repeat}"

                http_artifact = output_dir / f"{base_label}_http.json"
                webrtc_artifact = output_dir / f"{base_label}_webrtc.json"

                print(f"running {base_label} transport=http")
                http_report = await _run_one(
                    args,
                    calls=calls,
                    concurrency=concurrency,
                    transport="http",
                    webrtc_timeout_s=args.matrix_webrtc_timeout_s,
                    webrtc_reuse_session=True,
                    artifact_path=http_artifact,
                )

                print(
                    f"running {base_label} transport=webrtc timeout={args.matrix_webrtc_timeout_s} reuse=True"
                )
                webrtc_report = await _run_one(
                    args,
                    calls=calls,
                    concurrency=concurrency,
                    transport="webrtc",
                    webrtc_timeout_s=args.matrix_webrtc_timeout_s,
                    webrtc_reuse_session=True,
                    artifact_path=webrtc_artifact,
                )

                pairs.append(
                    {
                        "label": base_label,
                        "http_artifact": str(http_artifact),
                        "webrtc_artifact": str(webrtc_artifact),
                        "metadata": {
                            "calls": calls,
                            "concurrency": concurrency,
                            "repeat": repeat,
                            "phase": "matrix",
                            "webrtc_timeout_s": args.matrix_webrtc_timeout_s,
                            "webrtc_reuse_session": True,
                        },
                    }
                )

                matrix_runs.append(
                    {
                        "label": base_label,
                        "calls": calls,
                        "concurrency": concurrency,
                        "repeat": repeat,
                        "http_artifact": str(http_artifact),
                        "webrtc_artifact": str(webrtc_artifact),
                        "http_summary": http_report.get("summary", {}),
                        "webrtc_summary": webrtc_report.get("summary", {}),
                    }
                )

    profile_calls = args.profile_calls if args.profile_calls > 0 else max(calls_matrix)
    profile_concurrency = (
        args.profile_concurrency if args.profile_concurrency > 0 else max(concurrency_matrix)
    )

    profile_http_by_repeat: dict[int, Path] = {}
    for repeat in range(1, args.profile_repeats + 1):
        base_label = f"profile_baseline_calls{profile_calls}_conc{profile_concurrency}_rep{repeat}"
        http_artifact = output_dir / f"{base_label}_http.json"

        print(f"running {base_label} transport=http")
        await _run_one(
            args,
            calls=profile_calls,
            concurrency=profile_concurrency,
            transport="http",
            webrtc_timeout_s=args.matrix_webrtc_timeout_s,
            webrtc_reuse_session=True,
            artifact_path=http_artifact,
        )
        profile_http_by_repeat[repeat] = http_artifact

    for timeout_s in profile_timeouts:
        for reuse_session in profile_reuse_options:
            for repeat in range(1, args.profile_repeats + 1):
                label = (
                    "profile_"
                    f"calls{profile_calls}_conc{profile_concurrency}_"
                    f"timeout{timeout_s:g}_reuse{int(reuse_session)}_rep{repeat}"
                )
                webrtc_artifact = output_dir / f"{label}_webrtc.json"

                print(
                    "running "
                    f"{label} transport=webrtc timeout={timeout_s} reuse={reuse_session}"
                )
                webrtc_report = await _run_one(
                    args,
                    calls=profile_calls,
                    concurrency=profile_concurrency,
                    transport="webrtc",
                    webrtc_timeout_s=timeout_s,
                    webrtc_reuse_session=reuse_session,
                    artifact_path=webrtc_artifact,
                )

                http_artifact = profile_http_by_repeat[repeat]
                pairs.append(
                    {
                        "label": label,
                        "http_artifact": str(http_artifact),
                        "webrtc_artifact": str(webrtc_artifact),
                        "metadata": {
                            "calls": profile_calls,
                            "concurrency": profile_concurrency,
                            "repeat": repeat,
                            "phase": "profile",
                            "webrtc_timeout_s": timeout_s,
                            "webrtc_reuse_session": reuse_session,
                        },
                    }
                )

                profile_runs.append(
                    {
                        "label": label,
                        "webrtc_timeout_s": timeout_s,
                        "webrtc_reuse_session": reuse_session,
                        "repeat": repeat,
                        "artifact": str(webrtc_artifact),
                        "summary": webrtc_report.get("summary", {}),
                    }
                )

    manifest = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "benchmark": "transport_ab_matrix",
        "all_metrics": list(ALL_METRICS),
        "pairs": pairs,
        "matrix_runs": matrix_runs,
        "profile_runs": profile_runs,
    }

    manifest_path = output_dir / "paired_manifest.json"
    _save_json(manifest_path, manifest)

    comparison_report = build_comparison_report(pairs)
    comparison_md = render_markdown(comparison_report)

    comparison_json_path = output_dir / "comparison_report.json"
    comparison_md_path = output_dir / "comparison_report.md"
    comparison_json_path.write_text(json.dumps(comparison_report, indent=2), encoding="utf-8")
    comparison_md_path.write_text(comparison_md, encoding="utf-8")

    profile_summary = _build_profile_summary(profile_runs)
    profile_summary_path = output_dir / "profile_tuning_summary.json"
    profile_md_path = output_dir / "profile_tuning_summary.md"
    _save_json(profile_summary_path, profile_summary)
    profile_md_path.write_text(_render_profile_markdown(profile_summary), encoding="utf-8")

    return {
        "output_dir": str(output_dir),
        "manifest": str(manifest_path),
        "comparison_json": str(comparison_json_path),
        "comparison_md": str(comparison_md_path),
        "profile_json": str(profile_summary_path),
        "profile_md": str(profile_md_path),
        "pair_count": len(pairs),
        "matrix_run_count": len(matrix_runs),
        "profile_run_count": len(profile_runs),
    }


def parse_args() -> argparse.Namespace:
    now = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    parser = argparse.ArgumentParser(
        description=(
            "Run a larger HTTP vs WebRTC benchmark matrix, profile WebRTC timeout/session reuse, "
            "and generate paired artifact comparison reports."
        )
    )

    parser.add_argument("--calls-matrix", type=str, default="8,16,24")
    parser.add_argument("--concurrency-matrix", type=str, default="1,2,4")
    parser.add_argument("--matrix-repeats", type=int, default=2)
    parser.add_argument("--matrix-webrtc-timeout-s", type=float, default=120.0)

    parser.add_argument("--profile-calls", type=int, default=0)
    parser.add_argument("--profile-concurrency", type=int, default=0)
    parser.add_argument("--profile-timeouts", type=str, default="30,60,120")
    parser.add_argument("--profile-reuse-options", type=str, default="true,false")
    parser.add_argument("--profile-repeats", type=int, default=2)

    parser.add_argument("--sample-pool", type=int, default=4)
    parser.add_argument("--dataset-source", type=str, default="kaggle", choices=["hf", "kaggle", "tts", "mock"])
    parser.add_argument("--hf-dataset", type=str, default="hf-internal-testing/librispeech_asr_dummy")
    parser.add_argument("--hf-config", type=str, default="clean")
    parser.add_argument("--hf-split", type=str, default="validation")

    parser.add_argument("--asr-url", type=str, default="http://127.0.0.1:8111")
    parser.add_argument("--llm-base-url", type=str, default="http://127.0.0.1:8100/v1")
    parser.add_argument("--llm-model", type=str, default="HuggingFaceTB/SmolLM2-360M-Instruct")
    parser.add_argument("--llm-api-key", type=str, default="EMPTY")
    parser.add_argument(
        "--system-prompt",
        type=str,
        default="You are a concise and compliant banking collection assistant.",
    )
    parser.add_argument("--temperature", type=float, default=0.2)
    parser.add_argument("--max-tokens", type=int, default=100)
    parser.add_argument("--tts-url", type=str, default="http://127.0.0.1:8112")

    parser.add_argument("--asr-webrtc-offer-url", type=str, default="")
    parser.add_argument("--llm-webrtc-offer-url", type=str, default="")
    parser.add_argument("--tts-webrtc-offer-url", type=str, default="")

    parser.add_argument("--output-dir", type=str, default=f"results/transport_ab_matrix_{now}")

    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = asyncio.run(run_matrix(args))
    print(json.dumps(result, indent=2))
