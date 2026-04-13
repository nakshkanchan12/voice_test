from __future__ import annotations

import argparse
import asyncio
import json

from voice_pipeline import (
	compare_asr_models,
	run_load_benchmark,
	save_compare_report,
	save_results,
	save_selected_model,
)
from voice_pipeline.compare_cli import add_compare_args, build_compare_config


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Voice pipeline benchmark and ASR model selection CLI.",
	)
	parser.add_argument(
		"--calls",
		type=int,
		default=40,
		help="Total turn requests to execute in the benchmark.",
	)
	parser.add_argument(
		"--concurrency",
		type=int,
		default=20,
		help="Maximum number of concurrent turns.",
	)
	parser.add_argument(
		"--output",
		type=str,
		default="",
		help="Optional JSON path for benchmark or compare results.",
	)

	add_compare_args(parser)
	return parser.parse_args()


async def run() -> None:
	args = parse_args()

	if args.compare_asr:
		config = build_compare_config(args)
		report = await compare_asr_models(config)
		report_path = save_compare_report(report, output_path=args.output or None)
		selected_path = save_selected_model(report, output_path=args.selected_asr_output)

		print(json.dumps(report["selection"], indent=2))
		print(f"compare_results_saved_to={report_path}")
		print(f"selected_asr_saved_to={selected_path}")
		return

	report = await run_load_benchmark(
		total_calls=args.calls,
		concurrency=args.concurrency,
		asr_mode=args.asr_mode,
		asr_model=(args.asr_model or None),
	)
	target = save_results(report, output_path=args.output or None)

	print(json.dumps(report["summary"], indent=2))
	print(f"asr_mode={report.get('asr_mode')}")
	print(f"asr_model={report.get('asr_model')}")
	print(f"results_saved_to={target}")


if __name__ == "__main__":
    asyncio.run(run())
