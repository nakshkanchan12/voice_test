from __future__ import annotations

import argparse

from .asr_compare import ASRCompareConfig


def add_compare_args(parser: argparse.ArgumentParser) -> None:
    parser.add_argument(
        "--compare-asr",
        action="store_true",
        help="Run ASR model comparison and select a winner.",
    )
    parser.add_argument(
        "--dataset-source",
        type=str,
        default="hf",
        choices=["mock", "tts", "hf", "kaggle"],
        help="Dataset source used for ASR comparison.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=12,
        help="Number of comparison samples.",
    )
    parser.add_argument(
        "--wer-threshold",
        type=float,
        default=5.0,
        help="WER threshold for winner selection.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "cpu"],
        help="Execution device for real ASR comparison.",
    )
    parser.add_argument(
        "--hf-dataset",
        type=str,
        default="hf-internal-testing/librispeech_asr_dummy",
        help="Hugging Face dataset id when --dataset-source hf is used.",
    )
    parser.add_argument(
        "--hf-config",
        type=str,
        default="clean",
        help="Hugging Face dataset config name.",
    )
    parser.add_argument(
        "--hf-split",
        type=str,
        default="validation",
        help="Hugging Face dataset split.",
    )
    parser.add_argument(
        "--candidate-models",
        type=str,
        default="tiny,small,medium",
        help="Comma-separated faster-whisper model ids to compare.",
    )
    parser.add_argument(
        "--model-load-timeout",
        type=float,
        default=240.0,
        help="Max seconds allowed to load each ASR model.",
    )
    parser.add_argument(
        "--sample-timeout",
        type=float,
        default=120.0,
        help="Max seconds allowed per audio sample transcription.",
    )
    parser.add_argument(
        "--selected-asr-output",
        type=str,
        default="configs/selected_asr.json",
        help="Path to store selected ASR model config.",
    )
    parser.add_argument(
        "--asr-mode",
        type=str,
        default="mock",
        choices=["mock", "real-faster-whisper", "http-asr"],
        help="ASR provider mode for benchmark run.",
    )
    parser.add_argument(
        "--asr-model",
        type=str,
        default="",
        help="Override ASR model profile for mock benchmark runs.",
    )


def build_compare_config(args: argparse.Namespace) -> ASRCompareConfig:
    candidates = tuple(
        token.strip()
        for token in str(args.candidate_models).split(",")
        if token.strip()
    )

    return ASRCompareConfig(
        dataset_source=args.dataset_source,
        sample_count=max(1, args.samples),
        wer_threshold_pct=max(0.0, args.wer_threshold),
        device=args.device,
        hf_dataset=args.hf_dataset,
        hf_config=args.hf_config,
        hf_split=args.hf_split,
        candidate_models=candidates,
        model_load_timeout_s=max(1.0, args.model_load_timeout),
        sample_timeout_s=max(1.0, args.sample_timeout),
    )
