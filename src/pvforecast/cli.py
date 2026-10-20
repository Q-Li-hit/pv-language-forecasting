"""Command-line entry points for prepared-data experiments."""

from __future__ import annotations

import argparse
from dataclasses import asdict
import importlib.metadata
import json
from pathlib import Path
import sys

import numpy as np

from .data import ScalarData, SequenceData


def _json(path):
    return json.loads(Path(path).read_text(encoding="utf-8")) if path else {}


def _sequence_model(spec, *, model_path=None, device="cpu"):
    import torch
    from .sequence import ForecastModel, SequenceConfig

    torch.manual_seed(spec["seed"])
    return ForecastModel(spec["variant"], spec["history_length"], spec["features"], spec["horizons"],
                         config=SequenceConfig(**spec["architecture"]), model_path=model_path,
                         revision=spec.get("revision"), random_backbone=spec.get("random", False),
                         head_seed=spec["seed"],
                         model_dtype=torch.bfloat16 if str(device).startswith("cuda") else torch.float32)


def train_sequence(args):
    from .sequence import SequenceConfig
    from .training import fit, TrainingConfig, file_sha256, write_json

    data = SequenceData.load(args.data)
    if args.variant == "time_llm_qwen" and (not args.model_path or data.ids is None):
        raise ValueError("Language sequence models require a local model directory and tokenized prompts.")
    spec = {"family": "sequence", "variant": args.variant, "seed": args.seed, "random": args.random,
            "history_length": data.history.shape[1], "features": data.future.shape[-1],
            "horizons": data.target.shape[1], "revision": args.revision,
            "architecture": asdict(SequenceConfig(**_json(args.architecture)))}
    if args.variant == "time_llm_qwen":
        from .utils import model_digest
        spec["model_sha256"] = model_digest(args.model_path)
    if args.random and args.variant != "time_llm_qwen":
        raise ValueError("The random-parameter control applies to the language model.")
    model = _sequence_model(spec, model_path=args.model_path, device=args.device)
    settings = _json(args.training)
    if args.learning_rate is not None:
        settings["learning_rate"] = args.learning_rate
    report = fit(model, data, args.output, config=TrainingConfig(**settings), seed=args.seed,
                 device=args.device, model_spec=spec)
    report["data_sha256"] = file_sha256(args.data)
    write_json(Path(args.output) / "fit.json", report)


def train_weather(args):
    import torch
    from .training import fit, TrainingConfig, file_sha256, write_json
    from .weather import CapacityMatchedControl, TransformerConfig

    data = ScalarData.load(args.data)
    architecture = TransformerConfig(**_json(args.architecture))
    torch.manual_seed(args.seed)
    model = CapacityMatchedControl(data.history.shape[1], data.state.shape[1], architecture)
    settings = {"batch_size": 1024, "microbatch": 1024, "minimum_epochs": 15,
                "maximum_epochs": 150, "patience": 12}
    settings.update(_json(args.training))
    if args.learning_rate is not None:
        settings["learning_rate"] = args.learning_rate
    spec = {"family": "weather", "seed": args.seed, "lag_count": data.history.shape[1],
            "features": data.state.shape[1], "architecture": asdict(architecture)}
    report = fit(model, data, args.output, config=TrainingConfig(**settings), seed=args.seed,
                 device=args.device, model_spec=spec)
    report["data_sha256"] = file_sha256(args.data)
    write_json(Path(args.output) / "fit.json", report)


def _save_predictions(path, data, prediction):
    mask = data.evaluation_mask
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, target=data.target, prediction=prediction, mask=mask,
                        issue_dates=data.issue_dates, station_ids=data.station_ids, sample_keys=data.sample_keys)


def predict(args):
    from .training import load_task_parameters, predict as forward

    report = _json(Path(args.model) / "fit.json")
    spec = report["model"]
    if spec["family"] == "sequence":
        data = SequenceData.load(args.data)
        if spec["variant"] == "time_llm_qwen":
            from .utils import model_digest
            if not args.model_path or model_digest(args.model_path) != spec["model_sha256"]:
                raise ValueError("Frozen model snapshot differs from the training run.")
        model = _sequence_model(spec, model_path=args.model_path, device=args.device)
    else:
        from .weather import CapacityMatchedControl, TransformerConfig
        data = ScalarData.load(args.data)
        model = CapacityMatchedControl(spec["lag_count"], spec["features"], TransformerConfig(**spec["architecture"]))
    if data.metadata != report["input_metadata"]:
        raise ValueError("Input preprocessing metadata differs from the fitted model.")
    load_task_parameters(model, Path(args.model) / "parameters.pt")
    values = forward(model, data, device=args.device, batch_size=args.batch_size,
                     precision=report["training"]["precision"], capacity_mw=args.capacity)
    _save_predictions(args.output, data, values)


def evaluate(args):
    from .evaluation import metrics, paired_date_block_bootstrap
    from .utils import write_json

    with np.load(args.predictions, allow_pickle=False) as saved:
        y, prediction, mask = saved["target"], saved["prediction"], saved["mask"]
        dates, stations = saved["issue_dates"], saved["station_ids"]
        keys = saved["sample_keys"]
    if mask.dtype.kind != "b" or mask.shape != y.shape or prediction.shape != y.shape:
        raise ValueError("Predictions require aligned arrays and a boolean evaluation mask.")
    if not mask.any():
        raise ValueError("Evaluation mask is empty.")
    report = metrics(y[mask], prediction[mask], capacity_mw=args.capacity)
    if args.reference:
        with np.load(args.reference, allow_pickle=False) as saved:
            for key, value in (("target", y), ("mask", mask), ("issue_dates", dates),
                               ("station_ids", stations), ("sample_keys", keys)):
                if not np.array_equal(saved[key], value):
                    raise ValueError("Evaluated and reference targets, keys or masks differ.")
            reference = saved["prediction"]
        scalar_dates = np.broadcast_to(dates[:, None], y.shape)[mask] if y.ndim == 2 else dates[mask]
        report["paired_difference"] = paired_date_block_bootstrap(
            y[mask], prediction[mask], reference[mask], scalar_dates,
            repetitions=args.repetitions, confidence=args.confidence,
        )
    write_json(args.output, report)
    print(json.dumps(report, indent=2, default=float))


def prepare_weather(args):
    import pandas as pd
    from .preparation import prepare_weather as prepare
    prepare(pd.read_csv(args.csv), args.output, representation=args.representation,
            preprocessor=args.preprocessor, embedding_bank=args.embedding_bank,
            capacity_mw=args.capacity)


def encode_weather(args):
    import pandas as pd
    from .weather import context_fields, context_codes_key, text_from_codes, frozen_embeddings
    from .utils import model_digest

    codes = context_fields(pd.read_csv(args.csv))[0]
    keys = np.unique(context_codes_key(codes))
    unique = np.asarray([[int(x) for x in key.split(",")] for key in keys])
    values = frozen_embeddings(text_from_codes(unique), args.model_path, random_seed=args.random_seed)
    metadata = {"representation": "text" if args.random_seed is None else "random",
                "declared_model_id": args.model_id, "declared_revision": args.revision,
                "snapshot_sha256": model_digest(args.model_path),
                "random_seed": args.random_seed, "pooling": "last-valid-token-l2", "max_length": 128}
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.output, keys=keys, embedding=values,
                        metadata=np.asarray(json.dumps(metadata)))


def tokenize(args):
    from transformers import AutoTokenizer
    from .sequence import tokenize_prompts

    data = SequenceData.load(args.data)
    tokenizer = AutoTokenizer.from_pretrained(args.model_path, revision=args.revision, local_files_only=True)
    data.ids, data.attention = tokenize_prompts(data.history, args.task, tokenizer, domain=args.domain)
    data.metadata["prompt"] = "domain" if args.domain else "basic"
    data.metadata["tokenizer_revision"] = args.revision
    data.save(args.output)


def tree(args):
    from .trees import fit_tree, predict_tree
    data = SequenceData.load(args.data)
    if args.action == "fit":
        fit_tree(data, args.output)
    else:
        if not args.model:
            raise ValueError("Tree prediction requires --model.")
        report = _json(Path(args.model) / "fit.json")
        if report["input_metadata"] != data.metadata:
            raise ValueError("Input preprocessing metadata differs from the fitted tree.")
        _save_predictions(args.output, data, predict_tree(data, Path(args.model) / "model.txt"))


def decompose(args):
    from .evaluation import mse_average_decomposition
    from .utils import write_json
    with np.load(args.data, allow_pickle=False) as archive:
        result = mse_average_decomposition(archive["predictions"], archive["target"])
    write_json(args.output, result)


def scalar_tree(args):
    from .scalar_baselines import fit_scalar_tree, predict_scalar_tree
    data = ScalarData.load(args.data)
    if args.action == "fit":
        fit_scalar_tree(data, args.output, objective=args.objective)
    else:
        if not args.model:
            raise ValueError("Scalar tree prediction requires --model.")
        _save_predictions(args.output, data, predict_scalar_tree(data, args.model))


def demo(args):
    from .demo import run_demo
    run_demo(args.output)


def doctor(args):
    from . import __version__
    print(f"pv-language-forecasting {__version__}; Python {sys.version.split()[0]}")
    for name in ("numpy", "scipy", "pandas", "scikit-learn", "torch", "transformers", "lightgbm"):
        try:
            value = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            value = "not installed"
        print(f"{name}: {value}")


def main(argv=None):
    parser = argparse.ArgumentParser(prog="pvforecast", description="Equivalent PV forecasting experiments")
    commands = parser.add_subparsers(dest="command", required=True)
    commands.add_parser("doctor", help="Show installed dependency versions").set_defaults(func=doctor)
    p = commands.add_parser("demo", help="Run a small synthetic sequence example")
    p.add_argument("--output", required=True)
    p.set_defaults(func=demo)
    for command, function in (("train-sequence", train_sequence), ("train-weather", train_weather)):
        p = commands.add_parser(command)
        p.add_argument("--data", required=True)
        p.add_argument("--output", required=True)
        p.add_argument("--architecture")
        p.add_argument("--training")
        p.add_argument("--learning-rate", type=float)
        p.add_argument("--seed", type=int, default=42)
        p.add_argument("--device", default="cpu")
        if command == "train-sequence":
            p.add_argument("--variant", choices=["dense_numeric", "patchtst", "time_llm_qwen"], required=True)
            p.add_argument("--model-path")
            p.add_argument("--revision", default="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
            p.add_argument("--random", action="store_true")
        p.set_defaults(func=function)
    p = commands.add_parser("predict")
    p.add_argument("--data", required=True)
    p.add_argument("--model", required=True, help="Fitted run directory")
    p.add_argument("--model-path", help="Local frozen language model, if needed")
    p.add_argument("--output", required=True)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=128)
    p.add_argument("--capacity", type=float, default=10.)
    p.set_defaults(func=predict)
    p = commands.add_parser("evaluate")
    p.add_argument("--predictions", required=True)
    p.add_argument("--reference")
    p.add_argument("--output", required=True)
    p.add_argument("--capacity", type=float, default=10.)
    p.add_argument("--repetitions", type=int, default=2000)
    p.add_argument("--confidence", type=float, default=.95)
    p.set_defaults(func=evaluate)
    p = commands.add_parser("prepare-weather")
    p.add_argument("--csv", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--representation", choices=["numeric", "category", "text", "random"], required=True)
    p.add_argument("--preprocessor")
    p.add_argument("--embedding-bank")
    p.add_argument("--capacity", type=float, default=10.)
    p.set_defaults(func=prepare_weather)
    p = commands.add_parser("encode-weather")
    p.add_argument("--csv", required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--model-id", default="Qwen/Qwen3-Embedding-0.6B")
    p.add_argument("--random-seed", type=int)
    p.add_argument("--revision", default="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
    p.add_argument("--output", required=True)
    p.set_defaults(func=encode_weather)
    p = commands.add_parser("tokenize")
    p.add_argument("--data", required=True)
    p.add_argument("--task", choices=["day_ahead", "rolling"], required=True)
    p.add_argument("--model-path", required=True)
    p.add_argument("--domain", action="store_true")
    p.add_argument("--revision", default="97b0c614be4d77ee51c0cef4e5f07c00f9eb65b3")
    p.add_argument("--output", required=True)
    p.set_defaults(func=tokenize)
    p = commands.add_parser("tree")
    p.add_argument("action", choices=["fit", "predict"])
    p.add_argument("--data", required=True)
    p.add_argument("--model")
    p.add_argument("--output", required=True)
    p.set_defaults(func=tree)
    p = commands.add_parser("decompose")
    p.add_argument("--data", required=True, help="NPZ with predictions[models,targets] and target[targets]")
    p.add_argument("--output", required=True)
    p.set_defaults(func=decompose)
    p = commands.add_parser("scalar-tree")
    p.add_argument("action", choices=["fit", "predict"])
    p.add_argument("--data", required=True)
    p.add_argument("--objective", choices=["direct", "residual"], default="direct")
    p.add_argument("--model")
    p.add_argument("--output", required=True)
    p.set_defaults(func=scalar_tree)
    args = parser.parse_args(argv)
    try:
        args.func(args)
    except (ValueError, FileNotFoundError, FileExistsError, ModuleNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")
