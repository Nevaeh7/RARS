"""The three-stage RARS workflow."""

import argparse
from pathlib import Path

import numpy as np

from rars.common import read_json, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=["data", "tokenization", "gr"])
    parser.add_argument("--config", type=Path, default=Path("configs/rars.json"))
    parser.add_argument("--work-dir", type=Path, default=Path("artifacts"))
    parser.add_argument(
        "--locales", nargs="+", choices=["us", "es", "jp"], default=["us", "es", "jp"]
    )
    parser.add_argument(
        "--hf-data-dir",
        type=Path,
        help="Directory containing joined ESCI train/test parquet shards",
    )
    parser.add_argument("--esci-s-json-zst", type=Path, help="ESCI-S category metadata")
    parser.add_argument(
        "--model-root",
        type=Path,
        help="Optional directory mirroring HF IDs, e.g. google-t5/t5-base",
    )
    parser.add_argument("--device", default="cuda:0", choices=["cpu", "cuda:0"])
    parser.add_argument(
        "--seeds",
        type=int,
        nargs="+",
        help="Defaults to all five configured training seeds",
    )
    parser.add_argument(
        "--max-steps",
        type=int,
        default=-1,
        help="Positive value for a GR smoke run only",
    )
    parser.add_argument(
        "--limit-queries",
        type=int,
        default=0,
        help="Smoke-only training/evaluation query limit",
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        help="Explicit GR batch override; recorded in each run",
    )
    parser.add_argument(
        "--evaluation-only",
        action="store_true",
        help="Evaluate existing final GR checkpoints",
    )
    args = parser.parse_args()
    config = read_json(args.config)
    args.work_dir = args.work_dir.resolve()
    if args.limit_queries and args.max_steps <= 0:
        parser.error("--limit-queries requires a positive --max-steps value")
    if args.max_steps == 0 or (args.batch_size is not None and args.batch_size <= 0):
        parser.error("Steps and batch size must be positive")

    def source(model_id):
        return str(args.model_root / model_id) if args.model_root else model_id

    if args.stage == "data":
        if args.hf_data_dir is None or args.esci_s_json_zst is None:
            parser.error("data requires --hf-data-dir and --esci-s-json-zst")
        from rars.data.prepare import prepare

        prepare(
            argparse.Namespace(
                hf_data_dir=args.hf_data_dir,
                esci_s_json_zst=args.esci_s_json_zst,
                output_root=args.work_dir / "data",
                log_every=200000,
                check_statistics=True,
            )
        )
        return
    if args.stage == "tokenization":
        from rars.data.embeddings import prepare_embeddings
        from rars.tokenization.run import train_and_index

        for locale in args.locales:
            dataset = "esci_" + locale
            directory = args.work_dir / "data" / dataset
            model_name = source(config["embedding_models"][locale])
            output = args.work_dir / "tokenization" / dataset
            if output.exists() and any(output.iterdir()):
                raise FileExistsError(
                    f"Use a fresh work directory; tokenization exists: {output}"
                )
            embedding_path = prepare_embeddings(
                directory, model_name, args.device, config["embedding_batch_size"]
            )
            train_and_index(
                embedding_path,
                output,
                config["tokenization"],
                device=args.device,
                seed=config["tokenization"]["seed"],
            )
        return
    from rars.gr.retrieve import evaluate
    from rars.gr.train import train

    settings = config["gr"].copy()
    if args.batch_size:
        settings["batch_size"] = args.batch_size
    results = {}
    for locale in args.locales:
        dataset = "esci_" + locale
        directory = args.work_dir / "data" / dataset
        index = args.work_dir / "tokenization" / dataset / "index.json"
        runs = []
        for seed in args.seeds or config["seeds"]:
            output = (
                args.work_dir
                / ("smoke-gr" if args.max_steps > 0 else "gr")
                / dataset
                / f"seed-{seed}"
            )
            if not args.evaluation_only:
                train(
                    directory,
                    index,
                    output,
                    settings,
                    source=source(config["backbones"][locale]),
                    seed=seed,
                    device=args.device,
                    max_steps=args.max_steps,
                    limit_queries=args.limit_queries,
                )
            runs.append(
                evaluate(
                    directory,
                    index,
                    output / "final",
                    output,
                    config["retrieval"],
                    device=args.device,
                    limit_queries=args.limit_queries,
                )
            )
        results[locale] = {
            rule: {
                metric: {
                    "mean": float(np.mean([r["metrics"][rule][metric] for r in runs])),
                    "std": float(
                        np.std([r["metrics"][rule][metric] for r in runs], ddof=1)
                    )
                    if len(runs) > 1
                    else 0.0,
                }
                for metric in runs[0]["metrics"][rule]
            }
            for rule in ["all_levels"]
        }
    write_json(
        args.work_dir
        / ("smoke_results.json" if args.max_steps > 0 else "results.json"),
        results,
    )
    print(results)


if __name__ == "__main__":
    main()
