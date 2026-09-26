# Learning Multiresolution Relevance for Hierarchical Generative Retrieval

This repository contains the source code accompanying the paper **"Learning Multiresolution Relevance for Hierarchical Generative Retrieval."**

RARS learns query-level relevance distributions over semantic identifier hierarchies. This repository supports ESCI-US, ESCI-ES, and ESCI-JP through three stages: data preparation, tokenization, and generative retrieval.

## Setup

Use Python 3.10 or later (Python 3.12 tested) and a CUDA-enabled PyTorch installation for training.

The training pipeline uses one GPU and one process. On a machine with multiple GPUs, set `CUDA_VISIBLE_DEVICES=0` before running the scripts.

```bash
cd /path/to/RARS
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e .
```

## Data Preparation

Prepare local copies of the joined ESCI parquet files and the full ESCI-S product-category metadata. Arrange the input files as follows:

```text
raw/
├── esci/data/train-*.parquet
├── esci/data/test-*.parquet
└── esci.json.zst
```

Prepare the three locale-specific datasets:

```bash
bash scripts/1_prepare_data.sh \
  --hf-data-dir raw/esci/data \
  --esci-s-json-zst raw/esci.json.zst
```

The script performs category filtering, builds product mappings, and exports training and evaluation data to `artifacts/data/`.

## Tokenization

Generate product embeddings with BERT for US and multilingual BERT for ES/JP, train the RQ-VAE tokenizer, and export four-level semantic identifiers:

```bash
bash scripts/2_tokenization.sh
```

Tokenizer checkpoints and semantic identifiers are saved to `artifacts/tokenization/`.

## Generative Retrieval

Train RARS with T5-base for US and mT5-base for ES/JP, then evaluate the final checkpoints using all-level retrieval scoring:

```bash
bash scripts/3_gr.sh
```

The default run processes all three locales and five seeds sequentially. Model checkpoints, retrieval rankings, and Recall/NDCG metrics are saved to `artifacts/gr/`. Aggregated results are written to `artifacts/results.json`.

To run a single locale and seed:

```bash
bash scripts/3_gr.sh --locales us --seeds 2020
```

To evaluate existing final checkpoints:

```bash
bash scripts/3_gr.sh --evaluation-only
```

Training and retrieval settings are defined in `configs/rars.json`. Use `--work-dir /path/to/run` consistently across all three stages to select a different output directory. Steps 2 and 3 also accept `--locales us`, `--locales es`, or `--locales jp`.

For offline execution, pass `--model-root /path/to/models` to steps 2 and 3. Arrange the pretrained models as follows:

```text
models/
├── google-bert/bert-base-uncased/
├── google-bert/bert-base-multilingual-cased/
├── google-t5/t5-base/
└── google/mt5-base/
```

## Quick Check

After data preparation and tokenization, run a short training and evaluation check:

```bash
bash scripts/3_gr.sh --locales us es jp --seeds 2020 \
  --max-steps 2 --limit-queries 4 --batch-size 2
```

## License

This project is released under the MIT License. The license text is provided in `LICENSE`.

The latent retriever follows CaLIR. ESCI, ESCI-S, pretrained models, and third-party libraries retain their respective licenses.
