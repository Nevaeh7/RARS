"""
Generate BERT embeddings for ESCI items.

Encode product metadata with BERT or multilingual BERT. Each output matrix row
uses the product row ID defined in product_id_to_index.json.
"""

import json
import os
from pathlib import Path

import numpy as np
import torch
from tqdm import tqdm

from rars.common import read_json, sha256, write_json


def prepare_embeddings(directory, model_name, device, batch_size):
    """Build or validate embeddings bound to the exact product rows and text."""
    directory = Path(directory)
    items_path = directory / f"{directory.name}.item.json"
    mapping_path = directory / "product_id_to_index.json"
    mapping = read_json(mapping_path)
    if set(mapping.values()) != set(range(len(mapping))):
        raise ValueError("Invalid product row mapping for embeddings")
    expected = dict(
        schema=1,
        model=str(model_name),
        item_metadata_sha256=sha256(items_path),
        product_mapping_sha256=sha256(mapping_path),
        pooling="masked_mean",
        max_length=512,
    )
    path = directory / "embeddings.npy"
    receipt_path = directory / "embeddings.json"
    if path.exists():
        if not receipt_path.is_file():
            raise ValueError(
                "Existing embeddings are missing embeddings.json; use a fresh data directory"
            )
        receipt = read_json(receipt_path)
        if any(receipt.get(key) != value for key, value in expected.items()):
            raise ValueError(
                "Cached embeddings belong to different data or a different model"
            )
        if receipt.get("embedding_sha256") != sha256(path):
            raise ValueError(
                "Cached embeddings checksum does not match embeddings.json"
            )
        return path

    from transformers import AutoModel, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name).to(device).eval()
    embeddings = generate_item_embedding(
        preprocess_text(items_path), tokenizer, model, device, batch_size
    )
    if (
        embeddings.shape != (len(mapping), model.config.hidden_size)
        or not np.isfinite(embeddings).all()
    ):
        raise ValueError(
            "Generated embeddings have invalid rows, dimensions, or values"
        )
    temporary = path.with_name(path.name + f".tmp-{os.getpid()}")
    with temporary.open("xb") as stream:
        np.save(stream, embeddings, allow_pickle=False)
    os.replace(temporary, path)
    write_json(
        receipt_path,
        dict(expected, shape=list(embeddings.shape), embedding_sha256=sha256(path)),
    )
    del model
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return path


def load_json(file_path):
    """Load JSON file."""
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)


def clean_text(raw_text):
    """Clean text by removing HTML tags and special characters."""
    import html
    import re

    if raw_text is None:
        return ""

    if isinstance(raw_text, list):
        new_raw_text = []
        for raw in raw_text:
            raw = html.unescape(raw)
            raw = re.sub(r"</?\w+[^>]*>", "", raw)
            raw = re.sub(r'["\n\r]*', "", raw)
            new_raw_text.append(raw.strip())
        cleaned_text = " ".join(new_raw_text)
    else:
        if isinstance(raw_text, dict):
            cleaned_text = str(raw_text)[1:-1].strip()
        else:
            cleaned_text = raw_text.strip()
        cleaned_text = html.unescape(cleaned_text)
        cleaned_text = re.sub(r"</?\w+[^>]*>", "", cleaned_text)
        cleaned_text = re.sub(r'["\n\r]*', "", cleaned_text)

    # Ensure text ends with period
    index = -1
    while -index < len(cleaned_text) and cleaned_text[index] == ".":
        index -= 1
    index += 1
    if index == 0:
        cleaned_text = cleaned_text + "."
    else:
        cleaned_text = cleaned_text[:index] + "."

    # Omit fields with 2000 or more characters.
    if len(cleaned_text) >= 2000:
        cleaned_text = ""

    return cleaned_text


def set_device(gpu_id):
    """Set device for computation."""
    if gpu_id == -1:
        return torch.device("cpu")
    else:
        return torch.device(
            "cuda:" + str(gpu_id) if torch.cuda.is_available() else "cpu"
        )


def load_data_esci(item_json_path):
    """Load ESCI item data."""
    item2feature = load_json(item_json_path)
    return item2feature


def generate_text(item2feature, features):
    """Generate text from item features."""
    item_text_list = []
    for item_idx in item2feature:
        data = item2feature[item_idx]
        text = []
        for meta_key in features:
            if meta_key in data:
                meta_value = clean_text(data[meta_key])
                if meta_value.strip():
                    text.append(f"{meta_key}:{meta_value.strip()}")

        item_text_list.append([int(item_idx), " ".join(text)])

    return item_text_list


def preprocess_text(item_json_path):
    """Preprocess text data."""
    print("Processing text data...")
    item2feature = load_data_esci(item_json_path)
    # Use product_title, product_brand, product_color, product_description
    item_text_list = generate_text(
        item2feature,
        ["product_title", "product_brand", "product_color", "product_description"],
    )
    return item_text_list


def generate_item_embedding(item_text_list, tokenizer, model, device, batch_size=32):
    """Generate BERT embeddings for items."""
    print("Generating BERT embeddings...")

    items, texts = zip(*item_text_list)

    # Create ordered text list indexed by item index
    max_item_idx = max(items)
    order_texts = [""] * (max_item_idx + 1)
    for item, text in zip(items, texts):
        order_texts[item] = text

    # Verify all items have text
    for i, text in enumerate(order_texts):
        if not text:
            print(f"Warning: Item {i} has no text")

    embeddings = []
    start = 0

    print(f"Total items: {len(order_texts)}")

    with torch.no_grad():
        for start in tqdm(
            range(0, len(order_texts), batch_size), desc="Generating embeddings"
        ):
            end = min(start + batch_size, len(order_texts))
            batch_texts = order_texts[start:end]
            # Tokenize batch
            encoded_sentences = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512,
            ).to(device)

            # Get model output
            outputs = model(**encoded_sentences)

            # Use last hidden state with attention mask for mean pooling
            last_hidden_state = outputs.last_hidden_state
            attention_mask = encoded_sentences["attention_mask"]

            # Apply attention mask and compute mean
            masked_output = last_hidden_state * attention_mask.unsqueeze(-1)
            mean_output = masked_output.sum(dim=1) / attention_mask.sum(
                dim=-1, keepdim=True
            )

            embeddings.append(mean_output.cpu())

    # Concatenate all embeddings
    embeddings = torch.cat(embeddings, dim=0).to(torch.float32).numpy()
    print(f"Embeddings shape: {embeddings.shape}")

    return embeddings
