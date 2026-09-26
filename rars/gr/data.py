"""ESCI pair training with complete, query-level relevance distributions."""

import hashlib
import re
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import Dataset

from rars.common import read_json, sha256

from .tree import PrefixCatalog, project_documents

GRADES = {"E": 3.0, "S": 2.0, "C": 1.0, "I": 0.0}
PROMPT = "<retrieval>"


def parse_code(tokens):
    if len(tokens) != 4:
        raise ValueError("Expected four SID tokens")
    values = []
    for letter, token in zip("abcd", tokens):
        match = re.fullmatch(rf"<{letter}_(\d+)>", token)
        if not match or not 0 <= int(match[1]) < 256:
            raise ValueError(f"Invalid SID token: {token}")
        values.append(int(match[1]))
    return tuple(values)


def is_validation(query, fraction=0.1, seed=42):
    digest = hashlib.sha256(f"{seed}\0{query}".encode()).digest()
    return int.from_bytes(digest[:8], "big") / (1 << 64) < fraction


class Catalog:
    def __init__(self, directory, index_path):
        self.directory = Path(directory)
        self.dataset = self.directory.name
        self.product_to_id = read_json(self.directory / "product_id_to_index.json")
        self.indices = read_json(index_path)
        count = len(self.product_to_id)
        if set(self.product_to_id.values()) != set(range(count)) or set(
            self.indices
        ) != {str(i) for i in range(count)}:
            raise ValueError(
                "Product rows and SID rows must be identical contiguous IDs"
            )
        self.codes = np.asarray(
            [parse_code(self.indices[str(i)]) for i in range(count)], dtype=np.int64
        )
        self.prefix = PrefixCatalog(self.codes)
        raw_categories = read_json(self.directory / "product_categories.json")
        self.category_names = [{} for _ in range(3)]
        self.categories = {}
        # Assign category IDs in input order and save the mapping with each checkpoint.
        for product, path in raw_categories.items():
            if product not in self.product_to_id:
                raise ValueError(f"Unknown category product: {product}")
            result = []
            for depth, name in enumerate(path[:3]):
                mapping = self.category_names[depth]
                if name not in mapping:
                    mapping[name] = len(mapping)
                result.append(mapping[name])
            self.categories[product] = result
        self.category_sizes = [len(level) for level in self.category_names]
        if any(n == 0 for n in self.category_sizes):
            raise ValueError("Three nonempty category levels are required")
        self.category_items = defaultdict(list)
        sid_representatives = {}
        for product, path in self.categories.items():
            row = self.product_to_id[product]
            sid = tuple(self.indices[str(row)])
            sid_representatives[sid] = (row, path)
        # A shared SID uses the category path of its last catalog representative.
        for row, path in sid_representatives.values():
            if len(path) == 3:
                self.category_items[path[2]].append(row)

    def data_hashes(self):
        names = (
            "product_id_to_index.json",
            "product_categories.json",
            f"{self.dataset}.item.json",
        )
        return {name: sha256(self.directory / name) for name in names}

    def category_children(self):
        result = []
        for depth in (1, 2):
            groups = [set() for _ in range(self.category_sizes[depth - 1])]
            for path in self.categories.values():
                if len(path) > depth:
                    groups[path[depth - 1]].add(path[depth])
            result.append([sorted(group) for group in groups])
        return result

    def new_tokens(self):
        tokens = {PROMPT}
        tokens.update(token for sid in self.indices.values() for token in sid)
        # Include the task prompt and category tokens in the tokenizer vocabulary.
        tokens.update(
            f"<class_{d}_{i}>"
            for d, n in enumerate(self.category_sizes)
            for i in range(n)
        )
        return sorted(tokens)

    def documents(self, judgments):
        documents = {}
        for product, grade in judgments:
            product = str(product)
            if grade not in GRADES:
                raise ValueError(f"Unknown ESCI grade {grade}")
            if product not in self.product_to_id:
                raise ValueError(f"Judged product missing from catalog: {product}")
            if GRADES[grade] == 0 or len(self.categories.get(product, [])) < 3:
                continue
            row_id = self.product_to_id[product]
            row = dict(
                document=product,
                weight=GRADES[grade],
                category=self.categories[product],
                code=self.codes[row_id].tolist(),
                sid="".join(self.indices[str(row_id)]),
            )
            if product not in documents or row["weight"] > documents[product]["weight"]:
                documents[product] = row
        return sorted(documents.values(), key=lambda r: r["document"])


class TrainDataset(Dataset):
    def __init__(
        self, catalog, *, split="train", validation_fraction=0.1, limit_queries=0
    ):
        raw = read_json(catalog.directory / f"{catalog.dataset}.train.json")
        self.records = []
        self.pairs = []
        self.new_tokens = catalog.new_tokens()
        for query, judgments in raw.items():
            if is_validation(query, validation_fraction) != (split == "valid"):
                continue
            documents = catalog.documents(judgments)
            if not documents:
                continue
            index = len(self.records)
            self.records.append(dict(query=query, documents=documents))
            self.pairs.extend((index, i) for i in range(len(documents)))
            if limit_queries and len(self.records) >= limit_queries:
                break
        if not self.records:
            raise ValueError(f"Empty {split} query split")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        query_id, document_id = self.pairs[index]
        return self.records[query_id], document_id


class Collator:
    def __init__(self, tokenizer, catalog, max_length=512):
        self.tokenizer, self.catalog, self.max_length = tokenizer, catalog, max_length

    def __call__(self, batch):
        queries, positions, representatives, query_ids = [], {}, [], []
        for record, document_id in batch:
            if record["query"] not in positions:
                positions[record["query"]] = len(queries)
                queries.append(record)
            query_ids.append(positions[record["query"]])
            representatives.append(record["documents"][document_id])
        encoded = self.tokenizer(
            [PROMPT + r["query"] for r in queries],
            padding=True,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        labels = self.tokenizer(
            [r["sid"] for r in representatives], padding=True, return_tensors="pt"
        )["input_ids"]
        if labels.shape[1] != 5:
            raise ValueError("SIDs must encode as four atomic tokens followed by EOS")
        encoded["labels"] = labels.masked_fill(
            labels == self.tokenizer.pad_token_id, -100
        )
        encoded["pair_query_ids"] = torch.tensor(query_ids)
        encoded["category_labels"] = torch.tensor(
            [r["category"] for r in representatives]
        )
        category_sets = [
            sorted({tuple(d["category"]) for d in r["documents"]}) for r in queries
        ]
        groups = torch.full(
            (len(batch), 3, max(map(len, category_sets))), -100, dtype=torch.long
        )
        for i, qid in enumerate(query_ids):
            categories = category_sets[qid]
            groups[i, :, : len(categories)] = torch.tensor(categories).T
        encoded["group_category_labels"] = groups
        tasks = []
        for qid, record in enumerate(queries):
            docs = record["documents"]
            tasks.extend(
                (qid, task)
                for task in project_documents(
                    [d["document"] for d in docs],
                    [d["code"] for d in docs],
                    [d["weight"] for d in docs],
                )
            )
        targets = torch.zeros(len(tasks), 256)
        legal = torch.zeros(len(tasks), 256, dtype=torch.bool)
        parents = torch.full((len(tasks), 3), -1, dtype=torch.long)
        for i, (_, task) in enumerate(tasks):
            parent = task["parent"]
            if parent:
                parents[i, : len(parent)] = torch.tensor(parent)
            legal[i, list(self.catalog.prefix.children(parent))] = True
            for child, mass in task["child_mass"].items():
                targets[i, child] = mass / task["parent_mass"]
        encoded.update(
            tree_query_ids=torch.tensor([qid for qid, _ in tasks]),
            tree_depths=torch.tensor([len(t["parent"]) for _, t in tasks]),
            tree_parents=parents,
            tree_targets=targets,
            tree_legal=legal,
            tree_weights=torch.tensor([t["parent_mass"] for _, t in tasks]),
        )
        return encoded
