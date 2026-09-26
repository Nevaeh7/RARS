"""Trie-constrained beam search, including EOS and optional prefix compatibility."""

import math
from collections import defaultdict

import numpy as np
import torch
from transformers import AutoTokenizer

from rars.common import read_json, sha256, write_json

from .compatibility import LexicalCatalog
from .data import GRADES, PROMPT, Catalog
from .model import load_retriever
from .prefix import aggregate_prefix_feasibility, build_prefix_assignments


def metrics(ranking, targets):
    ranking = list(dict.fromkeys(ranking))
    if not targets:
        raise ValueError("Cannot evaluate a query with no positive judgments")
    result = {
        f"recall@{k}": 100 * sum(sid in targets for sid in ranking[:k]) / len(targets)
        for k in (5, 10, 100)
    }
    for k in (10, 100):
        ideal = sum(
            g / math.log2(i + 2)
            for i, g in enumerate(sorted(targets.values(), reverse=True)[:k])
        )
        actual = sum(
            targets.get(sid, 0) / math.log2(i + 2) for i, sid in enumerate(ranking[:k])
        )
        result[f"ndcg@{k}"] = 100 * actual / ideal if ideal else 0.0
    return result


def reorder_cache(cache, beam_indices):
    if hasattr(cache, "reorder_cache"):
        cache.reorder_cache(beam_indices)
        return cache
    return tuple(
        tuple(t.index_select(0, beam_indices) for t in layer) for layer in cache
    )


class Retriever:
    def __init__(
        self,
        model,
        tokenizer,
        catalog,
        *,
        beam_size=100,
        category_width=3,
        compatibility=None,
    ):
        self.model, self.tokenizer, self.catalog = model, tokenizer, catalog
        self.device = next(model.parameters()).device
        self.beam_size, self.category_width = beam_size, category_width
        if beam_size <= 0 or category_width <= 0:
            raise ValueError("Beam size and category width must be positive")
        token_codes = []
        for row in range(len(catalog.indices)):
            ids = [
                tokenizer.encode(token, add_special_tokens=False)
                for token in catalog.indices[str(row)]
            ]
            if any(len(v) != 1 for v in ids):
                raise ValueError("Non-atomic SID token")
            token_codes.append([v[0] for v in ids])
        self.token_codes = np.asarray(token_codes)
        self.assignments = build_prefix_assignments(self.token_codes)
        self.compatibility = compatibility
        self.trie_cache = {}

    def trie(self, categories):
        categories = tuple(sorted(categories))
        if categories not in self.trie_cache:
            trie = defaultdict(set)
            for category in categories:
                for row in self.catalog.category_items[category]:
                    code = tuple(map(int, self.token_codes[row]))
                    for depth in range(4):
                        trie[code[:depth]].add(code[depth])
                    trie[code].add(self.tokenizer.eos_token_id)
            if not trie:
                raise ValueError("Predicted categories have no catalog SIDs")
            # Bound caching during large-catalog evaluation.
            if len(self.trie_cache) >= 128:
                self.trie_cache.clear()
            self.trie_cache[categories] = {key: sorted(v) for key, v in trie.items()}
        return self.trie_cache[categories]

    @torch.no_grad()
    def search(self, query, *, weight=2.0, depth_mask=(1, 1, 1, 1)):
        model = self.model
        encoded = self.tokenizer(
            PROMPT + query,
            return_tensors="pt",
            truncation=True,
            max_length=512,
            return_token_type_ids=False,
        ).to(self.device)
        hidden = model.encoder(**encoded, return_dict=True).last_hidden_state
        mask = encoded["attention_mask"]
        current = model.shared(
            torch.tensor([[model.config.decoder_start_token_id]], device=self.device)
        )
        cache = None
        for depth in range(model.num_latent_steps):
            result = model.decoder(
                inputs_embeds=current,
                past_key_values=cache,
                encoder_hidden_states=hidden,
                encoder_attention_mask=mask,
                use_cache=True,
                return_dict=True,
            )
            current, cache = result.last_hidden_state, result.past_key_values
        category_logits, _ = model.category_heads[-1](current[:, -1])
        # Only categories represented by complete category paths can admit SIDs.
        available = sorted(self.catalog.category_items)
        category_ids = torch.tensor(available, device=self.device)
        choices = (
            category_logits[0, category_ids]
            .topk(min(self.category_width, len(available)))
            .indices
        )
        trie = self.trie([available[i] for i in choices.tolist()])
        scores_by_prefix = None
        if weight != 0 and self.compatibility is not None:
            item_scores = self.compatibility.item_scores(
                self.compatibility.compile(query)
            )
            scores_by_prefix = aggregate_prefix_feasibility(
                item_scores, self.assignments
            )
        if model.config.tie_word_embeddings:
            current = current * model.model_dim**-0.5
        logits = model.lm_head(current[:, -1]).float()
        paths = [()]
        cumulative = torch.zeros(1, device=self.device)
        for depth in range(5):  # four semantic tokens followed by EOS
            logp = torch.log_softmax(logits, -1)
            owners, candidate_tokens, biases = [], [], []
            for beam, path in enumerate(paths):
                for token in trie.get(path, []):
                    owners.append(beam)
                    candidate_tokens.append(token)
                    bias = 0.0
                    if depth < 4 and scores_by_prefix is not None and depth_mask[depth]:
                        row = self.assignments[depth].key_to_row[path + (token,)]
                        bias = weight * float(scores_by_prefix[depth][row])
                    biases.append(bias)
            if not owners:
                raise RuntimeError("Beam has no legal continuation")
            owner_tensor = torch.tensor(owners, device=self.device)
            token_tensor = torch.tensor(candidate_tokens, device=self.device)
            scores = cumulative[owner_tensor] + logp[owner_tensor, token_tensor]
            scores = scores + scores.new_tensor(biases)
            values, order = scores.topk(min(self.beam_size, len(owners)))
            parent_ids = owner_tensor[order]
            tokens = token_tensor[order]
            selected = [(owners[i], candidate_tokens[i], 0) for i in order.tolist()]
            paths = [paths[i] + (token,) for i, token, _ in selected]
            cumulative = values
            if depth == 4:
                break
            cache = reorder_cache(cache, parent_ids)
            hidden, mask = (
                hidden.index_select(0, parent_ids),
                mask.index_select(0, parent_ids),
            )
            result = model.decoder(
                input_ids=tokens[:, None],
                past_key_values=cache,
                encoder_hidden_states=hidden,
                encoder_attention_mask=mask,
                use_cache=True,
                return_dict=True,
            )
            cache = result.past_key_values
            current = result.last_hidden_state[:, -1]
            if model.config.tie_word_embeddings:
                current = current * model.model_dim**-0.5
            logits = model.lm_head(current).float()
        return [
            self.tokenizer.decode(list(path[:-1]), skip_special_tokens=True).replace(
                " ", ""
            )
            for path in paths
        ]


def evaluation_queries(catalog, protocol="released"):
    """Select queries whose first positive SID has a complete category path.

    A complete path contains at least three category levels. If a SID represents
    multiple products, its category path comes from the last catalog entry.
    """
    if protocol != "released":
        raise ValueError("Only the released query protocol is supported")
    sid_categories = {}
    for product, path in catalog.categories.items():
        sid = "".join(catalog.indices[str(catalog.product_to_id[product])])
        sid_categories[sid] = path
    raw = read_json(catalog.directory / f"{catalog.dataset}.test.seen.json")
    selected = {}
    for query, judgments in raw.items():
        first_sid = next(
            (
                "".join(catalog.indices[str(catalog.product_to_id[str(p)])])
                for p, grade in judgments
                if GRADES[grade] > 0
            ),
            None,
        )
        if first_sid is not None and len(sid_categories.get(first_sid, [])) >= 3:
            selected[query] = judgments
    return selected


def evaluate(
    directory,
    index_path,
    checkpoint,
    output,
    settings,
    *,
    device="cuda:0",
    limit_queries=0,
):
    catalog = Catalog(directory, index_path)
    tokenizer = AutoTokenizer.from_pretrained(checkpoint, use_fast=False)
    model = load_retriever(checkpoint, device)
    if model.config.rars_index_sha256 != sha256(index_path):
        raise ValueError("Checkpoint was trained with a different SID index")
    if model.config.rars_category_mapping != catalog.category_names:
        raise ValueError("Checkpoint category mapping differs from the catalog")
    if getattr(model.config, "rars_data_sha256", None) != catalog.data_hashes():
        raise ValueError(
            "Checkpoint catalog data hashes are missing or differ from the current data"
        )
    compatibility = LexicalCatalog(
        str(catalog.directory / f"{catalog.dataset}.item.json"),
        max_lexical_features=settings["lexical_features"],
        lexical_weight=settings["lexical_weight"],
        max_lexical_df_ratio=settings.get("max_lexical_df_ratio", 0.02),
        unicode_nfkc=True,
        enable_measurements=True,
        enable_negation=False,
    )
    retriever = Retriever(
        model,
        tokenizer,
        catalog,
        beam_size=settings["beam_size"],
        category_width=settings["category_width"],
        compatibility=compatibility,
    )
    queries = evaluation_queries(catalog, settings["query_protocol"])
    rows = []
    for query, judgments in queries.items():
        targets = {}
        for product, grade in judgments:
            if GRADES[grade] <= 0:
                continue
            sid = "".join(catalog.indices[str(catalog.product_to_id[str(product)])])
            # Repeated SID keys take the last positive judgment in input order.
            targets[sid] = GRADES[grade]
        if not targets:
            continue
        row = dict(query=query)
        for name, weight in [("all_levels", settings["compatibility_weight"])]:
            ranking = retriever.search(query, weight=weight)
            row[name] = dict(metrics=metrics(ranking, targets), ranking=ranking)
        rows.append(row)
        if limit_queries and len(rows) >= limit_queries:
            break
    if not rows:
        raise ValueError("No evaluable test queries")
    aggregate = {
        name: {
            metric: float(np.mean([r[name]["metrics"][metric] for r in rows]))
            for metric in rows[0][name]["metrics"]
        }
        for name in ["all_levels"]
    }
    result = dict(
        queries=len(rows),
        metrics=aggregate,
        settings=settings,
        tree_head_used=False,
        evaluation_unit="unique SID; last positive judgment for shared SIDs",
        data_sha256=model.config.rars_data_sha256,
        test_judgments_sha256=sha256(
            catalog.directory / f"{catalog.dataset}.test.seen.json"
        ),
    )
    write_json(output / "evaluation.json", result)
    write_json(output / "rankings.json", rows)
    return result
