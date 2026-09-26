"""Project document relevance onto a collision-tolerant four-level SID trie."""

import math
from collections import defaultdict
from functools import lru_cache

import numpy as np
import torch
import torch.nn.functional as F
from torch import nn


def pack(code):
    value = 0
    if len(code) != 4:
        raise ValueError("expected a four-level SID")
    for digit in code:
        if not 0 <= int(digit) < 256:
            raise ValueError("SID digit outside [0,255]")
        value = (value << 8) | int(digit)
    return value


class PrefixCatalog:
    """Legal-child catalog over unique paths; product collisions are allowed."""

    def __init__(self, codes):
        values = np.asarray(codes)
        if (
            values.ndim != 2
            or values.shape[1] != 4
            or not np.issubdtype(values.dtype, np.integer)
        ):
            raise ValueError("integer Nx4 SID paths required")
        if not len(values) or values.min() < 0 or values.max() > 255:
            raise ValueError("invalid SID catalog")
        keys = np.asarray([pack(row) for row in values], dtype=np.uint64)
        self.keys = np.unique(keys)

    @lru_cache(maxsize=100000)
    def children(self, parent):
        parent = tuple(map(int, parent))
        depth = len(parent)
        if depth >= 4:
            return ()
        value = 0
        for digit in parent:
            if not 0 <= digit < 256:
                raise ValueError("invalid prefix")
            value = (value << 8) | digit
        shift = 8 * (4 - depth)
        bounds = np.asarray([value << shift, (value + 1) << shift], dtype=np.uint64)
        left, right = np.searchsorted(self.keys, bounds)
        return tuple(
            int(x)
            for x in np.unique((self.keys[left:right] >> (8 * (3 - depth))) & 255)
        )


def project_documents(document_ids, paths, weights):
    """Deduplicate judgments by document, then sum document mass at SID paths.

    Distinct products sharing a SID contribute additive relevance mass.
    Repeated judgments for the same product use the maximum relevance weight.
    """
    if not len(document_ids) == len(paths) == len(weights):
        raise ValueError("document/path/evidence cardinality mismatch")
    documents = {}
    for raw_id, raw_path, raw_weight in zip(document_ids, paths, weights):
        document = str(raw_id)
        path = tuple(map(int, raw_path))
        pack(path)
        weight = float(raw_weight)
        if not document or not math.isfinite(weight) or weight < 0:
            raise ValueError("invalid document evidence")
        if document in documents and documents[document][0] != path:
            raise ValueError("one document maps to multiple SIDs")
        if weight > 0:
            documents[document] = (
                path,
                max(weight, documents.get(document, (path, 0.0))[1]),
            )
    path_weight = defaultdict(float)
    for path, weight in documents.values():
        path_weight[path] += weight
    total = sum(path_weight.values())
    if total <= 0:
        raise ValueError("query has no positive relevance")
    children = defaultdict(lambda: defaultdict(float))
    for path, weight in sorted(path_weight.items()):
        mass = weight / total
        for depth in range(4):
            children[path[:depth]][path[depth]] += mass
    rows = []
    for parent in sorted(children, key=lambda value: (len(value), value)):
        masses = dict(sorted(children[parent].items()))
        rows.append(
            {
                "parent": parent,
                "parent_mass": sum(masses.values()),
                "child_mass": masses,
            }
        )
    return rows


def parent_weighted_kl(
    logits, targets, weights, query_ids, legal, batch_size, negatives=32
):
    if logits.shape != targets.shape or logits.shape != legal.shape or logits.ndim != 2:
        raise ValueError("tree tensor shape mismatch")
    if batch_size <= 0 or negatives < 0:
        raise ValueError("invalid tree loss settings")
    logits = logits.float()
    targets = targets.float()
    weights = weights.float()
    positive = targets > 0
    if bool((positive & ~legal).any()) or bool((targets < 0).any()):
        raise ValueError("illegal or negative tree target")
    if not torch.allclose(targets.sum(-1), torch.ones_like(weights), atol=1e-6):
        raise ValueError("conditional target must sum to one")
    if bool((weights <= 0).any()) or not bool(torch.isfinite(weights).all()):
        raise ValueError("parent mass must be positive and finite")
    active = positive.clone()
    if negatives:
        candidate = logits.detach().masked_fill(~legal | positive, -torch.inf)
        values, ids = candidate.topk(min(negatives, logits.shape[-1]), dim=-1)
        active |= torch.zeros_like(active).scatter_(1, ids, torch.isfinite(values))
    logp = F.log_softmax(logits.masked_fill(~active, -torch.inf), -1)
    safe = logp.masked_fill(~positive, 0.0)
    kl = (targets * targets.clamp_min(1e-30).log() - targets * safe).sum(-1)
    per_query = logits.new_zeros(batch_size).index_add(0, query_ids, weights * kl)
    return per_query.mean()


class ParentConditionedHead(nn.Module):
    def __init__(self, hidden_size, rank=64):
        super().__init__()
        self.query_projection = nn.ModuleList(
            nn.Linear(hidden_size, rank, bias=False) for _ in range(4)
        )
        self.child = nn.Parameter(torch.empty(4, 256, rank))
        self.parent = nn.Parameter(torch.empty(3, 256, rank))
        self.bias = nn.Parameter(torch.zeros(4, 256))
        self.rank = rank
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.normal_(self.child, mean=0.0, std=0.02)
        nn.init.normal_(self.parent, mean=0.0, std=0.02)
        nn.init.zeros_(self.bias)

    def forward(self, query_features, query_ids, depths, parent_codes):
        scores = query_features.new_zeros((len(depths), 256))
        for depth in range(4):
            selected = (depths == depth).nonzero().flatten()
            if not len(selected):
                scores = (
                    scores + 0 * self.query_projection[depth](query_features[:1]).sum()
                )
                continue
            query = self.query_projection[depth](query_features)[query_ids[selected]]
            context = query.new_zeros(query.shape)
            for position in range(depth):
                context += self.parent[position, parent_codes[selected, position]]
            query = query * (1 + torch.tanh(context))
            logits = (
                query @ self.child[depth].T / math.sqrt(self.rank) + self.bias[depth]
            )
            scores = scores.index_copy(0, selected, logits.to(scores.dtype))
        return scores
