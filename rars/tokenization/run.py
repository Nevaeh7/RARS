"""Train the residual quantizer and export the four-level semantic index."""

import random
from collections import defaultdict

import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from rars.common import fresh_directory, sha256, write_json

from .rqvae import RQVAE


def train_and_index(embedding_path, output, settings, device="cuda:0", seed=42):
    output = fresh_directory(output)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    matrix = np.load(embedding_path, allow_pickle=False)
    if matrix.ndim != 2 or not len(matrix) or not np.isfinite(matrix).all():
        raise ValueError("Expected finite, nonempty item embedding matrix")
    if len(matrix) < 256:
        raise ValueError(
            "Four 256-entry codebooks require at least 256 product embeddings"
        )
    data = torch.from_numpy(matrix.astype(np.float32, copy=False))
    options = dict(
        in_dim=matrix.shape[1],
        num_emb_list=[256] * 4,
        e_dim=settings["code_dim"],
        layers=settings["layers"],
        dropout_prob=0.0,
        bn=False,
        loss_type="mse",
        quant_loss_weight=1.0,
        kmeans_init=True,
        kmeans_iters=10,
        sk_epsilons=[0.0, 0.0, 0.0, 0.003],
        sk_iters=50,
    )
    model = RQVAE(**options).to(device)
    # Bound the encoder initialization memory while keeping full-catalog k-means.
    with torch.no_grad():
        encoded = torch.cat(
            [
                model.encoder(batch.to(device)).cpu()
                for (batch,) in DataLoader(
                    TensorDataset(data), batch_size=settings["batch_size"]
                )
            ]
        ).to(device)
        model.rq.vq_ini(encoded)
        del encoded
    labels = {str(depth): [] for depth in range(4)}
    loader = DataLoader(
        TensorDataset(data), batch_size=settings["batch_size"], shuffle=True
    )
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=settings["learning_rate"], weight_decay=1e-4
    )
    history = []
    for epoch in range(settings["epochs"]):
        model.train()
        total = 0.0
        for (batch,) in loader:
            batch = batch.to(device)
            optimizer.zero_grad(set_to_none=True)
            reconstruction, quantization, _, dense = model(batch, labels)
            loss, _, _, _ = model.compute_loss(
                reconstruction, quantization, None, dense, xs=batch
            )
            if not torch.isfinite(loss):
                raise FloatingPointError("Nonfinite RQ-VAE loss")
            loss.backward()
            if any(
                p.grad is not None and not torch.isfinite(p.grad).all()
                for p in model.parameters()
            ):
                raise FloatingPointError("Nonfinite RQ-VAE gradient")
            optimizer.step()
            total += float(loss.detach()) * len(batch)
        history.append(dict(epoch=epoch + 1, loss=total / len(data)))
        print(
            f"Tokenization epoch {epoch + 1}: loss={total / len(data):.6f}", flush=True
        )
    checkpoint = output / "rqvae.pt"
    torch.save(
        dict(
            config=options,
            state_dict=model.state_dict(),
            seed=seed,
            epochs=settings["epochs"],
        ),
        checkpoint,
    )
    # Generate identifiers from the saved quantizer checkpoint.
    saved = torch.load(checkpoint, map_location="cpu", weights_only=True)
    model = RQVAE(**saved["config"]).to(device)
    model.load_state_dict(saved["state_dict"], strict=True)
    model.eval()
    with torch.no_grad():
        codes = torch.cat(
            [
                model.get_indices(batch.to(device), labels, use_sk=False).cpu()
                for (batch,) in DataLoader(
                    TensorDataset(data), batch_size=settings["batch_size"]
                )
            ]
        ).numpy()
        # Refine colliding codes using Sinkhorn assignments at the final level.
        for _ in range(settings["collision_refinements"]):
            groups = defaultdict(list)
            for row, code in enumerate(codes):
                groups[tuple(code)].append(row)
            collisions = [rows for rows in groups.values() if len(rows) > 1]
            if not collisions:
                break
            for rows in collisions:
                codes[rows] = (
                    model.get_indices(data[rows].to(device), labels, use_sk=True)
                    .cpu()
                    .numpy()
                )
    index = {
        str(i): [f"<{letter}_{int(value)}>" for letter, value in zip("abcd", code)]
        for i, code in enumerate(codes)
    }
    write_json(output / "index.json", index)
    unique = len({tuple(code) for code in codes})
    receipt = dict(
        items=len(data),
        unique_sids=unique,
        collision_rate=1 - unique / len(data),
        embedding_sha256=sha256(embedding_path),
        index_sha256=sha256(output / "index.json"),
        checkpoint_sha256=sha256(checkpoint),
        settings=settings,
        seed=seed,
        history=history,
    )
    write_json(output / "tokenization.json", receipt)
    return receipt
