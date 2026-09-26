"""Train RARS for a fixed duration and save the final checkpoint."""

import math
import os

import torch
from transformers import AutoTokenizer, Trainer, TrainingArguments, set_seed

from rars.common import fresh_directory, sha256, write_json

from .data import Catalog, Collator, TrainDataset
from .model import load_pretrained


class FiniteTrainer(Trainer):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # compute_loss returns a minibatch mean; it does not consume Trainer's
        # accumulated token count. Let Trainer normalize gradient accumulation.
        self.model_accepts_loss_kwargs = False

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        output = model(**inputs)
        if not torch.isfinite(output.loss):
            raise FloatingPointError("Nonfinite training loss")
        return (output.loss, output) if return_outputs else output.loss

    def log(self, logs, *args, **kwargs):
        if "grad_norm" in logs and not math.isfinite(float(logs["grad_norm"])):
            raise FloatingPointError("Nonfinite gradient norm")
        super().log(logs, *args, **kwargs)


def train(
    directory,
    index_path,
    output,
    settings,
    *,
    source,
    seed=42,
    device="cuda:0",
    max_steps=-1,
    limit_queries=0,
):
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This pipeline requires a single training process")
    if device != "cpu" and torch.cuda.is_available() and torch.cuda.device_count() != 1:
        raise ValueError(
            "Select one GPU with CUDA_VISIBLE_DEVICES for this single-GPU pipeline"
        )
    output = fresh_directory(output)
    set_seed(seed)
    catalog = Catalog(directory, index_path)
    dataset = TrainDataset(
        catalog,
        validation_fraction=settings["validation_fraction"],
        limit_queries=limit_queries,
    )
    tokenizer = AutoTokenizer.from_pretrained(source, use_fast=False)
    tokenizer.add_tokens(catalog.new_tokens())
    model = load_pretrained(source, tokenizer, catalog, settings)
    model.config.rars_category_mapping = catalog.category_names
    model.config.rars_index_sha256 = sha256(index_path)
    model.config.rars_dataset = catalog.dataset
    model.config.rars_data_sha256 = catalog.data_hashes()
    cpu = device == "cpu"
    if not cpu and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA was requested but is unavailable; pass --device cpu for a smoke test"
        )
    options = TrainingArguments(
        output_dir=str(output),
        seed=seed,
        data_seed=seed,
        learning_rate=settings["learning_rate"],
        num_train_epochs=settings["epochs"],
        per_device_train_batch_size=settings["batch_size"],
        gradient_accumulation_steps=settings["gradient_accumulation"],
        weight_decay=0.01,
        warmup_ratio=0.05,
        lr_scheduler_type="cosine",
        max_grad_norm=0.5,
        save_strategy="epoch",
        save_total_limit=2,
        eval_strategy="no",
        logging_steps=1 if max_steps > 0 else 20,
        logging_first_step=True,
        bf16=not cpu and settings["bf16"],
        use_cpu=cpu,
        report_to="none",
        remove_unused_columns=False,
        dataloader_num_workers=0,
        optim="adamw_torch",
        disable_tqdm=True,
        max_steps=max_steps,
        logging_nan_inf_filter=False,
    )
    trainer = FiniteTrainer(
        model=model,
        args=options,
        train_dataset=dataset,
        processing_class=tokenizer,
        data_collator=Collator(tokenizer, catalog, settings["max_length"]),
    )
    if not cpu:
        torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    trainer.save_model(str(output / "final"))
    tokenizer.save_pretrained(output / "final")
    trainer.save_state()
    for name, param in model.named_parameters():
        if not torch.isfinite(param).all():
            raise FloatingPointError("Nonfinite saved parameter: " + name)
    receipt = dict(
        status="smoke_passed" if max_steps > 0 else "complete",
        optimizer_steps=trainer.state.global_step,
        epochs=trainer.state.epoch,
        seed=seed,
        pairs=len(dataset),
        queries=len(dataset.records),
        dataset=catalog.dataset,
        settings=settings,
        final_components=model.last_components,
        metrics=result.metrics,
        index_sha256=sha256(index_path),
        data_sha256=model.config.rars_data_sha256,
        checkpoint_selection="final",
        peak_gpu_bytes=torch.cuda.max_memory_allocated() if not cpu else 0,
        torch=torch.__version__,
    )
    write_json(output / "training.json", receipt)
    return receipt
