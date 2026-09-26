"""RARS augments the CaLIR full-SID objective with query-level tree KL."""

import torch
from transformers import AutoConfig
from transformers.modeling_outputs import BaseModelOutput

from .backbone import LatentMT5, LatentT5
from .tree import ParentConditionedHead, parent_weighted_kl


class RARSMixin:
    def _init_weights(self, module):
        super()._init_weights(module)
        if isinstance(module, torch.nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        if isinstance(module, torch.nn.LayerNorm):
            torch.nn.init.ones_(module.weight)
            torch.nn.init.zeros_(module.bias)
        if isinstance(module, ParentConditionedHead):
            module.reset_parameters()

    def __init__(self, config):
        super().__init__(config)
        self.tree_head = ParentConditionedHead(config.d_model, config.rars_tree_rank)
        self.tree_head.apply(self._init_weights)
        self.last_components = {}
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        labels=None,
        pair_query_ids=None,
        tree_query_ids=None,
        tree_depths=None,
        tree_parents=None,
        tree_targets=None,
        tree_weights=None,
        tree_legal=None,
        **kwargs,
    ):
        if labels is None or pair_query_ids is None:
            return super().forward(
                input_ids=input_ids,
                attention_mask=attention_mask,
                labels=labels,
                **kwargs,
            )
        hidden = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, return_dict=True
        ).last_hidden_state
        outputs = super().forward(
            attention_mask=attention_mask[pair_query_ids],
            labels=labels,
            encoder_outputs=BaseModelOutput(last_hidden_state=hidden[pair_query_ids]),
            **kwargs,
        )
        mask = attention_mask.unsqueeze(-1).to(hidden.dtype)
        pooled = (hidden * mask).sum(1) / mask.sum(1).clamp_min(1)
        logits = self.tree_head(pooled, tree_query_ids, tree_depths, tree_parents)
        tree = parent_weighted_kl(
            logits,
            tree_targets,
            tree_weights,
            tree_query_ids,
            tree_legal,
            len(input_ids),
            self.config.rars_sibling_negatives,
        )
        outputs.loss = outputs.loss + self.config.rars_tree_weight * tree
        self.last_components = dict(
            leaf=float(outputs.loss_lm.detach()),
            category=float(outputs.loss_category.detach()),
            contrastive=float(outputs.loss_contrastive.detach()),
            tree=float(tree.detach()),
        )
        if not torch.isfinite(outputs.loss):
            raise FloatingPointError("Non-finite RARS objective")
        return outputs


class RARST5(RARSMixin, LatentT5):
    pass


class RARSMT5(RARSMixin, LatentMT5):
    pass


def configure(config, category_sizes, settings):
    if config.model_type not in ("t5", "mt5"):
        raise ValueError("RARS requires a T5 or mT5 encoder-decoder backbone")
    if config.model_type == "mt5":
        config.tie_word_embeddings = False
    config.rars_latent_steps = 3
    config.rars_category_sizes = list(category_sizes)
    config.rars_base_weight = settings["base_weight"]
    config.rars_tree_weight = settings["tree_weight"]
    config.rars_tree_rank = settings["tree_rank"]
    config.rars_sibling_negatives = settings["sibling_negatives"]
    return config


def model_class(config):
    return RARST5 if config.model_type == "t5" else RARSMT5


def load_pretrained(source, tokenizer, catalog, settings):
    config = configure(
        AutoConfig.from_pretrained(source), catalog.category_sizes, settings
    )
    config.rars_category_children = catalog.category_children()
    model = model_class(config).from_pretrained(source, config=config)
    model.resize_token_embeddings(len(tokenizer), mean_resizing=False)
    if (
        config.model_type == "mt5"
        and model.shared.weight.data_ptr() == model.lm_head.weight.data_ptr()
    ):
        raise ValueError("mT5 input/output weights must remain untied")
    return model


def load_retriever(checkpoint, device):
    config = AutoConfig.from_pretrained(checkpoint)
    model = model_class(config).from_pretrained(checkpoint).to(device).eval()
    del model.tree_head
    return model
