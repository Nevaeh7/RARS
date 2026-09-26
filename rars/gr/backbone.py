import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers.modeling_outputs import Seq2SeqLMOutput
from transformers.models.mt5.modeling_mt5 import MT5ForConditionalGeneration
from transformers.models.t5.modeling_t5 import T5ForConditionalGeneration


def multi_positive_contrastive_loss(
    query_emb, group_emb, group_mask=None, temperature=0.07
):
    B, D = query_emb.shape
    _, G, _ = group_emb.shape

    query_norm = F.normalize(query_emb, dim=-1)
    group_norm = F.normalize(group_emb, dim=-1)

    group_flat = group_norm.view(-1, D)

    logits = torch.matmul(query_norm, group_flat.T) / temperature

    block_mask = (
        torch.eye(B, device=query_emb.device).repeat_interleave(G, dim=1).bool()
    )

    if group_mask is not None:
        flat_valid_mask = group_mask.view(-1).unsqueeze(0)
        final_pos_mask = block_mask & flat_valid_mask
    else:
        final_pos_mask = block_mask

    if group_mask is not None:
        logits = logits.masked_fill(~flat_valid_mask, -torch.inf)
    if not final_pos_mask.any():
        return query_emb.sum() * 0
    log_probs = F.log_softmax(logits.float(), dim=1)

    pos_log_probs = log_probs.masked_select(final_pos_mask)

    if pos_log_probs.numel() == 0:
        return torch.tensor(0.0, device=query_emb.device, requires_grad=True)

    counts = final_pos_mask.sum(1)
    per_query = -(log_probs.masked_fill(~final_pos_mask, 0)).sum(1) / counts.clamp_min(
        1
    )
    loss = per_query[counts > 0].mean()
    return loss


class CategoryClassifier(nn.Module):
    def __init__(self, hidden_size, num_classes, dropout=0.1):
        super().__init__()
        self.projector = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.LayerNorm(hidden_size),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        embedding = self.projector(x)
        logits = self.classifier(embedding)
        return logits, embedding


class LatentDecoderMixin:
    def __init__(self, config):
        super().__init__(config)
        self.num_latent_steps = config.rars_latent_steps
        self.category_heads = nn.ModuleList(
            [CategoryClassifier(config.d_model, n) for n in config.rars_category_sizes]
        )
        self.alpha = self.beta = config.rars_base_weight
        self.loss_fct = CrossEntropyLoss(ignore_index=-100)
        self.category_heads.apply(self._init_weights)
        self.post_init()

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        decoder_input_ids=None,
        past_key_values=None,
        encoder_outputs=None,
        labels=None,
        use_cache=None,
        return_dict=None,
        category_labels=None,
        group_category_labels=None,
        **kwargs,
    ):
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = (
            return_dict if return_dict is not None else self.config.use_return_dict
        )

        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                input_ids=input_ids,
                attention_mask=attention_mask,
                return_dict=True,
                **kwargs,
            )
        is_inference_start = (past_key_values is None) or (
            isinstance(past_key_values, tuple) and len(past_key_values) == 0
        )

        if not is_inference_start and hasattr(past_key_values, "get_seq_length"):
            if past_key_values.get_seq_length() == 0:
                is_inference_start = True

        if labels is not None:
            return self._forward_training(
                input_ids,
                attention_mask,
                decoder_input_ids,
                labels,
                category_labels,
                group_category_labels,
                encoder_outputs,
                return_dict,
                **kwargs,
            )

        elif is_inference_start:
            return self._forward_inference_start(
                decoder_input_ids, encoder_outputs, attention_mask, return_dict
            )

        else:
            return self._forward_inference_step(
                decoder_input_ids,
                past_key_values,
                encoder_outputs,
                attention_mask,
                return_dict,
            )

    def _get_top_k_indices(self, logits, k=100):
        top_k_values, top_k_indices = torch.topk(
            logits, k=min(k, logits.size(-1)), dim=-1
        )
        return top_k_indices

    def _forward_training(
        self,
        input_ids,
        attention_mask,
        decoder_input_ids,
        labels,
        category_labels,
        group_category_labels,
        encoder_outputs,
        return_dict,
        **kwargs,
    ):
        if decoder_input_ids is None and labels is not None:
            decoder_input_ids = self._shift_right(labels)

        decoder_inputs_embeds = self.shared(decoder_input_ids)
        current_input_embeds = decoder_inputs_embeds[:, 0:1, :]
        past_key_values = None

        total_category_loss = torch.tensor(0.0, device=decoder_input_ids.device)
        total_contrastive_loss = torch.tensor(0.0, device=decoder_input_ids.device)

        for i in range(self.num_latent_steps):
            outputs = self.decoder(
                inputs_embeds=current_input_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )

            last_hidden_state = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

            current_latent_state = last_hidden_state.squeeze(1)

            if category_labels is not None:
                step_logits, step_query_emb = self.category_heads[i](
                    current_latent_state
                )

                step_target = category_labels[:, i]
                children = getattr(self.config, "rars_category_children", None)
                if i > 0 and children is not None:
                    legal = torch.zeros_like(step_logits, dtype=torch.bool)
                    for row, parent in enumerate(category_labels[:, i - 1].tolist()):
                        if parent >= 0:
                            legal[row, children[i - 1][parent]] = True
                        else:
                            legal[row] = True
                    step_logits = step_logits.masked_fill(~legal, -torch.inf)
                valid = step_target != -100
                step_ce_loss = (
                    self.loss_fct(step_logits[valid], step_target[valid])
                    if valid.any()
                    else step_logits.sum() * 0
                )
                total_category_loss += step_ce_loss

                if group_category_labels is not None:
                    step_group_ids = group_category_labels[:, i, :]

                    valid_mask = step_group_ids != -100

                    safe_group_ids = step_group_ids.clone()
                    safe_group_ids[~valid_mask] = 0

                    class_weights = self.category_heads[i].classifier.weight

                    step_group_embs = F.embedding(safe_group_ids, class_weights)

                    step_contra_loss = multi_positive_contrastive_loss(
                        query_emb=step_query_emb,
                        group_emb=step_group_embs,
                        group_mask=valid_mask,
                        temperature=0.1,
                    )
                    total_contrastive_loss += step_contra_loss

            current_input_embeds = last_hidden_state

        latent_output = current_input_embeds
        if self.config.tie_word_embeddings:
            latent_output = latent_output * (self.model_dim**-0.5)
        logits_first_token = self.lm_head(latent_output)

        text_inputs_embeds = decoder_inputs_embeds[:, 1:, :]
        lm_logits_rest = None

        if text_inputs_embeds.shape[1] > 0:
            outputs = self.decoder(
                inputs_embeds=text_inputs_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                use_cache=False,
                return_dict=True,
            )
            sequence_output = outputs.last_hidden_state
            if self.config.tie_word_embeddings:
                sequence_output = sequence_output * (self.model_dim**-0.5)
            lm_logits_rest = self.lm_head(sequence_output)

        if lm_logits_rest is not None:
            lm_logits = torch.cat([logits_first_token, lm_logits_rest], dim=1)
        else:
            lm_logits = logits_first_token

        lm_loss = self.loss_fct(lm_logits.view(-1, lm_logits.size(-1)), labels.view(-1))

        final_loss = lm_loss.clone()
        if category_labels is not None:
            final_loss += self.alpha * total_category_loss
        if group_category_labels is not None:
            final_loss += self.beta * total_contrastive_loss

        if not return_dict:
            return (final_loss, lm_logits)

        output = Seq2SeqLMOutput(
            loss=final_loss,
            logits=lm_logits,
            past_key_values=None,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

        output.loss_lm = lm_loss
        output.loss_category = self.alpha * total_category_loss
        output.loss_contrastive = self.beta * total_contrastive_loss

        self._last_loss_lm = lm_loss
        self._last_loss_category = self.alpha * total_category_loss
        self._last_loss_contrastive = self.beta * total_contrastive_loss

        return output

    def _forward_inference_start(
        self, decoder_input_ids, encoder_outputs, attention_mask, return_dict
    ):
        current_input_embeds = self.shared(decoder_input_ids)
        past_key_values = None
        for _ in range(self.num_latent_steps):
            outputs = self.decoder(
                inputs_embeds=current_input_embeds,
                past_key_values=past_key_values,
                encoder_hidden_states=encoder_outputs.last_hidden_state,
                encoder_attention_mask=attention_mask,
                use_cache=True,
                return_dict=True,
            )
            current_input_embeds = outputs.last_hidden_state
            past_key_values = outputs.past_key_values

        sequence_output = current_input_embeds
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)

        lm_logits = self.lm_head(sequence_output)
        if not return_dict:
            return (lm_logits, past_key_values)

        return Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )

    def _forward_inference_step(
        self,
        decoder_input_ids,
        past_key_values,
        encoder_outputs,
        attention_mask,
        return_dict,
    ):
        outputs = self.decoder(
            input_ids=decoder_input_ids,
            past_key_values=past_key_values,
            encoder_hidden_states=encoder_outputs.last_hidden_state,
            encoder_attention_mask=attention_mask,
            use_cache=True,
            return_dict=True,
        )
        sequence_output = outputs.last_hidden_state
        if self.config.tie_word_embeddings:
            sequence_output = sequence_output * (self.model_dim**-0.5)
        lm_logits = self.lm_head(sequence_output)

        if not return_dict:
            return (lm_logits, outputs.past_key_values)

        return Seq2SeqLMOutput(
            logits=lm_logits,
            past_key_values=outputs.past_key_values,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
        )


class LatentT5(LatentDecoderMixin, T5ForConditionalGeneration):
    pass


class LatentMT5(LatentDecoderMixin, MT5ForConditionalGeneration):
    pass
