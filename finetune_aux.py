import argparse
import os
import torch
import numpy as np
import pandas as pd
import json
import random
from tqdm import tqdm

from residual_stream.utils import load_model_and_tokenizer
from residual_stream.utils import load_datasets

from functools import partial
from typing import Any, Dict, List, Optional, Tuple, Union
from transformers import Cache
from transformers.modeling_outputs import BaseModelOutputWithPast, ModelOutput
from transformers.processing_utils import Unpack

from transformers.models.llama.modeling_llama import KwargsForCausalLM

from transformers import DynamicCache

from typing import Optional, Tuple

from functools import partial


class BaseModelOutputWithResidualStream(BaseModelOutputWithPast):
    """
    继承 BaseModelOutputWithPast，并增加 residual_stream, logits, predicted_token
    """
    def __init__(
        self,
        last_hidden_state: torch.FloatTensor,
        past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None,
        hidden_states: Optional[Tuple[torch.FloatTensor]] = None,
        attentions: Optional[Tuple[torch.FloatTensor]] = None,
        residual_stream: Optional[list] = None,  # 记录最后一层的第一 token 残差
        logits: Optional[torch.FloatTensor] = None,  # 记录 logits
        predicted_token: Optional[int] = None,  # 记录预测的 token id
        aux_loss: float = 0
    ):
        super().__init__(
            last_hidden_state=last_hidden_state,
            past_key_values=past_key_values,
            hidden_states=hidden_states,
            attentions=attentions
        )
        self.residual_stream = residual_stream if residual_stream is not None else []
        self.logits = logits if logits is not None else None
        self.predicted_token = predicted_token
        self.aux_loss = aux_loss if aux_loss is not None else None
        
        
def forward1(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    **kwargs
) -> Union[Tuple, BaseModelOutputWithPast]:
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    use_cache = use_cache if use_cache is not None else self.config.use_cache
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict

    if (input_ids is None) ^ (inputs_embeds is not None):
        raise ValueError("You must specify exactly one of input_ids or inputs_embeds")

    if self.gradient_checkpointing and self.training and use_cache:
        logger.warning_once(
            "`use_cache=True` is incompatible with gradient checkpointing. Setting `use_cache=False`."
        )
        use_cache = False

    if inputs_embeds is None:
        inputs_embeds = self.embed_tokens(input_ids)

    # kept for BC (non `Cache` `past_key_values` inputs)
    return_legacy_cache = False
    if use_cache and not isinstance(past_key_values, Cache):
        return_legacy_cache = True
        if past_key_values is None:
            past_key_values = DynamicCache()
        else:
            past_key_values = DynamicCache.from_legacy_cache(past_key_values)
            logger.warning_once(
                "We detected that you are passing `past_key_values` as a tuple of tuples. This is deprecated and "
                "will be removed in v4.47. Please convert your cache or use an appropriate `Cache` class "
                "(https://huggingface.co/docs/transformers/kv_cache#legacy-cache-format)"
            )

    if cache_position is None:
        past_seen_tokens = past_key_values.get_seq_length() if past_key_values is not None else 0
        cache_position = torch.arange(
            past_seen_tokens, past_seen_tokens + inputs_embeds.shape[1], device=inputs_embeds.device
        )
    if position_ids is None:
        position_ids = cache_position.unsqueeze(0)

    causal_mask = self._update_causal_mask(
        attention_mask, inputs_embeds, cache_position, past_key_values, output_attentions
    )
    hidden_states = inputs_embeds

    # create position embeddings to be shared across the decoder layers
    position_embeddings = self.rotary_emb(hidden_states, position_ids)

    # decoder layers
    all_hidden_states = () if output_hidden_states else None
    all_self_attns = () if output_attentions else None
    next_decoder_cache = None

    residual_stream = {}

    # for i, decoder_layer in enumerate(self.layers):
    # for i, decoder_layer in enumerate(tqdm(self.layers, desc="Processing Decoder Layers")):
    for i, decoder_layer in enumerate(self.layers):
        residual_stream[i] = hidden_states[:, -1, :] # batch, sequence, vector
        
        if output_hidden_states:
            all_hidden_states += (hidden_states,)

        if self.gradient_checkpointing and self.training:
            layer_outputs = self._gradient_checkpointing_func(
                decoder_layer.__call__,
                hidden_states,
                causal_mask,
                position_ids,
                past_key_values,
                output_attentions,
                use_cache,
                cache_position,
                position_embeddings,
            )
        else:
            layer_outputs = decoder_layer(
                hidden_states,
                attention_mask=causal_mask,
                position_ids=position_ids,
                past_key_value=past_key_values,
                output_attentions=output_attentions,
                use_cache=use_cache,
                cache_position=cache_position,
                position_embeddings=position_embeddings,
            )

        hidden_states = layer_outputs[0]

        if use_cache:
            next_decoder_cache = layer_outputs[2 if output_attentions else 1]

        if output_attentions:
            all_self_attns += (layer_outputs[1],)

    hidden_states = self.norm(hidden_states)

    # add hidden states from the last decoder layer
    if output_hidden_states:
        all_hidden_states += (hidden_states,)

    next_cache = next_decoder_cache if use_cache else None
    if return_legacy_cache:
        next_cache = next_cache.to_legacy_cache()

    def compute_structural_loss(residual_stream, layer_idx=10, k=3, gamma=1.0, l=5):
        """
        Compute the structural auxiliary loss over a range of layers up to `layer_idx`.

        Args:
            residual_stream (List[Tensor]): list of residual tensors per layer, each of shape (B, d)
            layer_idx (int): which layer to select residual from (inclusive)
            k (int): how many top singular values to keep
            gamma (float): loss weight
            l (int): how many previous layers (including current) to use

        Returns:
            aux_loss (Tensor): scalar loss value
        """
        # Sanity check
        start_idx = max(layer_idx - l + 1, 0)

        # Ensure keys are sorted integers (if residual_stream is a dict)
        selected = [residual_stream[i].float() for i in range(start_idx, layer_idx + 1)]

        # Concatenate residuals from selected layers along batch dimension
        R = torch.cat(selected, dim=0)  # shape: (B * l, d)

        # Step 1: Center
        R_centered = R - R.mean(dim=0, keepdim=True)

        # Step 2: SVD
        _, S, _ = torch.linalg.svd(R_centered, full_matrices=False)

        # Step 3: Loss
        S_squared = S ** 2
        loss = -gamma * (S_squared[:k].sum() / S_squared.sum())

        return loss



    # [!]
    aux_loss = compute_structural_loss(residual_stream)
    # [!]

    return BaseModelOutputWithResidualStream(
        last_hidden_state=hidden_states,
        past_key_values=next_decoder_cache if use_cache else None,
        hidden_states=all_hidden_states,
        attentions=all_self_attns,
        residual_stream=residual_stream,  # 传入记录的残差流
        aux_loss = aux_loss
    )

from dataclasses import dataclass

@dataclass
class CausalLMOutputWithPast(ModelOutput):
    aux_loss: float = 0
    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

def forward(
    self,
    input_ids: torch.LongTensor = None,
    attention_mask: Optional[torch.Tensor] = None,
    position_ids: Optional[torch.LongTensor] = None,
    past_key_values: Optional[Union[Cache, List[torch.FloatTensor]]] = None,
    inputs_embeds: Optional[torch.FloatTensor] = None,
    labels: Optional[torch.LongTensor] = None,
    use_cache: Optional[bool] = None,
    output_attentions: Optional[bool] = None,
    output_hidden_states: Optional[bool] = None,
    return_dict: Optional[bool] = None,
    cache_position: Optional[torch.LongTensor] = None,
    num_logits_to_keep: int = 0,
    **kwargs: Unpack[KwargsForCausalLM],
) -> Union[Tuple, CausalLMOutputWithPast]:
    r"""
    Args:
        labels (`torch.LongTensor` of shape `(batch_size, sequence_length)`, *optional*):
            Labels for computing the masked language modeling loss. Indices should either be in `[0, ...,
            config.vocab_size]` or -100 (see `input_ids` docstring). Tokens with indices set to `-100` are ignored
            (masked), the loss is only computed for the tokens with labels in `[0, ..., config.vocab_size]`.

        num_logits_to_keep (`int`, *optional*):
            Calculate logits for the last `num_logits_to_keep` tokens. If `0`, calculate logits for all
            `input_ids` (special case). Only last token logits are needed for generation, and calculating them only for that
            token can save memory, which becomes pretty significant for long sequences or large vocabulary size.

    Returns:

    Example:

    ```python
    >>> from transformers import AutoTokenizer, LlamaForCausalLM

    >>> model = LlamaForCausalLM.from_pretrained("meta-llama/Llama-2-7b-hf")
    >>> tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-hf")

    >>> prompt = "Hey, are you conscious? Can you talk to me?"
    >>> inputs = tokenizer(prompt, return_tensors="pt")

    >>> # Generate
    >>> generate_ids = model.generate(inputs.input_ids, max_length=30)
    >>> tokenizer.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
    "Hey, are you conscious? Can you talk to me?\nI'm not conscious, but I can talk to you."
    ```"""
    output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
    output_hidden_states = (
        output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
    )
    return_dict = return_dict if return_dict is not None else self.config.use_return_dict
    
    # [!]
    self.model.forward = partial(forward1, self.model)
    # [!]

    # decoder outputs consists of (dec_features, layer_state, dec_hidden, dec_attn)
    outputs = self.model(
        input_ids=input_ids,
        attention_mask=attention_mask,
        position_ids=position_ids,
        past_key_values=past_key_values,
        inputs_embeds=inputs_embeds,
        use_cache=use_cache,
        output_attentions=output_attentions,
        output_hidden_states=output_hidden_states,
        return_dict=return_dict,
        cache_position=cache_position,
        **kwargs,
    )

    hidden_states = outputs[0]
    # Only compute necessary logits, and do not upcast them to float if we are not computing the loss
    logits = self.lm_head(hidden_states[:, -num_logits_to_keep:, :])

    loss = None
    if labels is not None:
        loss = self.loss_function(logits=logits, labels=labels, vocab_size=self.config.vocab_size, **kwargs)

    if not return_dict:
        output = (logits,) + outputs[1:]
        return (loss,) + output if loss is not None else output

    return CausalLMOutputWithPast(
        loss=loss,
        logits=logits,
        past_key_values=outputs.past_key_values,
        hidden_states=outputs.hidden_states,
        attentions=outputs.attentions,
        aux_loss = outputs.aux_loss
    )
