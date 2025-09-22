from typing import Optional, Tuple, List, Union
import torch
import torch.nn.functional as F
from torch import nn
from loguru import logger
from dataclasses import dataclass

from transformers import (
    GenerationMixin,
    PreTrainedModel,
    AutoModelForCausalLM,
    StoppingCriteriaList,
    GenerationConfig,
)
from transformers.generation.utils import (
    GreedySearchDecoderOnlyOutput,
)
from transformers.utils import ModelOutput

@dataclass
class MemoryDecoderOutput(ModelOutput):

    loss: Optional[torch.FloatTensor] = None
    logits: Optional[torch.FloatTensor] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    knn_past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    hidden_states: Optional[Tuple[torch.FloatTensor, ...]] = None
    attentions: Optional[Tuple[torch.FloatTensor, ...]] = None

class MemoryDecoder(PreTrainedModel, GenerationMixin):
    """
    A light wrapper around **two** causal‑LMs that fuses their logits:

        logits_joint = logaddexp(logits_base + log(1‑λ),
                                 logits_knn  + log(λ))

    Greedy decoding chooses argmax over `logits_joint`.
    """

    def __init__(
        self,
        base_lm,
        knn_generator,
        lmbda: float = 0.25,
        knn_temp: float = 1.0,
    ):
        super().__init__(base_lm.config)

        self.base_lm = base_lm
        self.knn_generator = knn_generator
        self.lmbda = float(lmbda)
        self.knn_temp = float(knn_temp)
        
    # ------------------------------------------------------------------ #
    #                       1. forward()
    # ------------------------------------------------------------------ #
    def forward(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Tuple] = None,
        knn_past_key_values: Optional[Tuple] = None,
        use_cache: bool = True,
        **kwargs,
    ):
        """
        Forward pass that returns **fused log‑probs** as logits.
        We keep separate caches for each sub‑model.
        """
        base_outputs = self.base_lm(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            **kwargs,
        )
        knn_outputs = self.knn_generator(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=knn_past_key_values,
            use_cache=use_cache,
            **kwargs,
        )

        # Temperature on k‑NN logits only
        logits_base = base_outputs.logits      # (B, T, V)
        logits_knn = knn_outputs.logits
        if self.knn_temp != 1.0:
            logits_knn = logits_knn / self.knn_temp

        # Convert to log‑probabilities first (numerically stable when fusing)
        logp_base = F.log_softmax(logits_base, dim=-1)
        logp_knn = F.log_softmax(logits_knn, dim=-1)

        logp_joint = torch.logaddexp(
            logp_base + torch.log(torch.tensor(1.0 - self.lmbda, device=logp_base.device)),
            logp_knn + torch.log(torch.tensor(self.lmbda, device=logp_base.device)),
        )

        return MemoryDecoderOutput(
            logits=logp_joint,
            past_key_values=base_outputs.past_key_values,
            knn_past_key_values=knn_outputs.past_key_values,
            hidden_states=None,
            attentions=None
        )
        
    # ------------------------------------------------------------------ #
    #                       2. generate()
    # ------------------------------------------------------------------ #
    @torch.no_grad()
    def generate(  # type: ignore[override]
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        max_new_tokens: int = 20,
        stopping_criteria: Optional[StoppingCriteriaList] = None,
        do_sample: bool = False,             # must be False (greedy) for now
        generation_config: Optional[GenerationConfig] = None,
        **kwargs,
    ):
        """
        Greedy decoding with **shared** stopping criteria.
        We keep two independent KV caches (one per sub‑model) and extend them
        step‑by‑step.
        """
        if do_sample:
            raise ValueError("MemoryDecoder.generate only supports greedy decoding (do_sample=False).")

        device = input_ids.device
        batch_size = input_ids.shape[0]

        # Initialise caches with a single forward.
        outputs = self.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            **kwargs,
        )
        next_token_logits = outputs["logits"][:, -1, :]            # (B, V)

        base_past = outputs["past_key_values"]
        knn_past = outputs["knn_past_key_values"]

        # Greedy select
        next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)  # (B,1)
        generated = torch.cat([input_ids, next_tokens], dim=-1)              # (B,T+1)

        # --- main loop -------------------------------------------------- #
        num_new_token = 0
        while True:
            if stopping_criteria is not None and False not in stopping_criteria(generated, None):
                break
            if num_new_token >= max_new_tokens:
                break

            outputs = self.forward(
                input_ids=next_tokens,
                attention_mask=None,          # past manages causal masking
                past_key_values=base_past,
                knn_past_key_values=knn_past,
                use_cache=True,
                **kwargs,
            )
            next_token_logits = outputs["logits"][:, -1, :]
            base_past = outputs["past_key_values"]
            knn_past = outputs["knn_past_key_values"]

            next_tokens = torch.argmax(next_token_logits, dim=-1).unsqueeze(-1)
            generated = torch.cat([generated, next_tokens], dim=-1)
            num_new_token += 1

        return generated