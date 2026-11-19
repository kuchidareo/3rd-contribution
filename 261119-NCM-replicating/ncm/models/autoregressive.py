"""Causal text baseline reusing the upstream transformer blocks and RoPE."""
import math

import torch
from torch import nn

from models.common import Block, LayerNorm, precompute_rope_cos_sin


def sample_logits(logits, temperature=1.0, top_k=0, top_p=1.0):
    if temperature < 0 or not 0 < top_p <= 1 or top_k < 0:
        raise ValueError('Require temperature >= 0, top_k >= 0 and 0 < top_p <= 1')
    if temperature == 0:
        return logits.argmax(-1, keepdim=True)
    scores = logits.float() / temperature
    if top_k:
        threshold = scores.topk(min(top_k, scores.shape[-1]), dim=-1).values[..., -1:]
        scores = scores.masked_fill(scores < threshold, -torch.inf)
    if top_p < 1:
        ordered, indexes = scores.sort(descending=True, dim=-1)
        probs = ordered.softmax(-1)
        ordered = ordered.masked_fill(probs.cumsum(-1) - probs > top_p, -torch.inf)
        scores = torch.full_like(scores, -torch.inf).scatter(-1, indexes, ordered)
    return torch.multinomial(scores.softmax(-1), 1)


class AutoregressiveTransformer(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.wte = nn.Embedding(config.vocab_size, config.n_embd)
        self.drop = nn.Dropout(config.dropout)
        self.blocks = nn.ModuleList([Block(config) for _ in range(config.n_layer)])
        self.ln_f = LayerNorm(config.n_embd, bias=config.bias)
        self.head = nn.Linear(config.n_embd, config.vocab_size, bias=False)
        if config.get('tie_embeddings', True):
            self.head.weight = self.wte.weight
        cos, sin = precompute_rope_cos_sin(torch.arange(config.context_length),
                                         config.n_embd // config.n_head, config.rope_base)
        self.register_buffer('rope_cos', cos, persistent=False)
        self.register_buffer('rope_sin', sin, persistent=False)
        self.apply(self._initialize)
        for name, p in self.named_parameters():
            if name.endswith('c_proj.weight'):
                nn.init.normal_(p, std=0.02 / math.sqrt(2 * config.n_layer))

    @staticmethod
    def _initialize(module):
        if isinstance(module, (nn.Linear, nn.Embedding)):
            nn.init.normal_(module.weight, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                nn.init.zeros_(module.bias)

    def forward(self, ids):
        if ids.shape[1] > self.config.context_length:
            raise ValueError('Input exceeds configured context length')
        x = self.drop(self.wte(ids))
        for block in self.blocks:
            # Upstream attention is causal when no explicit mask is supplied.
            x = block(x, rope_cos_sin=(self.rope_cos, self.rope_sin))
        return self.head(self.ln_f(x))

    @torch.no_grad()
    def generate(self, batch_size, length, eos_id, *, use_kv_cache=True, **sampling):
        if self.training:
            raise RuntimeError('Call eval() before generation')
        if not 0 < length <= self.config.context_length:
            raise ValueError('Generation length exceeds configured context')
        ids = torch.full((batch_size, 1), eos_id, dtype=torch.long, device=self.wte.weight.device)
        generated = []
        for block in self.blocks:
            block.attn.kv_caching(use_kv_cache)
        try:
            for _ in range(length):
                logits = self(ids[:, -1:] if use_kv_cache else ids)[:, -1]
                token = sample_logits(logits, **sampling)
                generated.append(token)
                ids = torch.cat((ids, token), dim=1)
        finally:
            for block in self.blocks:
                block.attn.kv_caching(False)
        # Fixed compute length for both models; EOS truncation is applied only for text scoring.
        return torch.cat(generated, dim=1)
