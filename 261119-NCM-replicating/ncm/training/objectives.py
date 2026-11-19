"""Training objectives separate from model structure and CLI orchestration."""
from contextlib import nullcontext

import torch
import torch.nn.functional as F


def ar_loss(model, tokens, eos_id):
    prefix = torch.full_like(tokens[:, :1], eos_id)
    inputs = torch.cat((prefix, tokens[:, :-1]), dim=1)
    # Full positions including padding match the paper's token-budget convention.
    return F.cross_entropy(model(inputs).reshape(-1, model.config.vocab_size), tokens.reshape(-1))


def corrupt_codes(codes, levels, sizes, start, end):
    if not (0 <= start <= 1 and 0 <= end <= 1):
        raise ValueError('Corruption rates must be in [0,1]')
    noisy = codes.clone()
    rates = torch.linspace(start, end, len(levels), device=codes.device)
    offset = 0
    for length, size, rate in zip(levels, sizes, rates):
        chunk = noisy[:, offset:offset + length]
        mask = torch.rand(chunk.shape, device=codes.device) < rate
        replacement = torch.randint(size, chunk.shape, device=codes.device)
        chunk[mask] = replacement[mask]
        offset += length
    return noisy


def ncm_loss(model, vqvae, codes, eos_id, *, corruption_start, corruption_end, alpha, training):
    levels = list(vqvae.vq.levels)
    # Frozen released VQ weights were not trained with FP16; residual
    # reconstruction must not inherit the generator autocast context.
    with torch.no_grad(), (torch.autocast(device_type=codes.device.type, enabled=False)
                           if model.selective_fp32 else nullcontext()):
        conditioning = corrupt_codes(codes, levels, vqvae.vq.vqvae_vocab_sizes,
                                     corruption_start, corruption_end) if training else codes
        x = vqvae.vq.idx_to_ncm_input(conditioning)
        prefix = torch.full((len(codes), model.prefix_len), eos_id, device=codes.device, dtype=torch.long)
        z_prefix = vqvae.encoder.wte(prefix)
        x = torch.cat((z_prefix, x), dim=1)
    logits = model(x)
    targets = codes[:, levels[0]:]
    losses = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), targets.reshape(-1), reduction='none')
    losses = losses.view_as(targets)
    weights = torch.cat([torch.full((length,), float(length) ** (-alpha), device=codes.device)
                         for length in levels[1:]])
    weights = weights / weights.sum()
    loss = (losses * weights).sum(-1).mean()
    if not torch.isfinite(loss):
        def stats(tensor):
            finite = torch.isfinite(tensor)
            values = tensor[finite].float()
            return {'dtype': str(tensor.dtype), 'nonfinite': int((~finite).sum()),
                    'finite_abs_max': float(values.abs().max()) if values.numel() else None}
        raise FloatingPointError(
            f'NCM nonfinite forward loss: conditioning={stats(x)}, logits={stats(logits)}. '
            'Gradient scaling cannot fix forward overflow. Preserve this diagnostic output.')
    return loss


def tokenizer_loss(model, tokens):
    logits, vq_loss, *_ = model(tokens)
    recon = F.cross_entropy(logits.reshape(-1, logits.shape[-1]), tokens.reshape(-1))
    return recon + vq_loss, {'reconstruction_ce': recon.detach(), 'vq_loss': vq_loss.detach(),
                            'token_accuracy': (logits.argmax(-1) == tokens).float().mean().detach()}
