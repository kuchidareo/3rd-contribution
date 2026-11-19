"""Synchronized fixed-shape generation timing, excluding scoring and text decoding."""
from contextlib import contextmanager
import statistics
import time

import torch


def synchronize(device):
    if torch.device(device).type == 'cuda':
        torch.cuda.synchronize(device)


@contextmanager
def forward_counter(generator, decoder=None):
    counts = {'generator': 0, 'decoder': 0}
    def hook(key):
        def count(*_):
            counts[key] += 1
        return count
    # NCM.generate invokes its blocks directly, not the module's forward method.
    first = generator.transformer.blocks[0] if hasattr(generator, 'transformer') else generator.blocks[0]
    handles = [first.register_forward_hook(hook('generator'))]
    if decoder is not None:
        handles.append(decoder.register_forward_hook(hook('decoder')))
    try:
        yield counts
    finally:
        for handle in handles:
            handle.remove()


def benchmark(generate, model, decoder, device, batch_size, length, warmup=2, repeats=5):
    if repeats < 1 or warmup < 0:
        raise ValueError('Require repeats >= 1 and warmup >= 0')
    for _ in range(warmup):
        generate(batch_size)
    synchronize(device)
    cuda = torch.device(device).type == 'cuda'
    if cuda:
        torch.cuda.reset_peak_memory_stats(device)
    latencies, calls = [], []
    with forward_counter(model, decoder) as counter:
        for _ in range(repeats):
            before = dict(counter)
            synchronize(device)
            started = time.perf_counter()
            ids = generate(batch_size)
            synchronize(device)
            latencies.append(time.perf_counter() - started)
            if tuple(ids.shape) != (batch_size, length):
                raise ValueError('Generation did not produce the requested fixed shape')
            calls.append({key: counter[key] - before[key] for key in counter})
            del ids
    median = statistics.median(latencies)
    return {'batch_size': batch_size, 'generated_length': length, 'repeats': repeats,
            'warmup_batches': warmup, 'batch_latency_seconds': latencies,
            'median_batch_latency_seconds': median,
            'p95_batch_latency_seconds': sorted(latencies)[max(0, __import__('math').ceil(0.95 * repeats) - 1)],
            'amortized_seconds_per_sample': median / batch_size,
            'tokens_per_second': batch_size * length / median,
            'forward_passes_per_batch': calls,
            'peak_allocated_bytes': torch.cuda.max_memory_allocated(device) if cuda else None,
            'peak_reserved_bytes': torch.cuda.max_memory_reserved(device) if cuda else None,
            'timing_scope': 'initial code/prefix construction + generator + NCM latent reconstruction/decoder; excludes CPU text decoding, loading and scorer'}
