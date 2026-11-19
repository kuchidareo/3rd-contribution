# TinyStories baseline: AR vs NCM

This fork keeps the released NCM implementation and adds a single-device experiment
pipeline. Kaggle is the execution environment; notebooks only invoke the CLI.
No dataset, pretrained weights, or reference language model is downloaded by setup
or by the synthetic tests. Explicit acquisition commands below run on Kaggle.

## Sources and reproduction limits

- [NCM paper, Appendices B–E](https://arxiv.org/html/2610.14162v1)
- Upstream architecture: `config/eval/tinystories.yaml`, `models/next_concept.py`,
  `models/vqvae.py`, `models/quant.py`.
- Official scoring: `evaluation/gen_ppl.py`, `scripts/eval_merge.py`.
- Released data: <https://huggingface.co/datasets/hazyresearch/ncm-datasets/tree/main/tinystories>
- Released weights/BPE: <https://huggingface.co/hazyresearch/ncm-checkpoints/tree/main/tinystories>

Paper settings used: 128 text positions, custom BPE vocabulary 8,192; generator
12 layers, width 768, 12 heads, dropout 0.1. Generator training uses 50,000 updates,
global batch 2,048, peak/floor LR 3e-4/3e-5, warmup 500, AdamW betas 0.9/0.95,
weight decay 0.1, gradient norm limit 1. Tokenizer training uses 30,000 updates,
batch 4,096, peak/floor LR 5e-4/5e-5, warmup 1,000, weight decay 0.01.
Seed 42, BF16 and forward compilation are the default training settings.
The full official architecture configuration is reused, including quantizer levels.

These are implementation choices where a complete training recipe is unavailable:

- The release contains no training entry point. This fork supplies the loops.
- Both models use the released token IDs. Source records are cut into nonoverlapping
  128-position chunks, retaining the final partial chunk with right padding. No new
  EOS is inserted. This policy is explicit; exact upstream training-window sampling
  is not published in this checkout.
- Official training records remain training-only. Official validation records are
  deterministically divided equally into validation and test with seed 42, **before
  chunking**. For a flat token stream without record boundaries, nonoverlapping
  windows are split instead; the manifest records this distinction. This does not
  guarantee deduplication of duplicate text in the original corpus.
- The training sampler draws rows with replacement. A private counter-based seed
  gives both generators identical input rows and makes checkpoint resume reproducible.
- LR follows cosine decay after linear warmup. The exact upstream LR curve is absent.
- NCM conditioning-code replacement decreases linearly from 0.9 to 0.1 across
  levels; targets stay clean. Per-code loss weights are equal (`alpha=0`). These
  endpoints are an explicit reconstruction choice, not a verified released recipe.
  Validation uses clean conditioning and is labelled accordingly.
- AR reuses upstream `Block`, LayerNorm, GELU MLP, QK normalization and RoPE. Its text
  embedding and output head share weights. Exact AR embedding/head tying is not
  specified by the public release. This yields roughly 91M parameters, rather than
  asserting an exact 89M reproduction; commands record actual counts. NCM's released
  generator is approximately 88.6M. Frozen tokenizer parameters are reported separately.
- All text positions, including padding, contribute to training loss. Losses between
  AR and NCM are in different vocabularies and are not directly comparable.

Matching generator token exposure does not equal matching FLOPs. Full generator
exposure is 50,000 × 2,048 × 128 = 13,107,200,000 text positions each. NCM additionally
uses a tokenizer (full recipe: 30,000 × 4,096 × 128 = 15,728,640,000 positions).
Report both budgets; a reused official tokenizer has externally incurred pretraining
cost. A short Kaggle run is a pilot, not a reproduction of the paper's final scores.

## Kaggle setup and data preparation

Enable a GPU and Internet in the Kaggle notebook settings. Run shell commands in
notebook cells with `!` or `%%bash`. Kaggle working storage holds resumable outputs.
Use the fork's committed revision containing these changes:

```bash
cd /kaggle/working
git clone https://github.com/kuchidareo/3rd-contribution.git 3rd-contribution
cd 3rd-contribution/261119-NCM-replicating/ncm
bash scripts/kaggle_setup.sh

# Explicit Kaggle-only download: TinyStories assets/checkpoints and tokenized data.
# Data and model revisions are resolved and written to downloads.json.
python -m baseline_data.prepare --download
```

Training uses the release's binary token stream and document offsets through a
memory map (explicit `uint16` dtype, validated token range/offsets). Validation uses
the smaller official pickle. This avoids loading the large training pickle into
Python objects. If the release format changes, inspect its metadata and explicitly
set `--bin-dtype`, or use `--train-format pickle --train <path>`.

The first preparation produces `data/tinystories-prepared/{train,validation,test}.npy`
and `manifest.json`. The manifest records source/BPE/split SHA256 hashes and split
policy. Existing prepared manifests are not overwritten. For attached Kaggle inputs,
omit `--download`, supply `--train` and `--valid`, and set asset/output paths:

```bash
python -m baseline_data.prepare --train-format pickle \
  --train /kaggle/input/ncm-data/train_bpe_8192.pickle \
  --valid /kaggle/input/ncm-data/valid_bpe_8192.pickle \
  --set paths.artifacts=/kaggle/input/ncm-assets/tinystories \
  --set paths.data=/kaggle/working/tinystories-prepared
```

Pickles and PyTorch checkpoints must come from a trusted source. No new BPE is
trained; both models use the exact same released BPE assets.

## Smoke test before training

Synthetic tests use no real data or pretrained models:

```bash
python -m pytest tests -q
```

They check causal attention, cached vs uncached AR logits, tokenizer gradients,
NCM code corruption, final decoder equivalence, split isolation, forward-call
counting, and uninterrupted vs resumed weights. They do not establish generation
quality or GPU throughput.

For an actual-data pilot, keep the full LR horizon and stop after two updates:

```bash
python -m training.train_ar --out /kaggle/working/runs/ar-pilot --max-steps 2 \
  --set training.generator.global_batch_size=8 \
  --set training.generator.micro_batch_size=2 --set training.compile=false
```

A Kaggle T4 lacks native BF16 support. Add `--set precision=fp16` consistently to
training and generation, and label this departure from the paper. FP32 is also
available. On FP16 gradient overflow, the trainer lowers the loss scale and retries
the same accumulated batch without advancing the optimizer update or LR schedule.
Retries are bounded (default 16 per update); forward nonfinite losses still raise
because gradient scaling cannot repair forward overflow.

## AR training

```bash
python -m training.train_ar --out /kaggle/working/runs/ar
# Resume in another session after restoring the output files:
python -m training.train_ar --out /kaggle/working/runs/ar \
  --resume /kaggle/working/runs/ar/latest.pt
```

All trainers accept `--config`, repeated `--set key=value`, `--out`, `--resume`, and
`--max-steps` (absolute stop update). Model/optimizer/scaler/RNG state and completed
updates are saved atomically to `latest.pt`; the prior version survives until replace.
`run.json` records configuration/environment/provenance. `metrics.jsonl` records
training loss, clean validation loss, LR and text positions. Logs may repeat updates
if a process dies after logging but before the next checkpoint; use the last entry
per step. Checkpoints occur at update boundaries; unfinished updates are rerun.

Micro batches accumulate to the configured global batch. The full default NCM
sequence of latent codes is much longer than the AR text sequence; start with a
small micro batch, keeping global batch fixed when memory tuning. Cache/source
fingerprints are checked before training. Resume rejects policy/data changes, but
allows paths to be relocated with `--set paths.*=...`. Eager CPU regression tests
verify resume; compiled GPU kernels may differ across hardware/software versions.

## NCM tokenizer and generator training

Reuse the official tokenizer initially. Cache its FP32 nearest-neighbor assignments
for every split and compute the first-level empirical prior on **training only**:

```bash
python -m baseline_data.encode_codes --batch-size 32
python -m training.train_ncm --out /kaggle/working/runs/ncm
```

The cache is an accelerator for the shared token rows, not a separate dataset.
It records the tokenizer/config/data hashes and stores codes as uint16 when possible.
Budget working storage for the prepared rows and code cache; the latter contains
741 codes per text row in this configuration. NCM's first code is sampled from
`inference.level0_unigram`; the 11 remaining levels are generated by the transformer.
The local trainer emits checkpoint keys compatible with the upstream evaluator.

To train a new tokenizer instead:

```bash
python -m training.train_tokenizer --out /kaggle/working/runs/tokenizer
python -m baseline_data.encode_codes \
  --set paths.tokenizer_checkpoint=/kaggle/working/runs/tokenizer/latest.pt \
  --set paths.codes=/kaggle/working/codes-new-tokenizer
python -m training.train_ncm --out /kaggle/working/runs/ncm-new-tokenizer \
  --set paths.tokenizer_checkpoint=/kaggle/working/runs/tokenizer/latest.pt \
  --set paths.codes=/kaggle/working/codes-new-tokenizer
```

Tokenizer reconstruction CE, VQ loss and exact token accuracy are logged during
validation. Its EMA codebook state and the otherwise nonpersistent fine-drop step
are included in resume state. Stage 2 freezes every tokenizer parameter and uses
clean cached labels with corrupted conditioning. No test rows affect fitting or
first-level frequency counts.

Run both generators with identical `training.generator.*` overrides for a fair
Kaggle pilot. Reducing `--max-steps` preserves the planned full LR horizon; reducing
`training.generator.steps` changes that horizon. Training a full 13B-position
baseline may require multiple sessions; save Kaggle outputs before sessions end.

## Generation, speed and quality

Run models in separate processes so GPU memory measurements do not include the
other generator. Use the same GPU, precision, batch size, output length and sampling.

```bash
python -m evaluation.evaluate --model ar \
  --checkpoint /kaggle/working/runs/ar/latest.pt --out /kaggle/working/results/ar
python -m evaluation.evaluate --model ncm \
  --checkpoint /kaggle/working/runs/ncm/latest.pt --out /kaggle/working/results/ncm

# Fresh processes: generator memory is released before GPT-2-large scoring.
# These commands download the reference model on Kaggle if absent from its cache.
python -m evaluation.score --run /kaggle/working/results/ar
python -m evaluation.score --run /kaggle/working/results/ncm

python -m evaluation.compare \
  --ar /kaggle/working/results/ar --ncm /kaggle/working/results/ncm \
  --out /kaggle/working/results/comparison
```

Use the same overrides as training for precision, paths and any architecture changes.
`evaluation.score --ref-revision <full-HF-commit>` pins the reference model; record the
resolved revision for both runs. `--batch-size` reduces scoring memory independently.
Generation does not compile either model; training compilation is independent of
inference measurement. AR uses a KV cache; NCM follows the release's eager generation
path with KV caching disabled. These choices are recorded in generation metadata.

Outputs:

- `samples.jsonl`: seed/index, generated text, raw fixed-length BPE IDs, re-encoded
  text IDs for entropy. Quality text is truncated at EOS and excludes PAD/EOS.
- `generation.json`: model/tokenizer parameter counts, checkpoint/BPE hashes, training
  provenance, sampling, environment, timed runs and peak GPU memory.
- `quality.json`: GPT-2-large generative perplexity and mean per-sample unigram
  entropy in nats, per seed and aggregated. It reuses `generation_perplexity` from
  upstream, including its 1,024-reference-token limit. Empty text counts are explicit;
  entropy excludes empty sequences as in the release.
- `comparison.json`, `comparison.csv`, `comparison.png`: common metrics and charts.

Timing includes initial code/prefix preparation, generator calls and NCM's final
latent reconstruction/decoder. It excludes loading, CPU text decoding and reference
scoring. GPU synchronization brackets each repeat, warmup is excluded, and throughput
is batch × 128 **generated text positions** divided by median batch latency. Both
models generate all 128 positions even after EOS; also inspect actual text lengths.
Per-sample time is amortized batch time, not interactive request latency. Peak
allocated/reserved memory includes resident generation models, not optimizer/scorer.
Forward counts use hooks on the first transformer block plus the NCM decoder:
expect AR 128 generator calls and NCM 11 generator + 1 decoder call per batch.

The comparison rejects different tokenizers, generation/scoring protocols, GPU/
PyTorch/CUDA environments, training data and generator token budgets. An official
released NCM checkpoint can be smoke-tested with:

```bash
python -m evaluation.evaluate --model ncm \
  --checkpoint artifacts/tinystories/generator.pt --out /kaggle/working/results/ncm-official
```

Its exact training provenance is not embedded in our schema. To compare it against
a locally trained AR model, `--allow-unmatched` explicitly labels the comparison
exploratory. No paper speedup or quality score is claimed before measured runs.

## Structure and extension points

`models/autoregressive.py` adds AR while reusing upstream blocks. `baseline_data/`
provides shared data and cached codes. `training/objectives.py` contains loss/input
construction; `training/train.py` supplies optimization and resume. Evaluation is
split into generation/benchmarking, reference scoring and comparison. Future NCM
architecture experiments can replace the generator without rewriting data/scoring.
Level-specific subnetworks and distributed execution are outside this implementation.

## Validation status

Implementation and synthetic tests are provided. Real TinyStories training,
released-checkpoint parity, GPU timing, checkpoints and comparison scores require
execution on Kaggle. No dataset/checkpoints were downloaded locally, and no
unmeasured quality/speed results are included. The workspace Python venv is
`/home/user/kuchida/3rd-contribution/venv`; local dependency availability and test
results must be recorded separately from Kaggle results.

Local validation on 2026-11-19: the specified workspace venv passed all four
dependency-free comparison tests (`python -m unittest discover -s tests
-p test_comparison.py -v`), Python compilation, shell syntax and `git diff --check`.
The full synthetic PyTorch suite could not run: the venv lacks PyTorch/pytest, and
package installation failed because the package hosts were inaccessible. No other
venv was used for these checks. Run `python -m pytest tests -q` after Kaggle setup
before downloading the data. This status is not a claim that training has been validated.

## Two Kaggle notebooks: prepare once, train many variants

- [`01_prepare_tinystories_kaggle.ipynb`](../notebooks/01_prepare_tinystories_kaggle.ipynb)
  clones the fork, checks synthetic tests, downloads/prepares the official data, and
  saves reusable artifacts in `/kaggle/working/ncm-tinystories/`. It includes optional
  FP32 VQ-code caching and removes redundant raw downloads after hash verification.
  Save the notebook version with outputs and manually create a **private** Kaggle
  Dataset from that folder. The notebook does not publish or configure privacy.
- [`02_train_tinystories_kaggle.ipynb`](../notebooks/02_train_tinystories_kaggle.ipynb)
  reads that Dataset directly from `/kaggle/input/`, launches both training CLIs,
  supports checkpoints from previous notebook outputs, and generates comparison
  results. Edit `DATASET_ROOT` to match your attachment. Default training is a
  100-update, global-batch-32 pilot; select `PROFILE='paper'` for the full budget.

The saved data includes `prepared/`, `codes/` (if enabled),
`artifacts/tinystories/`, download revisions, and environment records. NCM generator
architecture changes reuse the same code cache. A changed VQ tokenizer requires new
codes from the saved prepared rows, **not another TinyStories download**.

Both notebooks are saved without execution outputs. Their JSON structure and Python
cell syntax were checked locally; Kaggle execution and model tests remain pending.
This folder is now ordinary content in the parent repository (see `UPSTREAM.md`).
Set notebook `REPO_URL` to the parent remote and `REPO_REF` to its branch/revision.
The local implementation must be committed and pushed before a notebook can clone
it. No notebook has been uploaded, no Dataset has been created, and no remote
repository changes have been pushed by this work.

## Notebook 2 paper profile (2026-11-20)

Change only `PROFILE = 'pilot'` to `PROFILE = 'paper'` after setting the attached
Dataset path. `training/profiles.py` explicitly selects the published 50,000-update,
2,048-global-batch recipe, seed 42, AdamW settings, explicit precision, compiled training, and
official sampling/evaluation configuration (5 seeds, 256 samples/seed, generation
batch 128, GPT-2-large). Scoring uses the release's batch size of 32. The profile
retains the default 128-position text length and unmodified upstream NCM architecture.

Set `PRECISION='fp16'` on T4; the full paper budget is retained and the deviation
from BF16 is recorded. Set `'bf16'` on native BF16-capable hardware to match paper
precision. There is no silent fallback. With micro batch 2 it accumulates
1,024 micro batches per optimizer update. Runs are stored in
`/kaggle/working/ncm-experiment/paper-fp16/` for FP16 or `paper/` for BF16; pilot
runs use the separate `pilot/` folder. Resume requires matching precision.
Existing pilot weights cannot be resumed as paper weights. Session stopping does
not shorten the full learning-rate horizon. Final paper evaluation checks both
checkpoint headers and requires all 50,000 updates for each generator.

The selected profile and explicit training assumptions are printed and saved in
`experiment.yaml` and checkpoint metadata. Cosine LR decay, corruption endpoints
0.9→0.1, loss alpha 0, AR embedding tying, optimizer grouping and data policies remain
reconstruction choices. Selecting this profile does not certify exact equivalence
to the authors' unpublished trainer or guarantee their final PPL. No data preparation
rerun is needed, and the official pretrained tokenizer is reused.

Local validation: 10 dependency-free profile/notebook-policy tests and the existing
4 comparison tests pass in the specified workspace venv. Notebook code-cell syntax
is checked. Full PyTorch/GPU execution remains untested locally because this venv
does not have the required packages; no data or model weights were downloaded.

### T4 update: explicit full-budget FP16

Notebook 2 now exposes `PRECISION='fp16'` alongside `PROFILE`. For T4 use:

```python
PROFILE = 'paper'
PRECISION = 'fp16'
```

This keeps 50,000 applied optimizer updates and global batch 2,048 for each model.
Gradient overflows reduce the dynamic scale and retry the same rows and RNG state,
without consuming another LR-schedule update. Tokenizer training additionally rolls
back its EMA state on a rejected update. Checkpoints persist scale/retry state.
`amp_events.jsonl` records overflows; metrics distinguish completed text-position
exposure from attempted exposure including retries. The final comparison still
requires both generators to complete training.

FP16 replaces paper BF16 and is explicitly recorded in `experiment.hardware_deviations`.
Use a new FP16 paper run; a pilot or BF16 checkpoint is not interchangeable.
The existing Kaggle Dataset and official frozen tokenizer remain reusable.
Offline profile, checkpoint-policy and scaler-policy tests pass; actual T4 execution
requires Kaggle and has not been verified locally.

## Validation early stopping and best checkpoints

Notebook 2 now exposes these controls:

```python
VALIDATE_EVERY = 100       # 200 is also supported
CHECKPOINT_EVERY = 100
EARLY_STOPPING = True
EARLY_STOP_PATIENCE = 5
EARLY_STOP_MIN_DELTA = 0.001
EARLY_STOP_MIN_STEPS = 500
EVALUATION_CHECKPOINT = 'best'
```

Every validation check uses the fixed clean validation subset. `best.pt` is saved
when this loss reaches a new minimum; `latest.pt` is saved at the checkpoint interval
and on session completion or early stopping. Both include model/optimizer/scaler/RNG
and selection state. The 50,000-update recipe remains the maximum and the LR horizon;
early stopping can end training sooner. It waits until at least the configured
minimum and warmup, and counts validation checks rather than individual updates.
Patience resets on an improvement exceeding `min_delta`; even a smaller new minimum
still updates `best.pt`. NCM clean validation loss is latent-code CE, not Gen. PPL.

On T4 the files are under `ncm-experiment/paper-fp16/runs/{ar,ncm}/`.
Save **both** `latest.pt` and `best.pt` when transferring a run between sessions.
The trainer permits changing validation/checkpoint/log intervals and early-stopping
controls when resuming; optimization settings, validation rows and dataset remain
locked. An old checkpoint without selection history is validated at resume and saved
as the initial `best.pt`; this is best since resume, not recovered historical best.

Notebook evaluation selects `best.pt` by default and accepts explicit early stopping.
A session stopped only by `STOP_AT_UPDATE` still needs resuming before final paper-mode
comparison. Selected checkpoints below the full target are labelled partial-training
comparisons; differing AR/NCM training exposure is retained in the report. These runs
are not represented as completed, equal-budget paper reproductions.

The model-level best/latest regression test is provided for Kaggle's PyTorch test
suite. Local validation covers the dependency-free patience/resume/selection logic;
actual model training is not executed locally.
