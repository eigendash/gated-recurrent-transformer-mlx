# gated-recurrent-transformer-mlx

A small MLX implementation of a recurrent-depth transformer: a fixed prelude
and coda bracket a *single shared core* that is applied R times, and a learned
elementwise gate conditioned on the current state, the frozen prelude output
and noise resampled at every step decides how much of the core's proposal is
written back. It follows *Gated Recurrent Transformers: Expressive Depth
through Recurrent Modulation* (Amr Hegazy, Amr Alanwar, Mostafa Elhoushi,
arXiv:2608.15062). In the paper a 3-layer GRT matches a 12-layer GPT-2 Small
under matched FLOPs, and deeper recurrence beats a non-recurrent counterpart at
matched parameters (2.76 vs 2.84 validation loss).

This repo is **not** a reproduction of those results. Everything here is toy
scale: 64-dimensional models of 0.16M-0.71M parameters, 24-token synthetic
sequences, 2000 training steps on a laptop. The point is that the mechanism is
implemented exactly as the paper describes it, that its invariants are tested,
and that the two regimes the paper argues over (isoFLOPs and isoParams) can be
measured honestly at a scale where every run finishes in about a minute.

## What is in here

**Blocks** (`grt/blocks.py`). The nanoGPT block the paper builds on: pre-norm
attention and MLP with residual adds, learned absolute position embeddings,
causal mask. `CountingBlock` wraps a block to count real executions, which is
how the compute proxy is measured rather than assumed.

**Recurrent projection and gate** (`grt/modulation.py`). The paper's Section 3.3
and Listing A.1, in equations:

```
h_tilde^(r) = W_proj [ h^(r-1) + eps_x , h^(pre) ]              (2)
o^(r)       = B_shared( h_tilde^(r) )                            (3)
h^(r)       = g^(r) * h^(r-1) + (1 - g^(r)) * o^(r)              (4)
g^(r)       = sigmoid( f_g([LN(h^(r-1)), LN(h^(pre))]) / tau + eps_g )   (5)
```

`W_proj` is a bias-free `d x 2d` linear, as Eq. 2 writes it. `f_g` is a
two-layer MLP with SiLU over the two layer-normed inputs, hidden width `d`, and
its last layer's bias is initialised to +4 so the gate starts at `sigmoid(4) ~
0.98`: the copy branch dominates and the residual stream passes through the
recurrence almost unchanged at initialisation. `eps_x ~ N(0, sigma_x^2)` is
added to the state before the projection and `eps_g ~ N(0, sigma_g^2)` is a
single scalar per step added after the temperature division, both resampled
every step. Noise is only injected when the caller passes an MLX PRNG key, so
an unkeyed forward pass is exactly deterministic and a keyed one is
reproducible from its key.

`g` is the *copy* gate: `g -> 1` keeps the state and ignores the shared block,
`g -> 0` overwrites the state with the block output. `forward(..., gate_override=)`
exists so the two ends and everything between can be probed; the tests assert
both ends exactly.

**Model** (`grt/model.py`). `GRT` runs embeddings + prelude once to obtain the
anchor `h^(pre)`, sets `h^(0) = h^(pre)` (Listing A.1), applies the shared core
`R` times with the gated update, then the coda and the head. `R` is a
forward-pass argument and never a weight, so the parameter count is
`Theta((n_pre + n_core + n_coda) * d^2)` and does not move with `R`.
`forward(..., return_trace=True)` returns the anchor, every per-step state and
every per-step gate, which the tests use.

`PlainStack` is the non-recurrent baseline: `L` unique blocks, each visited
once, with the same block class, embeddings and head. `PlainStack.from_grt`
reuses a GRT's own blocks so the `R = 1` clean limit can be compared bit for
bit.

**Task** (`grt/tasks.py`). Multi-query associative recall. A sequence is eight
key/value pairs followed by the keys again in a shuffled order, and the model
must produce each value; values are drawn independently of the keys, so the
only way to answer is to look the key up in the context. Loss and accuracy are
computed at the eight query positions only. With 16 keys over a 32-token
vocabulary, chance is 0.031 over the whole vocabulary (0.063 if a model emitted
only value tokens).

**Training** (`grt/train.py`). AdamW, linear warmup into a cosine decay to 10%
of the peak, gradient norm clipping, and optional depth sampling (Section 3.4:
draw `r ~ Uniform{1..R}` at every step) which the committed sweep leaves off so
that the comparison against a fixed-depth baseline changes one thing at a time.

## Checks

`pytest` runs 66 tests. The ones that matter:

- **The clean limit.** With the projection and the gate ablated, `R = 1` is
  *bit-identical* to the non-recurrent prelude + core + coda stack built from
  the same blocks (`mx.array_equal`, not `allclose`). A constant gate of 0
  lands on the same logits.
- **Parameters do not depend on R.** The parameter tree, its shapes and its
  size are unchanged by running at R = 1, 2, 3, 8, 16, and two configs that
  differ only in their default recurrence depth hold identical trees. Only
  unique blocks add parameters.
- **Compute scales linearly with R.** Block executions are *counted* during
  real forward passes and equal `n_pre + n_core * R + n_coda`, with equal
  increments of `n_core` per extra step.
- **The gate is a gate.** At `g = 1` every step's state equals the anchor
  exactly; at `g = 0` the final state equals the shared block's output
  recomputed by hand; `gated_update` matches Eq. 4 elementwise.
- **Noise contract.** Without a key two forwards are bit-identical; the same
  key reproduces the same noise; two different keys differ; forcing both noise
  scales to zero reproduces the noiseless forward exactly; state noise and gate
  noise each change the result on their own.
- **The prelude conditions every recurrence.** `LN` makes the gate invariant to
  constant shifts of the anchor, but a real perturbation changes the gate and
  the projection at *every* step, and nudging a prelude weight moves the state
  at every step, not just the first.
- **Gradients.** Gradients of each per-step state energy reach the shared core
  with a nonzero norm from every iteration, and the LM loss gradients are
  finite for R = 1, 2, 5.
- Shapes and dtypes (float32 logits, gates in [0,1], trace lengths equal R),
  attention causality, a hand-computed uniform-attention pattern, a
  hand-computed masked cross-entropy, and validation of bad R and temperature.

## Experiment

`scripts/train_recall.py` trains one model per configuration on the recall task
and writes `results/isoflops_isoparams.txt` and `.json`. Every number below is
in those files.

Setup: dim 64, 4 heads, mlp ratio 4, batch 64, 2000 steps of AdamW at peak
learning rate 3e-3 with 100 warmup steps, gradient clip 1.0, state noise
`sigma_x = 0.1` and gate noise `sigma_g = 0.1` (the paper's values), gate
temperature 1.0. One seed (0) shared by every run, so all models see the same
batch stream in the same order. Evaluation is noise-free, on 16 held-out
batches (1024 sequences, 8192 scored positions), at the full R. The GRT has
`n_prelude = n_coda = 1` and sweeps the core depth `n_core in {1, 2}` and
`R in {1, 2, 4, 6}`; the baseline is a plain stack with either the same number
of block applications (isoFLOPs) or the same number of unique blocks
(isoParams).

The GRT sweep. "Applications to solve" is the first evaluation step with
held-out accuracy >= 0.99, times the applications per step:

| run | core | R | unique blocks | params | apps/step | held-out loss | accuracy | solve step | applications to solve |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grt-c1-R1 | 1 | 1 | 3 | 177,344 | 3 | 0.0001 | 1.0000 | 750 | 2,250 |
| grt-c1-R2 | 1 | 2 | 3 | 177,344 | 4 | 0.0001 | 1.0000 | 1000 | 4,000 |
| grt-c1-R4 | 1 | 4 | 3 | 177,344 | 6 | 0.0001 | 1.0000 | 1250 | 7,500 |
| grt-c1-R6 | 1 | 6 | 3 | 177,344 | 8 | 0.0001 | 1.0000 | 1000 | 8,000 |
| grt-c2-R1 | 2 | 1 | 4 | 227,328 | 4 | 0.0001 | 1.0000 | 1500 | 6,000 |
| grt-c2-R2 | 2 | 2 | 4 | 227,328 | 6 | 1.7955 | 0.2595 | never | not reached |
| grt-c2-R4 | 2 | 4 | 4 | 227,328 | 10 | 1.7955 | 0.2582 | never | not reached |
| grt-c2-R6 | 2 | 6 | 4 | 227,328 | 14 | 0.0001 | 1.0000 | 1250 | 17,500 |

The non-recurrent baselines, same budget and same batch stream:

| run | unique layers | params | apps | held-out loss | accuracy | solve step |
|---|---:|---:|---:|---:|---:|---:|
| plain-3L | 3 | 156,480 | 3 | 1.7956 | 0.2578 | never |
| plain-4L | 4 | 206,464 | 4 | 1.6380 | 0.3429 | never |
| plain-6L | 6 | 306,432 | 6 | 1.7957 | 0.2640 | never |
| plain-8L | 8 | 406,400 | 8 | 1.7960 | 0.2618 | never |
| plain-10L | 10 | 506,368 | 10 | 1.7962 | 0.2600 | never |
| plain-14L | 14 | 706,304 | 14 | 1.7962 | 0.2614 | never |

isoFLOPs (same number of block applications, different parameter count):

| GRT | plain | apps | params GRT | params plain | plain/GRT | loss GRT | loss plain | dLoss | acc GRT | acc plain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grt-c1-R1 | plain-3L | 3 | 177,344 | 156,480 | 0.88 | 0.0001 | 1.7956 | -1.7956 | 1.0000 | 0.2578 |
| grt-c1-R2 | plain-4L | 4 | 177,344 | 206,464 | 1.16 | 0.0001 | 1.6380 | -1.6380 | 1.0000 | 0.3429 |
| grt-c1-R4 | plain-6L | 6 | 177,344 | 306,432 | 1.73 | 0.0001 | 1.7957 | -1.7956 | 1.0000 | 0.2640 |
| grt-c1-R6 | plain-8L | 8 | 177,344 | 406,400 | 2.29 | 0.0001 | 1.7960 | -1.7960 | 1.0000 | 0.2618 |
| grt-c2-R1 | plain-4L | 4 | 227,328 | 206,464 | 0.91 | 0.0001 | 1.6380 | -1.6379 | 1.0000 | 0.3429 |
| grt-c2-R2 | plain-6L | 6 | 227,328 | 306,432 | 1.35 | 1.7955 | 1.7957 | -0.0002 | 0.2595 | 0.2640 |
| grt-c2-R4 | plain-10L | 10 | 227,328 | 506,368 | 2.23 | 1.7955 | 1.7962 | -0.0007 | 0.2582 | 0.2600 |
| grt-c2-R6 | plain-14L | 14 | 227,328 | 706,304 | 3.11 | 0.0001 | 1.7962 | -1.7961 | 1.0000 | 0.2614 |

GRT lower held-out loss in 8/8 cells (mean delta -1.3075).

isoParams (same number of unique blocks):

| GRT | plain | apps | params GRT | params plain | plain/GRT | loss GRT | loss plain | dLoss | acc GRT | acc plain |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| grt-c1-R1 | plain-3L | 3 | 177,344 | 156,480 | 0.88 | 0.0001 | 1.7956 | -1.7956 | 1.0000 | 0.2578 |
| grt-c1-R2 | plain-3L | 4 | 177,344 | 156,480 | 0.88 | 0.0001 | 1.7956 | -1.7956 | 1.0000 | 0.2578 |
| grt-c1-R4 | plain-3L | 6 | 177,344 | 156,480 | 0.88 | 0.0001 | 1.7956 | -1.7955 | 1.0000 | 0.2578 |
| grt-c1-R6 | plain-3L | 8 | 177,344 | 156,480 | 0.88 | 0.0001 | 1.7956 | -1.7956 | 1.0000 | 0.2578 |
| grt-c2-R1 | plain-4L | 4 | 227,328 | 206,464 | 0.91 | 0.0001 | 1.6380 | -1.6379 | 1.0000 | 0.3429 |
| grt-c2-R2 | plain-4L | 6 | 227,328 | 206,464 | 0.91 | 1.7955 | 1.6380 | +0.1575 | 0.2595 | 0.3429 |
| grt-c2-R4 | plain-4L | 10 | 227,328 | 206,464 | 0.91 | 1.7955 | 1.6380 | +0.1575 | 0.2582 | 0.3429 |
| grt-c2-R6 | plain-4L | 14 | 227,328 | 206,464 | 0.91 | 0.0001 | 1.6380 | -1.6379 | 1.0000 | 0.3429 |

GRT lower held-out loss in 6/8 cells (mean delta -1.2679).

**What these numbers mean.** In this run 6 of the 8 GRT configurations reached
99% held-out accuracy within 2000 steps and **no** plain stack did: the
baselines all sat on the same plateau (loss 1.64-1.80, accuracy 0.26-0.34),
including `plain-14L` with 706,304 parameters and 14 applications per step
against `grt-c1-R1` with 177,344 parameters and 3. That is the shape of story
the paper tells under isoFLOPs, and the isoFLOPs table goes the same way in
8/8 cells. In six of those eight cells the plain stack holds 1.16x to 3.11x
more parameters at matched block applications; in the two smallest cells (3 and
4 applications) the GRT is the larger model (ratios 0.88 and 0.91), because its
projection and gate are a bigger fraction of a 3- or 4-block model than they
are at GPT-2 scale. Under isoParams the picture is weaker: 6/8 cells
favour the GRT, and `grt-c2-R2` and `grt-c2-R4` are 0.1575 nats *worse* than
`plain-4L` at the same unique block count, because that configuration never
escaped the plateau. The cheapest cell that solved the task was `grt-c1-R1` at
2,250 block applications (750 steps x 3); the most expensive was `grt-c2-R6` at
17,500. So more recurrence was not monotonically better here: with `n_core = 1`,
R = 1, 2, 4 and 6 all solved it, while with `n_core = 2` only R = 1 and R = 6
did.

**What they do not mean.** This is one synthetic task, one seed, one model
width, and a 2,000-step budget: it is not a scaling study and says nothing
about the paper's 124M-774M-parameter comparisons. The plain stacks may simply
have needed a different learning rate or more steps to escape the plateau,
which would be an optimisation story rather than an architecture one; I did not
run that ablation. Wall-clock is not comparable either: `grt-c1-R1` took 53.8s
against 14.2s for `plain-3L` at the same number of block applications, because
each recurrence is a separate Python-level pass with projection and gate work
on top, which the block-application proxy does not charge for.

The results file also holds a length-shifted evaluation (12 pairs instead of
the 8 seen in training, `shift L` / `shift A` columns). It should be read with
care: the model configs were built with `max_seq_len = 36` but training only
ever visited positions 0-23, so the position embeddings used by that split were
never trained. It mostly measures confidence under that shift: the
configurations that solved the task answer confidently and score a shifted loss
of 4.3671-5.7772, while the ones stuck on the plateau score 3.0326-3.1402.

## Where this may differ from the paper

- **The gate's name.** Eq. 4's `g` is the copy gate: `g = 1` passes the state
  through and `g = 0` takes the core output. Read as "how much of the core
  output is taken", the gate is `1 - g`.
- **Initial state.** Section 3.2's prose says the first step is initialised with
  noise conditioned on the prelude output, while Listing A.1 sets the state to
  the prelude output and resamples noise inside the loop at every step. I
  followed the listing and the abstract's "noise resampled at every step":
  `h^(0) = h^(pre)`, with `eps_x` added before `W_proj` on every step.
- **Gate MLP details.** The paper specifies `f_g` as a two-layer SiLU MLP of
  hidden width `d` over `[LN(state), LN(anchor)]`. The official implementation
  prepends an extra layer norm over the concatenated features; I did not, and I
  initialise the final bias to +4 as the paper describes.
- **`W_proj` bias.** Eq. 2 defines a `d x 2d` matrix, so my projection has no
  bias (the reference implementation uses a linear layer with bias). This
  changes the parameter count slightly in the GRT's favour.
- **Noise at evaluation.** The paper always resamples during training; here
  training passes a key every step (sigma_x = sigma_g = 0.1) and evaluation is
  noise-free, which keeps the held-out numbers deterministic.
- **Depth sampling.** Implemented (`depth_sampling=True`) but switched off in
  the committed sweep, so that the GRT and the fixed-depth baseline differ only
  in the recurrence.
- **What is not implemented.** No KV-cache sharing, no early-exit analysis, no
  bfloat16, no compiled kernels, no depth-sampling schedules, no MoR / Poisson
  / RRT / Ouro baselines, and no language-modelling corpus: the task is a
  synthetic recall problem, not text. Weight tying is off, so both models carry
  a separate head.
- **Scale.** `dim = 64` and 3-14 blocks, against the paper's 768-1280 and
  3-36 unique blocks at GPT-2 Small/Medium/Large scale with ~9.8B training
  tokens.

## Running it

```
/Users/dash/Documents/dev/ai_papers/.venv/bin/python -m pytest
/Users/dash/Documents/dev/ai_papers/.venv/bin/python scripts/train_recall.py
```

The default experiment (`--steps 2000 --batch 64 --pairs 8 --keys 16 --lr 3e-3
--seed 0 --core-depths 1 2 --recurrences 1 2 4 6`) runs 14 trainings in about
13 minutes on an M-series GPU and rewrites the two files in `results/`.
