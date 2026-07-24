# Training budget and batching

How to size a training run, and what batch size actually changes. This is the
reasoning behind [`tools/calc_training.py`](tools.md#training-settings-calculator-toolscalc_trainingpy).

Running example throughout: a **31-image Krea 2 character LoKr**, trained as a 512
bulk pass (batch 2) plus a 1024 refinement pass (batch 1).

## The vocabulary

- **Step** — one optimizer update. The trainer shows the model a batch, computes a
  gradient, and nudges the weights once.
- **Batch size** — how many images are averaged into that one gradient.
- **Epoch** — one pass over the whole dataset. `steps_per_epoch = images / batch`.
- **Effective (batch-1-equivalent) steps** — a pass of `S` steps at batch `B`
  counts as `S × B`. This puts passes with different batch sizes on one comparable
  scale, so you can add them into a single budget (e.g. "1250 total").

`drop_last`: when `images` isn't divisible by `batch`, most trainers drop the
trailing partial batch, so `steps_per_epoch = floor(images / batch)`. For 31 images
at batch 2 that's `floor(15.5) = 15` steps/epoch (× 30 epochs = 450 steps). Some
trainers instead pad/keep it (`ceil`). The calculator floors by default and takes
`--keep-last` to ceil.

## Effective steps = data exposure

`steps × batch` is exactly the number of **image presentations** over the run, and
it equals `epochs × dataset_size`. That's a sound, standard way to size a run, and
it's what the calculator budgets against.

A useful consequence: because `steps_per_epoch × batch ≈ images`, a pass's
effective steps work out to **≈ `images × epochs` regardless of batch size**. The
batch size cancels out of the budget math — it only changes the *reported raw step
count*, not how much data the model sees. So the real dial the calculator exposes
is "how many passes over the data, split across resolutions."

## Effective steps vs gradient updates

These are two different things, and conflating them is the usual source of
confusion:

| Quantity | Formula | What batching does to it |
|----------|---------|--------------------------|
| Images seen (= effective steps) | `steps × batch` = `epochs × dataset` | **unchanged** for a given number of epochs |
| Gradient updates | `steps` alone | **fewer** with a bigger batch |

One step at batch `B` shows the model `B` images and does **one** update. So a
bigger batch means *fewer but lower-variance* updates for the same data exposure.

Therefore `steps × batch` is a solid measure of **data exposure**, but only a rough
heuristic for **training progress / convergence**. Whether batch-2 × 450 steps
truly "equals" batch-1 × 900 steps depends on learning-rate scaling (below) and on
staying under the "critical batch size" — at tiny batches like 1–2 the linear
approximation is fine. Treat batch-as-a-multiplier as an image count, not a
convergence law.

## Learning-rate scaling

The exposure-equivalence above assumes you **scale the learning rate with the batch
size**. Doubling the batch halves the number of updates, so each update must move
~2× as far to cover the same ground:

> **Linear scaling rule** (Goyal et al., 2017): multiply batch by `k` → multiply LR
> by `k`.

Without it, batch-2 × 450 steps *under-trains* relative to batch-1 × 900 (fewer
updates, same per-update size → less cumulative movement). So if you run the 512
pass at batch 2 with 2× LR, that's the rule in action.

Caveats:

- **Approximate**, and usually wants LR **warmup** to stay stable early on.
- **Optimizer-dependent.** Linear (`×k`) is the classic result for SGD. LoRA/LoKr
  training usually uses **AdamW** (or an adaptive optimizer like Prodigy /
  DAdaptation), which normalizes by gradient magnitude — for those, **√k scaling**
  (≈1.41× for batch 2) is often the better fit, or the optimizer auto-tunes the LR
  and manual scaling is moot. For a 1→2 jump the difference is small and ×2 is a
  perfectly defensible choice.

## Gradient accumulation

`accum` micro-batches are run forward+backward and their gradients **summed** before
a single optimizer update, so the **effective batch = `batch × accum`** — at
batch-1 VRAM cost, but ~`accum`× slower.

It's the cheap way to hit a target effective batch when VRAM caps the *real* batch.
Concrete case: 1024 res is VRAM-limited to batch 1, but you want it to match the
effective batch 2 of the 512 pass — run `batch 1, accum 2`.

What it changes and doesn't:

- **Unchanged:** images seen (the budget) and the epoch count — those depend on the
  image count, not batching.
- **Changed:** the number of optimizer updates drops (`steps_per_epoch =
  floor(images / (batch × accum))`), so the reported step count falls. In the
  example the 1024 pass goes from 310 steps → 150.
- **LR scaling** keys off the *effective* batch: scale the LR by `batch × accum`,
  not `batch` alone. This is exactly why accum is handy — it lets one LR be
  correctly scaled across passes that would otherwise have different batch sizes.

Trade-off summary: `batch 1, accum 2` ≈ `batch 2, accum 1` in training math
(same effective steps, same LR, same result up to tiny numerical/norm details that
don't matter for LoRA/diffusion), differing only in **VRAM** (accum is cheaper) and
**speed** (accum is slower).

## Sizing the running example

For the 31-image LoKr with a 1250 effective-step budget, split 3/4 – 1/4:

```
res   batch  steps/epoch  epochs  steps  effective
512   2      15           30      450    900
1024  1      31           10      310    310
                                  760    1210
```

Epochs come from `round(fraction × total / images)` — derived from the image count,
so independent of batching mode. Reproduce and tweak this with
[`tools/calc_training.py`](tools.md#training-settings-calculator-toolscalc_trainingpy).
