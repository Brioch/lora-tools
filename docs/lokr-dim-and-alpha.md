# LoKr training: dim, alpha, and factor

What the three knobs on a LyCORIS **LoKr** actually do, why a "high" learning
rate stays stable, and how to set them for a character. Running case throughout:
a **31-image Krea 2 character** at `dim 16`, `alpha 4`, `factor -1`, `LR 0.002`,
~100 epochs.

## The one number that matters: the scale

LoKr injects a delta into each frozen base weight and multiplies it by a fixed
scalar before it reaches the output:

```
y = W_base·x  +  s · (ΔW)·x        s = alpha / dim
```

For the running case, `s = 4 / 16 = 0.25` — the adapter's contribution is
attenuated to a quarter. This single ratio explains most of LoKr's training
behaviour, so **think in `alpha/dim`, not in alpha or dim alone**. Halving dim
*or* doubling alpha both double the scale; the community rule "hold `alpha/dim`
constant when you change dim" is just "hold the scale fixed."

[`tools/lora_health.py`](tools.md#lora-weight-health-toolslora_healthpy) folds
this exact `alpha/dim` factor back in when it reports `‖ΔW‖_F`, so its magnitudes
are the *effective* contribution — no need to multiply by the scale yourself.

## Why a high LR (0.002) doesn't blow up

`2e-3` sounds hot only if you're anchored to full-fine-tune LRs (`1e-5`–`5e-5`).
For LoKr it's normal-to-conservative, and several things keep it stable:

1. **The scale caps output swing.** With `s = 0.25`, whatever the adapter learns
   is throttled to a quarter on the way to the output. A high LR can't translate
   into a big per-step change at the layer output — the loss curve stays calm by
   construction.
2. **`B` starts at zero.** The adapter begins as a no-op and eases in, so early
   high-LR steps don't lurch the frozen base off its manifold.
3. **Adam normalizes gradient magnitude.** The step is `≈ lr · m̂/√v̂`, so the raw
   gradient scale matters far less than the LR itself.
4. **Warmup + few trainable params** → a smooth, low-dimensional landscape with
   little room for a chaotic jump.

What *would* still diverge even at `2e-3`: a large scale (high `alpha/dim`), no
warmup, tiny batch with noisy gradients, or training too many modules at once.

### A subtlety: alpha is *not* simply a second learning rate

The folk wisdom "alpha acts like an LR multiplier" is true under **plain SGD**,
where the scale hits both the gradient into the params *and* the params' effect
on the output — compounding to roughly an `s²` damping. Under **Adam/AdamW**
(what you're almost certainly using) a constant factor `s` on the gradient
appears in both `m̂` and `√v̂` and **cancels** in the ratio, so the *parameter*
step size is essentially alpha-independent. What does **not** cancel is the
forward attenuation: the adapter's weights move at the same speed, but their
**reach into the output is capped at `s`**. So low alpha under Adam means the
adapter must grow **larger internal weights** (or train longer) to reach a given
effective strength — it sets the concept's **strength ceiling**, not the raw
update speed. Practically: **when likeness is soft, raise alpha, not LR.**

## dim — the inner rank, not a plain LoRA rank

`dim` (a.k.a. `lora_dim`) is the rank of the low-rank decomposition applied to
the **larger of the two Kronecker blocks** — not a flat LoRA rank over the whole
weight. Because the Kronecker structure multiplies capacity, **`dim 16` on a LoKr
is more expressive than LoRA rank 16 sounds.** For a single character, 16 is
generous; the identity of one subject rarely needs more. Raising dim mostly buys
capacity you'll spend on overfitting, and it *lowers* the scale (`alpha/dim`)
unless you bump alpha to match.

## factor — how the weight is split

LoKr decomposes each weight `W ≈ A ⊗ B` (Kronecker product of two smaller
blocks). `factor` caps the size of one block:

- **`factor = -1`** (the running case) removes the cap → the **most balanced
  split** the shape allows → the **most parameter-efficient / smallest file**,
  and the strongest built-in regularization. Great default for a character.
- **A smaller `factor` (e.g. 8 or 4)** keeps one block larger → more raw capacity
  and a bigger file. Reach for it only if alpha bumps alone can't lock likeness.

`dim` then applies its low-rank decomposition to the larger resulting block, so
`factor` and `dim` together set how much the adapter *can* represent, while
`alpha/dim` sets how hard it *pushes*.

## Regularization: you already have plenty

Kronecker structure + `factor -1` + a low scale is a lot of built-in
regularization, which is why a **near-zero dropout is fine** on this config:

- **`dropout 0.01` is effectively off** — too small to regularize, not zero
  either. Either set it to `0` and rely on the architecture (reasonable here), or
  make it *do* something at `0.05–0.1`.
- Bump toward **`0.05`** only if a real train/val gap opens up — most likely with
  **many epochs over a small set** (100 × 31 qualifies), so it's a defensible
  nudge if generations start looking rigid. Otherwise leave it.

## Setting it for a character — recommended order

Tune in this order; stop when likeness is right:

1. **alpha** first — it's the strength knob. `alpha 4`/`dim 16` (scale 0.25) is
   conservative for a face. If likeness is *close but not quite them* or drifts
   between generations, try **`alpha 8`** (scale 0.5), then **`alpha 16`**
   (scale 1.0). This is almost always the right lever before touching LR.
2. **factor** next — drop `-1 → 8` for more capacity only if alpha maxed out
   doesn't get likeness there.
3. **dropout** — raise to `0.05` only against a widening train/val gap.
4. **LR / epochs** last — the config is stable, so these rarely need attention.
   If anything, this recipe tolerates long training so well you can usually
   **trim epochs** (likeness often locks well before your last checkpoint)
   rather than push harder.

## Reading the run: what "stable" does and doesn't prove

A calm loss and a damped scale make LoKr forgiving — but "not diverging" ≠
"optimal," and a low-alpha config can quietly **underfit fine detail** while
looking perfectly healthy. Two independent reads catch what the loss won't:

- **Structural** — sweep
  [`tools/lora_health.py`](tools.md#lora-weight-health-toolslora_healthpy) across
  checkpoints. `‖ΔW‖_F` climbing then flattening = the adapter saturated; **stable
  rank falling** over epochs = updates collapsing onto fewer directions, the
  fingerprint of over-cooking. On the running case both stayed clean — magnitude
  rose smoothly to a gentle plateau and stable rank held flat — which is exactly
  the signature of the damped `s = 0.25` config: no collapse, wide safe window.
- **Behavioural** — generate at checkpoints. Structural health can't see
  overfitting-as-memorization; only samples can. See
  [Monitoring training](monitoring-training.md) for the deterministic validation
  loss and held-out-set method.

Because a well-behaved LoKr strengthens **smoothly** (magnitude up, rank flat),
the likeness↔flexibility trade-off is a **gradient, not a cliff**: later
checkpoints = maximum likeness, slightly less prompt-flexible; earlier = more
flexible, less locked. Pick by eye across the last several checkpoints rather
than reflexively grabbing the final one.

## Cheat sheet

| Knob | Running case | Does | Turn it when |
|---|---|---|---|
| `alpha/dim` (scale) | `4/16 = 0.25` | strength ceiling / output reach | likeness soft → raise alpha |
| `dim` | 16 | inner rank of the big Kron block | rarely; 16 is generous for one subject |
| `alpha` | 4 | numerator of the scale | first knob for stronger likeness |
| `factor` | −1 | Kronecker split → capacity & file size | −1 default; smaller only if alpha maxed |
| `dropout` | ~0.01 (≈off) | regularization | widening train/val gap → 0.05 |
| `LR` | 0.002 | step size (Adam-normalized) | rarely; stable by design |

### Sources

- [LyCORIS — GitHub](https://github.com/KohakuBlueleaf/LyCORIS) and its
  [algorithm details](https://github.com/KohakuBlueleaf/LyCORIS/blob/main/docs/Algo-Details.md)
  (LoKr = Kronecker-product decomposition; `factor`/`dim`/`alpha` semantics).
- Yeh et al., *Navigating Text-To-Image Customization: From LyCORIS Fine-Tuning
  to Model Evaluation* — the paper introducing LoKr/LoHa.
- Companion notes: [Monitoring training](monitoring-training.md) and
  [Training budget and batching](training-budget-and-batching.md).
