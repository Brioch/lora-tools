# Monitoring training

How to tell whether a LoRA/LoKr is actually learning — because the training loss
mostly won't tell you.

## Why the training loss looks flat (~0.1)

For diffusion / flow-matching models (Krea 2 included), a smoothed training loss
that hovers around ~0.1 and barely dips to ~0.09 over a whole run is **normal and
expected**, not a sign of a stuck model. The loss is a poor progress meter here,
for four reasons:

1. **Each step's loss is a random draw, not a convergence measure.** Every step
   samples a random **timestep** (noise level) and fresh random **noise**, then
   asks the model to predict it. Difficulty swings wildly with the timestep —
   near-pure-noise steps are almost impossible, low-noise steps are easy — so the
   per-step loss is dominated by *which timestep got sampled*, not by how well your
   concept is learned. The smoothed value is roughly the mean over that
   distribution: a near-constant number.
2. **The minimum achievable loss isn't 0.** At high noise levels many different
   clean images could have produced a given noisy latent, so even a *perfect* model
   can't drive the MSE to zero. The loss plateaus at a positive floor (your ~0.1).
3. **The concept signal is tiny relative to that floor.** A LoRA/LoKr on ~31 images
   adjusts a small low-rank subspace and only nudges predictions. Its contribution
   is a small fraction of the total MSE — often *inside* the noise floor. So a
   0.10 → 0.09 move can represent real, sufficient learning.
4. **Small dataset + random sampling** → high step-to-step variance, which
   smoothing flattens into a near-horizontal line.

## What the loss does and doesn't tell you

- It does **not** cleanly indicate "trained enough" or "overfit." You won't see
  overfitting as rising *training* loss.
- It's mainly useful for catching **breakage** — a divergence spike, or a flat line
  at the wrong magnitude (bad LR/config).
- The reliable signals are **generated samples at checkpoints** (the gold standard)
  and a **deterministic validation loss** (below).

## Deterministic validation loss

The stochastic training loss hides the overfit turn; a **fixed-seed, fixed-timestep
validation loss** reveals it. Evaluate loss on a held-out set the same way every
time (same seed, same timestep) so successive validations are comparable. That
curve trends **down** while the model generalizes, then turns **back up** when it
starts memorizing the training set — and the checkpoint just before the turn is
usually your best one.

## Held-out datasets

"Held-out" means images the model **never trains on**. Split your pictures into a
training pile (what the optimizer sees) and a validation pile it never touches.
Loss on images it *has* trained on just keeps dropping as it memorizes them — that
tells you nothing. Loss on *unseen* images is what reveals generalization vs
memorization.

What makes a good validation set for a **character** LoKr:

- **Same character**, just images kept out of training — you're measuring "does it
  predict *this* concept better over time."
- **Representative and varied** — a few different poses/angles/lighting, so the loss
  reflects general likeness, not one specific shot.
- **Captioned the same way** as training images (same trigger/caption style).
- **Distinct captures, not near-duplicate frames** — a near-identical val image
  behaves like a training image and hides overfitting (see below).
- **Small is fine** — 3–10 images; the loss is averaged, so more just smooths it.
- Same **resolution/bucketing** as training, roughly.

### How close to the training data is too close?

The test isn't "how different from training" — it's "is this a distinct capture,
or basically the same photo?" What blunts the signal is **visual near-duplication**,
not shared wardrobe.

- ✅ **Same outfit, clearly different pose/angle/expression/framing** — ideal. It's
  unseen, and predicting it well requires generalizing the character's *identity*,
  which is exactly what you want to measure. Outfit isn't what validation tests, so
  sharing it is fine (and keeps the concept consistent).
- ⚠️ **Same pose with only minor variation** (near-dupe / burst-shot sibling) —
  weak. As the model memorizes the matching training frame, its prediction on the
  near-clone improves too, so the validation loss keeps *dropping* even while it
  overfits — a falsely reassuring curve that hides the turn.
- ❌ **The same frame, or a crop of a training image** — don't use it; it behaves
  like a training image.

This is why images you left *out of training as redundant* are often perfect for
validation: "redundant for training" (added no new signal) is not the same as
"redundant for validation" (still tests generalization) — as long as they aren't
near-clones of a training frame.

> To catch near-clones you missed by eye, run
> [`tools/compare_datasets.py`](../README.md#checking-a-validation-set-for-duplicates-toolscompare_datasetspy)
> on your two folders — it perceptual-hashes both sets and flags validation images
> too close to any training image.

It's a spectrum, not pass/fail. If your only spares lean toward near-duplicates,
the curve is still *directionally* useful — the overfit inflection just shows up
subtler and later than it truly occurs, so read it as a soft signal rather than a
precise checkpoint-picker.

### What about images left out for low quality?

A weak source — for a different reason than closeness. "Never trained on" is
satisfied, but low quality degrades the *loss signal itself*:

- **The degradation dominates the loss.** On a blurry / noisy / compressed /
  low-res image, much of what the model must reconstruct is that junk, which has
  nothing to do with the character. The loss is driven by the irreducible
  difficulty of the degraded content, so the character-generalization component
  (already a small fraction — the reason training loss looks flat) becomes an even
  smaller fraction of a bigger, noisier number, and the overfit inflection gets
  buried.
- **Distribution shift.** If training images are clean and validation images are
  degraded, the val loss partly measures "how well does the model handle degraded
  images," not "has it learned the character." Even a perfect LoRA can score
  erratically on an off-distribution image.
- **Saving grace:** deterministic validation compares each image *to itself* over
  time, so quality doesn't break comparability — the *trend* still means something.
  But the signal-to-noise is worse: a higher, noisier curve where the turn is
  harder to pin down.

Degree matters: mildly soft / slightly worse lighting but sharp and clearly framed
is basically fine; heavy blur, strong compression, very low res, or a tiny/occluded
face drowns the signal. Preference order: a few **extra decent-quality** shots →
good **same-outfit-different-pose** hold-outs → low-quality rejects only as a last
resort, read loosely. And don't let one very degraded image dominate the averaged
concept loss.

So: "left out for redundancy" is a great validation source; "left out for low
quality" is a weak one — same unseen status, but it dilutes the exact signal you
want to read.

### Captions and masks: mirror your training setup

Validation loss is only a fair measurement if it's computed **the same way as
training loss**. So the validation set must reproduce the same conditioning and
loss-region as training:

- **Captions — always.** The diffusion loss is a *conditional* prediction (noised
  latent + timestep + **text embedding** → predict the noise), so the caption
  shapes the loss. Caption validation images the **same way as training** — same
  trigger word, same style. Uncaptioned images get scored under an empty/
  unconditional prompt, which measures something different from how you trained.
- **Masks — only if you train with masks, then mirror them.** Masks exist only
  because of **masked training** (loss computed over the white/subject region,
  ignoring the black/background). If you use masked training, give validation
  images masks too, made the same way (ClipSeg/Rembg or by hand) — otherwise the
  val loss includes the background you deliberately excluded, a different objective
  that adds noise and misaligns the curve. If you're not using masked training,
  validation images need no masks.

Through-line: whatever conditioning and loss-region training uses, reproduce it on
the validation set — that's what keeps the loss a like-for-like measurement and the
overfit inflection clean.

**The small-dataset catch.** Validation images come *out* of training, and pulling
5 from 31 measurably weakens a character LoRA. Best move: source a few **extra**
images of the character you wouldn't have trained on anyway (redundant poses, an
alternate shot, slightly lower quality) and make *those* the validation set — you
lose nothing from training. If you can't, hold out ~3–5 of your weakest/most
redundant shots, or skip formal validation and judge by samples (a defensible
choice for a 31-image set). Validation loss is most worth the trade-off when you
want to pin down the exact epoch where overfitting begins.

## OneTrainer validation setup

OneTrainer implements exactly this deterministic validation loss:

1. **General tab** → enable **validation** and set a **validation interval** (every
   N steps, like the sampling interval).
2. Add a **validation concept**, flagged as a validation concept and captioned
   normally — using **held-out images, not your training set**.
3. Watch the per-concept **validation loss graph in TensorBoard**; the inflection
   where it stops dropping / turns up is the overfitting onset.

It's deliberately **deterministic** — it seeds to a fixed value (0) and locks the
timestep — so validations are comparable run-to-run.

**Known limitation** ([issue #772](https://github.com/Nerogar/OneTrainer/issues/772)):
the current implementation uses the *same single timestep* for all validation steps
rather than a fixed *distribution* across the noise schedule. So it measures loss at
essentially one noise level — still perfectly comparable between validations (good
for spotting the overfit turn), but treat it as a consistent *slice*, not a
full-schedule average. The ideal (deterministic *between* validations, distributed
*within* each) is still under discussion.

### Sources

- [Training — OneTrainer Wiki](https://github.com/Nerogar/OneTrainer/wiki/Training)
- [PR #660: Make validation deterministic](https://github.com/Nerogar/OneTrainer/pull/660)
- [Issue #772: Same timestep used for all validation steps](https://github.com/Nerogar/OneTrainer/issues/772)
- [Discussion #594: What does validation do?](https://github.com/Nerogar/OneTrainer/discussions/594)
