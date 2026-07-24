# Docs

Reference and concepts for LoRA/LoKr training with these tools.

- **[Tools reference](tools.md)** — usage, examples, and flags for every script in
  [`tools/`](../tools/).
- **[Training budget and batching](training-budget-and-batching.md)** — how to size
  a run (steps, epochs, batch, effective steps), the difference between data
  exposure and gradient updates, learning-rate scaling, and gradient accumulation.
  This is the theory behind [`tools/calc_training.py`](tools.md#training-settings-calculator-toolscalc_trainingpy).
- **[Monitoring training](monitoring-training.md)** — why diffusion/flow training
  loss looks flat, what it does and doesn't tell you, and how to use a deterministic
  validation loss with a held-out dataset (incl. OneTrainer setup) to actually spot
  overfitting.
- **[LoKr training: dim, alpha, and factor](lokr-dim-and-alpha.md)** — what the
  three LoKr knobs do, why `alpha/dim` (the scale) is the number that matters, why
  a high LR stays stable, and how to set them for a character.

These are working notes, not gospel — the numeric examples use a 31-image Krea 2
character LoKr as the running case.
