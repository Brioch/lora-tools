# Changelog

All notable changes to this project are documented here. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and the project aims to
follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [Unreleased]

### Added
- `caption_stats` — tag distribution across a dataset's captions, calling out
  ubiquitous tags (no discriminative signal) and rare ones (tagger noise), using the
  same tokenizer as `edit_captions` so the counts match what an edit would touch.
- `split_dataset` — hold out a validation set without leaking near-duplicates, by
  splitting on perceptual-hash clusters rather than individual images so a frame and
  its near-twins never straddle the split.
- `compare_loras` — per-module ΔW cosine similarity, magnitude ratio and relative
  change between two adapters, separating "learned something different" from "same
  thing, louder". Computed without materializing ΔW.
- `lint_dataset`: `--trigger WORD` warns about captions missing the trigger word
  (matched as a whole tag), and near-blank images are now flagged by grayscale
  dynamic range (`--min-contrast`, on by default).
- Dataset preparation tools: `lint_dataset` (validate image + caption pairs, exits
  non-zero on real problems), `prepare_images` (EXIF orientation, RGB flattening,
  format conversion, max-edge downscaling; writes to a sibling `<dir>.prepared/`
  unless `--in-place` is given), `dedupe_images` (exact and perceptual
  near-duplicate removal), `edit_captions` (tag-aware batch caption edits) and
  `balance_regularization` (match a regularization set to the training image count),
  plus the shared `dsutils` helper module they import.

### Changed
- `edit_metadata` docs and `--help` now use a generic placeholder LoRA title and
  trigger word in their examples.
- Test suite (`pytest`) with a coverage gate, strict `mypy` type checking, Ruff
  lint/format, pre-commit hooks, and GitHub Actions CI (lint, test, dependency
  audit).
- `docs/tools.md` reference and a slimmed-down README.
- Community health files (contributing, security policy, issue/PR templates) and
  Dependabot configuration.

## [0.1.0]

### Added
- Initial toolkit: `inspect_lora`, `edit_metadata`, `metadata_ui`,
  `embed_workflow`, `strip_metadata`, `calc_training`, `compare_datasets`.
- Training notes under `docs/`.
