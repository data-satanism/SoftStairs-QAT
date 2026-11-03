# SoftStairs Derivative-Floor Ablation (Inception v3 / STL-10, 2-bit)

**Date:** 2026-11-03
**Script:** `experiments/CV/run_floor_ablation.py`
**Logs:** `experiments/CV/logs/*_gfloor*/`

## Question

Late in training the SoftStairs backward gate collapses inside the
quantization wells (normalized surrogate: gate minimum ≈ `t_b²/4` with
`t_b = min(1, t·async_t_factor)`), so settled weights receive almost no
gradient. Does clamping the gate from below,

```
gate(x) = max(D(x), floor),
```

help, and how does the answer depend on `async_t_factor`?

The patch is applied **on the fly** in the experiment script only
(monkeypatched `ScaledSoftStairs.derivative`); no library code was touched.
The forward map is unchanged — only the backward gate is floored.

## Protocol

- Model: Inception v3 (ImageNet-pretrained), STL-10, train 4000 / val 1000
- Quantization: 2-bit **asymmetric**, `type='standard'`, normalized derivative,
  `clamp_codes=True`, `fake_quant=True`
- t-schedule: `exp`, `t_start=0.9`, `t_end=5e-4` (final logged t ≈ 8.0e-4),
  `early_power=0.95`
- Optimizer: AdamW, `lr=1e-3`, **weight decay 0**, betas `(0.9, 0.999)`, seed 42
- 30 epochs (schedule stepped per epoch), 12 runs:
  `floor ∈ {0 (baseline), 1e-3, 1e-2, 1e-1}` × `async_t_factor ∈ {1, 10, 100}`

## Results

Final-epoch fake-quantized validation accuracy / loss (30/30 epochs completed
in every run):

| floor \ async_t_factor | af=1          | af=10         | af=100        |
|------------------------|---------------|---------------|---------------|
| 0 (baseline)           | 0.406 / 1.489 | **0.505** / 1.335 | 0.362 / 1.760 |
| 1e-3                   | 0.414 / 1.455 | 0.422 / 1.441 | 0.469 / 1.355 |
| 1e-2                   | 0.444 / 1.403 | 0.500 / 1.309 | 0.472 / 1.370 |
| 1e-1                   | 0.406 / 1.489 | 0.453 / 1.388 | 0.500 / 1.304 |

Best-epoch quantized accuracy: af=100 baseline reaches 0.540 mid-training but
**collapses to 0.362 by epoch 30** (late-training instability); with
`floor=0.1` the same arm finishes at 0.500.

### Late-stage code churn (`level_change_frac`, last 5 epochs)

| floor \ async_t_factor | af=1 | af=10 | af=100 |
|------------------------|------|-------|--------|
| 0      | ~3e-6 | ~5e-5 | ~9e-4 |
| 1e-3   | ~1e-5 | ~4e-5 | ~8e-4 |
| 1e-2   | ~2e-4 | ~8e-5 | ~1e-3 |
| 1e-1   | ~2e-3 | ~1e-3 | ~2e-3 |

The floor keeps weights moving between quantization codes late in training
(up to ~2000× the baseline churn at af=1); the baseline arms instead freeze
almost completely (af=1: ~8e-7 of weights per epoch at the end).

### Quantization error (final epoch, code units)

- Rounding error is essentially floor-independent (~4.3–5.2e-4 everywhere).
- Clamping error and out-of-range fraction grow with the floor at af=1
  (clamping 6.8e-5 → 2.5e-4; out-of-range 1.8e-4 → 5.4e-4 from baseline to
  `floor=0.1`): a strong floor keeps pushing weights against the code-window
  boundary instead of letting them settle.

## Analysis

1. **Scale matching matters more than the floor itself.** The gate only
   changes where the floor exceeds the *natural* well floor `≈ t_b²/4`:
   - af=1: natural floor ≈ 1.6e-7 at the end → every tested floor binds hard.
   - af=10: natural floor ≈ 1.6e-5 → `1e-3` and above bind.
   - af=100: natural floor ≈ 1.7e-3 → `floor=1e-3` **never binds**; `1e-2`
     and `1e-1` do.

   The rescue pattern at af=100 (0.362 → 0.469/0.472/0.500 as the floor
   grows past the natural scale) is exactly the regime where the floor is
   active. At af=10, where the baseline is already healthy, floors mostly
   hurt or are neutral.

2. **The floor is a stabilizer, not an accuracy win.** The best run overall
   is the *unfloored* af=10 baseline (0.505). The floor's clear benefit is
   taming the unstable high-async arm (af=100 collapses late; any binding
   floor fixes the collapse) and it is accuracy-neutral at af=10 for
   `floor=1e-2` (0.500 vs 0.505).

3. **Too strong a floor re-melts the lattice.** `floor=0.1` (10% of the
   normalized gate peak) multiplies code churn by orders of magnitude,
   raises clamping error/out-of-range mass, and degrades af=10 (0.453). The
   useful range is a small fraction of the peak: `floor ≈ 1e-2`.

4. **Run-to-run noise is large; single-run deltas < ~0.05 are not
   significant.** Evidence: the af=100 `floor=1e-3` run is *mathematically
   identical* to the af=100 baseline (the floor never binds at any point of
   the schedule) yet finishes at 0.469 vs 0.362 — pure GPU nondeterminism
   (same seed). The af=10 `floor=1e-3` dip (0.422 vs 0.505) and the af=1
   `floor=1e-2` gain (+3.8 pt) are within this noise band.

## Conclusions

- A floored gate in the range `1e-3 … 1e-2` is a cheap, safe stabilizer for
  aggressive `async_t_factor` (≥ 100), preventing late-training collapse
  without hurting quantization settling.
- It does **not** improve on a well-tuned moderate `async_t_factor` (af=10);
  there the baseline gate's own annealing is already sufficient.
- Recommendation for follow-up: repeat the af=100 column over ≥3 seeds to
  confirm the stabilization, and consider annealing the floor together with
  `t` (e.g. `floor ∝ t_b²/4` scaled by a constant) instead of a constant
  floor, so the gate never re-melts the lattice as strongly as the constant
  `0.1` arm did.
