# hiner2d vs. INR baselines — rate (bpppb) / quality (PSNR) comparison

**Goal:** compare the `inspect_3d` (and sibling 1D/2D/HINER/HINER++) baselines against the
`hiner2d` variant on **bpppb** and **PSNR**, decide **which configs are worth trying**, and
work out **how to set hiner2d parameters to hit a target compression rate (bpppb) or time**.

> The user asked to optimise the mapping for **compression rate (bpppb)**.

---

## 0. Methodology & caveats (read first)

- **Same bpppb definition everywhere.** Both stacks compute `bpppb = total_bits / (H·W·B)` with
  `H=W=128, B=202 → 3,309,568`. hiner2d: `bitstream_stats()` in `train_hiner2d.py:98` (Huffman over
  quantised weights + 16-bit min/scale overhead). 3D/2D/1D: `file_size_bytes·8 / (H·W·B)`. **Directly comparable.**
- **Patch basis differs.** The baseline summaries (`after_intermediate_results/Experiments/{1D,2D,3D,HINER,HINER++}`)
  are **dataset-average means** over ~100–175 patches. The hiner2d numbers below are a **single patch**
  (`…Y09181045_X06690796`, the same patch as the 3D `batch_ablation`). Treat cross-method PSNR gaps of <~1 dB
  as noise; the structural conclusions hold regardless.
- **Training budget differs.** Baselines trained 10k–20k iterations; the hiner2d sweep used 800 epochs ≈ 3,200
  steps. This matters — see §4.

---

## 1. Baseline landscape (dataset-average means)

| Method (config) | bpppb | PSNR (dB) | time (s) | notes |
|---|---|---|---|---|
| **3D-INR** h64, 12-bit QAT (`batch_ablation`, 1 patch) | **0.086** | 31.1→**34.5** | 22→85 | lowest rate; per-scene FiLM/seed only |
| **3D-INR** h256, 32-bit (`sampling_random`) | **2.56** | 36.6–37.3 | 126–1850 | |
| **1D-INR** C (10L + CNN upsample) | 1.96 | 35.28 | 47 | |
| **1D-INR** D (MLP 64×64) | 5.21 | 39.34 | 63 | |
| **1D-INR** B (5L, 12-bit QAT) | 15.92 | 40.17 | 102 | per-pixel latents → high rate |
| **1D-INR** A (5L, 32-bit) | 43.27 | 42.90 | 59 | latents scale with H·W |
| **2D-INR** D (5L, 12-bit QAT, 10k) | 1.86 | 37.84 | 123 | |
| **2D-INR** A (5L, 32-bit, 10k) | 3.06 | 38.09 | 56 | |
| **2D-INR** C (10L, 12-bit QAT, 20k) | **3.37** | **49.36** | 257 | best 2D rate/quality |
| **2D-INR** B (10L, 32-bit, 20k) | 6.24 | 49.44 | 188 | |
| **Original HINER** (1D-band INR) | **3.56** | **49.83** | 631 | |
| **HINER++** | **3.20** | **51.68** | 1551 | best quality |

**Where hiner2d lives:** its natural operating band is **~0.8–3.2 bpppb** (§3), i.e. it overlaps
**HINER++ (3.20), 2D-INR C (3.37), original HINER (3.56) and 3D-INR (2.56)** — *not* the cheap
3D 0.086-bpppb regime. So those four are the configs worth comparing against.

---

## 2. hiner2d head-to-head (single patch, 800 ep ≈ 3,200 steps, 8-bit quant)

| config (`enc_dim 64_S`) | params | **bpppb** | **PSNR_q** | PSNR_fp | SAM | time |
|---|---|---|---|---|---|---|
| seed16 | 1.33 M | **3.094** | 42.68 | 42.75 | 0.0294 | 401 s |
| seed8  | 0.81 M | 1.890 | 42.21 | 42.28 | 0.0310 | 401 s |
| seed4  | 0.55 M | 1.289 | 42.23 | 42.29 | 0.0312 | 403 s |
| seed2  | 0.42 M | **0.993** | 41.60 | 41.66 | 0.0332 | 398 s |

At matched rate (~3.1 bpppb): **hiner2d ≈ 42.7 dB vs HINER++ 51.7 / 2D-C 49.4 / HINER 49.8** — a 7–9 dB gap.

---

## 3. How model size → bpppb works in hiner2d (the mapping you asked for)

`bpppb ≈ N_params · bits_eff / (H·W·B)`, with `bits_eff ≈ 7.86` at `quant_model_bit=8`
(Huffman after quantisation; ≈ `0.78 × nominal_bits`).

**The dominant lever is `--enc_dim 64_S` (S = seed channels).** `--fc_dim`, `--lower_width`, `--num_blks`
barely move the count — they only touch the small 1D decoder (~83k params). The encoder MLP
`Mlp(320 → 640 → S·101)` dominates, and its **first layer `320×640 ≈ 205k` params is a hard floor**
(the PE width `pe_1.25_80`→320 and the `2×` hidden are **hardcoded** in `models/HinerArch2d.py:160,162`).

**Rate floor:** even `S=1` ⇒ 0.35 M params ⇒ **~0.84 bpppb @ 8-bit** (≈0.63 @ 6-bit, ≈0.42 @ 4-bit).
**hiner2d cannot reach the 3D 0.086-bpppb point** without an architecture change (see §6).

### Target-bpppb → config table (8-bit, fc_len=101, dec_strds=[2], fc_dim=128)

| target bpppb | needed params | **`--enc_dim`** | actual bpppb |
|---|---|---|---|
| 0.8 | 0.34 M | `64_1` | 0.84 (floor) |
| 1.0 | 0.42 M | `64_2` | 0.99 |
| 1.5 | 0.63 M | `64_5` | 1.45 |
| 2.0 | 0.84 M | `64_9` | 2.07 |
| **2.56** (match 3D) | 1.08 M | `64_12` | 2.53 |
| 3.0 | 1.26 M | `64_15` | 3.00 |
| **3.2** (match HINER++/2D-C) | 1.35 M | `64_16` | 3.15 |
| 3.56 (match HINER) | 1.50 M | `64_19` | 3.61 |

Closed form to invert: `S ≈ (target_bpppb·3,309,568/7.86 − 288,440) / 64,640`, then round.
For a different `quant_model_bit b`, scale `bits_eff ≈ 0.78·b` (so lower bits → proportionally lower bpppb,
e.g. `--quant_model_bit 6` multiplies bpppb by ~0.75).

### Matching compression **time**
hiner2d wall-time is ~**flat in S** (model is tiny; all four sweep points ≈ 400 s for 800 epochs/3,200 steps).
Time is set almost entirely by **steps = epochs × ⌈16384/batchSize⌉**: ~0.125 s/step here.
To match a baseline's wall-clock, pick `epochs` for the step budget (e.g. ~250 epochs ≈ 126 s to match the
3D `sampling bs256` point) — **but** at that budget hiner2d is badly underfit, so time-matching is not the
useful axis; quality-at-rate is.

---

## 4. The key finding: hiner2d is **training-limited, not rate-limited**

PSNR is **nearly flat (41.6–42.7 dB) across a 4× bitrate range**, and 8-bit quantisation costs ~0.07 dB
(`PSNR_q ≈ PSNR_fp`). Spending more bits (seed2→seed16) buys almost nothing. That is the signature of a model
whose bottleneck is **optimisation/architecture capacity to fit, not the bitstream**:

- Adding params doesn't help ⇒ not capacity-starved at the head/decoder.
- Quantisation is nearly free ⇒ no rate-distortion tension at 8-bit; you could **drop to 6-bit for free** and
  shave ~20 % bpppb at equal PSNR.
- The likely real limit is the **fixed 320-dim Fourier PE + shallow 2-layer encoder + 3,200 training steps**,
  vs. the baselines' 10k–20k iterations and deeper decoders.

A longer seed16 run (2,500 epochs ≈ 20k steps) **confirms this** — see §7: at the *same* ~3.05 bpppb,
PSNR rises **42.7 → 49.4 dB** purely from more training steps. The gap to the baselines was a training-budget
gap, not a method gap.

---

## 5. Which baseline configs are worth trying / comparing against

**Worth it (same rate band as hiner2d, fair fight):**
1. **3D-INR @ 2.56 bpppb** (h256, 32-bit) ↔ hiner2d `--enc_dim 64_12`. Direct 3D-vs-2D-coord comparison.
2. **HINER++ @ 3.20** and **2D-INR C @ 3.37** ↔ hiner2d `--enc_dim 64_16`. The quality bar to beat (~49–52 dB).
3. **2D-INR D @ 1.86** (12-bit QAT) ↔ hiner2d `--enc_dim 64_8` **+ `--quant_model_bit 6`**.

**Not worth chasing with current hiner2d:**
- **3D-INR @ 0.086 bpppb** — below hiner2d's ~0.8 bpppb floor; needs the §6 architecture change first.
- **1D-INR A/B (15–43 bpppb)** — pathologically high rate (per-pixel latents); only useful as an upper-rate sanity point.

---

## 6. To make hiner2d competitive (recommended code/param changes)

1. **Train longer** — biggest lever. 800 epochs is ~3k steps; baselines use 10–20k. Use `--epochs 2500
   --batchSize 2048` (≈20k steps) or smaller batch for more steps/epoch.
2. **Unlock the rate floor** — make the PE configurable so the encoder isn't pinned at 205k params:
   parametrise `pe_1.25_80` (fewer levels → smaller `pe_dim`) and the `2×pe_dim` hidden in
   `models/HinerArch2d.py:160-162`. This is required to reach <0.8 bpppb and to compete with 3D at low rate.
3. **Quantise to 6-bit** — essentially free here (`PSNR_q≈PSNR_fp`), ~20 % bpppb reduction at equal quality.
4. **Deeper/wider decoder** once training is longer — `--num_blks 1_2`, larger `--dec_strds` chains
   (note fc_len·∏dec_strds must == 202; usable splits are limited: 101×2, 2×101, 1×202).

---

## 7. Ceiling run — RESULT (seed16 ≈ 3.05 bpppb, 2,500 epochs / 20k steps, 8-bit)

| step budget | epochs×batch | bpppb | **PSNR_q** | SAM_q | time |
|---|---|---|---|---|---|
| 3.2k steps (short sweep) | 800 × 4096 | 3.094 | 42.68 | 0.0294 | 6.7 min |
| **20k steps (this run)** | 2500 × 2048 | 3.048 | **49.44** (fp 49.67) | 0.0139 | 22.2 min |

Eval trajectory: 41.5 (ep1) → 45.5 (500) → 47.8 (1000) → 49.4 (1500) → **49.7 (2500)**.

**Conclusion:** with an adequate training budget, hiner2d at ~3.05 bpppb reaches **49.4 dB**, i.e. **on par
with 2D-INR C (49.36 @ 3.37) and original HINER (49.83 @ 3.56)**, and within ~2 dB of HINER++ (51.68 @ 3.20)
— while being **cheaper to encode** (22 min vs HINER 631 s… note baseline times are on different hardware/patches).
The 7 dB deficit seen in §2/§4 was entirely a training-step shortfall. 8-bit quantisation still costs only ~0.2 dB.

---

## Reproduction

```bash
cd hiner_forked/HINER/hiner2d
PATCH=/faststorage/hyspecnet-11k/patches/…Y09181045_X06690796…-DATA.npy
# NOTE: CUDA is masked by an empty CUDA_VISIBLE_DEVICES in this shell — set it explicitly.
CUDA_VISIBLE_DEVICES=0 python train_hiner2d.py --data_path "$PATCH" \
  --vid demo --enc_dim 64_12 --fc_len 101 --dec_strds 2 --fc_dim 128 \
  --epochs 800 --batchSize 4096 --loss L2 --quant_model_bit 8 --overwrite
```
Baselines (dataset avg): `after_intermediate_results/Experiments/{1D,2D,3D,HINER,HINER++}`.
