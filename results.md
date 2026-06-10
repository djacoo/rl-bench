# HalfCheetah Results — SAC vs MBPO vs MACURA

**Authors:** Jacopo Parretti · Cesare Fraccaroli  
**Affiliation:** University of Verona — MSc in Artificial Intelligence  
**Companion:** see [`presentation.md`](presentation.md) for slide content.

---

## Executive summary

Single-seed benchmark on **HalfCheetah-v5**, **300k** environment steps, seed **0**.

| Rank | Algorithm | Final return @ 300k | vs SAC | vs MBPO |
|:----:|-----------|:-------------------:|:------:|:-------:|
| 1 | **MACURA** | **11,888** | **+7,195 (+153%)** | **+1,685 (+17%)** |
| 2 | **MBPO**   | **10,203** | **+5,510 (+117%)** | — |
| 3 | **SAC**    | **4,693**  | — | **−5,510** |

Both model-based methods strongly outperform model-free SAC. **MACURA beats MBPO** on final return, consistent with the paper's claim that uncertainty-gated rollouts help over a fixed horizon.

![Learning curves — seed 0, 300k steps](rl-bench/gifs%20and%20imgs%20for%20presentation/halfcheetah_seed0_curves_dark.png)

---

## Final results @ 300k

| Algorithm | Return | Eval source | Rollout / updates |
|-----------|-------:|---------------|-------------------|
| **MACURA** | **11,887.5** | `runs/macura_halfcheetah_seed0/results.csv` (epoch 299) | uncertainty-gated, T_max = 10, ξ = 2.0 |
| **MBPO**   | **10,203.0** | `runs/mbpo_halfcheetah_seed0/results.csv` (epoch 299) | fixed horizon = 1, G = 8 |
| **SAC**    | **4,693.0**  | `runs/sac_halfcheetah_seed0/eval.csv` (step 300k) | 1 update / env step, no model |

**Best checkpoint during training (not final step):**

| Algorithm | Peak return | At step |
|-----------|------------:|--------:|
| MACURA | 12,323 | ~288k |
| MBPO   | 10,618 | ~287k |
| SAC    | 4,852  | ~280k |

---

## Learning curves — selected checkpoints

SAC is evaluated every **10k** steps (10 episodes). MBPO and MACURA log **epoch reward every 1k** env steps (upstream `results.csv`).

| Step | SAC | MBPO | MACURA | Leader |
|-----:|----:|-----:|-------:|:------:|
| 50k  | 4,092 | 4,579 | 5,395 | MACURA |
| 100k | 4,623 | 7,151 | 9,234 | MACURA |
| 150k | 4,492 | 8,790 | 10,532 | MACURA |
| 200k | 4,744 | 9,683 | 11,098 | MACURA |
| 250k | 4,762 | 10,299 | 11,835 | MACURA |
| **300k** | **4,693** | **10,203** | **11,888** | **MACURA** |

**Trend:** model-based methods pull away after ~50k. SAC plateaus near **~4.7k** from ~100k onward. MACURA leads MBPO throughout the second half of training.

---

## Setup (shared)

| Parameter | Value |
|-----------|-------|
| Environment | `HalfCheetah-v5` |
| Seed | 0 |
| Env steps | 300,000 |
| Obs normalization | on |
| Exploration | pink noise, scale = 0.05 |
| Warmup | 5,000 random steps |
| Actor / critic width | 256 × 2 |
| Ensemble (MBPO/MACURA) | 7 PNNs, 4 × 200 hidden |

### Algorithm-specific

| Parameter | SAC | MBPO | MACURA |
|-----------|:---:|:----:|:------:|
| Real data in SAC batch | 100% | 10% | 10% |
| SAC updates / env step | 1 | 8 | ≤ 8 (adaptive G) |
| Rollout horizon | — | **1** (fixed) | **≤ 10** (GJS-gated) |
| MACURA ξ / ζ | — | — | 2.0 / 0.95 |

Configs: `rl-bench/configs/{sac,mbpo,macura}_halfcheetah.yaml`

---

## Comparison with paper claims

| Claim | Paper | Our run (seed 0) | Match? |
|-------|-------|------------------|:------:|
| MACURA > MBPO | Yes (Figure 4) | +1,685 @ 300k | ✅ |
| Model-based >> SAC on HalfCheetah | Yes | ~2.2–2.5× | ✅ |
| MACURA ≈ SAC asymptotically | Often comparable | SAC far below | ⚠️ (our SAC config / budget) |

**Caveat:** these are **laptop-scale configs** (256-wide nets, G=8, 300k steps, single seed) — not a paper-faithful reproduction of Figure 4. Rankings among our three runs are internally fair (same env, steps, exploration).

---

## Qualitative results (policy GIFs)

Final policies @ 300k — folder `rl-bench/gifs and imgs for presentation/`:

| Algorithm | GIF |
|-----------|-----|
| MACURA | `macura.gif` |
| MBPO   | `mbpo.gif` |
| SAC    | `sac.gif` |

| MACURA | MBPO | SAC |
|:------:|:----:|:---:|
| ![MACURA](rl-bench/gifs%20and%20imgs%20for%20presentation/macura.gif) | ![MBPO](rl-bench/gifs%20and%20imgs%20for%20presentation/mbpo.gif) | ![SAC](rl-bench/gifs%20and%20imgs%20for%20presentation/sac.gif) |

---

## Artifacts

| Run | Directory | Key files |
|-----|-----------|-----------|
| SAC | `rl-bench/runs/sac_halfcheetah_seed0/` | `eval.csv`, `sac.pth`, `videos/step_0300000.mp4` |
| MBPO | `rl-bench/runs/mbpo_halfcheetah_seed0/` | `results.csv`, `sac.pth`, `videos/step_0300000.mp4` |
| MACURA | `rl-bench/runs/macura_halfcheetah_seed0/` | `results.csv`, `sac.pth`, `videos/step_0300000.mp4` |

**Plot:** `rl-bench/gifs and imgs for presentation/halfcheetah_seed0_curves_dark.png`

---

## Notes

- **Eval frequency differs:** SAC `eval.csv` every 10k; MBPO/MACURA `results.csv` every 1k epoch — the plot smooths MBPO/MACURA accordingly.
- **SAC dip @ ~80k:** return drops to ~3.4k (min −502 on one eval episode) then recovers — visible on the curve.
- **Single seed:** paper reports 5-seed mean ± std; these numbers are one run only.
