# Presentation: Trust the Model Where It Trusts Itself (MACURA)

> Input spec for slide generation. One `##` block = one slide.
> Keep slides visual and sparse: short bullets, one formula/figure/code block max per slide.
> Code blocks are real excerpts from our repo and must be shown verbatim as code.
> **Media folder:** `rl-bench/gifs and imgs for presentation/` — use these paths for all figures and GIFs.

---

## Slide 1 — Title

**Trust the Model Where It Trusts Itself**
Reproducing & benchmarking **MACURA** (ICML 2024) vs **MBPO** and **SAC** on MuJoCo.

- Subtitle: *Uncertainty-aware rollout length adaption in Dyna-style model-based RL*
- Authors: **Jacopo Parretti · Cesare Fraccaroli**
- Affiliation: **University of Verona — MSc in Artificial Intelligence**
- Visual: clean title slide; optional small HalfCheetah still from `rl-bench/gifs and imgs for presentation/macura.gif`.

---

## Slide 2 — When vs. Where to trust the model

**Dyna-style MBRL in two lines:** an agent collects real data, trains a *dynamics model*, then uses the model to generate cheap synthetic rollouts that train the policy. Real data is expensive, model data is free — but only useful if the model is accurate.

**The core question — how far can we roll out the model before its errors corrupt the policy?**

- **MBPO (Janner 2019) → "*When* to trust?"** — time-based: increase rollout length on a fixed schedule.
- **MACURA (this paper) → "*Where* to trust?"** — spatial: roll out only while the model is *locally* certain.

> The whole project is about this **rollout mechanism**: fixed horizon (MBPO) vs. uncertainty-gated horizon (MACURA).

---

## Slide 3 — MBPO (Janner et al., 2019)

State-of-the-art Dyna-style MBRL: **Probabilistic Ensemble dynamics model + SAC agent**.

Reproduce the **Figure 1** loop (blocks + arrows):

- **Environment M** ⟶ agent interaction data ⟶ **D_env**
- **D_env** ⟶ supervised learning ⟶ **model p̃**
- **model p̃** + policy π ⟶ branched rollouts ⟶ **D_mod**
- **D_mod** ⟶ reinforcement learning ⟶ **policy π** (loops back to env)

> Two coupled learning problems: data trains the *model*; model rollouts train the *agent*.

---

## Slide 4 — What we built

- We implemented the **MuJoCo benchmark**, focusing on **HalfCheetah-v5**.
- We wrote a **thin wrapper (`rl-bench`)** around the original MACURA repository.
- The upstream repo already contains the three pieces we compare: **MBPO**, **MACURA**, and the shared **SAC** core.
- Our wrapper adds: flat YAML configs, Gymnasium env wrappers, logging, evaluation, video/plots — so all three algorithms share one harness.

> We **do not** reimplement the ensembles/SAC; we orchestrate, configure, and benchmark them fairly.

---

## Slide 5 — SAC is the model-free core

All three algorithms share the **same model-free engine: Soft Actor-Critic (SAC)**.

- **SAC** = maximum-entropy off-policy actor-critic (twin Q-functions, learned temperature α).
- **MBPO / MACURA** = a **model-based structure wrapped around SAC**: the ensemble model and the buffers **D_env** (real) and **D_mod** (imagined) feed SAC extra training data.
- Difference between the three is *what data SAC trains on*, not the SAC update itself.

> Visual: SAC box in the center, MBPO/MACURA machinery as a ring around it (D_env → model → D_mod → SAC).

---

## Slide 6 — SAC in brief: pseudocode vs. our code

**Pseudocode (course material):**

```text
init policy π_φ, twin critics Q_θ1, Q_θ2 (+ targets), temperature α
for each step:
    a ~ π_φ(·|s);  s', r ← env.step(a);  store (s,a,r,s') in D
    sample batch from D
    a' ~ π_φ(·|s')
    y = r + γ ( min_i Q_θ'_i(s',a')  −  α log π_φ(a'|s') )      # entropy-regularized target
    update critics:  minimize (Q_θi(s,a) − y)²
    update actor:    maximize  min_i Q_θi(s, ã) − α log π_φ(ã|s)
    update α toward target entropy;  soft-update target critics
```

**Our training loop (`src/rl_bench/train_sac.py`):**

```python
a = policy.act(obs, deterministic=det)
if ex_cfg["kind"] in ("white", "pink"):          # optional exploration noise
    a = np.clip(a + noise(t), -1.0, 1.0)
next_obs, r, term, trunc, _ = env.step(a)
memory.add(obs, a, next_obs, r, term, trunc)
...
for _ in range(upd_per_step):                     # SAC gradient steps
    agent.sac_agent.update_parameters(memory, batch, updates, reverse_mask=True)
```

> The SAC actor/critic + `update_parameters` come from upstream; we provide the loop, replay buffer, eval and logging.

---

## Slide 7 — Probabilistic Ensemble (PE)

- **E probabilistic neural networks** (PNNs), here **E = 7**.
- Each PNN *e* predicts a **Gaussian** over the next state: p̃_θe(s′|s,a) = N(μ_θe, Σ_θe).
- Each has its **own parameters θ_e**, trained on a **bootstrapped subset of D_env** with a **negative log-likelihood (NLL)** loss.
- **Key idea:** the **disagreement** between ensemble members measures *epistemic* uncertainty → tells us **where** to truncate rollouts.

> Visual: E little Gaussians; tight overlap = certain (trust), spread out = uncertain (stop).

---

## Slide 8 — How MBPO rolls out (fixed horizon)

**Algorithm 1 — "Vanilla" branched model-based rollouts:**

```text
s0 ~ U(D_env)                       # start from a real state
for t = 0 ... T_max − 1:            # FIXED horizon
    e_t ~ U(1..E)                   # pick a random ensemble member
    a_t ~ π(·|s_t)                  # action from current policy
    s_{t+1} ~ p̃_θe_t(·|s_t, a_t)    # next state from that PNN
    r_{t+1} = r(s_t, a_t)
    D_mod ← D_mod ∪ {(s_t, a_t, r_{t+1}, s_{t+1})}
```

- Many short rollouts (**M ≈ hundreds**) branched in parallel from real states.
- Horizon **T_max is preset / scheduled by training time** — same length everywhere, regardless of local model error.

---

## Slide 9 — MBPO rollout in the original code

Main loop of `rollout_model_and_populate_sac_buffer` (upstream `mbrl/algorithms/mbpo.py`):

```python
for i in range(rollout_horizon):                 # fixed horizon, no early stop
    action = agent.act(obs, sample=sac_samples_action, batched=True)
    pred_next_obs, pred_rewards, pred_terminated, model_state = \
        model_env.step(action, model_state, sample=True)
    sac_buffer.add_batch(
        obs[~accum_terminated], action[~accum_terminated],
        pred_next_obs[~accum_terminated], pred_rewards[~accum_terminated, 0],
        pred_terminated[~accum_terminated, 0], pred_truncated[~accum_terminated],
    )
    obs = pred_next_obs
    accum_terminated |= pred_terminated.squeeze()  # only env termination ends a rollout
```

> Note: a rollout only stops on **predicted termination** — never on model uncertainty.

---

## Slide 10 — The purpose of MACURA

**Define a region E ⊆ S of low model uncertainty** = where the model dynamics are accurate enough to trust.

- **While the rollout stays in E** → keep going (long rollouts are beneficial).
- **As soon as it leaves E** → **stop and discard** (risk of model exploitation outweighs the benefit).

So rollout length becomes **adaptive and local**: long where the model is confident, short (even < 1 step) where it isn't.

> Visual: yellow blob E around the data; trajectories continue inside, get cut at the boundary.

---

## Slide 11 — The metric: GJS divergence + threshold κ

**Uncertainty** = pairwise **Geometric Jensen-Shannon (GJS)** divergence between the ensemble's Gaussians (closed-form, cheap, computed every transition):

$$u_{\text{GJS}}(s,a) = \frac{2}{E(E-1)} \sum_{e=1}^{E}\sum_{f<e} D_{\text{GJS}}\big(\mathcal{N}_e \,\|\, \mathcal{N}_f\big)$$

**Region E:**  E = { s : u_GJS(s,a) < κ }

**Self-tuning threshold κ** (from the *first* rollout step, averaged over rounds):

$$\kappa = \frac{\xi}{K}\sum_{k=1}^{K}\hat{u}_{\text{GJS},k}, \qquad \hat{u}_{\text{GJS},k} = \zeta\text{-quantile of first-step uncertainties}$$

**Algorithm 2:** roll out step-by-step; at each step keep the transition if `u_GJS < κ`, else **break** that rollout.

> Hyperparameters: **T_max = 10**, **ζ = 95%** fixed for all tasks; only **ξ** is tuned.
> Visual: **`rl-bench/gifs and imgs for presentation/gjs gif.gif`** — distribution overlap + Jensen–Shannon distance over time (intuition for the GJS metric).

![GJS intuition — distribution drift & Jensen–Shannon distance](rl-bench/gifs%20and%20imgs%20for%20presentation/gjs%20gif.gif)

---

## Slide 12 — Our code: GJS metric + adaptive threshold

**GJS between two Gaussians** (upstream `mbrl/util/distance_measures.py`, condensed):

```python
def calc_uncertainty_score_genShen(mu1, var1, mu2, var2):   # GJS, α = 0.5
    al = 0.5
    sigAL = 1 / ((1 - al) / var1 + al / var2)
    muAl  = sigAL * ((1 - al) / var1 * mu1 + al / var2 * mu2)
    t1 = ((1 - al) * mu1 / var1 * mu1).sum(1)
    t2 = (al * mu2 / var2 * mu2).sum(1)
    t3 = (muAl / sigAL * muAl).sum(1)
    log_term = ((1 - al) * log(var1).sum(1) + al * log(var2).sum(1) - log(sigAL).sum(1))
    return 0.5 * (t1 + t2 - t3 + log_term)
# averaged over all ensemble pairs → u_GJS  (calc_pairwise_symmetric_...)
```

**Threshold + gated rollout** (upstream `mbrl/algorithms/macura.py`, condensed — this is Algorithm 2):

```python
uncertainty_score = pairwise_GJS(means, vars, E)         # u_GJS per parallel rollout
if i == 0:                                               # first rollout step only
    zeta_percentile = np.percentile(uncertainty_score, zeta)  # ζ = 95% quantile
    border_for_this_rollout = zeta_percentile * xi            # × ξ
    threshold = running_average(border_for_this_rollout)      # κ (avg over rounds)
certain = uncertainty_score < threshold                  # inside E ?
if certain.sum() == 0:
    break                                                # all rollouts left E → stop
sac_buffer.add_batch(... obs[certain] ...)               # keep only certain transitions
```

---

## Slide 13 — Exploration / exploitation: the ξ knob

- **ξ scales the threshold κ** → it sets *how big* the trusted region E is.
  - **ξ too large** → E too permissive → **model exploitation** (uses bad data).
  - **ξ too small** → E too strict → **narrow D_mod**, critics overfit → instability.
  - Sweet spot is broad; ξ is the **single hyperparameter** that needs tuning.
- **Exploration via pink noise** (Eberhard et al., 2023): temporally-correlated noise (between white noise and Brownian motion) added to actions in the *real* environment → richer D_env → **expands E**.
- **Fixed for all tasks:** T_max = 10, ζ = 95%. We used **ξ = 2.0**, pink-noise scale **0.05** on HalfCheetah.

> Visual: the ξ-sweep curve (too low / good / too high), plus a tiny pink-noise vs white-noise trace.

---

## Slide 14 — Our results on HalfCheetah

**Setup:** HalfCheetah-v5, seed 0, **300k** env steps, pink noise, obs-norm on. (Laptop-scale, **not** paper-faithful configs.)

**Ranking @ 300k:  MACURA (11,888) > MBPO (10,203) > SAC (4,693)**

| Algorithm | Final return @ 300k | Rollouts |
|-----------|:------------------:|:--------:|
| **MACURA** | **11,888** | uncertainty-gated (≤ T_max = 10) |
| **MBPO**   | **10,203** | fixed horizon = 1 |
| **SAC**    | **4,693**  | model-free, no model |

**Takeaways (match the paper's story):**

- **Both model-based methods crush model-free SAC** (~2.2–2.5×) — the model massively boosts data efficiency at 300k.
- **MACURA > MBPO by ~+1,700** — the *where-to-trust* uncertainty gate beats the fixed 1-step horizon.
- **SAC plateaus ~4.7k**, while MACURA/MBPO are still climbing — cost is speed (model-based is ~10× slower per step).

**Figures & GIFs (presentation folder):**

| Asset | Path | Use on slide |
|-------|------|--------------|
| Learning curves (dark) | `rl-bench/gifs and imgs for presentation/halfcheetah_seed0_curves_dark.png` | main plot |
| MACURA policy @ 300k | `rl-bench/gifs and imgs for presentation/macura.gif` | side-by-side or strip |
| MBPO policy @ 300k | `rl-bench/gifs and imgs for presentation/mbpo.gif` | side-by-side or strip |
| SAC policy @ 300k | `rl-bench/gifs and imgs for presentation/sac.gif` | side-by-side or strip |

> Layout idea: **curve on top**, three **GIFs in a row** below (MACURA · MBPO · SAC) — shows both *how well* and *how it moves*.

![HalfCheetah learning curves — seed 0, 300k steps](rl-bench/gifs%20and%20imgs%20for%20presentation/halfcheetah_seed0_curves_dark.png)

| MACURA | MBPO | SAC |
|:------:|:----:|:---:|
| ![MACURA policy](rl-bench/gifs%20and%20imgs%20for%20presentation/macura.gif) | ![MBPO policy](rl-bench/gifs%20and%20imgs%20for%20presentation/mbpo.gif) | ![SAC policy](rl-bench/gifs%20and%20imgs%20for%20presentation/sac.gif) |

---

## Assets (presentation folder)

All media for the design tool lives in **`rl-bench/gifs and imgs for presentation/`**:

| File | Type | Slide(s) |
|------|------|----------|
| `halfcheetah_seed0_curves_dark.png` | PNG | **14** — learning curves (MACURA 11,888 · MBPO 10,203 · SAC 4,693) |
| `macura.gif` | GIF | **14** (and optional **1**) — final MACURA policy on HalfCheetah |
| `mbpo.gif` | GIF | **14** — final MBPO policy |
| `sac.gif` | GIF | **14** — final SAC policy |
| `gjs gif.gif` | GIF | **11** — GJS / distribution-drift intuition |

- ✅ **Final results** (seed 0, 300k): MACURA 11,888 · MBPO 10,203 · SAC 4,693.
- ✅ **Author names + course/affiliation** — Slide 1.
- Optional / not in folder: paper **Figure 1** (Dyna loop), **Figure 2** (region E), **ξ-sweep** plot — still described in text on slides 3, 10, 13.
