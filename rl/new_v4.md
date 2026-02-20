# Structured Credit Assignment PPO (SC-PPO) for MultiDiscrete Actions (MetaWorld)

> **Goal**: Provide a detailed, code-oriented specification for implementing **structured advantage decomposition** and **per-dimension credit assignment** on top of PPO for **MultiDiscrete** actions.
>
> Current target environment: **MetaWorld**  
> - **State**: `obs ∈ R^39`  
> - **Action**: MultiDiscrete with **D=4** dimensions, each dimension has **K=256** discrete tokens  
> - Policy outputs a single tensor **`logits: [B, 4, 256]`**  
>
> Future target: **LIBERO** (likely `D=7`, action sizes unknown) — method is dimension-agnostic.

---

## 0. Problem Motivation

In many robotic control tasks, **success requires joint coordination across all action dimensions**:
- Even if the policy factorizes as `π(a|s)=∏_i π_i(a_i|s)`, the task reward is **joint** and often **sparse**.
- Standard PPO uses a **single scalar advantage** `Â_t` shared by all dimensions:
  \[
  \sum_{i=1}^{D} \nabla_\theta \log \pi_i(a_i|s)\; \hat A
  \]
  This causes:
  - **Credit assignment failure**: cannot tell which dimension caused success/failure
  - **Learning inefficiency**: irrelevant dimensions receive noisy gradients
  - **Slow and unstable training**, especially under sparse success signals

We want:
- Keep **joint policy** and **joint PPO ratio** (to preserve PPO’s near-trust-region stability)
- But improve credit assignment by **decomposing advantage** into **unary + interaction terms**, and optionally drive **dimension-specific gradient channels**.

---

## 1. Setup and Notation

### 1.1 MultiDiscrete action
- `D=4`, `K=256`
- action sample: `a = (a_0, a_1, a_2, a_3)` where each `a_i ∈ {0,...,255}`

### 1.2 Factorized policy (most common)
\[
\pi_\theta(a|s) = \prod_{i=1}^{D} \pi_{\theta,i}(a_i|s)
\]
Given logits tensor `logits: [B,D,K]`, each dimension corresponds to a categorical distribution.

### 1.3 Joint log-probability
Let `logp_i = log π_i(a_i|s)`. Then:
\[
\log \pi_\theta(a|s) = \sum_{i=1}^{D} \log \pi_{\theta,i}(a_i|s)
\]

---

## 2. PPO Refresher (Joint Ratio)

Given behavior policy `π_old` that generated the batch, standard PPO uses:

### 2.1 Joint ratio
\[
r(\theta) = \frac{\pi_\theta(a|s)}{\pi_{\text{old}}(a|s)}
\]
For factorized policy:
\[
r(\theta)=\exp\Big(\sum_i (\log\pi_{\theta,i}(a_i|s)-\log\pi_{\text{old},i}(a_i|s))\Big)
\]

### 2.2 PPO-Clip objective (scalar advantage)
\[
L^{\text{CLIP}}(\theta)=\mathbb{E}\left[\min(r\hat A, \operatorname{clip}(r,1-\epsilon,1+\epsilon)\hat A)\right]
\]

We keep **joint ratio** unchanged in our method.

---

## 3. Key Idea: Structured Advantage Decomposition

We replace the black-box scalar advantage estimator with a structured model:

### 3.1 Decomposition (truncate to second-order interactions)
For `D=4` we use unary + pairwise:
\[
A_\phi(s,a)=\sum_{i=1}^{D}A_\phi^{(i)}(s,a_i)+\sum_{i<j}A_\phi^{(ij)}(s,a_i,a_j)
\]

- Unary term: “single-dimension contribution”
- Pairwise term: “coordination / interaction contribution”

For `D=4`, number of pairs = C(4,2)=6:
- (0,1), (0,2), (0,3), (1,2), (1,3), (2,3)

---

## 4. Training Targets: Use Old Failure Pool + Few Successes Safely

### Why not fit `A_phi` directly to old-policy GAE?
Old failures may be far from current policy distribution → using raw GAE from old data is unstable and biased.

### Recommended approach: learn a proxy outcome/Q model offline
Use stored data (many failures + few successes) to train a stable evaluator:

#### Option A (simplest): success probability model
Train:
\[
\hat p_\psi(\text{succ}|s,a)
\]
Then define a scalar proxy value:
- `Q_hat(s,a) = logit(p_hat)` (often better scaling)
- or `Q_hat(s,a) = p_hat`

#### Option B (advanced): offline Q learning (FQE/IQL/CQL)
Train:
\[
\hat Q_\psi(s,a)
\]
using sparse success reward.

**This document assumes Option A** for MVP.

---

## 5. Constructing Advantage Targets from Proxy Q

We want a target advantage `Â_tgt(s,a)`.

Simple centered target:
\[
\hat A_{\text{tgt}}(s,a)=\hat Q(s,a)-\mathbb{E}_{\tilde a\sim \pi_{\text{old}}(\cdot|s)}[\hat Q(s,\tilde a)]
\]

Practical approximation:
- If you only have sampled `a` in batch, approximate baseline by batch mean:
\[
\hat A_{\text{tgt}} \approx \hat Q(s,a) - \text{mean}_\text{batch}(\hat Q)
\]

---

## 6. Train the Structured Advantage Model `A_phi`

### 6.1 Loss
\[
L_{\text{adv}}(\phi)=\mathbb{E}\left[\left(A_\phi(s,a)-\hat A_{\text{tgt}}(s,a)\right)^2\right]+\lambda\Omega
\]

### 6.2 Regularization `Ω` (important)
To prevent pairwise terms from overfitting:
- L2 penalty on pairwise outputs:
\[
\Omega = \mathbb{E}\left[\sum_{i<j}\left(A^{(ij)}_\phi(s,a_i,a_j)\right)^2\right]
\]
- optionally group-lasso over pairs to encourage sparsity

### 6.3 Gauge fixing / identifiability (recommended)
Decomposition is not unique; enforce mean-zero constraints to stabilize contributions:
- Unary: `E_{a_i~π_old}[A^{(i)}(s,a_i)] = 0`
- Pairwise: marginal means zero w.r.t each member:
  - `E_{a_i}[A^{(ij)}(s,a_i,a_j)] = 0`
  - `E_{a_j}[A^{(ij)}(s,a_i,a_j)] = 0`

Implementation can approximate these expectations using Top-K over `π_old`.

---

## 7. Per-Dimension Credit Assignment from Structured Terms

### 7.1 Define “terms involving dimension i”
\[
C_i(s,a)=A^{(i)}_\phi(s,a_i)+\sum_{j\neq i}A^{(ij)}_\phi(s,a_i,a_j)
\]

For D=4 with pair order:
`pairs = [(0,1),(0,2),(0,3),(1,2),(1,3),(2,3)]`:

- `C0 = u0 + p01 + p02 + p03`
- `C1 = u1 + p01 + p12 + p13`
- `C2 = u2 + p02 + p12 + p23`
- `C3 = u3 + p03 + p13 + p23`

### 7.2 Counterfactual baseline (per-dimension)
We want a baseline that does **not depend on the sampled a_i**, so we can use it as a control variate:

\[
b_i(s,a_{-i}) = \mathbb{E}_{\tilde a_i \sim \pi_{\text{old},i}(\cdot|s)} \left[ C_i\big(s,(\tilde a_i,a_{-i})\big) \right]
\]

### 7.3 Per-dimension advantage
\[
\hat A^{(i)}(s,a)=C_i(s,a)-b_i(s,a_{-i})
\]

This is the cleanest definition of:
- “How much this chosen token a_i improved relative to the average choice under π_old”
- Provides credit assignment while avoiding “action-dependent baseline” issues.

### 7.4 Top-K approximation (required because K=256)
Compute `TopK` candidates from `π_old,i` for each dimension:

- `topk_idx: [B,D,Ktop]`
- `topk_prob: [B,D,Ktop]` normalized to sum to 1 within TopK

Then:
\[
b_i \approx \sum_{k\in \text{TopK}} \pi_{\text{old},i}(k|s)\;C_i(s,(k,a_{-i}))
\]

---

## 8. Policy Update: Two Variants

We recommend implementing both and running ablations.

### Variant A (strict joint PPO, structured A only as better estimator)
Use scalar `A_gate = A_phi(s,a)` inside PPO-clip:
\[
L^{\text{CLIP}}(\theta)=\mathbb{E}\left[\min(rA_\phi, \operatorname{clip}(r)A_\phi)\right]
\]
This preserves the classic PPO structure exactly.

### Variant B (dimension-specific gradient channels; main credit assignment variant)
We still compute the joint ratio `r` and a joint clipped weight `w`, but update each dimension using its own `Â^(i)`.

#### 8.1 Compute PPO-style gate weight `w`
Use `A_gate = stopgrad(A_phi)` to choose min/max branch:

\[
w=
\begin{cases}
\min(r, r_{\text{clip}}), & A_{\text{gate}}\ge 0\\
\max(r, r_{\text{clip}}), & A_{\text{gate}}<0
\end{cases}
\]

#### 8.2 Credit policy loss
Define loss using per-dim logprob and per-dim advantages:

\[
L_{\pi}(\theta)= -\mathbb{E}\left[ \;\text{stopgrad}(w)\;\sum_{i=1}^{D}\text{stopgrad}(\hat A^{(i)}(s,a))\;\log \pi_{\theta,i}(a_i|s)\right]
\]

Notes:
- Detaching `w` removes additional gradient paths that complicate interpretation and stability.
- This loss gives each dimension its own credit signal.

---

## 9. Concrete Tensor Shapes and Computations (MetaWorld)

### 9.1 Policy logprobs
Inputs:
- `logits_new: [B,4,256]`
- `logits_old: [B,4,256]`
- `actions: [B,4]`

Compute:
- `logp_new: [B,4]`
- `logp_old: [B,4]`
- `ratio: [B]`

### 9.2 Top-K from old policy
- `prob_old = softmax(logits_old, dim=-1)`: `[B,4,256]`
- `topk_prob, topk_idx = topk(prob_old, Ktop, dim=-1)`:
  - `[B,4,Ktop]`, `[B,4,Ktop]`
- renormalize `topk_prob` across TopK so sum is 1.

### 9.3 Structured advantage model outputs
`AdvNet.forward_components(obs, actions)` returns:
- `u: [B,4]` unary terms evaluated at chosen tokens
- `p: [B,6]` pair terms evaluated at chosen tokens
- `A_phi: [B] = u.sum(-1)+p.sum(-1)`

Additionally provide:
- `AdvNet.compute_C(obs, actions)` returns `C: [B,4]`

### 9.4 Counterfactual baseline with Top-K
For each dimension `i`:
1. Build `actions_i_topk: [B,Ktop,4]` by copying `actions` and replacing dim `i` with `topk_idx[:,i,:]`.
2. Compute `C_i_topk: [B,Ktop]` by running `AdvNet` on these actions and extracting `C_i`.
3. Baseline:
   - `b_i = sum_k topk_prob[:,i,k] * C_i_topk[:,k]` → `[B]`

Stack:
- `b: [B,4]`
- `A_dim = C_actual - b` → `[B,4]`

---

## 10. Model Architecture Recommendations

### 10.1 Action token embeddings
Because each dimension has 256 tokens, use embeddings:
- `Embedding(K=256, E)` for each dimension (can share or separate)
- Typically `E=32..128`

### 10.2 State encoder
Given `obs: [B,39]`:
- `MLP(39 → H)` where `H=128..512`

### 10.3 Unary term
For each dimension `i`:
- input: concat(`[h, emb_i]`) → MLP → scalar
- output: `u_i`

### 10.4 Pairwise term
For each pair `(i,j)`:
- input: concat(`[h, emb_i, emb_j]`) → MLP → scalar
- output: `p_ij`

---

## 11. Training Schedule

We recommend three interleaved loops:

### Phase 1: Pretrain proxy outcome model `p_hat(s,a)`
- dataset: old buffer (many failures + few successes)
- loss: weighted BCE or focal loss
- objective: stable evaluation of “how close to success”

### Phase 2: Train structured advantage `A_phi`
- target: `A_tgt` from `Q_hat = logit(p_hat)`
- regularize pairwise terms
- optionally enforce mean-zero constraints via Top-K marginal penalties

### Phase 3: PPO training
- collect on-policy-ish data
- refresh proxy and adv models periodically (or EMA update)

**Practical**: start with a simple alternating schedule:
- Every N PPO updates, do M steps of proxy/adv model updates.

---

## 12. Diagnostics / Logging (Must-have)

### 12.1 Energy ratio: unary vs pairwise
- `E_unary = mean(|u|)`
- `E_pair = mean(|p|)`
- `E_pair / (E_unary + E_pair)`

### 12.2 Per-dimension credit statistics
- mean/var of `A_dim[:,i]`
- correlation between `A_dim[:,i]` and scalar `A_phi`

### 12.3 Gradient share per head (policy)
Measure gradient norms contributed to each dimension’s logits head:
- `||grad_i|| / sum ||grad||`

### 12.4 Pairwise interaction interpretability
For each pair `(i,j)`, log:
- mean `A^(ij)` in success vs failure batches
- optionally heatmap over token pairs for top tokens

---

## 13. Implementation Skeleton (Pseudo-code)

### 13.1 Policy logp and ratio

```python
logp_new = F.log_softmax(logits_new, dim=-1).gather(-1, actions[..., None]).squeeze(-1)  # [B,4]
logp_old = F.log_softmax(logits_old, dim=-1).gather(-1, actions[..., None]).squeeze(-1)  # [B,4]

logp_new_joint = logp_new.sum(-1)  # [B]
logp_old_joint = logp_old.sum(-1)  # [B]
ratio = torch.exp(logp_new_joint - logp_old_joint)  # [B]
ratio_clip = ratio.clamp(1 - eps, 1 + eps)  # [B]
```

### 13.2 Top-K from π_old

```python
prob_old = F.softmax(logits_old, dim=-1)  # [B,4,256]
topk_prob, topk_idx = prob_old.topk(Ktop, dim=-1)  # [B,4,Ktop], [B,4,Ktop]
topk_prob = topk_prob / (topk_prob.sum(-1, keepdim=True) + 1e-8)
```

### 13.3 Structured advantage components

```python
u, p, A_phi = adv_net.forward_components(obs, actions)   # u:[B,4], p:[B,6], A_phi:[B]
C_actual = adv_net.compute_C_from_components(u, p)       # [B,4]
```

### 13.4 Counterfactual baseline (Top-K)

For each i:

```python
# actions_i_topk: [B,Ktop,4]
actions_i_topk = actions[:, None, :].repeat(1, Ktop, 1)
actions_i_topk[:, :, i] = topk_idx[:, i, :]  # replace dim i

# flatten for batch compute
obs_rep = obs[:, None, :].repeat(1, Ktop, 1).reshape(B * Ktop, 39)
act_flat = actions_i_topk.reshape(B * Ktop, 4)

u_k, p_k, _ = adv_net.forward_components(obs_rep, act_flat)  # u_k:[B*Ktop,4], p_k:[B*Ktop,6]
C_k = adv_net.compute_C_from_components(u_k, p_k)            # [B*Ktop,4]
C_i_k = C_k[:, i].reshape(B, Ktop)                           # [B,Ktop]

b_i = (topk_prob[:, i, :] * C_i_k).sum(-1)                   # [B]
```

Stack `b_i` to get `b: [B,4]`, then:

```python
A_dim = C_actual - b  # [B,4]
```

### 13.5 Variant B credit loss (dimension-specific channels)

```python
A_gate = A_phi.detach()  # [B]
w = torch.where(
    A_gate >= 0,
    torch.min(ratio, ratio_clip),
    torch.max(ratio, ratio_clip),
)  # [B]

loss_pi_credit = -(w.detach() * (A_dim.detach() * logp_new).sum(-1)).mean()
```

## 14. Notes on Stability & Practical Tips

Start with `Ktop = 8`, then try 4/16.

Pairwise regularization `lam_pair` is crucial; start around `1e-3` to `1e-2` and tune.

Use `detach()` for:

- `A_gate` in gate decision
- `w` in credit loss (recommended)
- `A_dim` in policy loss (advantage should not backprop to `adv_net` through policy update)

Refresh proxy model periodically to reduce distribution shift issues.

For extreme success imbalance, use:

- weighted BCE (`pos_weight`)
- focal loss
- balanced sampling

## 15. Extension to LIBERO (D=7)

Changes:

- `D = 7`
- number of pairs = `C(7,2) = 21`
- same formulas, same Top-K baseline; computation grows linearly with `D * (D - 1) * Ktop`

Only needed:

- define list of pairs and mapping from pair outputs to each `C_i`
- action sizes per dimension might differ → use separate embeddings per dim

## 16. Deliverables Checklist for Code Modification

To implement this in an existing PPO codebase, the model should add:

- Outcome/Q proxy model:
  - `forward(obs, actions) -> logit_success`
  - training loop on replay (failures + successes)
- Structured Advantage Network:
  - `forward_components(obs, actions) -> (u:[B,D], p:[B,n_pairs], A:[B])`
  - `compute_C(u, p) -> C:[B,D]`
- Top-K counterfactual baseline:
  - given `logits_old` and `actions`, compute `A_dim:[B,D]`
- Policy loss variant B (optional):
  - use `w` from joint ratio + `A_dim` per dimension
- Logging:
  - unary/pair energy ratio
  - per-dim credit stats
  - per-dim gradient share
  - pairwise stats for interpretation

## 17. Suggested Minimal Implementation Order (MVP)

- Implement AdvNet unary+pair and compute scalar `A_phi` only (no per-dim credit yet).
- Implement `compute_C` and Top-K baseline → get `A_dim`.
- Plug in Variant B credit loss (keep ratio joint).
- Add proxy model `p_hat` and train AdvNet using `Q_hat` targets.
- Add diagnostics and ablations.

End of document.
