# LoopSlate — Experiment Plan v1

**Project:** LoopSlate (*Persistent Working Memory for Looped Transformers: From Recurrent Computation to Reusable Computation*)  
**Version:** v1.0  
**Date:** 2026-11-25  
**Scope:** End-to-end experimental protocol covering Phase 0 (Sanity & Causality), Phase 1 (Core PWM Validation vs Vanilla Loop & LoopRes), Phase 2 (Selectivity Ablations & Causal Mechanistic Probes), and Phase 3 (Integration with Variable-Width Recurrent Cells: `LoopSlate-X`).

---

## 1. Core Research Question & Falsifiable Hypotheses

**Core Research Question:**  
*Can a looped Transformer improve test-time compute efficiency by selectively preserving and reusing intermediate computational results in a bounded, overwritable working memory?*

| ID | Hypothesis | Primary Evidence Required | Falsification / Kill Criterion |
|---|---|---|---|
| **H1 (Value of Bounded PWM)** | Under matched executed FLOPs, `LoopSlate (SelectivePWM)` outperforms `Loop-Vanilla (Iso-FLOP)` on multi-step compositional / state-reuse tasks and improves compute-to-accuracy scaling. | Phase 1 accuracy & Iso-FLOP Pareto frontier across synthetic DAG/multi-hop benchmarks and LM validation loss. | `LoopSlate` fails to beat `Loop-Vanilla (Iso-FLOP)` by a statistically significant margin across $\ge 3$ seeds. |
| **H2 (Advantage over LoopRes)** | Bounded, overwritable $O(K)$ working memory matches or outperforms unbounded append-only `LoopRes`, and decisively beats `LoopRes` under **loop extrapolation ($R_{\text{eval}} > R_{\text{train}}$)** and **hypothesis revision / overwrite tasks**. | Phase 1–2 extrapolation curves ($R_{\text{train}}=4 \to R_{\text{eval}} \in \{4,6,8,12\}$), peak memory scaling, and overwrite-heavy benchmarks. | `LoopRes` strictly dominates `LoopSlate` at both in-distribution $R$ and extrapolated $R$ despite unbounded state growth. |
| **H3 (Selectivity & True Reuse)** | Gains come from **selective writing, slot competition, and gated overwriting** (not merely extra residual parameters), and slots causally store reusable sub-goals. | `SelectivePWM > FIFO / EMA / Read-Only / Uncompetitive Write` AND positive **Causal Slot Surgery** + **Sub-goal Linear Probe separation** ($M_r$ retains earlier sub-goals while $h_r$ advances). | `FIFO`/`EMA` matches `SelectivePWM`, OR Causal Slot Surgery fails to steer downstream predictions above random slot noise. |
| **H4 (Complementarity with Variable-Width)** | Pairing `SelectivePWM` with a Variable-Width (`W→N→W`) recurrent cell (`LoopSlate-X`) yields better Iso-FLOP & KV-cache efficiency than either mechanism alone and beats `SMELT-Unif`. | Phase 3 2×2 factorial (`{Uniform, VW} × {NullMemory, SelectivePWM}`) + `SMELT-Unif` / `WideIO` / `Diamond` controls. | `LoopSlate-X` degrades below `LoopSlate (Uniform)` at matched FLOPs, or `VW` and `PWM` exhibit destructive interference. |

---

## 2. Exact Mathematical Specification of Model Variants

All autoregressive models operate at **Per-Token Slot Granularity** $M_r \in \mathbb{R}^{B \times N \times K \times d_m}$ to guarantee **zero causal leakage** across token positions. Cross-token communication happens inside the recurrent Transformer cell $F_\theta$; cross-loop intermediate storage happens inside each token's $K$ persistent slots.

### 2.1 Shared Backbone Template (`LoopedTransformerModel`)
Given token embeddings $h_0 \in \mathbb{R}^{B \times N \times D}$ and optional unshared `Prelude` / `Coda` blocks:
1. Initialize memory state $M_0 = \text{memory.init\_state}(B, N)$.
2. For loop $r = 0, \dots, R-1$:
   - **Read:** $c_r = \text{memory.read}(h_r, M_r, r)$; inject via zero-initialized gate $\tilde{h}_r = h_r + g_{\text{read}, r} \odot c_r$.
   - **Compute:** $u_r = F_\theta(\tilde{h}_r; e_r)$ using a `BaseRecurrentCell` with lightweight `AdaRMSNorm` loop conditioning.
   - **Write & Update:** $M_{r+1} = \text{memory.write\_and\_update}(u_r, M_r, r)$.
   - **Advance:** $h_{r+1} = u_r$ (with optional input injection ablation $h_{r+1} = u_r + \lambda_r h_0$).
3. Apply `Coda` (if enabled), final RMSNorm, and LM Head.

### 2.2 Polymorphic `BaseWorkingMemory` Implementations

| Class Name | State Complexity | Mechanism | Role in Plan |
|---|---|---|---|
| `NullMemory` | $O(1)$ | Identity read ($c_r = 0$), no-op write. | Vanilla Looped Transformer baseline (width-matched for Iso-FLOP). |
| `LoopResMemory` | $O(R \cdot D)$ | Appends $u_r$ to history buffer $[h_0, u_0, \dots, u_{r-1}]$; reads via cross-attention over all past loop states. | **Primary external competitor baseline.** |
| `FIFOMemory` | $O(K \cdot d_m)$ | Fixed $K$-slot ring buffer; writes $\operatorname{Proj}_w(u_r)$ deterministically to slot $r \bmod K$ (no selectivity). | Phase 2 control (tests bounded storage without learned selection). |
| `EMAMemory` | $O(K \cdot d_m)$ | Fixed $K$-slot buffer updated with fixed or scalar exponential decay without slot competition. | Phase 2 control (tests smooth averaging vs discrete slot allocation). |
| `SelectivePWM` | $O(K \cdot d_m)$ | Learned slot prototypes $E_{\text{slot}} \in \mathbb{R}^{K \times d_m}$; content-based attention **Read**; competitive softmax slot addressing $a_r = \operatorname{Softmax}(q_w(u_r) M_r^\top / \sqrt{d_m})$ + scalar/vector write gate $\gamma_r = \sigma(w_g^\top [u_r, \bar{m}_r] + b_g)$; gated overwrite $M_{r+1} = (1 - w_r) \odot M_r + w_r \odot \operatorname{Proj}_w(u_r)$ where $w_r = \gamma_r \cdot a_r$. | **Proposed primary method (LoopSlate).** |

### 2.3 Anti-Collapse Safeguards for `SelectivePWM`
- **Zero-Init Read Gate** ($g_{\text{read}, r} = 0$ at step 0) for stable warm-start from vanilla recurrent dynamics.
- **Distinct Learnable Initial Slot Prototypes** $E_{\text{slot}} \sim \mathcal{N}(0, 0.02)$ to break slot permutation symmetry.
- **Optional Auxiliary Load-Balance / Orthogonality Regularizer** (ablated in Phase 2): small penalty $\lambda_{\text{bal}}$ only if slot utilization collapses (monitored via write entropy & effective active slots).

---

## 3. Fairness, Causality & Compute Accounting Rules

1. **Zero Causal Leakage Gate (Mandatory Pre-Flight):**
   Every model & memory class must pass `tests/test_causality.py`: perturbing input token $x_{t+1}$ must yield exact `0.0` difference on logits and memory tensors at all positions $\le t$.
2. **Strict Iso-FLOP Alignment ($\le 1\%$ tolerance):**
   Because `SelectivePWM` (with $K=4, d_m = D/2$) adds ~3–5% forward FLOPs over a bare loop, `Loop-Vanilla (Iso-FLOP)` receives a matched FLOP budget (either slightly wider FFN hidden dim $d_{\text{ff}}$ or matched width via `ModelProfiler`) so PWM never wins merely by spending extra compute.
3. **Dual Accounting Report:**
   Every table reports:
   - Body stored parameters & Total stored parameters (noting tied embeddings);
   - Executed FLOPs/token (decomposed into Attention, MLP, Memory Read/Write, IO);
   - Peak KV-Cache bytes/token + Working Memory state bytes/token;
   - Wall-clock throughput (tokens/sec) and decode latency (ms/token).
4. **WideIO Confound Control (Phase 3):**
   Whenever `VariableWidthTransformerCell` ($D_{\max} > D_{\text{rms}}$) is compared to uniform baselines ($D_{\text{rms}}$), include `Loop-Unif + WideIO` (identical $D_{\max}$ Embedding/LM Head + linear bridge $D_{\max} \leftrightarrow D_{\text{rms}}$) to eliminate embedding capacity confounds.

---

## 4. Three-Tier Benchmark & Diagnostic Suite

### Tier 1 — Synthetic Intermediate-Reuse Suite (Phase 1–2 Primary Signal)
Designed specifically so **preserving and reusing computed sub-results** provides a direct algorithmic advantage over repeatedly recomputing or overwriting a single residual stream:

1. **DAG Expression Composition (`DAG-Reuse`) — Flagship Synthetic Task**
   - *Structure:* Sequences define nodes and operations over a finite group/modular ring, where intermediate nodes have **fan-out $\ge 2$** (e.g., $x_1 = f(a, b)$, $x_2 = f(x_1, c)$, $x_3 = f(x_2, d)$, $y = g(x_1, x_3)$).
   - *Why it discriminates:* Computing $x_2 \to x_3$ requires mutating the active workspace while preserving $x_1$ across multiple loops until the merge step $g(x_1, x_3)$. Provides exact ground-truth intermediate labels (`x1`, `x2`, `x3`) for **Sub-goal Linear Probes** and **Causal Slot Surgery**.
2. **Multi-Hop Graph / Pointer Chasing (`MultiHop-Chain`)**
   - *Structure:* Given randomized edge list and query $(s, k)$, compute $k$-hop target $v_0 \to v_1 \to \dots \to v_k$.
   - *Extrapolation test:* Train on hops $k \in [2, 4]$ with $R_{\text{train}} = 4$; evaluate on $k \in [5, 8]$ with $R_{\text{eval}} \in \{6, 8, 10\}$.
3. **Overwrite-Heavy State Tracking (`StateTrack-Revise`)**
   - *Structure:* Variable register machine with $M$ registers undergoing assignment, conditional swap, and **overwrite / cancellation** updates.
   - *Why it discriminates against `LoopRes`:* Old values become stale and act as distractors in append-only history buffers, testing `SelectivePWM`'s explicit **Update/Overwrite** capability.
4. **Multi-Query Associative Recall (`MQAR`)**
   - *Purpose:* Sanity check that adding working memory (and later narrow bottlenecks in Phase 3) does not hurt crisp in-context key-value recall.

### Tier 2 — Autoregressive Language Modeling (Phase 1 Validation & Phase 3 Scaling)
- **Dataset:** FineWeb-Edu (tokenized with standard BPE, context length $N = 1024$, scaling to $2048$ in Scale-Up).
- **Budgets:**
  - *Pilot LM:* 1B–2B tokens (~40M–80M active/stored scale) for fast Phase 1–2 validation.
  - *Main LM:* 5B–10B tokens (~150M equivalent executed scale) for Phase 3 & paper main tables.
- **Metrics:** Validation Loss / Bits-per-Byte (BPB), Iso-FLOP loss curve, Early-Exit / Per-Loop Logit-Lens convergence curve.

### Tier 3 — Downstream Reasoning & Depth Extrapolation
- **Zero/Few-shot Eval:** LAMBADA, HellaSwag, PIQA, ARC-Easy/Challenge, WinoGrande (plus GSM8K / ProntoQA-style reasoning at Main scale).
- **Test-Time Compute Scaling:** Evaluate checkpoints trained at $R_{\text{train}} = 4$ across $R_{\text{eval}} \in \{2, 3, 4, 6, 8, 12\}$ to plot accuracy vs inference FLOPs.

---

## 5. Phased Experiment Matrix & Stage Gates

### Phase 0 — Engineering Verification & Sanity Suite (Target: Days 1–2)
- **Goal:** Verify OOP architecture, exact FLOP/param accounting, zero causal leakage, gradient health, and non-collapsed slot utilization on tiny synthetic smoke runs.
- **Checks (All Must Pass Before Phase 1):**
  - `pytest` green on all `BaseWorkingMemory` & `BaseRecurrentCell` classes.
  - Causality diff `== 0.0` for all autoregressive variants.
  - Iso-FLOP delta `< 1.0%` verified by `ModelProfiler`.
  - Overfit single batch of `DAG-Reuse` cleanly with non-zero read gate activation and write entropy $> 0.3 \ln K$.

---

### Phase 1 — Core Proof of Value: Vanilla Loop vs LoopRes vs LoopSlate (Target: Days 3–7)
Fixed uniform width throughout to isolate the working memory mechanism cleanly.

#### Model Scale: `Tier-Pilot`
- Cell: `UniformTransformerCell` with $L_{\text{cell}} = 4$ layers, $D = 384$ ($n_h = 6, d_{\text{head}} = 64$), $R_{\text{train}} = 4$ ($L_{\text{exec}} = 16$).
- Memory defaults: $K = 4$ slots, $d_m = 192$ ($D/2$).
- Seeds: 3 random seeds per run on Tier 1 synthetic tasks; 1–2 seeds on Pilot LM (1B tokens).

| Exp ID | Model Variant | Memory Module | Alignment | Purpose |
|---|---|---|---|---|
| `P1-01` | `Std-Unshared` | `NullMemory` ($L=16, R=1$) | Iso-FLOP | Non-looped upper/reference baseline |
| `P1-02` | `Loop-Vanilla` | `NullMemory` ($L=4, R=4$) | Same $D$ | Unadjusted vanilla loop baseline |
| `P1-03` | `Loop-Vanilla-IsoFLOP` | `NullMemory` ($L=4, R=4$, scaled $d_{\text{ff}}$) | **Iso-FLOP ($\le 1\%$)** | **Strict compute-fair vanilla baseline** |
| `P1-04` | `LoopRes` | `LoopResMemory` ($L=4, R=4$) | Iso-FLOP | **Primary history-residual competitor** |
| `P1-05` | `LoopSlate (PWM)` | `SelectivePWM` ($K=4, d_m=D/2$) | **Reference** | **Proposed method** |

**Phase 1 Stage Gate (Go / Iterate / Pivot):**
- **Go to Phase 2:** `P1-05` beats `P1-03` on at least 2 of 3 reasoning tasks (`DAG-Reuse`, `MultiHop-Chain`, `StateTrack-Revise`) AND beats or matches `P1-04 (LoopRes)` with superior $R_{\text{eval}} > R_{\text{train}}$ extrapolation.
- **Iterate:** If `P1-05` shows slot collapse (all slots identical or write gate $\approx 0$), tune write temperature, initialization bias $b_g$, and slot orthogonality regularization before proceeding.

---

### Phase 2 — Selectivity Ablations & Mechanistic Reuse Probes (Target: Week 2)

#### 2A. Controlled Mechanism Ablations (all Iso-FLOP matched on Tier 1 + Pilot LM)
| Exp ID | Ablation Axis | Variants Tested | Research Question Answered |
|---|---|---|---|
| `P2-A1` | **Write Selectivity** | `SelectivePWM` vs `FIFOMemory` vs `EMAMemory` | Does learned selective writing beat mechanical ring-buffer / smoothing? |
| `P2-A2` | **Component Removal** | Full `PWM` vs `No-Overwrite (Add-only)` vs `No-Competition (Sigmoid per slot)` vs `Read-Only (Fixed learned memory bank)` | Are **Read**, **Competitive Write**, and **Gated Update/Overwrite** each necessary? |
| `P2-A3` | **Slot Capacity Scaling** | $K \in \{1, 2, 4, 8, 16\}$ (fixed $d_m$) | How does performance scale with slot count $K$ vs task branching factor? |
| `P2-A4` | **Slot Dimension Scaling** | $d_m \in \{D/4, D/2, D\}$ (fixed $K=4$) | Minimum memory width needed for lossless intermediate storage. |
| `P2-A5` | **Input Injection Interaction** | `{With, Without Input Injection} × {Loop-Vanilla, LoopSlate}` | Does PWM reduce dependence on explicit raw-input re-injection? |

#### 2B. Three Mandatory Mechanistic Reuse Probes (`src/loopslate/probes/`)
1. **Probe 1 — Sub-Goal Linear Probe Trajectory (`SubgoalLinearProbe`):**
   - On `DAG-Reuse` ($x_1 \to x_2 \to x_3 \to y$), train frozen linear probes on $h_r$ vs $M_r$ across loops $r = 1 \dots R$.
   - *Expected signature:* $h_r$ peaks on the currently computed variable and decays on earlier ones; $M_r$ retains high linear decodability of $x_1$ across loops until the merge step.
2. **Probe 2 — Causal Slot Surgery (`CausalSlotSurgeryProbe`):**
   - Construct minimal counterfactual pairs differing only in intermediate sub-goal $x_1$ ($x_1^{(A)}$ vs $x_1^{(B)}$).
   - At loop $r^*$, surgically transplant the top-written slot(s) of $M_{r^*}^{(B)}$ into run $A$ (leaving $h_{r^*}^{(A)}$ untouched).
   - Measure **Causal Flip Rate**: fraction of examples whose final answer deterministically switches to $g(x_1^{(B)}, x_3^{(A)})$, compared against equal-norm random-slot / `LoopRes` / `FIFO` controls.
3. **Probe 3 — Write Sparsity & Read-Write Alignment (`WriteReadAlignmentProbe`):**
   - Compute write-gate sparsity, per-slot effective entropy, and the $(r_{\text{write}}, r_{\text{read}})$ cross-loop reuse matrix $\sum_k a_{\text{write}}(r_1, k) \cdot a_{\text{read}}(r_2, k)$ for $r_2 > r_1$.
   - Quantify **Intermediate Reuse Ratio**: mass of read attention directed to slots written $\ge 2$ loops ago without intermediate overwrite.

---

### Phase 3 — `LoopSlate-X`: Variable-Width Cell Integration & Scaling (Target: Weeks 3–4)

Once `SelectivePWM` is validated on uniform backbones, integrate `VariableWidthTransformerCell` to test whether **Variable Width (lower compute & KV-cache)** and **PWM (reusable computation)** compose synergistically.

#### Exact Head-Aligned Configuration (`Main-Mid`, $K_{\text{layers}}=8, d_{\text{head}}=64, R=3, L_{\text{exec}}=24$)
- **Variable-Width Profile:** `[832, 576, 448, 320, 320, 448, 576, 832]` $\Rightarrow$ $D_{\text{rms}} = \mathbf{576.0}$ (exact integer match!), $D_{\text{mean}} = 544$ ($-5.6\%$ attention FLOPs & KV-cache).
- **Uniform Matched Width:** $H_{\text{unif}} = 576$.

| Exp ID | Model Name | Topology & Width | Memory | Key Question |
|---|---|---|---|---|
| `P3-01` | `Loop-Unif` | `[576 × 8] × 3` | `NullMemory` | Uniform loop baseline |
| `P3-02` | `Loop-Unif+WideIO` | `[576 × 8] × 3` + $832$ IO | `NullMemory` | Eliminates Embed/Head confound |
| `P3-03` | `Loop-VW` | `[832→320→832] × 3` | `NullMemory` | Variable-width only (implicit bypass memory) |
| `P3-04` | `Loop-VW-Diamond` | `[320→832→320] × 3` | `NullMemory` | Phase-shifted geometry control |
| `P3-05` | `LoopSlate (Unif)` | `[576 × 8] × 3` | `SelectivePWM` | PWM only |
| `P3-06` | `LoopSlate-X (Full)` | `[832→320→832] × 3` | `SelectivePWM` | **Full synergy (VW + PWM)** |
| `P3-07` | `SMELT-Unif` | `Pre(4,832)→[576×8]×2→Coda(4,832)` | `NullMemory` | **SMELT-isomorphic baseline** |
| `P3-08` | `SMELT-LoopSlate-X` | `Pre(4,832)→[VW×8]×2→Coda(4,832)` | `SelectivePWM` | **Direct head-to-head vs SMELT** |

*Bonus Phase 3 Diagnostic:* Compare **Implicit Bypass Memory** (`P3-03`, nested prefix slicing without PWM) vs **Explicit Content-Addressed Memory** (`P3-05`, uniform + PWM) vs **Hybrid** (`P3-06`).

---

## 6. OOP Code-to-Experiment Traceability

Every experiment in this plan maps 1-to-1 to declarative configs and OOP components in `src/loopslate/`:

| Experiment Need | OOP Abstraction (`AGENT.md`) | Planned Location |
|---|---|---|
| Memory baselines & ablations | `BaseWorkingMemory` (`NullMemory`, `LoopResMemory`, `FIFOMemory`, `EMAMemory`, `SelectivePWM`) | `src/loopslate/memory/` |
| Uniform / VW / Diamond cells | `BaseRecurrentCell` + `BaseSubspaceRouter` (`NestedPrefixRouter`, `RotatingSubspaceRouter`) | `src/loopslate/cells/` |
| Loop conditioning | `BaseLoopConditioner` (`AdaRMSNormConditioner`, `ResidualScaleConditioner`) | `src/loopslate/cells/conditioning.py` |
| Exact FLOP / Param / KV matching | `ModelProfiler` + `IsoFlopSolver` | `src/loopslate/accounting/` |
| Synthetic & LM datasets | `BaseTaskDataset` (`DAGReuseDataset`, `MultiHopChainDataset`, `StateTrackReviseDataset`, `MQARDataset`) | `src/loopslate/tasks/` |
| Mechanistic verification | `BaseProbe` (`SubgoalLinearProbe`, `CausalSlotSurgeryProbe`, `WriteReadAlignmentProbe`) | `src/loopslate/probes/` |
| Training & reproducibility | `Trainer`, `Evaluator`, `ReproducibilityManager` | `src/loopslate/training/` |

---

## 7. Planned Paper Artifacts (Working Backward from Target Figures/Tables)

1. **Figure 1:** Conceptual architecture comparing `Loop-Vanilla`, `LoopRes` (unbounded append-only), and `LoopSlate` (bounded competitive Read/Write/Overwrite slots).
2. **Table 1 (Phase 1 Main Results):** Iso-FLOP comparison on `DAG-Reuse`, `MultiHop-Chain`, `StateTrack-Revise`, `MQAR`, and Pilot LM (`Loop-Vanilla-IsoFLOP` vs `LoopRes` vs `LoopSlate`).
3. **Figure 2 (Test-Time Compute & Depth Extrapolation):** Accuracy & memory footprint vs $R_{\text{eval}} \in [2, 12]$, highlighting `LoopSlate`'s constant $O(K)$ memory and extrapolation advantage over `LoopRes`.
4. **Figure 3 (Mechanistic Evidence Panel):** (a) Sub-goal Linear Probe Trajectory ($h_r$ vs $M_r$), (b) Causal Slot Surgery Flip Rates, (c) Cross-Loop Read-Write Alignment Heatmap.
5. **Table 2 (Phase 2 Ablation Table):** Selectivity (`FIFO`/`EMA`), component ablations (`No-Overwrite`, `No-Competition`, `Read-Only`), and slot capacity $K$ / dimension $d_m$ sweep.
6. **Table 3 (Phase 3 `LoopSlate-X` & SMELT Comparison):** Full 2×2 synergy table (`Uniform`/`VW` × `Null`/`PWM`) + `WideIO`, `Diamond`, and `SMELT-Unif` controls with FLOPs, KV-Cache, and wall-clock latency.

---

## 8. Risk Register & Mitigation Plan

| Risk | Early Symptom | Mitigation Built into Plan |
|---|---|---|
| **Memory Collapse** (slots unused or identical) | Read gate stays $\approx 0$; pairwise slot cosine $> 0.95$; write entropy $\to 0$. | Distinct learnable $E_{\text{slot}}$, competitive softmax write addressing, temperature tuning, optional light load-balance regularizer. |
| **Perplexity Dilution** | Gains clear on synthetic DAG tasks but noisy on general LM PPL. | Evaluate LM on **long-range / reasoning-dense slices** (code, math, multi-step synthetic evals) alongside FineWeb-Edu BPB and per-loop Logit-Lens convergence. |
| **LoopRes Outperforms at Small $R$** | At $R=3,4$, unbounded attention over 4 vectors is cheap and strong. | Explicitly test $R_{\text{eval}} \in \{8, 12, 16\}$ extrapolation and `StateTrack-Revise` (where stale history actively hurts append-only attention). |
| **Training Instability at Higher $R$** | Gradient explosion or NaN during unrolled BPTT across loops. | Pre-Norm / Sandwich-Norm, Zero-Init read gates, gradient clipping (`1.0`), and optional truncated BPTT / checkpointing ablation for large $R$. |
| **Wall-Clock Overhead Exceeds FLOP Savings** | Memory-bound small slot attention adds kernel launch latency. | Keep $d_m \le D/2$, fuse QKV projections across slots, profile real tokens/sec alongside theoretical FLOPs from Phase 0 onward. |
