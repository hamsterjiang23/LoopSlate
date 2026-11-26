# LoopSlate

**Persistent Working Memory for Looped Transformers: From Recurrent Computation to Reusable Computation**

> **Core Research Question:**  
> *Can a looped Transformer improve test-time compute efficiency by selectively preserving and reusing intermediate computational results in a bounded, overwritable working memory?*

---

## 1. Motivation & Core Idea

Standard Looped (Recurrent-Depth) Transformers repeatedly apply a shared parameter core $F_\theta$ over a single residual stream $h_r$. Because the same residual stream must simultaneously act as an **active computation workspace (ALU)** and a **long-horizon scratchpad**, intermediate results from earlier loops are easily overwritten as loop depth $R$ grows.

**LoopSlate** decouples recurrent computation from intermediate result retention via a **bounded-capacity Persistent Working Memory (PWM)** attached to each token (ensuring 100% causal safety in autoregressive language modeling):

$$
\begin{aligned}
c_r &= \operatorname{Read}(h_r, M_r), & \tilde{h}_r &= h_r + g_{\text{read}, r} \odot c_r \\
u_r &= F_\theta(\tilde{h}_r; e_r) \\
W_r &= \operatorname{Write}(u_r, M_r), & M_{r+1} &= \operatorname{Update}(M_r, W_r)
\end{aligned}
$$

- **Bounded & Overwritable ($O(K)$ slots, independent of $R$):** Unlike append-only history residual attention (`LoopRes`), LoopSlate enforces fixed memory capacity and learns *what to write, which slot to allocate, and when to overwrite stale hypotheses*.
- **Orthogonal to Variable-Width Backbones (`LoopSlate-X`):** Once PWM is validated on a uniform-width looped backbone (Phase 1–2), we pair it with a Variable-Width (`Wide → Narrow → Wide`) recurrent cell (Phase 3) to reduce per-loop compute and KV-cache footprint.

---

## 2. Staged Research Roadmap

| Phase | Core Comparison | Primary Question |
|---|---|---|
| **Phase 1** | `Loop-Vanilla (Iso-FLOP)` vs `LoopRes` vs `LoopSlate (PWM)` | Does bounded selective working memory beat vanilla looping and unbounded history residuals at matched FLOPs? |
| **Phase 2** | `FIFO / EMA Memory` vs `Read-Only` vs `Selective Competitive PWM` | Are selective write gating and slot competition causally responsible for intermediate result reuse? |
| **Phase 3** | `LoopSlate (Uniform)` vs `LoopSlate-X (Variable-Width Cell)` | Do variable-width bottlenecks and persistent working memory complement each other under Iso-FLOP & Iso-Param budgets? |

### Key Diagnostic Metrics
1. **Iso-FLOP & Iso-Param Accuracy / Loss** (with `WideIO` control whenever boundary widths differ).
2. **Depth Extrapolation** ($R_{\text{eval}} > R_{\text{train}}$) and compute-to-accuracy efficiency.
3. **Mechanistic Reuse Probes**:
   - *Sub-goal Linear Probe Trajectory*: tracking intermediate variables ($A \to B \to C \to D$ or DAG merges) in $h_r$ vs $M_r$.
   - *Causal Slot Surgery*: swapping specific memory slots across paired inputs mid-loop to verify causal steering of downstream answers.
   - *Write Sparsity & Read-Write Alignment Matrix*: measuring non-collapsed slot specialization across loops.

---

## 3. Planned Object-Oriented Repository Layout

All source code follows strict **Object-Oriented Programming (OOP)** principles documented in [AGENT.md](AGENT.md).

```text
LoopSlate/
├── README.md                  # Project overview and roadmap
├── AGENT.md                   # Engineering, OOP design, and research-rigor rules
├── REFERENCES.md              # Categorized literature survey and verification status
├── experiment_plan/           # Versioned experimental designs (experiment_plan_v1.md)
├── pyproject.toml             # Package metadata and tooling configuration
├── configs/                   # Declarative YAML / dataclass experiment presets
├── src/
│   └── loopslate/
│       ├── core/              # Config dataclasses, registries, shared state containers
│       ├── cells/             # RecurrentCell ABC, UniformCell, VariableWidthCell, LoopConditioner
│       ├── memory/            # WorkingMemory ABC, SelectivePWM, FIFOMemory, LoopResMemory, NullMemory
│       ├── models/            # LoopedTransformerLM composer & baseline factory
│       ├── accounting/        # Exact ParamCounter, FlopProfiler, KVCacheEstimator
│       ├── tasks/             # Synthetic DAG/multi-hop/state-tracking tasks & LM datasets
│       ├── probes/            # LinearProbeRunner, SlotSurgeryIntervenor, AlignmentLogger
│       └── training/          # Trainer, Evaluator, CheckpointManager, ReproducibilityManager
└── tests/                     # Unit tests: causality, Iso-FLOP parity, shape & gradient checks
```

---

## 4. Literature & Positioning

For a comprehensive, categorized bibliography covering looped transformers, depth-wise residual mixing (`LoopRes`, `DenseFormer`, `Hyper-Connections`), neural memory slots (`NTM`, `DNC`, `RMT`, `TTM`, `Titans`), latent reasoning, and variable-width architectures, see [REFERENCES.md](REFERENCES.md).

---

## 5. Development Guidelines

Before implementing new modules or experiments, read [AGENT.md](AGENT.md) for mandatory OOP architecture contracts, causal safety checks, and experimental fairness rules.
