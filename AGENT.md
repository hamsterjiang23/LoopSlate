# AGENT.md — LoopSlate Engineering & Research Standards

This document defines the mandatory software engineering, **Object-Oriented Programming (OOP)**, and scientific rigor rules for all human and AI contributors working in `E:\LoopSlate`.

---

## 1. Mandatory Object-Oriented Programming (OOP) Architecture

Every component in `src/loopslate/` MUST be designed and implemented using clean, idiomatic Python / PyTorch Object-Oriented Programming. Procedural scripts full of global flags, nested `if mode == ...` branches, and duplicated forward loops are **strictly forbidden**.

### 1.1 Core OOP Principles
1. **Explicit Abstract Contracts (`abc.ABC` / `typing.Protocol`)**:
   - Define abstract base classes for every extensible research axis so baselines and ablations are interchangeable drop-in strategies:
     - `BaseRecurrentCell(nn.Module, ABC)`: Executes one recurrent step $u_r = F_\theta(\tilde{h}_r; e_r)$. Concrete subclasses: `UniformTransformerCell`, `VariableWidthTransformerCell`, `DiamondWidthTransformerCell`.
     - `BaseWorkingMemory(nn.Module, ABC)`: Manages cross-loop state $M_r$ with explicit methods:
       - `init_state(batch_size, seq_len, device, dtype) -> MemoryState`
       - `read(hidden, state, loop_idx) -> tuple[Tensor, MemoryDiagnostics]`
       - `write_and_update(hidden, state, loop_idx) -> tuple[MemoryState, MemoryDiagnostics]`
       - Concrete subclasses: `NullMemory` (vanilla loop), `LoopResMemory` (append-only attention residual), `FIFOMemory`, `EMAMemory`, `SelectivePWM` (full competitive slot write + gated update).
     - `BaseLoopConditioner(nn.Module, ABC)`: Applies lightweight loop conditioning (e.g., `NullConditioner`, `AdaRMSNormConditioner`, `ResidualScaleConditioner`, `LoopLoRAConditioner`).
     - `BaseSubspaceRouter(nn.Module, ABC)`: Maps between full residual stream $D_{\max}$ and layer width $d_\ell$ (e.g., `NestedPrefixRouter`, `RotatingSubspaceRouter`, `LearnedProjectionRouter`).
     - `BaseTaskDataset(Dataset, ABC)`: Generates or loads benchmarks with standardized metadata (`subgoal_labels`, `causal_pairs`) for mechanistic probing.
     - `BaseProbe(ABC)`: Implements read-only analysis or causal intervention (`SubgoalLinearProbe`, `CausalSlotSurgeryProbe`, `WriteReadAlignmentProbe`).
2. **Composition Over Deep Inheritance**:
   - Assemble full models (`LoopedTransformerModel`) by composing injected strategy objects (`embedder`, `prelude`, `cell`, `memory`, `coda`, `head`) rather than creating combinatorial subclasses like `SMELTVariableWidthSelectivePWMLoopedTransformer`.
   - Keep inheritance hierarchies shallow (max 2 levels: `ABC -> ConcreteImplementation`).
3. **Typed Configuration & State Value Objects (`@dataclass`)**:
   - Pass strongly typed, validated `@dataclass` configuration objects (`ModelConfig`, `MemoryConfig`, `CellConfig`, `TrainConfig`) rather than raw `dict`s or long positional argument lists.
   - Encapsulate loop runtime state in structured dataclasses (`LoopState`, `MemoryState`, `ForwardOutput`, `MemoryDiagnostics`) instead of untyped tuples of tensors.
4. **Factory & Registry Pattern for Experiments**:
   - Register cells, memories, routers, and tasks in typed registries (`ComponentRegistry[T]`) so YAML configs or CLI flags instantiate exact model variants cleanly without branching logic inside `forward()`.
5. **Single Responsibility & Encapsulation**:
   - Training loops live in `Trainer`; evaluation lives in `Evaluator`; FLOP/parameter/KV-cache math lives in `ModelProfiler`; plotting/probing lives in `probes/`. Model `nn.Module` classes must remain pure computational graphs with optional structured diagnostic returns.
   - Keep private helper methods prefixed with `_`, maintain full type annotations on all public methods, and write concise Google-style docstrings stating tensor shapes (e.g., `[B, N, K, d_m]`).

### 1.2 Anti-Patterns to Reject
- Do **not** scatter `if config.use_pwm:` or `if config.memory_type == "loopres":` inside `LoopedTransformerModel.forward()`. Use polymorphism via `BaseWorkingMemory`.
- Do **not** mutate shared config objects at runtime or rely on module-level global mutable state.
- Do **not** hardcode hyperparameters, file paths, or device strings (`"cuda"`) inside classes; inject them via config or constructor parameters.

---

## 2. Scientific & Experimental Rigor Rules

### 2.1 Zero Causal Leakage (Non-Negotiable)
- For all autoregressive language modeling tasks, memory slots MUST NOT transfer information from position $j > i$ to position $i$ across loops.
- Default `SelectivePWM` operates at **Per-Token Slot granularity** $M_r \in \mathbb{R}^{B \times N \times K \times d_m}$. Any global/prefix-shared scratchpad variant MUST live in a distinct class (`PrefixSharedPWM`) and assert an explicit causal/prefix mask.
- Every `BaseWorkingMemory` implementation MUST pass an automated **Future-Token Perturbation Invariance Test** (`tests/test_causality.py`): perturbing token $t+1$ must produce exactly `0.0` change in logits and memory states at all positions $\le t$.

### 2.2 Strict Compute & Parameter Accounting
- Every baseline comparison MUST report via `ModelProfiler`:
  1. **Body-only stored parameters** and **Total stored parameters** (explicitly noting tied vs untied embeddings).
  2. **Forward FLOPs per token** (broken down into Attention, MLP/Linear, Memory Read/Write, and IO Projections).
  3. **Theoretical KV-Cache bytes per token** across all executed loops.
- Whenever comparing models with different boundary widths ($D_{\max}$ vs $D_{\text{rms}}$), include the `WideIO` control baseline so Embedding/LM-Head capacity cannot confound results.
- For Iso-FLOP comparisons between `LoopSlate` and `Loop-Vanilla` / `LoopRes`, match total executed FLOPs within $\le 1\%$ tolerance.

### 2.3 Anti-Collapse & Training Stability Defaults
- Initialize memory read-injection gates (`g_read`) to zero (Zero-Init) so step-0 behavior matches a clean looped Transformer.
- Break slot symmetry with distinct learnable initial slot prototypes $E_{\text{slot}} \in \mathbb{R}^{K \times d_m}$.
- Log memory health diagnostics every eval step: write gate mean/std, slot write entropy, effective slot utilization, and pairwise slot cosine similarity.

### 2.4 Reproducibility
- Every training/evaluation run must be driven by a `ReproducibilityManager` setting Python, NumPy, PyTorch CPU/CUDA seeds and saving the resolved config + git commit hash alongside metrics.

---

## 3. Code Quality & Verification Workflow

Before considering any implementation task complete:
1. Run `pytest` covering:
   - Shape & gradient flow checks across all `BaseWorkingMemory` and `BaseRecurrentCell` implementations.
   - Strict causal invariance test (future-token perturbation = exact zero diff).
   - `ModelProfiler` Iso-FLOP / Iso-Param sanity assertions.
   - Memory slot surgery probe end-to-end smoke test.
2. Ensure clean typing and lint compliance (`ruff check`, `mypy src/loopslate` when installed).
3. Keep commits modular and aligned with the Phase 1 → Phase 2 → Phase 3 roadmap in [README.md](README.md).
