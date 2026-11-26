# REFERENCES.md — Comprehensive Literature Survey for LoopSlate

> **Citation Verification Policy:**  
> Entries marked **`[Verified]`** correspond to well-established published papers with confirmed titles/venues. Entries marked **`[Verify ID/Details]`** come from recent 2025–2026 notes or community references whose exact arXiv identifiers, author lists, or architectural formulas **must be verified directly against the original PDF** before inclusion in a manuscript.

---

## 1. Looped, Weight-Shared & Recurrent-Depth Transformers

Core lineage establishing weight sharing along depth, iterative latent computation, and test-time depth scaling.

1. **Universal Transformers** — Dehghani, Gouws, Vinyals, Uszkoreit, Kaiser (ICLR 2019, arXiv:1807.03819). `[Verified]`  
   *Relevance:* Foundational depth-recurrent Transformer applying a shared block across depth with adaptive halting.
2. **ALBERT: A Lite BERT for Self-supervised Learning of Language Representations** — Lan et al. (ICLR 2020, arXiv:1909.11942). `[Verified]`  
   *Relevance:* Early large-scale demonstration of cross-layer parameter sharing and factorized embedding parameterization.
3. **Deep Equilibrium Models (DEQ)** — Bai, Kolter, Koltun (NeurIPS 2019, arXiv:1909.01377). `[Verified]`  
   *Relevance:* Infinite-depth weight-tied fixed-point computation via root-finding and implicit differentiation.
4. **Looped Transformers as Programmable Computers** — Giannou, Rajput, Sohn, Lee, Lee, Papailiopoulos (ICML 2023, arXiv:2301.13196). `[Verified]`  
   *Relevance:* Constructive proof that looped Transformers can emulate programmable computers using explicit scratchpad subspaces inside the residual stream.
5. **Looped Transformers are Better at Learning Learning Algorithms** — Yang, Lee, Nowak, Papailiopoulos (2023/2024, arXiv:2311.12424). `[Verified]`  
   *Relevance:* Shows looped Transformers learn iterative optimization / in-context learning algorithms better than non-looped depth-matched baselines.
6. **Looped Transformers for Length Generalization** — Fan et al. (2024, arXiv:2409.15647). `[Verify ID/Details]`  
   *Relevance:* Demonstrates that adaptive looping enables length generalization on algorithmic tasks.
7. **Relaxed Recursive Transformers: Effective Parameter Sharing with Layer-wise LoRA** — Bae et al. (Google, Dec 2024, arXiv:2410.20672). `[Verified]`  
   *Relevance:* Adds lightweight layer/loop-specific LoRA adapters to break symmetry in weight-shared recursive Transformers.
8. **Scaling up Test-Time Compute with Latent Reasoning: A Recurrent Depth Approach (Huginn)** — Geiping, McLeish, Jain, Kirchenbauer, Singh, Bartoldson, Kailkhura, Bhatele, Goldstein (Feb 2025, arXiv:2502.05171). `[Verified]`  
   *Relevance:* Flagship large-scale recurrent-depth language model (`Prelude → Recurrent Core × R → Coda`) scaling test-time compute in latent space.
9. **Ouro (Looped / Recurrent-Depth Knowledge Manipulation)** — Recent looped Transformer study referenced in project notes (2025/2026). `[Verify ID/Details]`  
   *Relevance:* Shows increased loop depth enhances multi-step knowledge manipulation; motivates moving from *more loops* to *reusable computation across loops*.
10. **How to Loop Your Transformer** — Cited in prior notes as arXiv:2610.40325 (systematically studying full vs middle-layer looping, loop counts, and loop-specific low-rank adapters). `[Verify ID/Details]`  
    *Relevance:* Empirical design space for loop placement, loop conditioning, and loop saturation/instability.
11. **LoopVAE** — Cited in prior notes as arXiv:2609.11516 (loop-conditioned core, input injection, zero-initialized gating in visual tokenizers). `[Verify ID/Details]`  
    *Relevance:* Practical conditioning/stability recipes and empirical latency vs stored-parameter tradeoff analysis.

---

## 2. Depth-Wise Residual Mixing, Skip Aggregation & History Access (`LoopRes` Lineage)

Direct closest competitors to LoopSlate along the **depth / loop axis**—crucial to contrast against bounded, overwritable working memory.

12. **LoopRes / Attention Residuals over Loop History** — Referenced in project notes (2025/2026). `[Verify ID/Details]`  
    *Relevance:* **Primary baseline to beat.** Retains append-only historical loop states and reads via attention residual. LoopSlate differs by enforcing bounded $O(K)$ capacity and explicit selective write/overwrite/eviction.
13. **DenseFormer: Enhancing Information Flow in Transformers via Weighted Averaging of Past Activations** — Pagliardini et al. (2024, arXiv:2402.02622). `[Verified]`  
    *Relevance:* Depth-Weighted Averaging (DWA) across all preceding layer outputs; demonstrates the utility of preserving earlier-depth representations.
14. **Hyper-Connections** — Zhu, Huang, et al. (ByteDance Seed / Tsinghua, 2024, arXiv:2409.19606 / 2412.17748 — confirm exact arXiv ID). `[Verify ID/Details]`  
    *Relevance:* Expands the single residual stream into $n$ parallel streams with learned static/dynamic depth-wise mixing matrices; closely related to multi-slot depth state transport, though without explicit content-addressed write/read/eviction semantics.
15. **MUDDFormer: Breaking Residual Bottlenecks in Transformers via Multiway Dynamic Dense Connections** — (2025). `[Verify ID/Details]`  
    *Relevance:* Dynamic cross-layer dense connections addressing single-stream residual overwriting.
16. **Highway Networks** — Srivastava, Greff, Schmidhuber (2015, arXiv:1505.00387). `[Verified]`  
    *Relevance:* Classic gated carry/transform residual mechanism over depth.
17. **Residual Connections & Depth-Wise Attention (e.g., Layer Attention / RealFormer / LAuReL)** — He et al. (2020) / Men et al. (2024). `[Verify ID/Details]`  
    *Relevance:* Generalized residual connectivity across Transformer depth.

---

## 3. Neural Working Memory, External Memory & Memory-Slot Architectures

Classical and modern architectures implementing explicit Read / Write / Update memory slots. **Key distinction:** most operate along the *sequence/time* axis rather than the *recurrent loop-depth* axis.

18. **Neural Turing Machines (NTM)** — Graves, Wayne, Danihelka (2014, arXiv:1410.5401). `[Verified]`  
    *Relevance:* Canonical content- and location-addressed differentiable external memory with erase/add write heads.
19. **Hybrid Computing Using a Neural Network with Dynamic External Memory (DNC)** — Graves et al. (*Nature*, 2016). `[Verified]`  
    *Relevance:* Adds dynamic memory allocation, usage tracking, and temporal link matrices to prevent write interference.
20. **End-to-End Memory Networks** — Sukhbaatar, Szlam, Weston, Fergus (NeurIPS 2015, arXiv:1503.08895). `[Verified]`  
    *Relevance:* Multi-hop iterative memory readout over stored representations.
21. **Relational Recurrent Neural Networks (RMC)** — Santoro et al. (NeurIPS 2018, arXiv:1806.01822). `[Verified]`  
    *Relevance:* Multi-slot working memory whose slots interact via multi-head self-attention.
22. **Memory Transformers** — Burtsev, Kuratov, Peganov, Sapunov (2020, arXiv:2006.11527). `[Verified]`  
    *Relevance:* Augments Transformer inputs with learnable memory tokens.
23. **Memformer: A Memory-Augmented Transformer for Sequence Modeling** — Wu, Lan, Qian, Gu, Gerstein, Yu (Findings of AACL 2022, arXiv:2010.06891). `[Verified]`  
    *Relevance:* Fixed-size dynamic memory slots updated via cross-attention and memory-writing gates across segments.
24. **Recurrent Memory Transformer (RMT)** — Bulatov, Kuratov, Burtsev (NeurIPS 2022, arXiv:2207.06881 / 2207.12886 — verify exact ID). `[Verify ID/Details]`  
    *Relevance:* Passes recurrent memory tokens across sequence segments for long-context processing.
25. **Token Turing Machines (TTM)** — Ryoo, Gopalakrishnan, Kahatapitiya, Xiao, Rao, Stone, Lu, Ibarz, Arnab (CVPR 2023, arXiv:2211.09119). `[Verified]`  
    *Relevance:* Closest memory-slot transformer mechanism: fixed-capacity external token memory with learned Read and Write summarization/selection units across temporal steps.
26. **Memorizing Transformers** — Wu, Rabe, Hutchins, Szegedy (ICLR 2022, arXiv:2203.08913). `[Verified]`  
    *Relevance:* Non-parametric kNN retrieval over historical keys/values.
27. **Slot Attention / Object-Centric Learning with Slot Attention** — Locatello et al. (NeurIPS 2020, arXiv:2006.15055). `[Verified]`  
    *Relevance:* Competitive softmax normalization across slots to prevent slot collapse and encourage specialization—directly inspires LoopSlate's competitive write addressing.
28. **Titans: Learning to Memorize at Test Time** — Behrouz, Zhong, Mirrokni (Dec 2024 / Jan 2025, arXiv:2501.00663). `[Verified]`  
    *Relevance:* Surprise-gated neural long-term memory with forgetting decay (`MAC`, `MAG`, `MAL`) along the sequence dimension.
29. **Linear Transformers Are Secretly Fast Weight Programmers** — Schlag, Irie, Schmidhuber (ICML 2021, arXiv:2102.11174). `[Verified]`  
    *Relevance:* Theoretical bridge between outer-product key-value memory updates, delta rules, and fast weight programming.
30. **Gated Delta Networks / DeltaNet** — Yang et al. (2024/2025, arXiv:2412.06464). `[Verify ID/Details]`  
    *Relevance:* Selective erasure + writing in associative memory via gated delta rule; strong inspiration for selective overwrite dynamics.
31. **Global Workspace / Coordination Among Neural Modules Through a Shared Global Workspace** — Goyal et al. (ICLR 2022, arXiv:2103.01197). `[Verified]`  
    *Relevance:* Cognitive-science-grounded bottlenecked shared workspace with competitive write/broadcast dynamics.

---

## 4. Latent Reasoning, Explicit/Implicit Scratchpads & Test-Time Compute

Connects LoopSlate's "reusable latent scratchpad" to explicit CoT scratchpads and latent reasoning models.

32. **Show Your Work: Scratchpads for Intermediate Computation with Language Models** — Nye et al. (2021, arXiv:2112.00114). `[Verified]`  
    *Relevance:* Original explicit token-level scratchpad showing intermediate storage unlocks multi-step algorithmic execution.
33. **Chain-of-Thought Prompting Elicits Reasoning in Large Language Models** — Wei et al. (NeurIPS 2022, arXiv:2201.11903). `[Verified]`  
    *Relevance:* Token-level externalization of intermediate reasoning states.
34. **Think Before You Speak: Training Language Models With Pause Tokens** — Goyal et al. (ICLR 2024, arXiv:2310.02226). `[Verified]`  
    *Relevance:* Allocates extra horizontal compute before token emission without explicit text CoT.
35. **Quiet-STaR: Language Models Can Teach Themselves to Think Before Speaking** — Zelikman et al. (2024, arXiv:2403.09629). `[Verified]`  
    *Relevance:* Internal rationale generation for per-token reasoning.
36. **Coconut: Training Large Language Models to Reason in a Continuous Latent Space** — Hao, Sukhbaatar, Su, Li, Hu, Weston, Tian (Meta FAIR, Dec 2024, arXiv:2412.06769). `[Verified]`  
    *Relevance:* Feeds continuous hidden states back as latent thoughts, supporting superposition of reasoning paths.
37. **From Explicit CoT to Implicit CoT: Learning to Internalize CoT Step by Step** — Deng et al. (2024, arXiv:2405.14838). `[Verified]`  
    *Relevance:* Studies limits and mechanisms of internalizing intermediate reasoning steps into hidden activations.
38. **Scaling LLM Test-Time Compute Optimally can be More Effective than Scaling Model Parameters** — Snell et al. (2024, arXiv:2408.03314). `[Verified]`  
    *Relevance:* Frames test-time compute efficiency as a primary scaling axis.

---

## 5. Variable-Width, Bottleneck & Non-Uniform Capacity Transformers (Phase 3 Integration)

Essential for Phase 3 (`LoopSlate-X`) and for positioning against width-reduced looped baselines.

39. **Variable-Width Transformers (`><former`)** — Cited in prior notes as arXiv:2606.18246 (`Wide → Narrow → Wide` profile over a full-width residual stream via nested subspace updates). `[Verify ID/Details]`  
    *Relevance:* Direct foundation for our Phase 3 `VariableWidthTransformerCell` and implicit bypass-channel baseline.
40. **SMELT** — Cited in prior notes as arXiv:2609.01343 (`Front → [Middle]×2 → Back` with globally narrowed uniform hidden dimension $H$ to match FLOPs; *note: automated lookup returned a different paper under `2609.01343`, so verify exact ID/title directly*). `[Verify ID/Details]`  
    *Relevance:* Closest compute-matched looped baseline; contrasts global narrowing vs non-uniform in-loop width and explicit working memory.
41. **SpiralFormer** — Cited in prior notes as arXiv:2602.11698 (varying token sequence resolution across loop iterations). `[Verify ID/Details]`  
    *Relevance:* Orthogonal axis of variable compute inside loops (token resolution vs hidden width vs memory slots).
42. **Funnel-Transformer: Filtering out Sequential Redundancy for Efficient Language Processing** — Dai, Lai, Yang, Le (NeurIPS 2020, arXiv:2006.03236). `[Verified]`  
    *Relevance:* Classic hourglass/funnel compression-expansion pattern (along sequence length).
43. **Hourglass Transformers (Hierarchical Transformers Are More Efficient Language Models)** — Nawrot et al. (NAACL 2022, arXiv:2110.13711). `[Verified]`  
    *Relevance:* U-Net-like downsampling and upsampling within autoregressive language models.
44. **MatFormer: Nested Transformer for Elastic Inference** — Kudugunta et al. (2023, arXiv:2310.07707) & **Matryoshka Representation Learning** — Kusupati et al. (NeurIPS 2022, arXiv:2205.13147). `[Verified]`  
    *Relevance:* Nested prefix sub-dimensions inside a shared vector space; directly informs `NestedPrefixRouter` subspace slicing.
45. **Sandwich Transformer** — Press, Smith, Levy (2019, arXiv:1911.03864). `[Verified]`  
    *Relevance:* Early evidence that non-uniform layer ordering/allocation across depth improves perplexity under fixed budgets.
46. **Transformer Layers as Painters** — Sun et al. (2024, arXiv:2407.09298). `[Verified]`  
    *Relevance:* Empirical study of shared representation space across middle layers, supporting why middle layers can be narrowed or looped while early/late layers specialize.

---

## 6. Adaptive Computation, Dynamic Routing & Early Exit

Relevant for optional dynamic halting / compute allocation extensions and FLOP-efficiency comparisons.

47. **Adaptive Computation Time for Recurrent Neural Networks (ACT)** — Graves (2016, arXiv:1603.08983). `[Verified]`  
    *Relevance:* Per-step halting probabilities for variable recurrent computation.
48. **PonderNet: Learning to Ponder** — Banino, Balaguer, Blundell (2021, arXiv:2107.05407). `[Verified]`  
    *Relevance:* Unbiased low-variance probabilistic halting formulation for recurrent depth.
49. **Mixture-of-Depths (MoD): Dynamically Allocating Compute in Transformer-Based Language Models** — Raposo et al. (DeepMind, 2024, arXiv:2404.02258). `[Verified]`  
    *Relevance:* Token-level routing around Transformer blocks under fixed FLOP budgets.
50. **Mixture-of-Recursions (MoR)** — Recent dynamic per-token recursion depth architecture (2025). `[Verify ID/Details]`  
    *Relevance:* Adaptive per-token loop counts in recurrent-depth Transformers; complementary to LoopSlate's per-token working memory.

---

## 7. Algorithmic Reasoning, State Tracking & Synthetic Benchmarks

Benchmarks designed to expose **intermediate computation reuse** (where vanilla LM perplexity alone can have low signal-to-noise).

51. **Zoology: Measuring and Improving Recall in Efficient Language Models (MQAR)** — Arora et al. (ICLR 2024, arXiv:2312.04927). `[Verified]`  
    *Relevance:* Multi-Query Associative Recall benchmark for evaluating retrieval vs bottleneck tradeoffs.
52. **The Illusion of State in State-Space Models** — Merrill, Petty, Sabharwal (ICML 2024, arXiv:2404.08819). `[Verified]`  
    *Relevance:* Formal state-tracking ($S_5$, finite-state automata) benchmarks distinguishing true iterative state maintenance from shallow shortcuts.
53. **GSM-Symbolic / ProntoQA / Pointer Chasing / Graph Reachability Benchmarks** — Saparov & He (ICLR 2023, arXiv:2210.01240), Mirzadeh et al. (2024, arXiv:2410.05229). `[Verified]`  
    *Relevance:* Controllable multi-hop compositional reasoning and DAG dependency tasks where reusing sub-results directly cuts redundant loops.
54. **FineWeb / FineWeb-Edu** — Penedo et al. (HuggingFace, 2024, arXiv:2406.17557). `[Verified]`  
    *Relevance:* Standard open pretraining corpus for clean Iso-FLOP scaling evaluations.

---

## 8. Mechanistic Interpretability, Probing & Causal Verification

Methodology for proving that LoopSlate's slots store and causally reuse intermediate results (rather than acting as extra non-linear parameters).

55. **Interpreting GPT: The Logit Lens** — nostalgebraist (2020) & **Eliciting Latent Predictions from Transformers with the Tuned Lens** — Belrose et al. (2023, arXiv:2303.08112). `[Verified]`  
    *Relevance:* Per-loop convergence trajectories and early-exit diagnostics.
56. **Locating and Editing Factual Associations in GPT (Causal Tracing / Activation Patching)** — Meng et al. (NeurIPS 2022, arXiv:2202.05262) & **Towards Best Practices of Activation Patching** — Zhang & Nanda (ICLR 2024, arXiv:2309.16042). `[Verified]`  
    *Relevance:* Theoretical and methodological foundation for our **Causal Slot Surgery** probe (swapping memory slots across paired multi-hop instances mid-loop).
57. **Interpretability in the Wild / Path Patching** — Wang et al. (ICLR 2023, arXiv:2211.00593). `[Verified]`  
    *Relevance:* Isolating direct vs memory-mediated causal pathways across recurrent iterations.

---

## 9. Recommended Reading Priority Before Phase 1 Coding

If you want the tightest 6-paper reading list directly shaping LoopSlate's code and baselines:
1. **Geiping et al. (2025, `2502.05171`)** — Recurrent-depth architecture (`Prelude → Core × R → Coda`) and training stability.
2. **Ryoo et al. (2023, `2211.09119`, Token Turing Machines)** — Fixed-slot Read/Write Transformer memory mechanics.
3. **Locatello et al. (2020, `2006.15055`, Slot Attention)** — Competitive softmax slot allocation to prevent memory collapse.
4. **LoopRes & DenseFormer (`2402.02622`) / Hyper-Connections** — Exact depth-history baselines to outperform.
5. **Giannou et al. (2023, `2301.13196`)** — Theoretical grounding of scratchpads inside looped Transformers.
6. **Zhang & Nanda (2024, `2309.16042`)** — Rigorous causal patching protocol for `CausalSlotSurgeryProbe`.
