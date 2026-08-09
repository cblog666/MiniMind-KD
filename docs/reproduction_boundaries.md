# Reproduction boundaries

| Component | Status | Meaning |
|---|---|---|
| KDA recurrence, lower-bounded decay, ShortConv, full-rank gate | Paper-faithful reference | Same public equations; unfused sequential PyTorch path |
| 3:1 KDA/Gated-MLA and final global layer | Paper-faithful schedule | Width, depth and ranks are scaled down |
| NoPE Gated MLA | Paper-faithful mechanism | Eager quadratic attention, not production kernels/cache |
| Block AttnRes before Attention and MoE | Paper-faithful pseudocode | Local single-process representation store |
| Stable LatentMoE and SiTU-GLU | Paper-faithful mechanism | Fewer experts and simple expert loop |
| Quantile Balancing | Exact local variant | Exact minibatch quantile replaces K3's global histogram estimator |
| Per-head Muon | Functional small-scale implementation | No distributed optimizer-state sharding |
| MTP | Scaled approximation | Lightweight t+2 head, not a full backbone-shaped draft layer |
| Native MoonViT-V2 multimodality | Not implemented | v0.1 is a text backbone; no claim of native K3 multimodality |
| Fused/chunkwise KDA, prefix cache, expert parallelism | Not implemented | Required for long-context production performance |
| Domain SFT → GRPO experts | Functional framework | Results depend entirely on user data, rewards and compute |
| Generative reward model | Local interface | No reproduction of DeepSeek's private annotations/training |
| Multi-teacher full-vocabulary OPD | Functional implementation | Single-process teacher scheduling; local checkpoints only |
| MXFP4 QAT | Portable approximation | Fake E2M1 quantization, not hardware-native MXFP4/MXFP8 |
| V4-Flash-0731-specific recipe | Not claimed | Official update discloses re-post-training, not a separate recipe |

Passing the included tests establishes tensor-shape, causality, gradient and loss invariants. It does not establish benchmark parity, scaling efficiency, million-token quality, or equivalence to frontier-model training.
