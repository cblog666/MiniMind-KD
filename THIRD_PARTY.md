# Third-party sources and attribution

## MiniMind

MiniMind-KD is designed as a new, independent repository with familiar JSONL datasets and command-line training stages for users of [`jingyaogong/minimind`](https://github.com/jingyaogong/minimind). MiniMind is copyright its authors and distributed under Apache License 2.0. Its name is used only to describe compatibility and provenance. This repository does not overwrite, impersonate, or publish to the upstream repository.

## Moonshot AI / Kimi

The implementation is informed by these public primary sources:

- *Kimi K3: Open Frontier Intelligence*, arXiv:2607.24653;
- the official `moonshotai/Kimi-K3` ModelScope release page and model configuration;
- *Kimi Linear*, MoonshotAI/Kimi-Linear and its technical report;
- *Attention Residuals*, arXiv:2603.15031;
- the public Kimi K3 model configuration and modeling code, consulted to validate terminology and tensor shapes.

The code in this repository is a clean, small PyTorch implementation written for this project. It does not redistribute Kimi K3 weights, official training code, or Moonshot datasets. The `k3_shape_reference.yaml` file transcribes public architecture metadata for comparison only.

## DeepSeek-AI

The post-training pipeline is informed by *DeepSeek-V4: Towards Highly Efficient Million-Token Context Intelligence*, arXiv:2606.19348, and DeepSeek's public API update log. The report describes domain specialist fine-tuning, GRPO, reasoning-effort modes, generative rewards, tool-call formatting, interleaved thinking, and full-vocabulary multi-teacher on-policy distillation.

DeepSeek's public update for `DeepSeek-V4-Flash-0731` states that its architecture and size match the preview model and that it was re-post-trained. It does not disclose a separate 0731-only post-training recipe. MiniMind-KD therefore implements the public V4 report pipeline and does not label speculative details as official.

## Licenses and trademarks

MiniMind-KD source code is distributed under Apache License 2.0. Third-party weights, datasets, papers, code, logos, names, and trademarks retain their own terms. No right to use the MiniMind, Kimi, Moonshot AI, DeepSeek, or DeepSeek-AI trademarks is granted by this repository.
