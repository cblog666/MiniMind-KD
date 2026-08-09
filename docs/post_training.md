# DeepSeek V4-style post-training

## Pipeline

1. Start from one shared base checkpoint.
2. Fine-tune separate domain datasets, for example math, code, agent and instruction following.
3. Run GRPO separately for each domain and reasoning-effort policy.
4. Keep specialist checkpoints frozen as teachers.
5. Sample fresh trajectories from the unified student.
6. Route each prompt to applicable teachers and minimize full-vocabulary reverse KL.

The pipeline intentionally avoids mixed-domain RL as the final merger. That mirrors the key substitution described in the V4 report: specialist RL remains, while the final mixed RL stage is replaced by on-policy distillation.

## GRPO

For each prompt, $G$ completions receive rewards $r_i$. Group-relative advantages are:

$$A_i=\frac{r_i-\operatorname{mean}(r)}{\operatorname{std}(r)+\epsilon}.$$

The loss uses a clipped policy ratio and a reference-policy KL estimator. Built-in rewards verify exact/numeric answers, reasoning format, and the public `<|DSML|tool_calls>` / `invoke` / `parameter` XML-style schema. `LocalGenerativeReward` can score rubric-driven tasks through a caller-supplied local generator. No network client is present.

Reasoning effort controls prompting, rollout budget and optional length penalty. It does not magically reproduce V4's hidden training data or exact system prompt.

## On-policy distillation

The student samples its own trajectory $y\sim\pi_\theta$. Teachers are evaluated on the same prefix and sampled tokens. For teacher $i$:

$$
\mathcal L_i=\sum_t \sum_{v\in V}
p_\theta(v\mid x,y_{<t})
\left[\log p_\theta(v\mid x,y_{<t})-\log p_i(v\mid x,y_{<t})\right].
$$

Teacher weights are normalized per sample after domain routing. The implementation keeps the full vocabulary distribution instead of estimating KL only at sampled tokens.

## Deployment-aware post-training

When `expert_qat` is enabled, routed expert linear weights use a straight-through, block-scaled E2M1 fake quantizer during all stages. This emulates the optimization pressure of MXFP4 QAT but does not pack weights or provide FP4 kernels. K3/DeepSeek production infrastructure, distributed teacher scheduling, resumable sandboxes and million-token rollouts are outside this repository.
