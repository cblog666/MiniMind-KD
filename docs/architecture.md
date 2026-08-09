# Architecture notes

## Hybrid sequence mixing

Each repeated group contains three KDA layers and one Gated MLA layer. If model depth cuts a group early, the final layer is replaced by MLA so the backbone always ends with unrestricted global content mixing.

For one head, the readable KDA recurrence is:

$$
\bar S_t=\operatorname{Diag}(\alpha_t)S_{t-1},\qquad
S_t=\bar S_t+\beta_t k_t(v_t-\bar S_t^{\mathsf T}k_t)^{\mathsf T},\qquad
\tilde o_t=S_t^{\mathsf T}q_t.
$$

The implementation uses the algebraically equivalent delta update with $\bar S_t$ in the prediction term. Query and key are L2-normalized after a causal depthwise ShortConv and SiLU. K3's lower-bounded mapping is retained:

$$g_t=g_{\min}\,\sigma(e^A z_t),\quad \alpha_t=e^{g_t},\quad g_{\min}=-5.$$

The output is head-wise RMS-normalized, multiplied by a full-rank sigmoid gate, and projected back to model width. `kda.py` uses an $O(T)$ Python recurrence; it is a correctness/reference path, not K3's chunkwise fused kernel.

Periodic MLA compresses K/V to a latent vector, reconstructs per-head content K/V, performs causal global attention without positional encoding, then applies the same style of full-rank output gate. Q/K and V head widths are independently configurable. The K3 shape reference uses a 128-wide reconstructed Q/K channel, a shared 64-wide direct K channel, and a 128-wide V channel; despite the upstream configuration name `qk_rope_head_dim`, no RoPE is applied. KDA supplies recency and position sensitivity between the NoPE global layers.

## Block Attention Residuals

The embedding is block source $b_0$. Completed layer groups are stored as $b_1,\ldots,b_n$; outputs inside the current group are accumulated in a partial block. Before both Attention and MoE, a learned pseudo-query attends over RMS-normalized source keys:

$$
\alpha_i=\operatorname{softmax}_i\left(w^\top\operatorname{RMSNorm}(b_i)\right),
\qquad h=\sum_i\alpha_i b_i.
$$

This follows the public Block AttnRes pseudocode and replaces the standard residual stream. There is one learned depth query before Attention and another before MoE in every decoder layer.

## Stable LatentMoE

Shared experts process the full hidden width. The routed path first projects $x\in\mathbb R^d$ to $z\in\mathbb R^\ell$, sends it to Top-k latent experts, normalizes the weighted aggregate, and projects it back:

$$
u=\sum_{i\in T_k(x)}p_iE_i(W_\downarrow x),\qquad
y=\sum_jE_j^{\text{shared}}(x)+W_\uparrow\operatorname{RMSNorm}(u).
$$

Expert GLUs use SiTU's smooth caps $\beta_1=4$ and $\beta_2=25$. Router scores are sigmoid values. The persistent bias affects selection only, while unmodified scores determine mixture weights.

K3 estimates Quantile Balancing over a global batch with distributed histograms. At this scale the exact local minibatch quantile is cheaper. Bias derived from batch $t$ is copied only after routing, so it first affects batch $t+1$.

## Optimizer and objective

Muon updates two-dimensional matrices; Q/K/V-style matrices are split along their output-head axis before Newton–Schulz orthogonalization. Embeddings, LM/MTP heads, vectors and scalar parameters use AdamW. The compact optimizer uses a configurable target update RMS and does not reproduce K3's distributed optimizer or weight clipping.

Pre-training loss is next-token cross entropy plus a weighted t+2 auxiliary prediction. The auxiliary head is deliberately smaller than K3's backbone-shaped MTP/EAGLE-compatible layer and is documented as a scaled approximation.
