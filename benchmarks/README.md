# MiniMind-KD benchmarks

本目录只运行 MiniMind-KD。原版 MiniMind 的客观评测成绩直接引用其仓库的公开结果，不重复消耗算力。

## CPU 工程跑分

下面的命令测量随机 token 下的模型执行效率，不涉及 tokenizer、磁盘读取或数据集下载，也不能衡量语言能力：

```bash
python benchmarks/benchmark_kd_runtime.py \
  --config configs/small.yaml \
  --device cpu \
  --batch-size 1 \
  --sequence-length 64 \
  --threads 8 \
  --forward-warmup-steps 2 \
  --forward-steps 5 \
  --train-warmup-steps 1 \
  --train-steps 3 \
  --decode-prompt-length 16 \
  --decode-new-tokens 8
```

指标定义：

- `forward_tokens_per_second`：无标签、无梯度的整段前向吞吐；
- `training_tokens_per_second`：包含 next-token + MTP 前向、反向和一次 AdamW 更新；
- `decode_tokens_per_second`：当前教学版 `generate` 的单样本贪心解码速度。它每生成一个 token 都重算完整前缀，未实现 KDA state / MLA KV cache；
- `estimated_active_parameters_per_token`：总参数减去未选中的 routed-expert 参数。共享专家、路由器和主干参数均计入；这是结构估算，不是 profiler 的逐算子 FLOP 统计。

已提交的原始 JSON 位于 [`results/`](results/)，README 中的数字由该文件四舍五入得到。

## 能力评测边界

原版 MiniMind 使用 `lm-evaluation-harness` 在 C-Eval、CMMLU、ARC-Easy、PIQA、OpenBookQA、HellaSwag 和 Social-IQA 上公布了成绩。MiniMind-KD 只有架构与训练代码，仓库不附带经过完整预训练/后训练的权重，因此当前不能产出与其公平可比的七项准确率。

随机初始化或一步烟雾测试权重在选择题上的数字只反映随机下界；把它放进主对比表会误导。待有固定数据配方、训练 token 数和公开 checkpoint 后，应使用与 MiniMind README 相同的任务、`lm-evaluation-harness` 版本和 chat-template 口径补测，并同时公开 checkpoint 哈希和训练预算。
