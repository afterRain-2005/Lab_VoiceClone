# 多层离散 Token + GPT + DiffWave 语音克隆

该工程基于 Amphion 的论文复现代码，严格对应以下流程：

```text
3–10 秒参考音频
  ├─ EnCodec 8 层 RVQ ───────────────┐
  └─ ECAPA-TDNN speaker embedding ───┼─> Speaker-conditioned VALL-E
输入文本 ──> G2P/文本 Token ─────────┘             │
                                      AR 第 1 层 + NAR 其余 7 层声学 Token
                                                   │
                         8 层 Token + speaker embedding
                                                   │
                                                   v
                                      Token-conditioned DiffWave
                                                   │
                                         DDPM 逐步波形去噪
                                                   │
                                                   v
                                             24 kHz 波形
```

VALL-E 保留论文的“第一码本自回归、剩余码本非自回归”结构；第一层承担主要语义/时序预测，完整输出仍是 `[T, 8]` 多层声学 Token。若实验必须让八层全部展平后逐 Token 自回归，可切换 Amphion 内置的 DualCodec flattened-AR，但训练成本和序列长度会显著增加。

## 新增内容

- VALL-E 支持连续说话人 embedding 全局条件。
- 数据集支持离线加载每条参考音频的 ECAPA embedding。
- DiffWave 新增多码本 Token conditioner，不再依赖 mel 频谱。
- 新增 Emilia、LibriTTS、CV3-Eval 统一数据准备脚本。
- 新增目标说话人选择、特征提取、两阶段 VALL-E 与扩散声码器训练流程。

入口与详细命令见 `egs/tts/VALLE_DiffWaveClone/README.md`。
