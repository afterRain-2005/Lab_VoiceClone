# 数据集获取与目录

## Emilia

官方公开版本可从 Hugging Face 的 `amphion/Emilia-Dataset` 获取。完整数据规模很大，建议先下载目标语种或小规模 shard 验证流程，再扩展训练规模。

```bash
huggingface-cli download amphion/Emilia-Dataset \
  --repo-type dataset \
  --local-dir /data/Emilia
```

Emilia 用于增加口音、噪声、语速和真实录音环境多样性。目标说话人实验只会从包含明确 speaker ID 的记录中筛选。

## LibriTTS

从 OpenSLR 资源 60 获取。最小实验推荐：

```bash
mkdir -p /data/LibriTTS
cd /data/LibriTTS
wget https://www.openslr.org/resources/60/train-clean-100.tar.gz
wget https://www.openslr.org/resources/60/dev-clean.tar.gz
tar -xzf train-clean-100.tar.gz
tar -xzf dev-clean.tar.gz
```

LibriTTS 提供清晰文本、音频和说话人 ID。`prepare_datasets.py` 会递归解析 `*.trans.tsv`。

## CV3-Eval

从官方 `FunAudioLLM/CV3-Eval` 仓库按其说明获取音频和元数据。该数据只写入 `test.json`，不会参与目标说话人训练。

```bash
git clone https://github.com/FunAudioLLM/CV3-Eval.git /data/CV3-Eval
```

## 推荐目录

```text
/data/
├── Emilia/
├── LibriTTS/
│   └── LibriTTS/
└── CV3-Eval/
```

## 数据许可

下载前分别检查数据集许可、非商业限制、说话人授权与删除请求政策。CV3-Eval 是评测集，不应因为方便而混入训练集。生成音频需保留参考音频标识、模型版本、文本和生成时间等审计信息。
