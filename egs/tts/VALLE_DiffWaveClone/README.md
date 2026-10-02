# VALL-E + Token-DiffWave target-speaker cloning

数据集下载与许可说明见 `DATASETS_ZH.md`。

先按 Amphion 根目录的 `env.sh` 安装基础环境，再执行：

```bash
pip install -r egs/tts/VALLE_DiffWaveClone/requirements_voiceclone.txt
```

## 模型对应

1. **语音分词**：Amphion EnCodec 提取 8 层、1024 大小码本，Token 张量为 `[T, 8]`。
2. **说话人条件**：SpeechBrain ECAPA-TDNN 从 3–10 秒参考音频提取 192 维向量，投影后加到 VALL-E 文本与声学序列，并注入 DiffWave 残差块。
3. **文本语义编码**：Amphion G2P 与 VALL-E text embedding/Transformer。
4. **Token 生成**：VALL-E AR Transformer 逐 Token 预测第一码本，NAR Transformer补齐其余七码本。
5. **扩散精修**：Token-conditioned DiffWave 直接读取 8 层 Token，从高斯噪声执行 DDPM 去噪并输出 24 kHz 波形。

## 数据准备

```bash
python egs/tts/VALLE_DiffWaveClone/local/prepare_datasets.py \
  --emilia-root /data/Emilia \
  --libritts-root /data/LibriTTS \
  --cv3-root /data/CV3-Eval \
  --target-speaker 1089 \
  --output data/valle_diffwave_clone/voiceclone
```

Emilia 和 LibriTTS 用于目标说话人训练候选；CV3-Eval 固定写入 `test.json`，用于噪声与跨语言评测，不混入训练。没有传 `--target-speaker` 时，自动选择语音总时长最长且达到阈值的说话人。

## 特征提取

```bash
python egs/tts/VALLE_DiffWaveClone/local/extract_features.py \
  --config egs/tts/VALLE_DiffWaveClone/configs/exp_config.json \
  --device cuda
```

该步骤生成：

- `phones/*.phone`
- `symbols.dict`
- `acoutic_tokens/*.npy`
- `speaker_embeddings/*.npy`

## 训练

```bash
# VALL-E 第一层自回归模型
accelerate launch bins/tts/train.py \
  --config egs/tts/VALLE_DiffWaveClone/configs/exp_config.json \
  --exp_name voiceclone_ar --train_stage 1

# VALL-E 剩余七码本模型
accelerate launch bins/tts/train.py \
  --config egs/tts/VALLE_DiffWaveClone/configs/exp_config.json \
  --exp_name voiceclone_nar --train_stage 2 \
  --ar_model_ckpt_dir ckpts/tts/voiceclone_ar

# Token 条件扩散声码器
python egs/tts/VALLE_DiffWaveClone/local/train_token_diffwave.py \
  --config egs/tts/VALLE_DiffWaveClone/configs/exp_config.json \
  --output-dir ckpts/vocoder/token_diffwave
```

## 推理

```bash
accelerate launch bins/tts/inference.py \
  --config egs/tts/VALLE_DiffWaveClone/configs/exp_config.json \
  --mode single \
  --acoustics_dir ckpts/tts/voiceclone_nar \
  --vocoder_dir ckpts/vocoder/token_diffwave \
  --output_dir outputs \
  --text "This is the text to synthesize." \
  --text_prompt "Exact transcript of the reference audio." \
  --audio_prompt /path/to/reference_5s.wav
```

参考音频应为单人、3–10 秒、转写准确。仅使用已取得明确授权的声音。
