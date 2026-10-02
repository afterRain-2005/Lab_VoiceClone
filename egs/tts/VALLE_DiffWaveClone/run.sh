#!/usr/bin/env bash
set -euo pipefail

script_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)
work_dir=$(cd "${script_dir}/../../.." && pwd)
cd "${work_dir}"
export WORK_DIR="${work_dir}"
export PYTHONPATH="${work_dir}:${PYTHONPATH:-}"

stage=0
stop_stage=5
gpu=0
emilia_root=""
libritts_root=""
cv3_root=""
target_speaker=""
ar_experiment=voiceclone_ar
nar_experiment=voiceclone_nar
ar_checkpoint_dir=""
config=egs/tts/VALLE_DiffWaveClone/configs/exp_config.json
reference_wav=""
reference_text=""
target_text=""

while [[ $# -gt 0 ]]; do
  case "$1" in
    --stage) stage="$2"; shift 2 ;;
    --stop-stage) stop_stage="$2"; shift 2 ;;
    --gpu) gpu="$2"; shift 2 ;;
    --emilia-root) emilia_root="$2"; shift 2 ;;
    --libritts-root) libritts_root="$2"; shift 2 ;;
    --cv3-root) cv3_root="$2"; shift 2 ;;
    --target-speaker) target_speaker="$2"; shift 2 ;;
    --ar-checkpoint-dir) ar_checkpoint_dir="$2"; shift 2 ;;
    --reference-wav) reference_wav="$2"; shift 2 ;;
    --reference-text) reference_text="$2"; shift 2 ;;
    --target-text) target_text="$2"; shift 2 ;;
    -h|--help)
      echo "Stages: 0 metadata, 1 EnCodec/ECAPA features, 2 AR, 3 NAR, 4 DiffWave, 5 inference"
      exit 0 ;;
    *) echo "Unknown option: $1" >&2; exit 2 ;;
  esac
done

if [[ ${stage} -le 0 && ${stop_stage} -ge 0 ]]; then
  python egs/tts/VALLE_DiffWaveClone/local/prepare_datasets.py \
    --emilia-root "${emilia_root}" \
    --libritts-root "${libritts_root}" \
    --cv3-root "${cv3_root}" \
    --target-speaker "${target_speaker}" \
    --output data/valle_diffwave_clone/voiceclone
fi

if [[ ${stage} -le 1 && ${stop_stage} -ge 1 ]]; then
  CUDA_VISIBLE_DEVICES="${gpu}" python \
    egs/tts/VALLE_DiffWaveClone/local/extract_features.py \
    --config "${config}" --device cuda
fi

if [[ ${stage} -le 2 && ${stop_stage} -ge 2 ]]; then
  CUDA_VISIBLE_DEVICES="${gpu}" accelerate launch bins/tts/train.py \
    --config "${config}" --exp_name "${ar_experiment}" --train_stage 1
fi

if [[ ${stage} -le 3 && ${stop_stage} -ge 3 ]]; then
  if [[ -z "${ar_checkpoint_dir}" ]]; then
    ar_checkpoint_dir="ckpts/tts/${ar_experiment}"
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" accelerate launch bins/tts/train.py \
    --config "${config}" --exp_name "${nar_experiment}" --train_stage 2 \
    --ar_model_ckpt_dir "${ar_checkpoint_dir}"
fi

if [[ ${stage} -le 4 && ${stop_stage} -ge 4 ]]; then
  CUDA_VISIBLE_DEVICES="${gpu}" python \
    egs/tts/VALLE_DiffWaveClone/local/train_token_diffwave.py \
    --config "${config}" --output-dir ckpts/vocoder/token_diffwave
fi

if [[ ${stage} -le 5 && ${stop_stage} -ge 5 ]]; then
  if [[ -z "${reference_wav}" || -z "${reference_text}" || -z "${target_text}" ]]; then
    echo "Stage 5 requires --reference-wav, --reference-text and --target-text" >&2
    exit 1
  fi
  CUDA_VISIBLE_DEVICES="${gpu}" accelerate launch bins/tts/inference.py \
    --config "${config}" --mode single \
    --acoustics_dir "ckpts/tts/${nar_experiment}" \
    --vocoder_dir ckpts/vocoder/token_diffwave/best.pt \
    --output_dir outputs/valle_diffwave_clone \
    --text "${target_text}" --text_prompt "${reference_text}" \
    --audio_prompt "${reference_wav}"
fi
