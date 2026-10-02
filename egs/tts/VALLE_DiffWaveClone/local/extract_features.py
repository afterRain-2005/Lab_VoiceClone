#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from tqdm import tqdm

from models.tts.valle.speaker_encoder import ECAPASpeakerEncoder
from processors.acoustic_extractor import extract_utt_acoustic_features_serial
from processors.phone_extractor import extract_utt_phone_sequence
from utils.util import load_config


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract EnCodec tokens, phones and ECAPA embeddings")
    parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", default="voiceclone")
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()
    cfg = load_config(args.config)
    dataset_dir = Path(cfg.preprocess.processed_dir) / args.dataset
    metadata = []
    for split in ("train", "valid", "test"):
        path = dataset_dir / f"{split}.json"
        if path.is_file():
            metadata.extend(json.loads(path.read_text(encoding="utf-8")))
    if not metadata:
        raise RuntimeError(f"No metadata found under {dataset_dir}")

    extract_utt_acoustic_features_serial(metadata, str(dataset_dir), cfg)
    extract_utt_phone_sequence(args.dataset, cfg, metadata)

    output = dataset_dir / cfg.preprocess.spk_embedding_dir
    output.mkdir(parents=True, exist_ok=True)
    encoder = ECAPASpeakerEncoder(
        source=cfg.preprocess.speaker_encoder_source,
        savedir=cfg.preprocess.speaker_encoder_savedir,
        device=args.device,
    )
    for item in tqdm(metadata, desc="ECAPA speaker embeddings"):
        destination = output / f"{item['Uid']}.npy"
        if destination.is_file():
            continue
        embedding = encoder.encode_file(item["Path"])[0].cpu().numpy().astype(np.float32)
        np.save(destination, embedding)


if __name__ == "__main__":
    main()
