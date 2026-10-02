#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import torch
import torchaudio
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm

from models.vocoders.diffusion.diffwave.token_diffwave import TokenConditionedDiffWave
from utils.util import load_config


class TokenWaveDataset(Dataset):
    def __init__(self, cfg, split: str, training: bool) -> None:
        self.cfg = cfg
        self.training = training
        self.dataset_name = cfg.dataset[0]
        self.dataset_dir = Path(cfg.preprocess.processed_dir) / self.dataset_name
        self.items = json.loads((self.dataset_dir / f"{split}.json").read_text(encoding="utf-8"))
        self.token_dir = self.dataset_dir / cfg.preprocess.acoustic_token_dir
        self.speaker_dir = self.dataset_dir / cfg.preprocess.spk_embedding_dir
        self.sample_rate = int(cfg.preprocess.sample_rate)
        self.hop_size = int(cfg.preprocess.codec_hop_size)
        self.segment_samples = int(cfg.diffwave_train.segment_samples)
        self.segment_frames = self.segment_samples // self.hop_size

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        item = self.items[index]
        waveform, sample_rate = torchaudio.load(item["Path"])
        waveform = waveform.mean(dim=0)
        if sample_rate != self.sample_rate:
            waveform = torchaudio.functional.resample(waveform, sample_rate, self.sample_rate)
        tokens = torch.from_numpy(np.load(self.token_dir / f"{item['Uid']}.npy")).long()
        speaker = torch.from_numpy(np.load(self.speaker_dir / f"{item['Uid']}.npy")).float()

        usable_frames = min(tokens.shape[0], waveform.numel() // self.hop_size)
        tokens = tokens[:usable_frames]
        waveform = waveform[: usable_frames * self.hop_size]
        if usable_frames >= self.segment_frames:
            if self.training:
                start_frame = random.randint(0, usable_frames - self.segment_frames)
            else:
                start_frame = (usable_frames - self.segment_frames) // 2
            tokens = tokens[start_frame : start_frame + self.segment_frames]
            start_sample = start_frame * self.hop_size
            waveform = waveform[start_sample : start_sample + self.segment_samples]
        else:
            token_padding = self.segment_frames - usable_frames
            tokens = torch.nn.functional.pad(tokens, (0, 0, 0, token_padding))
            waveform = torch.nn.functional.pad(
                waveform, (0, self.segment_samples - waveform.numel())
            )
        peak = waveform.abs().max().clamp_min(1e-7)
        waveform = 0.95 * waveform / peak
        return {"audio": waveform, "tokens": tokens, "speaker_embedding": speaker}


@torch.no_grad()
def validate(model, loader, device: torch.device) -> float:
    model.eval()
    losses = []
    for batch in loader:
        losses.append(
            model.training_loss(
                batch["audio"].to(device),
                batch["tokens"].to(device),
                batch["speaker_embedding"].to(device),
            ).item()
        )
    return float(sum(losses) / max(len(losses), 1))


def main() -> None:
    parser = argparse.ArgumentParser(description="Train token-conditioned DiffWave")
    parser.add_argument("--config", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--resume", type=Path)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device_name = (
        "cuda" if torch.cuda.is_available() else "cpu"
    ) if args.device == "auto" else args.device
    device = torch.device(device_name)

    train_dataset = TokenWaveDataset(cfg, "train", training=True)
    valid_dataset = TokenWaveDataset(cfg, "valid", training=False)
    train_loader = DataLoader(
        train_dataset,
        batch_size=cfg.diffwave_train.batch_size,
        shuffle=True,
        num_workers=cfg.diffwave_train.num_workers,
        pin_memory=device.type == "cuda",
    )
    valid_loader = DataLoader(
        valid_dataset,
        batch_size=cfg.diffwave_train.batch_size,
        shuffle=False,
        num_workers=cfg.diffwave_train.num_workers,
    )
    model = TokenConditionedDiffWave(cfg.model.token_diffwave).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.diffwave_train.learning_rate
    )
    start_epoch = 0
    best_valid = float("inf")
    if args.resume:
        checkpoint = torch.load(args.resume, map_location="cpu")
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_epoch = int(checkpoint["epoch"]) + 1
        best_valid = float(checkpoint.get("best_valid", best_valid))

    args.output_dir.mkdir(parents=True, exist_ok=True)
    for epoch in range(start_epoch, int(cfg.diffwave_train.epochs)):
        model.train()
        running = 0.0
        progress = tqdm(train_loader, desc=f"Token-DiffWave epoch {epoch}")
        for step, batch in enumerate(progress, start=1):
            optimizer.zero_grad(set_to_none=True)
            loss = model.training_loss(
                batch["audio"].to(device),
                batch["tokens"].to(device),
                batch["speaker_embedding"].to(device),
            )
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.diffwave_train.grad_clip)
            optimizer.step()
            running += loss.item()
            progress.set_postfix(loss=f"{running / step:.5f}")

        valid_loss = validate(model, valid_loader, device)
        state = {
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "epoch": epoch,
            "valid_loss": valid_loss,
            "best_valid": min(best_valid, valid_loss),
            "config": args.config,
        }
        torch.save(state, args.output_dir / "latest.pt")
        if valid_loss < best_valid:
            best_valid = valid_loss
            state["best_valid"] = best_valid
            torch.save(state, args.output_dir / "best.pt")
        print(json.dumps({"epoch": epoch, "valid_loss": valid_loss}))


if __name__ == "__main__":
    main()
