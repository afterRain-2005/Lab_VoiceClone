from __future__ import annotations

from pathlib import Path

import torch
import torch.nn.functional as F
import torchaudio


class ECAPASpeakerEncoder:
    """Thin lazy wrapper around SpeechBrain's VoxCeleb ECAPA encoder."""

    def __init__(
        self,
        source: str = "speechbrain/spkrec-ecapa-voxceleb",
        savedir: str | Path = "pretrained/ecapa_voxceleb",
        device: str = "cuda",
    ) -> None:
        try:
            from speechbrain.inference.speaker import EncoderClassifier
        except ImportError as error:
            raise ImportError(
                "Speaker conditioning requires `pip install speechbrain`."
            ) from error
        self.device = torch.device(device)
        self.model = EncoderClassifier.from_hparams(
            source=source,
            savedir=str(savedir),
            run_opts={"device": str(self.device)},
        )

    @torch.inference_mode()
    def encode_waveform(self, waveform: torch.Tensor, sample_rate: int) -> torch.Tensor:
        if waveform.ndim == 1:
            waveform = waveform.unsqueeze(0)
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        if sample_rate != 16000:
            waveform = torchaudio.functional.resample(waveform, sample_rate, 16000)
        embedding = self.model.encode_batch(waveform.to(self.device)).squeeze(1)
        return F.normalize(embedding.float(), dim=-1)

    @torch.inference_mode()
    def encode_file(self, path: str | Path) -> torch.Tensor:
        waveform, sample_rate = torchaudio.load(str(path))
        return self.encode_waveform(waveform, sample_rate)
