from __future__ import annotations

from math import sqrt

import torch
import torch.nn as nn
import torch.nn.functional as F

from models.vocoders.diffusion.diffwave.diffwave import Conv1d, DiffusionEmbedding


class MultiCodebookConditioner(nn.Module):
    def __init__(
        self,
        num_quantizers: int,
        codebook_size: int,
        condition_dim: int,
    ) -> None:
        super().__init__()
        self.num_quantizers = num_quantizers
        self.embeddings = nn.ModuleList(
            [nn.Embedding(codebook_size, condition_dim) for _ in range(num_quantizers)]
        )
        self.projection = nn.Sequential(
            Conv1d(condition_dim, condition_dim, 3, padding=1),
            nn.SiLU(),
            Conv1d(condition_dim, condition_dim, 1),
        )

    def forward(self, tokens: torch.Tensor, output_length: int) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(f"tokens must have shape [B, T, Q], got {tokens.shape}")
        if tokens.shape[-1] != self.num_quantizers:
            raise ValueError(
                f"expected {self.num_quantizers} quantizers, got {tokens.shape[-1]}"
            )
        condition = 0
        for index, embedding in enumerate(self.embeddings):
            condition = condition + embedding(tokens[..., index].long())
        condition = condition.transpose(1, 2) / sqrt(self.num_quantizers)
        condition = self.projection(condition)
        return F.interpolate(
            condition,
            size=output_length,
            mode="linear",
            align_corners=False,
        )


class TokenResidualBlock(nn.Module):
    def __init__(
        self,
        condition_dim: int,
        speaker_embed_dim: int,
        residual_channels: int,
        dilation: int,
    ) -> None:
        super().__init__()
        self.dilated_conv = Conv1d(
            residual_channels,
            2 * residual_channels,
            3,
            padding=dilation,
            dilation=dilation,
        )
        self.diffusion_projection = nn.Linear(512, residual_channels)
        self.conditioner_projection = Conv1d(
            condition_dim, 2 * residual_channels, 1
        )
        self.speaker_projection = nn.Linear(
            speaker_embed_dim, 2 * residual_channels
        )
        self.output_projection = Conv1d(
            residual_channels, 2 * residual_channels, 1
        )

    def forward(
        self,
        x: torch.Tensor,
        diffusion_step: torch.Tensor,
        token_condition: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        diffusion_step = self.diffusion_projection(diffusion_step).unsqueeze(-1)
        speaker_condition = self.speaker_projection(speaker_embedding).unsqueeze(-1)
        y = self.dilated_conv(x + diffusion_step)
        y = y + self.conditioner_projection(token_condition) + speaker_condition
        gate, filter_value = torch.chunk(y, 2, dim=1)
        y = torch.sigmoid(gate) * torch.tanh(filter_value)
        residual, skip = torch.chunk(self.output_projection(y), 2, dim=1)
        return (x + residual) / sqrt(2.0), skip


class TokenConditionedDiffWave(nn.Module):
    """DiffWave conditioned directly on multi-layer codec tokens and speaker identity."""

    def __init__(self, cfg) -> None:
        super().__init__()
        self.num_quantizers = int(cfg.num_quantizers)
        self.codebook_size = int(cfg.codebook_size)
        self.codec_hop_size = int(cfg.codec_hop_size)
        self.speaker_embed_dim = int(cfg.speaker_embed_dim)
        self.residual_channels = int(cfg.residual_channels)
        self.train_steps = int(cfg.train_steps)

        self.token_conditioner = MultiCodebookConditioner(
            self.num_quantizers,
            self.codebook_size,
            int(cfg.condition_dim),
        )
        self.input_projection = Conv1d(1, self.residual_channels, 1)
        self.diffusion_embedding = DiffusionEmbedding(self.train_steps)
        self.residual_layers = nn.ModuleList(
            [
                TokenResidualBlock(
                    int(cfg.condition_dim),
                    self.speaker_embed_dim,
                    self.residual_channels,
                    2 ** (index % int(cfg.dilation_cycle_length)),
                )
                for index in range(int(cfg.residual_layers))
            ]
        )
        self.skip_projection = Conv1d(
            self.residual_channels, self.residual_channels, 1
        )
        self.output_projection = Conv1d(self.residual_channels, 1, 1)
        nn.init.zeros_(self.output_projection.weight)

        betas = torch.linspace(
            float(cfg.beta_start), float(cfg.beta_end), self.train_steps
        )
        alphas = 1.0 - betas
        self.register_buffer("betas", betas)
        self.register_buffer("alphas", alphas)
        self.register_buffer("alpha_cumprod", torch.cumprod(alphas, dim=0))

    def forward(
        self,
        audio: torch.Tensor,
        diffusion_step: torch.Tensor,
        tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        if audio.ndim != 2:
            raise ValueError(f"audio must have shape [B, S], got {audio.shape}")
        speaker_embedding = F.normalize(speaker_embedding.float(), dim=-1)
        token_condition = self.token_conditioner(tokens, audio.shape[-1])
        x = F.relu(self.input_projection(audio.unsqueeze(1)))
        diffusion_embedding = self.diffusion_embedding(diffusion_step)

        skip = None
        for layer in self.residual_layers:
            x, skip_connection = layer(
                x,
                diffusion_embedding,
                token_condition,
                speaker_embedding,
            )
            skip = skip_connection if skip is None else skip + skip_connection
        x = skip / sqrt(len(self.residual_layers))
        x = F.relu(self.skip_projection(x))
        return self.output_projection(x).squeeze(1)

    def training_loss(
        self,
        clean_audio: torch.Tensor,
        tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
    ) -> torch.Tensor:
        batch_size = clean_audio.shape[0]
        diffusion_step = torch.randint(
            0,
            self.train_steps,
            (batch_size,),
            device=clean_audio.device,
        )
        alpha_bar = self.alpha_cumprod[diffusion_step].unsqueeze(-1)
        noise = torch.randn_like(clean_audio)
        noisy_audio = alpha_bar.sqrt() * clean_audio + (1.0 - alpha_bar).sqrt() * noise
        predicted_noise = self(
            noisy_audio,
            diffusion_step,
            tokens,
            speaker_embedding,
        )
        return F.mse_loss(predicted_noise, noise)

    @torch.inference_mode()
    def sample(
        self,
        tokens: torch.Tensor,
        speaker_embedding: torch.Tensor,
        output_length: int | None = None,
    ) -> torch.Tensor:
        if output_length is None:
            output_length = tokens.shape[1] * self.codec_hop_size
        audio = torch.randn(
            tokens.shape[0],
            output_length,
            device=tokens.device,
        )
        for step in range(self.train_steps - 1, -1, -1):
            diffusion_step = torch.full(
                (tokens.shape[0],), step, device=tokens.device, dtype=torch.long
            )
            predicted_noise = self(
                audio,
                diffusion_step,
                tokens,
                speaker_embedding,
            )
            beta = self.betas[step]
            alpha = self.alphas[step]
            alpha_bar = self.alpha_cumprod[step]
            mean = (audio - beta * predicted_noise / (1.0 - alpha_bar).sqrt()) / alpha.sqrt()
            if step > 0:
                audio = mean + beta.sqrt() * torch.randn_like(audio)
            else:
                audio = mean
        return audio.clamp(-1.0, 1.0)
