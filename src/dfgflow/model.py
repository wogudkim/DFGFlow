from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

from .config import DirectDiffConfig


BRANCH_MODES = {"both", "direct", "differential"}
GATE_MODES = {"sigmoid", "tanh", "random", "zero", "one", "fixed", "lambda", "learned", "score"}


class ConvEncoder(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_dim: int,
        freq_bins: int,
        time_bins: int,
        hidden: int = 64,
    ) -> None:
        super().__init__()
        # Two stride-2 stages retain a 16x16 grid for the 64x64 datasets.
        # The former third stride-2 stage plus 4x4 pooling reduced 64x64 all
        # the way to 4x4 before the latent projection.
        grid_h = max(1, (freq_bins + 3) // 4)
        grid_w = max(1, (time_bins + 3) // 4)
        self.net = nn.Sequential(
            nn.Conv2d(in_channels, hidden, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, hidden * 2, 3, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden * 2, hidden * 4, 3, stride=1, padding=1),
            nn.SiLU(),
            nn.AdaptiveAvgPool2d((grid_h, grid_w)),
            nn.Flatten(),
            nn.Linear(hidden * 4 * grid_h * grid_w, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SpectrogramDecoder(nn.Module):
    def __init__(self, in_dim: int, channels: int, freq_bins: int, time_bins: int, hidden: int = 64) -> None:
        super().__init__()
        self.freq_bins = freq_bins
        self.time_bins = time_bins
        # Mirror the encoder's total 4x spatial reduction. For the standard
        # 64x64 representation this decodes from 16x16 with two learned
        # upsampling stages instead of reconstructing from an 8x8 seed.
        self.seed_h = max(1, (freq_bins + 3) // 4)
        self.seed_w = max(1, (time_bins + 3) // 4)
        self.fc = nn.Sequential(nn.Linear(in_dim, hidden * 4 * self.seed_h * self.seed_w), nn.SiLU())
        self.net = nn.Sequential(
            nn.ConvTranspose2d(hidden * 4, hidden * 2, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.ConvTranspose2d(hidden * 2, hidden, 4, stride=2, padding=1),
            nn.SiLU(),
            nn.Conv2d(hidden, channels, 3, padding=1),
        )

    def forward(self, z: torch.Tensor) -> torch.Tensor:
        x = self.fc(z).view(z.shape[0], -1, self.seed_h, self.seed_w)
        x = self.net(x)
        return F.interpolate(x, size=(self.freq_bins, self.time_bins), mode="bilinear", align_corners=False)


class GaussianHead(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(), nn.Linear(hidden, hidden), nn.SiLU())
        self.mu = nn.Linear(hidden, out_dim)
        self.logvar = nn.Linear(hidden, out_dim)

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        h = self.net(x)
        return self.mu(h), self.logvar(h).clamp(-6.0, 4.0)


class MLP(nn.Module):
    def __init__(self, in_dim: int, out_dim: int, hidden: int) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.SiLU(),
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Linear(hidden, out_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class SwiGLU(nn.Module):
    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.proj = nn.Linear(in_dim, out_dim * 2)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        value, gate = self.proj(x).chunk(2, dim=-1)
        return value * F.silu(gate)


def sample_gaussian(mu: torch.Tensor, logvar: torch.Tensor, temperature: float = 1.0) -> torch.Tensor:
    return mu + torch.randn_like(mu) * torch.exp(0.5 * logvar) * temperature


class ConditionalFlowMatcher(nn.Module):
    def __init__(self, cfg: DirectDiffConfig) -> None:
        super().__init__()
        self.cfg = cfg
        self.channel_token_mode = cfg.token_mode == "channel"
        cond_dim = cfg.channel_embed_dim if cfg.num_condition_channels > 0 else 0
        self.topk_ratio = cfg.cfm_topk_ratio
        self.base_attention = nn.MultiheadAttention(
            cfg.direct_dim, cfg.cfm_heads, dropout=cfg.attn_dropout, batch_first=True
        )
        self.base_norm = nn.LayerNorm(cfg.direct_dim)
        self.base_ffn = MLP(cfg.direct_dim, cfg.direct_dim, cfg.hidden_dim)
        # Differential residual attention (DRA), corresponding to Eqs. (13)--(15):
        # log-variation FFN -> self-attention -> stacked residual
        # cross-attention -> GRU.
        self.diff_norm = nn.LayerNorm(cfg.direct_dim)
        self.log_variation_ffn = MLP(cfg.direct_dim, cfg.direct_dim, cfg.hidden_dim)
        self.diff_self_attention = nn.MultiheadAttention(
            cfg.direct_dim, cfg.cfm_heads, dropout=cfg.attn_dropout, batch_first=True
        )
        self.diff_cross_attentions = nn.ModuleList(
            [
                nn.MultiheadAttention(
                    cfg.direct_dim,
                    cfg.cfm_heads,
                    dropout=cfg.attn_dropout,
                    batch_first=True,
                )
                for _ in range(cfg.dra_cross_layers)
            ]
        )
        self.gate_attention = nn.MultiheadAttention(
            cfg.direct_dim, cfg.cfm_heads, dropout=cfg.attn_dropout, batch_first=True
        )
        self.diff_gru = nn.GRU(cfg.direct_dim, cfg.direct_dim, batch_first=True)
        self.context_gate = SwiGLU(cfg.direct_dim * 2, cfg.direct_dim)
        context_width = cfg.direct_dim * (2 if cfg.cfm_injection == "concat" else 1)
        self.net = nn.Sequential(
            nn.Linear(cfg.direct_dim + context_width + 2 + cond_dim, cfg.hidden_dim),
            nn.LayerNorm(cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.hidden_dim),
            nn.SiLU(),
            nn.Linear(cfg.hidden_dim, cfg.direct_dim),
        )

    @staticmethod
    def adjacent_residual(tokens: torch.Tensor, group_size: int | None = None) -> torch.Tensor:
        residual = torch.zeros_like(tokens)
        residual[:, 1:] = tokens[:, 1:] - tokens[:, :-1]
        if group_size is not None and group_size > 0:
            residual[:, ::group_size] = 0.0
        return residual

    def temporal_group_size(self, token_count: int) -> int | None:
        if not self.channel_token_mode:
            return None
        channel_count = max(1, self.cfg.num_condition_channels)
        if token_count % channel_count != 0:
            raise ValueError(
                f"Channel-token count {token_count} is not divisible by {channel_count} channels."
            )
        return token_count // channel_count

    def flow_contexts(
        self,
        z_t: torch.Tensor,
        use_base: bool = True,
        use_differential: bool = True,
        gate_mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        base_input = self.base_norm(z_t)
        base_attended, _ = self.base_attention(base_input, base_input, base_input, need_weights=False)
        base = z_t + base_attended
        base = base + self.base_ffn(self.base_norm(base))

        residual = self.adjacent_residual(z_t, self.temporal_group_size(z_t.shape[1]))
        normalized = self.diff_norm(residual)
        # Eq. (13): a real-valued signed log compression and feature-axis
        # variance of the normalized residual, followed by an FFN. log1p keeps
        # the transform defined at zero and for negative residuals.
        log_residual = normalized.sign() * torch.log1p(normalized.abs())
        feature_variance = normalized.var(dim=-1, keepdim=True, unbiased=False)
        log_variation = (self.log_variation_ffn(log_residual + feature_variance)
                         if self.cfg.cfm_log_variation else torch.zeros_like(residual))

        # Eq. (14): self-attention initializes Y, then each cross-attention
        # stage adds a residual update while attending to Delta H.
        attended = residual
        if self.cfg.cfm_dra_attention:
            attended, _ = self.diff_self_attention(
                residual, residual, residual, need_weights=False
            )
            for cross_attention in self.diff_cross_attentions:
                update, _ = cross_attention(attended, residual, residual, need_weights=False)
                attended = attended + update
        combined = log_variation + attended
        differential = self.diff_gru(combined)[0] if self.cfg.cfm_gru else combined

        mode = DirectDiffModel.normalize_gate_mode(gate_mode or self.cfg.gate_mode)
        if mode == "zero" or not use_differential:
            selected_differential = torch.zeros_like(differential)
            sparse_mask = torch.zeros_like(differential)
        elif mode == "one":
            selected_differential = differential
            sparse_mask = torch.ones_like(differential)
        elif mode == "lambda":
            selected_differential = differential * float(self.cfg.gate_lambda)
            sparse_mask = torch.ones_like(differential)
        elif mode == "random":
            selected_differential = differential * (2.0 * torch.rand_like(differential))
            sparse_mask = torch.ones_like(differential)
        else:
            pooled = differential.mean(dim=1, keepdim=True).expand_as(differential)
            contextual = differential * pooled
            k = max(1, min(contextual.shape[-1], round(contextual.shape[-1] * self.topk_ratio)))
            sparse_mask = torch.ones_like(contextual)
            if self.cfg.cfm_topk:
                topk_indices = contextual.abs().topk(k, dim=-1).indices
                sparse_mask = torch.zeros_like(contextual).scatter_(-1, topk_indices, 1.0)
            sparse_context = contextual * sparse_mask
            refined_context = sparse_context
            if self.cfg.cfm_gate_attention:
                refined_context, _ = self.gate_attention(
                    sparse_context, differential, differential, need_weights=False
                )
            scores = self.context_gate(torch.cat([refined_context, differential], dim=-1))
            weights = 1.0 + torch.tanh(scores) if mode == "tanh" else torch.sigmoid(scores)
            if not self.cfg.cfm_learned_gate:
                weights = torch.ones_like(weights)
            selected_differential = differential * weights * sparse_mask

        if not use_base:
            base = torch.zeros_like(base)
        return base, selected_differential, sparse_mask

    def forward(
        self,
        z_t: torch.Tensor,
        t: torch.Tensor,
        steps: torch.Tensor,
        cond: torch.Tensor | None = None,
        use_base: bool = True,
        use_differential: bool = True,
        gate_mode: str | None = None,
    ) -> torch.Tensor:
        base, selected_differential, _ = self.flow_contexts(
            z_t, use_base, use_differential, gate_mode
        )
        if self.cfg.cfm_injection == "concat":
            contexts = [base, selected_differential]
        elif self.cfg.cfm_injection == "add":
            contexts = [base + selected_differential]
        else:
            raise ValueError(f"Unknown CFM injection: {self.cfg.cfm_injection}")
        parts = [z_t, t, steps, *contexts]
        if cond is not None:
            parts.append(cond)
        x = torch.cat(parts, dim=-1)
        v = self.net(x.reshape(-1, x.shape[-1]))
        return v.view_as(z_t)


class DirectDiffModel(nn.Module):
    """Differential feature-guided latent flow model.

    The base path encodes the target trajectory. The differential path encodes
    adjacent time-frequency residuals and applies sparse token-space guidance.
    """

    def __init__(self, cfg: DirectDiffConfig) -> None:
        super().__init__()
        self.cfg = cfg
        channels, freq_bins, time_bins = cfg.spec_shape
        if cfg.token_mode not in {"pair", "channel"}:
            raise ValueError(f"token_mode must be pair or channel, got {cfg.token_mode}")
        self.channel_token_mode = cfg.token_mode == "channel"
        if self.channel_token_mode:
            if channels % 2 != 0:
                raise ValueError("channel token mode expects complex channels arranged as [real0, imag0, real1, imag1, ...].")
            if cfg.num_condition_channels <= 0:
                cfg.num_condition_channels = channels // 2
            encoder_channels = 2
            differential_channels = 2
            decoder_channels = 2
        else:
            encoder_channels = channels
            differential_channels = channels * 2
            decoder_channels = channels
        cond_dim = cfg.channel_embed_dim if cfg.num_condition_channels > 0 else 0
        self.channel_embedding = nn.Embedding(cfg.num_condition_channels, cfg.channel_embed_dim) if cfg.num_condition_channels > 0 else None
        self.channel_to_direct = nn.Linear(cfg.channel_embed_dim, cfg.direct_dim) if cfg.num_condition_channels > 0 else None
        self.shared_encoder = ConvEncoder(
            encoder_channels, cfg.hidden_dim, freq_bins, time_bins
        )
        self.direct_branch = MLP(cfg.hidden_dim + 1 + cond_dim, cfg.direct_dim, cfg.hidden_dim)
        self.channel_base_branch = (
            MLP(cfg.direct_dim * 2 + 1, cfg.direct_dim, cfg.hidden_dim) if self.channel_token_mode else None
        )
        self.differential_encoder = ConvEncoder(
            differential_channels, cfg.hidden_dim, freq_bins, time_bins
        )
        self.differential_posterior = GaussianHead(
            cfg.hidden_dim + cfg.direct_dim + 1 + cond_dim,
            cfg.differential_dim,
            cfg.hidden_dim,
        )
        self.differential_projector = MLP(cfg.differential_dim, cfg.direct_dim, cfg.hidden_dim)
        self.differential_gru = nn.GRU(cfg.direct_dim, cfg.direct_dim, batch_first=True)
        self.gate_mlp = SwiGLU(cfg.direct_dim * 2 + 3, cfg.direct_dim)
        self.cfm = ConditionalFlowMatcher(cfg)
        self.out_decoder = SpectrogramDecoder(cfg.direct_dim, decoder_channels, freq_bins, time_bins)
        self.differential_decoder = SpectrogramDecoder(
            cfg.direct_dim, decoder_channels, freq_bins, time_bins
        )
        # learnable scalar to allow the model to scale gate outputs (helps when sigmoid range is too small)
        self.gate_scale = nn.Parameter(torch.tensor(1.0))
        self.register_buffer("flow_latent_mean", torch.zeros(1, 1, cfg.direct_dim))
        self.register_buffer("flow_latent_var", torch.ones(1, 1, cfg.direct_dim))
        self.register_buffer("flow_latent_updates", torch.zeros((), dtype=torch.long))

        # Attention modules: base self-attention (MLC) and diff cross-attention (MVC)
        self.base_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(cfg.direct_dim, cfg.attn_num_heads, dropout=cfg.attn_dropout, batch_first=False)
            for _ in range(cfg.attn_num_layers)
        ])
        self.base_attn_norms = nn.ModuleList([nn.LayerNorm(cfg.direct_dim) for _ in range(cfg.attn_num_layers)])

        self.diff_self_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(cfg.direct_dim, cfg.attn_num_heads, dropout=cfg.attn_dropout, batch_first=False)
            for _ in range(cfg.attn_num_layers)
        ])
        self.diff_self_attn_norms = nn.ModuleList([nn.LayerNorm(cfg.direct_dim) for _ in range(cfg.attn_num_layers)])

        self.diff_cross_attn_layers = nn.ModuleList([
            nn.MultiheadAttention(cfg.direct_dim, cfg.attn_num_heads, dropout=cfg.attn_dropout, batch_first=False)
            for _ in range(cfg.attn_num_layers)
        ])
        self.diff_cross_attn_norms = nn.ModuleList([nn.LayerNorm(cfg.direct_dim) for _ in range(cfg.attn_num_layers)])

        # Positional encodings (learnable)
        if self.channel_token_mode:
            max_tokens = getattr(cfg, 'seq_len', 16) * max(1, getattr(cfg, 'num_condition_channels', 1))
        else:
            max_tokens = getattr(cfg, 'seq_len', 16)
        self.pos_embed = nn.Parameter(torch.randn(max_tokens, 1, cfg.direct_dim) * 0.02)

    def apply_base_self_attention(self, tokens: torch.Tensor) -> tuple[torch.Tensor, list]:
        """Apply self-attention to base tokens (MLC).

        Args:
            tokens: (B, T, D)
        Returns:
            attended tokens: (B, T, D), attn weights
        """
        b, t, d = tokens.shape
        x = self.add_positional_encoding(tokens).permute(1, 0, 2)  # (T, B, D)
        attn_weights_all = []

        for attn_layer, norm_layer in zip(self.base_attn_layers, self.base_attn_norms):
            x_norm = norm_layer(x)
            x_attn, attn_w = attn_layer(x_norm, x_norm, x_norm, average_attn_weights=False)
            x = x + x_attn
            attn_weights_all.append(attn_w)

        return x.permute(1, 0, 2), attn_weights_all  # (B, T, D)

    def apply_diff_self_attention(self, tokens: torch.Tensor) -> torch.Tensor:
        """Apply self-attention to differential tokens (MVC self)."""
        b, t, d = tokens.shape
        x = self.add_positional_encoding(tokens).permute(1, 0, 2)  # (T, B, D)

        for attn_layer, norm_layer in zip(self.diff_self_attn_layers, self.diff_self_attn_norms):
            x_norm = norm_layer(x)
            x_attn, _ = attn_layer(x_norm, x_norm, x_norm, average_attn_weights=False)
            x = x + x_attn

        return x.permute(1, 0, 2)  # (B, T, D)

    def apply_diff_cross_attention(self, diff_tokens: torch.Tensor, base_tokens: torch.Tensor) -> tuple[torch.Tensor, list]:
        """Apply cross-attention where queries=diff, keys/values=base (MVC cross).

        Returns:
            attended diff tokens, cross-attn weights (for gate_hint score extraction)
        """
        b, t, d = diff_tokens.shape
        q = self.add_positional_encoding(diff_tokens).permute(1, 0, 2)  # (T, B, D)
        kv = self.add_positional_encoding(base_tokens).permute(1, 0, 2)  # (T, B, D)

        cross_attn_weights_all = []
        for attn_layer, norm_layer in zip(self.diff_cross_attn_layers, self.diff_cross_attn_norms):
            q_norm = norm_layer(q)
            cross_out, cross_w = attn_layer(q_norm, kv, kv, average_attn_weights=False)
            q = q + cross_out
            cross_attn_weights_all.append(cross_w)

        return q.permute(1, 0, 2), cross_attn_weights_all  # (B, T, D)

    def add_positional_encoding(self, tokens: torch.Tensor) -> torch.Tensor:
        token_count = tokens.shape[1]
        if token_count <= self.pos_embed.shape[0]:
            pos = self.pos_embed[:token_count].permute(1, 0, 2)
        else:
            pos = F.interpolate(
                self.pos_embed.permute(1, 2, 0),
                size=token_count,
                mode="linear",
                align_corners=False,
            ).permute(0, 2, 1)
        return tokens + pos.to(device=tokens.device, dtype=tokens.dtype)

    def common_input(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return 0.5 * (source + target)

    def channel_condition(
        self,
        channel_ids: torch.Tensor | None,
        batch: int,
        seq_len: int,
        device: torch.device,
    ) -> tuple[torch.Tensor | None, torch.Tensor]:
        if self.channel_embedding is None:
            return None, torch.zeros(batch, seq_len, self.cfg.direct_dim, device=device)
        if channel_ids is None:
            channel_ids = torch.zeros(batch, seq_len, dtype=torch.long, device=device)
        channel_ids = channel_ids.to(device=device, dtype=torch.long)
        if channel_ids.ndim == 0:
            channel_ids = channel_ids.view(1).expand(batch)
        if channel_ids.ndim == 1:
            if channel_ids.numel() == batch:
                channel_ids = channel_ids[:, None].expand(batch, seq_len)
            elif channel_ids.numel() == seq_len:
                channel_ids = channel_ids[None, :].expand(batch, seq_len)
            else:
                raise ValueError(f"channel_ids length must be batch or seq_len, got {channel_ids.numel()}.")
        elif channel_ids.shape != (batch, seq_len):
            channel_ids = channel_ids.reshape(batch, seq_len)
        emb = self.channel_embedding(channel_ids)
        token = self.channel_to_direct(emb.reshape(batch * seq_len, -1)).view(batch, seq_len, -1)
        return emb, token

    def channel_token_ids(self, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        channel_count = self.cfg.num_condition_channels
        ids = torch.arange(channel_count, device=device, dtype=torch.long)
        return ids[None, :, None].expand(batch, channel_count, seq_len).reshape(
            batch, channel_count * seq_len
        )

    def channel_token_steps(self, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        channel_count = self.cfg.num_condition_channels
        step = torch.linspace(0.0, 1.0, seq_len, device=device)
        return step[None, None, :, None].expand(batch, channel_count, seq_len, 1).reshape(
            batch, channel_count * seq_len, 1
        )

    def split_channel_tokens(self, specs: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int]:
        b, seq_len, channels, freq_bins, time_bins = specs.shape
        if channels != self.cfg.channels:
            raise ValueError(f"Expected {self.cfg.channels} channels, got {channels}.")
        if channels % 2 != 0:
            raise ValueError("Expected complex channels arranged as real/imag pairs.")
        channel_count = channels // 2
        # Channel-major order keeps consecutive tokens on the same temporal
        # trajectory: [ch0_t0, ch0_t1, ..., ch1_t0, ch1_t1, ...].
        tokens = specs.view(b, seq_len, channel_count, 2, freq_bins, time_bins)
        tokens = tokens.permute(0, 2, 1, 3, 4, 5).contiguous()
        tokens = tokens.reshape(b, seq_len * channel_count, 2, freq_bins, time_bins)
        return tokens, self.channel_token_ids(b, seq_len, specs.device), seq_len

    def merge_channel_tokens(self, tokens: torch.Tensor, seq_len: int) -> torch.Tensor:
        b, token_count, two, freq_bins, time_bins = tokens.shape
        channel_count = self.cfg.num_condition_channels
        if two != 2:
            raise ValueError(f"Expected decoded real/imag token channels=2, got {two}.")
        expected = seq_len * channel_count
        if token_count != expected:
            raise ValueError(f"Expected {expected} channel tokens, got {token_count}.")
        tokens = tokens.view(b, channel_count, seq_len, 2, freq_bins, time_bins)
        tokens = tokens.permute(0, 2, 1, 3, 4, 5).contiguous()
        return tokens.reshape(b, seq_len, channel_count * 2, freq_bins, time_bins)

    def decode_channel_tokens(self, tokens: torch.Tensor, seq_len: int) -> torch.Tensor:
        return self.merge_channel_tokens(self.decode_direct(tokens), seq_len)

    @torch.no_grad()
    def update_flow_latent_stats(self, tokens: torch.Tensor) -> None:
        """Track target moments so CFM sees a stable, standardized endpoint."""
        if not self.cfg.normalize_flow_latent or not self.training:
            return
        values = tokens.detach().float()
        mean = values.mean(dim=(0, 1), keepdim=True)
        var = values.var(dim=(0, 1), keepdim=True, unbiased=False).clamp_min(
            self.cfg.flow_latent_eps
        )
        if self.flow_latent_updates.item() == 0:
            self.flow_latent_mean.copy_(mean)
            self.flow_latent_var.copy_(var)
        else:
            momentum = float(self.cfg.flow_latent_momentum)
            self.flow_latent_mean.lerp_(mean, momentum)
            self.flow_latent_var.lerp_(var, momentum)
        self.flow_latent_updates.add_(1)

    def normalize_flow_token(self, tokens: torch.Tensor, update: bool = False) -> torch.Tensor:
        if not self.cfg.normalize_flow_latent:
            return tokens
        if update:
            self.update_flow_latent_stats(tokens)
        mean = self.flow_latent_mean.to(device=tokens.device, dtype=tokens.dtype)
        scale = self.flow_latent_var.to(device=tokens.device, dtype=tokens.dtype).add(
            self.cfg.flow_latent_eps
        ).sqrt()
        return (tokens - mean) / scale

    def denormalize_flow_token(self, tokens: torch.Tensor) -> torch.Tensor:
        if not self.cfg.normalize_flow_latent:
            return tokens
        mean = self.flow_latent_mean.to(device=tokens.device, dtype=tokens.dtype)
        scale = self.flow_latent_var.to(device=tokens.device, dtype=tokens.dtype).add(
            self.cfg.flow_latent_eps
        ).sqrt()
        return tokens * scale + mean

    def encode_channel_direct(self, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, int]:
        tiles, channel_ids, seq_len = self.split_channel_tokens(target)
        b, n, c, f, w = tiles.shape
        steps = self.channel_token_steps(b, seq_len, target.device)
        cond, cond_token = self.channel_condition(channel_ids, b, n, target.device)
        shared = self.shared_encoder(tiles.reshape(b * n, c, f, w)).view(b, n, -1)
        direct_input = [shared, steps]
        if cond is not None:
            direct_input.append(cond)
        raw = self.direct_branch(torch.cat(direct_input, dim=-1).reshape(b * n, -1)).view(b, n, -1)
        channel_raw = raw + cond_token

        # Apply base self-attention (MLC)
        channel_raw, _ = self.apply_base_self_attention(channel_raw)

        global_context = channel_raw.mean(dim=1, keepdim=True).expand_as(channel_raw)
        base_input = torch.cat([channel_raw, global_context, steps], dim=-1)
        if self.channel_base_branch is None:
            raise RuntimeError("channel_base_branch is only available in channel token mode.")
        base_delta = self.channel_base_branch(base_input.reshape(b * n, -1)).view(b, n, -1)
        return channel_raw + base_delta, tiles, channel_ids, seq_len

    def channel_variation_features(self, tiles: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        difference = self.temporal_difference(tiles)
        variation = torch.log1p(difference.abs() / (tiles.abs() + eps))
        return difference + 0.25 * variation

    def forward_channel_tokens(
        self,
        target: torch.Tensor,
        sample_posterior: bool,
        branch_mode: str,
        gate_mode: str,
    ) -> dict[str, torch.Tensor]:
        direct, tiles, channel_ids, seq_len = self.encode_channel_direct(target)
        direct_base = self.decode_channel_tokens(direct, seq_len)
        zero_difference = torch.zeros_like(target)
        zero_direct = torch.zeros_like(direct)
        zero_diff = target.new_zeros(*direct.shape[:2], self.cfg.differential_dim)
        target_difference = self.merge_channel_tokens(self.temporal_difference(tiles), seq_len)

        if branch_mode == "direct":
            zero_gate = torch.zeros_like(direct)
            z_target = direct
            cfm_target = self.normalize_flow_token(z_target, update=True)
            z0, z_t, velocity_target, velocity_pred, one_step_token = self.cfm_training_step(
                cfm_target, channel_ids, branch_mode, gate_mode
            )
            prior_reconstruction = self.decode_channel_tokens(
                self.denormalize_flow_token(one_step_token), seq_len
            )
            reconstruction = direct_base
            return {
                "direct": direct,
                "direct_mu": zero_direct,
                "direct_logvar": zero_direct,
                "direct_base": direct_base,
                "target_difference": target_difference,
                "target_transition": target_difference,
                "q_mu": zero_diff,
                "q_logvar": zero_diff,
                "p_mu": zero_diff,
                "p_logvar": zero_diff,
                "transition_hat": reconstruction - direct_base,
                "gate": zero_gate,
                "gate_hint": zero_gate,
                "effective_transition": zero_difference,
                "reconstruction": reconstruction,
                "prior_transition": zero_difference,
                "prior_gate": zero_gate,
                "prior_effective_transition": zero_difference,
                "prior_transition_hat": prior_reconstruction - direct_base,
                "differential_token": torch.zeros_like(direct),
                "prior_differential_token": torch.zeros_like(direct),
                "differential_base": zero_difference,
                "prior_differential_base": zero_difference,
                "z_target": z_target,
                "cfm_target_token": cfm_target,
                "cfm_z0": z0,
                "cfm_z_t": z_t,
                "cfm_velocity_target": velocity_target,
                "cfm_velocity_pred": velocity_pred,
                "cfm_endpoint_token": one_step_token,
                "additive_reconstruction": reconstruction,
                "prior_reconstruction": prior_reconstruction,
            }

        direct_context = torch.zeros_like(direct) if branch_mode == "differential" else direct
        features = self.channel_variation_features(tiles)
        q_mu, q_logvar = self.encode_differential(features, direct_context, channel_ids)
        differential = sample_gaussian(q_mu, q_logvar) if sample_posterior else q_mu
        differential_token = self.project_differential(differential)
        differential_base = self.merge_channel_tokens(
            self.decode_differential(differential_token), seq_len
        )
        z_target, gate, gate_hint = self.gate_tokens(direct_context, differential_token, branch_mode, gate_mode)
        reconstruction = self.decode_channel_tokens(z_target, seq_len)
        cfm_target = self.normalize_flow_token(z_target, update=True)
        z0, z_t, velocity_target, velocity_pred, one_step_token = self.cfm_training_step(
            cfm_target, channel_ids, branch_mode, gate_mode
        )
        prior_reconstruction = self.decode_channel_tokens(
            self.denormalize_flow_token(one_step_token), seq_len
        )
        return {
            "direct": direct,
            "direct_mu": zero_direct,
            "direct_logvar": zero_direct,
            "direct_base": direct_base,
            "target_difference": target_difference,
            "target_transition": target_difference,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": torch.zeros_like(q_mu),
            "p_logvar": torch.zeros_like(q_logvar),
            "transition_hat": reconstruction - direct_base,
            "gate": gate,
            "gate_hint": gate_hint,
            "effective_transition": zero_difference,
            "differential_token": differential_token,
            "differential_base": differential_base,
            "reconstruction": reconstruction,
            "prior_transition": zero_difference,
            "prior_gate": gate,
            "prior_effective_transition": zero_difference,
            "prior_transition_hat": prior_reconstruction - direct_base,
            "prior_differential_token": torch.zeros_like(differential_token),
            "prior_differential_base": zero_difference,
            "z_target": z_target,
            "cfm_target_token": cfm_target,
            "cfm_z0": z0,
            "cfm_z_t": z_t,
            "cfm_velocity_target": velocity_target,
            "cfm_velocity_pred": velocity_pred,
            "cfm_endpoint_token": one_step_token,
            "additive_reconstruction": reconstruction,
            "prior_reconstruction": prior_reconstruction,
        }

    def encode_direct(self, source: torch.Tensor, target: torch.Tensor, channel_ids: torch.Tensor | None = None) -> torch.Tensor:
        b, n, c, f, w = target.shape
        steps = self._step_column(b, n, target.device)
        cond, cond_token = self.channel_condition(channel_ids, b, n, target.device)
        common = self.common_input(source, target)
        shared = self.shared_encoder(common.reshape(b * n, c, f, w)).view(b, n, -1)
        direct_input = [shared, steps]
        if cond is not None:
            direct_input.append(cond)
        direct = self.direct_branch(torch.cat(direct_input, dim=-1).reshape(b * n, -1))
        direct = direct.view(b, n, -1) + cond_token
        direct, _ = self.apply_base_self_attention(direct)
        return direct

    def decode_direct(self, direct: torch.Tensor) -> torch.Tensor:
        b, n, d = direct.shape
        out = self.out_decoder(direct.reshape(b * n, d))
        return out.view(b, n, *out.shape[1:])

    def decode_differential(self, differential: torch.Tensor) -> torch.Tensor:
        b, n, d = differential.shape
        out = self.differential_decoder(differential.reshape(b * n, d))
        return out.view(b, n, *out.shape[1:])

    def encode_differential(
        self,
        differential_features: torch.Tensor,
        direct: torch.Tensor,
        channel_ids: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        b, n, c, f, w = differential_features.shape
        steps = self._step_column(b, n, differential_features.device)
        cond, _ = self.channel_condition(channel_ids, b, n, differential_features.device)
        transition_context = self.differential_encoder(differential_features.reshape(b * n, c, f, w)).view(b, n, -1)
        posterior_parts = [transition_context, direct, steps]
        if cond is not None:
            posterior_parts.append(cond)
        posterior_in = torch.cat(posterior_parts, dim=-1)
        mu, logvar = self.differential_posterior(posterior_in.reshape(b * n, -1))
        return mu.view(b, n, -1), logvar.view(b, n, -1)

    def project_differential(self, differential: torch.Tensor) -> torch.Tensor:
        b, n, d = differential.shape
        projected = self.differential_projector(differential.reshape(b * n, d))
        projected = projected.view(b, n, -1)
        refined, _ = self.differential_gru(projected)
        return refined

    def gate_tokens(
        self,
        direct: torch.Tensor,
        differential_token: torch.Tensor,
        branch_mode: str,
        gate_mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Compute fused token with gating.

        Returns:
            fused tokens, gate weights, gate_hint for supervision (same shape as gate)
        """
        gate_mode = self.normalize_gate_mode(gate_mode or self.cfg.gate_mode)
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {sorted(GATE_MODES)}, got {gate_mode}")
        if branch_mode == "direct" or gate_mode == "zero":
            gate = torch.zeros_like(direct)
            gate_hint = torch.zeros_like(direct)
            return direct, gate, gate_hint
        if branch_mode == "differential":
            direct = torch.zeros_like(direct)
        if gate_mode in {"one", "fixed"}:
            gate = torch.ones_like(direct)
            gate_hint = torch.ones_like(direct)
        elif gate_mode == "lambda":
            gate = torch.full_like(direct, float(self.cfg.gate_lambda))
            gate_hint = gate
        elif gate_mode == "random":
            gate = torch.rand_like(direct) * 2.0
            gate_hint = torch.ones_like(direct)
        else:
            # Apply attention only in channel token mode
            if self.channel_token_mode:
                # Apply self-attention to differential tokens (MVC self)
                diff_attn = self.apply_diff_self_attention(differential_token)

                # Apply cross-attention where queries=diff, keys/values=base (MVC cross)
                diff_cross, cross_attn_weights = self.apply_diff_cross_attention(diff_attn, direct)

                differential_token_for_gate = diff_cross
            else:
                # Pair mode: no attention
                differential_token_for_gate = differential_token

            diff_score = differential_token.abs().mean(dim=-1, keepdim=True)
            direct_score = direct.abs().mean(dim=-1, keepdim=True)
            ratio = diff_score / (direct_score + 1e-6)
            gate_input = torch.cat([direct, differential_token_for_gate, diff_score, direct_score, ratio], dim=-1)
            raw_gate = self.gate_mlp(gate_input.reshape(-1, gate_input.shape[-1])).view_as(direct)
            score = self.score_gate(differential_token)
            k = max(1, min(raw_gate.shape[-1], round(raw_gate.shape[-1] * self.cfg.sparse_topk_ratio)))
            topk_indices = raw_gate.abs().topk(k, dim=-1).indices
            sparse_mask = torch.zeros_like(raw_gate).scatter_(-1, topk_indices, 1.0)
            if gate_mode == "tanh":
                # keep tanh behavior but modulate by score and learnable scale
                gate_core = 1.0 + torch.tanh(raw_gate)
                gate = self.gate_scale * score * gate_core * sparse_mask
            elif gate_mode == "score":
                # Top-K already determines sparsity. Keep selected coordinates
                # active instead of multiplying by a score that can collapse.
                gate = self.gate_scale * torch.sigmoid(raw_gate) * sparse_mask
            else:
                # sigmoid / learned behavior: learned gating modulated by score
                gate = self.gate_scale * score * torch.sigmoid(raw_gate) * sparse_mask

            gate_hint = sparse_mask

        fused = direct + gate * differential_token
        return fused, gate, gate_hint

    @staticmethod
    def temporal_difference(spectrograms: torch.Tensor) -> torch.Tensor:
        difference = torch.zeros_like(spectrograms)
        difference[..., 1:] = spectrograms[..., 1:] - spectrograms[..., :-1]
        return difference

    def pair_difference(self, source: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        return target - source

    def variation_features(self, source: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        difference = self.pair_difference(source, target)
        common = self.common_input(source, target)
        variation = torch.log1p(difference.abs() / (common.abs() + eps))
        return torch.cat([difference, variation], dim=2)

    def cfm_training_step(
        self,
        target_token: torch.Tensor,
        channel_ids: torch.Tensor | None = None,
        branch_mode: str = "both",
        gate_mode: str | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        # Flow fitting must never reshape the encoder distribution to reduce
        # its regression loss. The reconstruction objective owns that space.
        target_token = target_token.detach()
        if self.channel_token_mode:
            seq_len = target_token.shape[1] // self.cfg.num_condition_channels
            steps = self.channel_token_steps(target_token.shape[0], seq_len, target_token.device)
        else:
            steps = self._step_column(target_token.shape[0], target_token.shape[1], target_token.device)
        cond, _ = self.channel_condition(channel_ids, target_token.shape[0], target_token.shape[1], target_token.device)
        z0 = torch.randn_like(target_token)
        # One flow time per sequence. Expansion keeps every token in a sample
        # on the same conditional probability path used during generation.
        t = self.sample_sequence_flow_time(target_token)
        sigma_min = float(self.cfg.cfm_sigma_min)
        z_t = (1.0 - (1.0 - sigma_min) * t) * z0 + t * target_token
        velocity_target = target_token - (1.0 - sigma_min) * z0
        velocity_pred = self.cfm(
            z_t,
            t,
            steps,
            cond,
            use_base=branch_mode != "differential",
            use_differential=branch_mode != "direct",
            gate_mode=gate_mode,
        )
        one_step_token = z_t + (1.0 - t) * velocity_pred
        return z0, z_t, velocity_target, velocity_pred, one_step_token

    @staticmethod
    def sample_sequence_flow_time(target_token: torch.Tensor) -> torch.Tensor:
        batch_size, token_count, _ = target_token.shape
        return torch.rand(
            batch_size,
            1,
            1,
            device=target_token.device,
            dtype=target_token.dtype,
        ).expand(batch_size, token_count, 1)

    def forward(
        self,
        source: torch.Tensor,
        target: torch.Tensor | None = None,
        sample_posterior: bool = True,
        branch_mode: str | None = None,
        gate_mode: str | None = None,
        channel_ids: torch.Tensor | None = None,
    ) -> dict[str, torch.Tensor]:
        branch_mode = branch_mode or self.cfg.branch_mode
        gate_mode = self.normalize_gate_mode(gate_mode or self.cfg.gate_mode)
        if branch_mode not in BRANCH_MODES:
            raise ValueError(f"branch_mode must be one of {sorted(BRANCH_MODES)}, got {branch_mode}")
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {sorted(GATE_MODES)}, got {gate_mode}")
        if target is None:
            target = source
        if self.channel_token_mode:
            return self.forward_channel_tokens(target, sample_posterior, branch_mode, gate_mode)
        direct = self.encode_direct(source, target, channel_ids)
        direct_base = self.decode_direct(direct)
        common_target = self.common_input(source, target)
        target_difference = self.pair_difference(source, target)
        target_correction = target - common_target
        differential_features = self.variation_features(source, target)
        zero_difference = torch.zeros_like(target)
        zero_direct = torch.zeros_like(direct)
        zero_diff = target.new_zeros(*target.shape[:2], self.cfg.differential_dim)

        if branch_mode == "direct":
            zero_gate = torch.zeros_like(direct)
            z_target = direct
            cfm_target = self.normalize_flow_token(z_target, update=True)
            z0, z_t, velocity_target, velocity_pred, one_step_token = self.cfm_training_step(
                cfm_target, channel_ids, branch_mode, gate_mode
            )
            prior_reconstruction = self.decode_direct(
                self.denormalize_flow_token(one_step_token)
            )
            reconstruction = direct_base
            return {
                "direct": direct,
                "direct_mu": zero_direct,
                "direct_logvar": zero_direct,
                "direct_base": direct_base,
                "target_difference": target_difference,
                "target_transition": target_correction,
                "q_mu": zero_diff,
                "q_logvar": zero_diff,
                "p_mu": zero_diff,
                "p_logvar": zero_diff,
                "transition_hat": reconstruction - direct_base,
                "gate": zero_gate,
                "gate_hint": zero_gate,
                "effective_transition": zero_difference,
                "reconstruction": reconstruction,
                "prior_transition": zero_difference,
                "prior_gate": zero_gate,
                "prior_effective_transition": zero_difference,
                "prior_transition_hat": prior_reconstruction - direct_base,
                "differential_token": torch.zeros_like(direct),
                "prior_differential_token": torch.zeros_like(direct),
                "differential_base": zero_difference,
                "prior_differential_base": zero_difference,
                "z_target": z_target,
                "cfm_target_token": cfm_target,
                "cfm_z0": z0,
                "cfm_z_t": z_t,
                "cfm_velocity_target": velocity_target,
                "cfm_velocity_pred": velocity_pred,
                "cfm_endpoint_token": one_step_token,
                "additive_reconstruction": reconstruction,
                "prior_reconstruction": prior_reconstruction,
            }

        direct_context = torch.zeros_like(direct) if branch_mode == "differential" else direct
        q_mu, q_logvar = self.encode_differential(differential_features, direct_context, channel_ids)
        differential = sample_gaussian(q_mu, q_logvar) if sample_posterior else q_mu
        differential_token = self.project_differential(differential)
        differential_base = self.decode_differential(differential_token)
        additive_reconstruction = direct_base + differential_base
        z_target, gate, gate_hint = self.gate_tokens(direct_context, differential_token, branch_mode, gate_mode)
        reconstruction = self.decode_direct(z_target)
        cfm_target = self.normalize_flow_token(z_target, update=True)
        z0, z_t, velocity_target, velocity_pred, one_step_token = self.cfm_training_step(
            cfm_target, channel_ids, branch_mode, gate_mode
        )
        prior_reconstruction = self.decode_direct(
            self.denormalize_flow_token(one_step_token)
        )
        return {
            "direct": direct,
            "direct_mu": zero_direct,
            "direct_logvar": zero_direct,
            "direct_base": direct_base,
            "target_difference": target_difference,
            "target_transition": target_correction,
            "q_mu": q_mu,
            "q_logvar": q_logvar,
            "p_mu": torch.zeros_like(q_mu),
            "p_logvar": torch.zeros_like(q_logvar),
            "transition_hat": reconstruction - direct_base,
            "gate": gate,
            "gate_hint": gate_hint,
            "effective_transition": reconstruction - direct_base,
            "differential_token": differential_token,
            "differential_base": differential_base,
            "reconstruction": reconstruction,
            "prior_transition": zero_difference,
            "prior_gate": gate,
            "prior_effective_transition": prior_reconstruction - direct_base,
            "prior_transition_hat": prior_reconstruction - direct_base,
            "prior_differential_token": torch.zeros_like(differential_token),
            "prior_differential_base": zero_difference,
            "z_target": z_target,
            "cfm_target_token": cfm_target,
            "cfm_z0": z0,
            "cfm_z_t": z_t,
            "cfm_velocity_target": velocity_target,
            "cfm_velocity_pred": velocity_pred,
            "cfm_endpoint_token": one_step_token,
            "additive_reconstruction": additive_reconstruction,
            "prior_reconstruction": prior_reconstruction,
        }

    @torch.no_grad()
    def generate(
        self,
        batch_size: int,
        seq_len: int | None = None,
        device: torch.device | str = "cpu",
        direct_temperature: float = 1.0,
        differential_temperature: float = 1.0,
        branch_mode: str | None = None,
        gate_mode: str | None = None,
        sample_steps: int | None = None,
        solver: str | None = None,
        channel_ids: torch.Tensor | None = None,
    ) -> torch.Tensor:
        branch_mode = branch_mode or self.cfg.branch_mode
        gate_mode = self.normalize_gate_mode(gate_mode or self.cfg.gate_mode)
        if branch_mode not in BRANCH_MODES:
            raise ValueError(f"branch_mode must be one of {sorted(BRANCH_MODES)}, got {branch_mode}")
        if gate_mode not in GATE_MODES:
            raise ValueError(f"gate_mode must be one of {sorted(GATE_MODES)}, got {gate_mode}")
        seq_len = seq_len or self.cfg.seq_len
        device = torch.device(device)
        sample_steps = sample_steps or self.cfg.cfm_sample_steps
        solver = solver or self.cfg.cfm_solver
        if solver not in {"euler", "heun"}:
            raise ValueError(f"solver must be euler or heun, got {solver}")
        if self.channel_token_mode:
            token_count = seq_len * self.cfg.num_condition_channels
            channel_ids = self.channel_token_ids(batch_size, seq_len, device)
            steps = self.channel_token_steps(batch_size, seq_len, device)
        else:
            token_count = seq_len
            steps = self._step_column(batch_size, token_count, device)
        if branch_mode == "both" and direct_temperature != differential_temperature:
            raise ValueError(
                "The fused CFM has one Gaussian prior, so direct and differential temperatures "
                "must match in branch_mode='both'."
            )
        temperature = differential_temperature if branch_mode == "differential" else direct_temperature
        z = torch.randn(batch_size, token_count, self.cfg.direct_dim, device=device) * temperature
        cond, _ = self.channel_condition(channel_ids, batch_size, token_count, device)
        dt = 1.0 / max(sample_steps, 1)
        for step_idx in range(sample_steps):
            t_value = step_idx * dt
            t = torch.full((batch_size, token_count, 1), t_value, device=device, dtype=z.dtype)
            velocity = self.cfm(
                z,
                t,
                steps,
                cond,
                use_base=branch_mode != "differential",
                use_differential=branch_mode != "direct",
                gate_mode=gate_mode,
            )
            if solver == "euler":
                z = z + velocity * dt
            else:
                predicted = z + velocity * dt
                t_next = torch.full_like(t, min((step_idx + 1) * dt, 1.0))
                next_velocity = self.cfm(
                    predicted,
                    t_next,
                    steps,
                    cond,
                    use_base=branch_mode != "differential",
                    use_differential=branch_mode != "direct",
                    gate_mode=gate_mode,
                )
                z = z + 0.5 * (velocity + next_velocity) * dt
        z = self.denormalize_flow_token(z)
        if self.channel_token_mode:
            return self.decode_channel_tokens(z, seq_len)
        return self.decode_direct(z)

    def score_gate(self, correction: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        score = correction.abs()
        lo = score.amin(dim=-1, keepdim=True)
        hi = score.amax(dim=-1, keepdim=True)
        return ((score - lo) / (hi - lo + eps)).clamp(0.0, 1.0)

    def gate_hint(self, correction: torch.Tensor, gate_mode: str) -> torch.Tensor:
        score = self.score_gate(correction)
        if gate_mode == "tanh":
            return 1.0 + score
        if gate_mode == "random":
            return torch.ones_like(score)
        if gate_mode in {"one", "fixed"}:
            return torch.ones_like(score)
        if gate_mode == "zero":
            return torch.zeros_like(score)
        return score

    @staticmethod
    def normalize_gate_mode(gate_mode: str) -> str:
        if gate_mode == "learned":
            return "sigmoid"
        if gate_mode == "fixed":
            return "one"
        return gate_mode

    def _step_column(self, batch: int, seq_len: int, device: torch.device) -> torch.Tensor:
        step = torch.linspace(0.0, 1.0, seq_len, device=device)
        return step[None, :, None].expand(batch, seq_len, 1)
