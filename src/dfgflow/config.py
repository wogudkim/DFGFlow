from __future__ import annotations

from dataclasses import dataclass, fields


@dataclass
class DirectDiffConfig:
    seq_len: int = 16
    channels: int = 2
    freq_bins: int = 64
    time_bins: int = 64
    direct_dim: int = 512
    differential_dim: int = 512
    hidden_dim: int = 512
    num_condition_channels: int = 0
    channel_embed_dim: int = 64
    # CFM-only ablations keep the frozen reconstruction model identical.
    cfm_log_variation: bool = True
    cfm_dra_attention: bool = True
    cfm_gru: bool = True
    cfm_topk: bool = True
    cfm_gate_attention: bool = True
    cfm_learned_gate: bool = True
    cfm_heads: int = 4
    cfm_topk_ratio: float = 0.25
    cfm_injection: str = "concat"
    prior_layers: int = 2
    cfm_sample_steps: int = 32
    cfm_solver: str = "heun"
    cfm_sigma_min: float = 1e-4
    sparse_topk_ratio: float = 0.25
    token_mode: str = "pair"
    branch_mode: str = "both"
    gate_mode: str = "score"
    gate_lambda: float = 1.0
    lr: float = 1e-4
    weight_decay: float = 0.0
    lambda_recon: float = 1.0
    lambda_direct: float = 1.0
    lambda_prior_recon: float = 0.25
    lambda_detail: float = 0.35
    lambda_diff_recon: float = 2.0
    lambda_kl: float = 1e-4
    lambda_cfm: float = 1.0
    lambda_cfm_endpoint: float = 1.0
    lambda_additive_recon: float = 1.0
    lambda_amplitude: float = 0.5
    # Strengthen gate supervision by default so score-based gating is effective
    lambda_gate_hint: float = 0.5
    lambda_gate_sparse: float = 0.0
    lambda_gate_tv: float = 0.02
    lambda_tv: float = 0.0
    lambda_raw: float = 1.0
    lambda_raw_detail: float = 1.0
    lambda_peak: float = 1.0
    lambda_consistency: float = 0.25
    lambda_freq: float = 0.35
    # Attention module hyperparameters
    attn_num_heads: int = 4
    attn_dropout: float = 0.1
    attn_num_layers: int = 1
    # Number of residual cross-attention refinements in the CFM DRA block.
    dra_cross_layers: int = 2
    # Train the CFM in a coordinate-wise standardized latent space while the
    # encoder and decoder retain their native latent coordinates.
    normalize_flow_latent: bool = False
    flow_latent_momentum: float = 0.01
    flow_latent_eps: float = 1e-5

    @property
    def spec_shape(self) -> tuple[int, int, int]:
        return self.channels, self.freq_bins, self.time_bins


def config_from_dict(values: dict) -> DirectDiffConfig:
    values = dict(values)
    values.setdefault("cfm_heads", values.get("attn_num_heads", 4))
    values.setdefault("cfm_topk_ratio", values.get("sparse_topk_ratio", 0.25))
    valid = {field.name for field in fields(DirectDiffConfig)}
    return DirectDiffConfig(**{key: value for key, value in values.items() if key in valid})
