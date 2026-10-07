from __future__ import annotations

import argparse
import csv
import random
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from .metadata import load_preprocessing_metadata
from .config import DirectDiffConfig
from .data import SpectrogramPairDataset, SpectrogramSequenceDataset
from .losses import (
    centered_complex_patch_sequence,
    gradient_detail_loss,
    frequency_weighted_l1,
    inverse_centered_complex_patch_sequence,
    inverse_complex_spectrogram_sequence,
    kl_normal,
    total_variation_loss,
)
from .model import DirectDiffModel


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train DFGFlow.", formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument("--data", type=str, required=True)
    p.add_argument("--out-dir", type=str, required=True)
    p.add_argument("--resume", type=str, default=None, help="Load model weights and continue with a fresh optimizer.")
    p.add_argument(
        "--reset-cfm",
        action="store_true",
        help="Reinitialize only the flow matcher after loading --resume.",
    )
    p.add_argument("--stage", choices=["autoencoder", "flow"], default="autoencoder",
                   help="Train reconstruction first, then freeze it and fit CFM with --stage flow --resume.")
    p.add_argument("--kl-warmup-steps", type=int, default=10000)
    p.add_argument("--steps", type=int, default=30000)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--seed", type=int, default=7)
    p.add_argument("--log-every", type=int, default=50)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--weight-decay", type=float, default=None)
    p.add_argument("--direct-dim", type=int, default=None)
    p.add_argument("--differential-dim", type=int, default=None)
    p.add_argument("--hidden-dim", type=int, default=None)
    p.add_argument("--channel-embed-dim", type=int, default=None)
    p.add_argument("--cfm-sample-steps", type=int, default=None)
    p.add_argument("--cfm-solver", choices=["euler", "heun"], default=None)
    p.add_argument("--cfm-sigma-min", type=float, default=None)
    p.add_argument(
        "--dra-cross-layers",
        type=int,
        default=None,
        help="Number of residual cross-attention stages in the CFM differential residual attention block.",
    )
    p.add_argument("--sparse-topk-ratio", type=float, default=None)
    for component in ("log-variation", "dra-attention", "gru", "topk", "gate-attention", "learned-gate"):
        p.add_argument(f"--cfm-{component}", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--cfm-heads", type=int, default=None)
    p.add_argument("--cfm-topk-ratio", type=float, default=None)
    p.add_argument("--cfm-injection", choices=["concat", "add"], default=None)

    p.add_argument("--lambda-amplitude", type=float, default=None)
    p.add_argument("--lambda-raw", type=float, default=None)
    p.add_argument("--lambda-raw-detail", type=float, default=None)
    p.add_argument("--lambda-peak", type=float, default=None)
    p.add_argument("--lambda-kl", type=float, default=None)
    p.add_argument("--lambda-diff-recon", type=float, default=None)
    p.add_argument("--lambda-cfm-endpoint", type=float, default=None)
    p.add_argument("--lambda-additive-recon", type=float, default=None)
    p.add_argument("--lambda-consistency", type=float, default=None)
    p.add_argument(
        "--token-mode",
        choices=["pair", "channel"],
        default="pair",
        help="pair: build tokens from sample pairs. channel: build tokens from real/imag channel pairs inside each sample.",
    )
    p.add_argument("--branch-mode", choices=["both", "direct", "differential"], default="both")
    p.add_argument(
        "--gate-mode",
        choices=["sigmoid", "tanh", "random", "zero", "one", "fixed", "lambda", "learned", "score"],
        default="score",
        help=(
            "Token gain ablation. zero=base-only, one/fixed=base+diff, "
            "lambda=base+gate_lambda*diff, "
            "sigmoid=base+sigmoid(score)*diff, tanh=base+(1+tanh(score))*diff, "
            "random=base+U(0,2)*diff. learned/score are kept as sigmoid aliases."
        ),
    )
    p.add_argument("--gate-lambda", type=float, default=1.0, help="Fixed lambda for --gate-mode lambda.")
    p.add_argument("--pair-gap", type=int, default=1, help="Pair source index i with target index i+pair_gap.")
    p.add_argument(
        "--pair-mode",
        choices=["self", "adjacent", "random", "random_same_record"],
        default="adjacent",
        help="Pair source A and target B. 'self' makes B-A zero and is intended only for direct-only ablations.",
    )
    p.add_argument("--pair-seed", type=int, default=7)
    p.add_argument("--amp", action="store_true", help="Use CUDA automatic mixed precision.")
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def reconstruct_raw(
    recon: torch.Tensor,
    dataset: SpectrogramSequenceDataset,
) -> torch.Tensor | None:
    if dataset.spec_mean is None or dataset.spec_std is None:
        return None
    meta = dataset.metadata
    if meta.get("representation") != "complex":
        return None
    if meta.get("spectrogram_layout", "window_stft") == "full_stft_patches":
        return inverse_centered_complex_patch_sequence(
            recon,
            original_channels=int(meta["original_channels"]),
            n_fft=int(meta["n_fft"]),
            spec_mean=dataset.spec_mean,
            spec_std=dataset.spec_std,
        )
    return inverse_complex_spectrogram_sequence(
        recon,
        original_channels=int(meta["original_channels"]),
        window_length=int(meta["window_length"]),
        n_fft=int(meta["n_fft"]),
        hop_length=int(meta["hop_length"]),
        spec_mean=dataset.spec_mean,
        spec_std=dataset.spec_std,
    )


def raw_reconstruction_loss(
    recon: torch.Tensor,
    raw_targets: torch.Tensor | None,
    dataset: SpectrogramSequenceDataset,
) -> torch.Tensor:
    if raw_targets is None:
        return recon.new_tensor(0.0)
    raw_recon = reconstruct_raw(recon, dataset)
    if raw_recon is None:
        return recon.new_tensor(0.0)
    dims = (1, 3)
    moment = F.l1_loss(raw_recon.mean(dims), raw_targets.mean(dims))
    moment = moment + F.l1_loss(
        raw_recon.var(dims, unbiased=False).add(1e-6).sqrt(),
        raw_targets.var(dims, unbiased=False).add(1e-6).sqrt(),
    )
    return F.l1_loss(raw_recon, raw_targets) + moment


def raw_reconstruction_terms(
    recon: torch.Tensor,
    raw_targets: torch.Tensor | None,
    dataset: SpectrogramSequenceDataset,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Waveform value, temporal-detail, and activity-weighted peak losses."""
    zero = recon.new_tensor(0.0)
    if raw_targets is None:
        return zero, zero, zero
    raw_recon = reconstruct_raw(recon, dataset)
    if raw_recon is None:
        return zero, zero, zero
    dims = (1, 3)
    moment = F.l1_loss(raw_recon.mean(dims), raw_targets.mean(dims))
    moment = moment + F.l1_loss(
        raw_recon.var(dims, unbiased=False).add(1e-6).sqrt(),
        raw_targets.var(dims, unbiased=False).add(1e-6).sqrt(),
    )
    value = F.l1_loss(raw_recon, raw_targets) + moment
    pred_delta = raw_recon[..., 1:] - raw_recon[..., :-1]
    target_delta = raw_targets[..., 1:] - raw_targets[..., :-1]
    detail = F.l1_loss(pred_delta, target_delta)
    if raw_recon.shape[-1] > 2:
        detail = detail + 0.5 * F.l1_loss(
            pred_delta[..., 1:] - pred_delta[..., :-1],
            target_delta[..., 1:] - target_delta[..., :-1],
        )
    activity = target_delta.abs()
    activity_scale = activity.mean(dim=(1, 3), keepdim=True).clamp_min(1e-4)
    weights = 1.0 + 4.0 * (activity / activity_scale).clamp(max=10.0)
    derivative_peak = (weights * (pred_delta - target_delta).abs()).mean()

    # Derivative-only weighting misses long plateaus such as appliance-on
    # events.  Also weight value errors by distance from the per-channel
    # temporal median and explicitly preserve extrema.
    flattened_target = raw_targets.transpose(1, 2).flatten(2)
    flattened_pred = raw_recon.transpose(1, 2).flatten(2)
    baseline = flattened_target.median(dim=-1, keepdim=True).values
    event_strength = (flattened_target - baseline).abs()
    event_scale = event_strength.mean(dim=-1, keepdim=True).clamp_min(1e-4)
    event_weights = 1.0 + 4.0 * (event_strength / event_scale).clamp(max=10.0)
    event_value = (event_weights * (flattened_pred - flattened_target).abs()).mean()
    extrema = F.l1_loss(flattened_pred.amax(dim=-1), flattened_target.amax(dim=-1))
    extrema = extrema + F.l1_loss(flattened_pred.amin(dim=-1), flattened_target.amin(dim=-1))
    peak = derivative_peak + 0.5 * event_value + 0.25 * extrema
    return value, detail, peak


def stft_consistency_loss(recon: torch.Tensor, dataset: SpectrogramSequenceDataset) -> torch.Tensor:
    meta = dataset.metadata
    if meta.get("representation") != "complex" or meta.get("spectrogram_layout") != "full_stft_patches":
        return recon.new_tensor(0.0)
    raw_recon = reconstruct_raw(recon, dataset)
    if raw_recon is None:
        return recon.new_tensor(0.0)
    projected = centered_complex_patch_sequence(
        raw_recon,
        n_fft=int(meta["n_fft"]),
        freq_bins=int(meta["freq_bins"]),
        spec_mean=dataset.spec_mean,
        spec_std=dataset.spec_std,
    )
    return F.l1_loss(projected, recon)


def token_gate_tv(gate: torch.Tensor) -> torch.Tensor:
    dim_tv = (gate[:, :, 1:] - gate[:, :, :-1]).abs().mean() if gate.shape[-1] > 1 else gate.new_tensor(0.0)
    seq_tv = (gate[:, 1:] - gate[:, :-1]).abs().mean() if gate.shape[1] > 1 else gate.new_tensor(0.0)
    return dim_tv + 0.25 * seq_tv


def amplitude_moment_loss(pred: torch.Tensor, target: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Match per-sample/channel mean and standard deviation."""
    dims = (1, 3, 4)
    pred_mean = pred.mean(dim=dims)
    target_mean = target.mean(dim=dims)
    pred_std = pred.var(dim=dims, unbiased=False).add(eps).sqrt()
    target_std = target.var(dim=dims, unbiased=False).add(eps).sqrt()
    return F.l1_loss(pred_mean, target_mean) + F.l1_loss(pred_std, target_std)


def configure_training_stage(model: DirectDiffModel, stage: str) -> list[torch.nn.Parameter]:
    if stage not in {"autoencoder", "flow"}:
        raise ValueError(stage)
    # Keep the frozen representation deterministic, including attention dropout.
    model.eval() if stage == "flow" else model.train()
    model.cfm.train(stage == "flow")
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.startswith("cfm.") == (stage == "flow"))
        parameter.grad = None
    return [parameter for parameter in model.parameters() if parameter.requires_grad]


def train() -> None:
    args = parse_args()
    if args.stage == "flow" and args.resume is None:
        raise ValueError("--stage flow requires a trained autoencoder checkpoint via --resume")
    if args.kl_warmup_steps < 0:
        raise ValueError("--kl-warmup-steps must be nonnegative")
    set_seed(args.seed)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    base_dataset = SpectrogramSequenceDataset(args.data)
    if args.token_mode == "channel":
        dataset = base_dataset
    else:
        dataset = SpectrogramPairDataset(
            base_dataset,
            pair_gap=args.pair_gap,
            pair_mode=args.pair_mode,
            pair_seed=args.pair_seed,
        )
    _, seq_len, channels, freq_bins, time_bins = base_dataset.data.shape
    if seq_len < 2 and args.branch_mode != "direct":
        raise ValueError(
            "The differential path requires at least two spectrogram tokens per sample. "
            "Regenerate the dataset with --window-length smaller than --series-length "
            "(for example, series length 256 and window length 64)."
        )
    cfg = DirectDiffConfig(seq_len=int(seq_len), channels=int(channels), freq_bins=int(freq_bins), time_bins=int(time_bins))
    cfg.token_mode = args.token_mode
    cfg.branch_mode = args.branch_mode
    cfg.gate_mode = args.gate_mode
    cfg.gate_lambda = args.gate_lambda
    if args.direct_dim is not None:
        cfg.direct_dim = args.direct_dim
    if args.differential_dim is not None:
        cfg.differential_dim = args.differential_dim
    if args.hidden_dim is not None:
        cfg.hidden_dim = args.hidden_dim
    if args.channel_embed_dim is not None:
        cfg.channel_embed_dim = args.channel_embed_dim
    if args.cfm_sample_steps is not None:
        cfg.cfm_sample_steps = args.cfm_sample_steps
    if args.cfm_solver is not None:
        cfg.cfm_solver = args.cfm_solver
    if args.cfm_sigma_min is not None:
        cfg.cfm_sigma_min = args.cfm_sigma_min
    if args.dra_cross_layers is not None:
        if args.dra_cross_layers < 1:
            raise ValueError("--dra-cross-layers must be at least 1.")
        cfg.dra_cross_layers = args.dra_cross_layers
    if args.sparse_topk_ratio is not None:
        if not 0.0 < args.sparse_topk_ratio <= 1.0:
            raise ValueError("--sparse-topk-ratio must be in (0, 1].")
        cfg.sparse_topk_ratio = args.sparse_topk_ratio
    if args.lambda_amplitude is not None:
        cfg.lambda_amplitude = args.lambda_amplitude
    if args.lambda_raw is not None:
        cfg.lambda_raw = args.lambda_raw
    if args.lambda_raw_detail is not None:
        cfg.lambda_raw_detail = args.lambda_raw_detail
    if args.lambda_peak is not None:
        cfg.lambda_peak = args.lambda_peak
    if args.lambda_kl is not None:
        cfg.lambda_kl = args.lambda_kl
    if args.lambda_diff_recon is not None:
        cfg.lambda_diff_recon = args.lambda_diff_recon
    if args.lambda_cfm_endpoint is not None:
        cfg.lambda_cfm_endpoint = args.lambda_cfm_endpoint
    if args.lambda_additive_recon is not None:
        cfg.lambda_additive_recon = args.lambda_additive_recon
    if args.lambda_consistency is not None:
        cfg.lambda_consistency = args.lambda_consistency
    if args.token_mode == "channel":
        if channels % 2 != 0:
            raise ValueError("channel token mode requires complex real/imag channel pairs.")
        cfg.num_condition_channels = int(channels // 2)
    elif base_dataset.sample_channel_ids is not None:
        cfg.num_condition_channels = int(base_dataset.metadata.get("num_condition_channels", int(base_dataset.sample_channel_ids.max()) + 1))
    if args.lr is not None:
        cfg.lr = args.lr
    if args.weight_decay is not None:
        cfg.weight_decay = args.weight_decay
    # Inherit CFM choices on resume unless explicitly overridden.
    resume_config = torch.load(args.resume, map_location="cpu", weights_only=False)["config"] if args.resume else {}
    ablation_fields = ("cfm_log_variation", "cfm_dra_attention", "cfm_gru", "cfm_topk",
                       "cfm_gate_attention", "cfm_learned_gate", "cfm_heads",
                       "cfm_topk_ratio", "cfm_injection")
    for field in ablation_fields:
        override = getattr(args, field)
        setattr(cfg, field, override if override is not None else resume_config.get(field, getattr(cfg, field)))
    if cfg.cfm_heads < 1 or cfg.direct_dim % cfg.cfm_heads:
        raise ValueError("--cfm-heads must divide --direct-dim")
    if not 0 < cfg.cfm_topk_ratio <= 1:
        raise ValueError("--cfm-topk-ratio must be in (0, 1]")
    device = torch.device(args.device)
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=True,
        num_workers=args.num_workers,
        pin_memory=device.type == "cuda",
    )
    model = DirectDiffModel(cfg).to(device)
    if args.resume is not None:
        resume = torch.load(args.resume, map_location=device, weights_only=False)
        state = resume["model"]
        if args.reset_cfm:
            state = {key: value for key, value in state.items() if not key.startswith("cfm.")}
        incompatible = model.load_state_dict(state, strict=False)
        unexpected = list(incompatible.unexpected_keys)
        missing = [key for key in incompatible.missing_keys if not key.startswith("flow_latent_") and not (args.reset_cfm and key.startswith("cfm."))]
        if missing or unexpected:
            raise RuntimeError(f"incompatible resume checkpoint: missing={missing}, unexpected={unexpected}")
        if args.reset_cfm:
            model.cfm = type(model.cfm)(cfg).to(device)
            print("reinitialized CFM for the standardized latent target")
        print(f"resumed model weights from {args.resume}")
    if args.stage == "flow":
        # Restore the representation settings of the frozen checkpoint exactly.
        cfg.normalize_flow_latent = resume["config"].get("normalize_flow_latent", False)
    trainable = configure_training_stage(model, args.stage)
    optim = torch.optim.Adam(trainable, lr=cfg.lr, weight_decay=cfg.weight_decay)
    use_amp = bool(args.amp and device.type == "cuda")
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    fields = [
        "step",
        "stage",
        "kl_weight",
        "loss",
        "recon",
        "prior_recon",
        "direct",
        "diff",
        "kl",
        "detail",
        "freq",
        "cfm",
        "cfm_endpoint",
        "additive_recon",
        "amplitude",
        "gate_hint",
        "gate_sparse",
        "gate_tv",
        "tv",
        "raw",
        "raw_detail",
        "peak",
        "consistency",
        "gate_mean",
        "direct_std",
        "diff_std",
    ]
    iterator = iter(loader)
    with (out_dir / "train_metrics.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        progress = tqdm(range(args.steps), desc="train")
        for step in progress:
            try:
                batch = next(iterator)
            except StopIteration:
                iterator = iter(loader)
                batch = next(iterator)
            if args.token_mode == "channel":
                if isinstance(batch, (list, tuple)):
                    target = batch[0].to(device)
                    raw_targets = batch[1].to(device) if len(batch) > 1 else None
                else:
                    target = batch.to(device)
                    raw_targets = None
                source = target
                source_raw = raw_targets
                channel_ids = None
            else:
                source = batch["source"].to(device)
                target = batch["target"].to(device)
                source_raw = batch.get("source_raw")
                source_raw = source_raw.to(device) if source_raw is not None else None
                raw_targets = batch.get("target_raw")
                raw_targets = raw_targets.to(device) if raw_targets is not None else None
                channel_ids = batch.get("channel_id")
                channel_ids = channel_ids.to(device) if channel_ids is not None else None
            with torch.amp.autocast("cuda", enabled=use_amp):
                out = model(source, target, sample_posterior=False, branch_mode=args.branch_mode, gate_mode=args.gate_mode, channel_ids=channel_ids)
                zero = target.new_tensor(0.0)
                prior_recon_loss = F.l1_loss(out["prior_reconstruction"], target) if args.branch_mode != "direct" else zero
                direct_target = target if args.token_mode == "channel" else 0.5 * (source + target)
                reconstruction_target = direct_target if args.branch_mode == "direct" else target
                recon_loss = F.l1_loss(out["reconstruction"], reconstruction_target)
                direct_loss = F.l1_loss(out["direct_base"], direct_target) if args.branch_mode != "differential" else zero
                if args.branch_mode != "direct":
                    diff_loss = F.l1_loss(out["differential_base"], out["target_transition"])
                    diff_loss = diff_loss + 0.5 * gradient_detail_loss(
                        out["differential_base"], out["target_transition"]
                    )
                    diff_loss = diff_loss + 0.25 * frequency_weighted_l1(
                        out["differential_base"], out["target_transition"]
                    )
                    kl_loss = kl_normal(out["q_mu"], out["q_logvar"], out["p_mu"], out["p_logvar"])
                    additive_recon_loss = F.l1_loss(out["additive_reconstruction"], target)
                    if args.gate_mode in {"sigmoid", "tanh", "learned", "score"}:
                        gate_hint_loss = F.mse_loss(out["gate"], out["gate_hint"])
                        gate_sparse_loss = (out["gate"] - 1.0).abs().mean() if args.gate_mode == "tanh" else out["gate"].mean()
                        gate_tv_loss = token_gate_tv(out["gate"])
                    else:
                        gate_hint_loss = zero
                        gate_sparse_loss = zero
                        gate_tv_loss = zero
                else:
                    diff_loss = zero
                    kl_loss = zero
                    additive_recon_loss = zero
                    gate_hint_loss = zero
                    gate_sparse_loss = zero
                    gate_tv_loss = zero
                detail_loss = gradient_detail_loss(out["reconstruction"], reconstruction_target)
                freq_loss = frequency_weighted_l1(out["reconstruction"], reconstruction_target)
                cfm_loss = F.mse_loss(out["cfm_velocity_pred"], out["cfm_velocity_target"])
                cfm_endpoint_loss = F.mse_loss(
                    out["cfm_endpoint_token"], out["cfm_target_token"].detach()
                )
                amplitude_loss = amplitude_moment_loss(out["reconstruction"], reconstruction_target)
                tv_loss = total_variation_loss(out["reconstruction"])
                raw_supervision = (
                    0.5 * (source_raw + raw_targets)
                    if args.branch_mode == "direct" and source_raw is not None and raw_targets is not None
                    else raw_targets
                )
                if cfg.lambda_raw > 0.0 or cfg.lambda_raw_detail > 0.0 or cfg.lambda_peak > 0.0:
                    raw_loss, raw_detail_loss, peak_loss = raw_reconstruction_terms(
                        out["reconstruction"], raw_supervision, base_dataset
                    )
                else:
                    raw_loss = raw_detail_loss = peak_loss = zero
                consistency_loss = (
                    stft_consistency_loss(out["reconstruction"], base_dataset)
                    if cfg.lambda_consistency > 0.0
                    else zero
                )
                kl_weight = cfg.lambda_kl * min(1.0, step / max(1, args.kl_warmup_steps)) if args.kl_warmup_steps else cfg.lambda_kl
                loss = (
                    cfg.lambda_recon * recon_loss
                    + cfg.lambda_direct * direct_loss
                    + cfg.lambda_diff_recon * diff_loss
                    + kl_weight * kl_loss
                    + cfg.lambda_detail * detail_loss
                    + cfg.lambda_freq * freq_loss
                    + cfg.lambda_additive_recon * additive_recon_loss
                    + cfg.lambda_amplitude * amplitude_loss
                    + cfg.lambda_gate_hint * gate_hint_loss
                    + cfg.lambda_gate_sparse * gate_sparse_loss
                    + cfg.lambda_gate_tv * gate_tv_loss
                    + cfg.lambda_tv * tv_loss
                    + cfg.lambda_raw * raw_loss
                    + cfg.lambda_raw_detail * raw_detail_loss
                    + cfg.lambda_peak * peak_loss
                    + cfg.lambda_consistency * consistency_loss
                )
                if args.stage == "flow":
                    loss = cfg.lambda_cfm * cfm_loss + cfg.lambda_cfm_endpoint * cfm_endpoint_loss
            optim.zero_grad(set_to_none=True)
            scaler.scale(loss).backward()
            scaler.unscale_(optim)
            torch.nn.utils.clip_grad_norm_(trainable, 1.0)
            scaler.step(optim)
            scaler.update()
            if step % args.log_every == 0 or step == args.steps - 1:
                row = {
                    "step": step,
                    "stage": args.stage,
                    "kl_weight": kl_weight,
                    "loss": float(loss.detach().cpu()),
                    "recon": float(recon_loss.detach().cpu()),
                    "prior_recon": float(prior_recon_loss.detach().cpu()),
                    "direct": float(direct_loss.detach().cpu()),
                    "diff": float(diff_loss.detach().cpu()),
                    "kl": float(kl_loss.detach().cpu()),
                    "detail": float(detail_loss.detach().cpu()),
                    "freq": float(freq_loss.detach().cpu()),
                    "cfm": float(cfm_loss.detach().cpu()),
                    "cfm_endpoint": float(cfm_endpoint_loss.detach().cpu()),
                    "additive_recon": float(additive_recon_loss.detach().cpu()),
                    "amplitude": float(amplitude_loss.detach().cpu()),
                    "gate_hint": float(gate_hint_loss.detach().cpu()),
                    "gate_sparse": float(gate_sparse_loss.detach().cpu()),
                    "gate_tv": float(gate_tv_loss.detach().cpu()),
                    "tv": float(tv_loss.detach().cpu()),
                    "raw": float(raw_loss.detach().cpu()),
                    "raw_detail": float(raw_detail_loss.detach().cpu()),
                    "peak": float(peak_loss.detach().cpu()),
                    "consistency": float(consistency_loss.detach().cpu()),
                    "gate_mean": float(out["gate"].mean().detach().cpu()),
                    "direct_std": float(out["direct"].std().detach().cpu()),
                    "diff_std": float(out["q_mu"].std().detach().cpu()),
                }
                writer.writerow(row)
                f.flush()
                progress.set_postfix(loss=row["loss"], recon=row["recon"], prior=row["prior_recon"])
    torch.save({"model": model.state_dict(), "config": cfg.__dict__, "data_path": args.data, "training_stage": args.stage, "training_args": vars(args), "preprocessing": load_preprocessing_metadata(args.data)}, out_dir / "directdiff_last.pt")
    print(f"saved checkpoint to {out_dir / 'directdiff_last.pt'}")


if __name__ == "__main__":
    train()
