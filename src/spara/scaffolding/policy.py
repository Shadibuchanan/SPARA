"""Scaffolding RL objective (eqs. 3-6, App. C) over tensors.

Nothing here depends on how the latent prior is produced, so the same terms can
later attach to other latent sources (e.g. the LIVR checkpoint).
"""
import math

import torch


def group_advantages(rewards, eps=1e-6):
    """Group-relative advantages; identical rewards give zero advantage."""
    rewards = torch.as_tensor(rewards, dtype=torch.float64)
    spread = rewards.std() if rewards.numel() > 1 else rewards.new_zeros(())
    return ((rewards - rewards.mean()) / (spread + eps)).float()


def gaussian_log_prob(action, prior, mean, sigma, reduction="sum"):
    """log p(a | x) with a = prior + delta, delta ~ N(mean, sigma^2), factorized over
    the K x d latent block (eq. 5). Computed in log space; the paper sums."""
    sigma = sigma.float()
    noise = (action.float() - prior.float() - mean.float()) / sigma
    per_dim = -0.5 * noise.square() - sigma.log() - 0.5 * math.log(2 * math.pi)
    if reduction == "sum":
        return per_dim.sum()
    if reduction == "mean":
        return per_dim.mean()
    raise ValueError(f"Unknown reduction: {reduction}")


def clipped_surrogate(log_ratio, advantage, epsilon):
    ratio = log_ratio.exp()
    return torch.minimum(ratio * advantage, ratio.clamp(1 - epsilon, 1 + epsilon) * advantage)


def clipped_fraction(log_ratio, epsilon):
    ratio = log_ratio.detach().exp()
    return ((ratio < 1 - epsilon) | (ratio > 1 + epsilon)).float().mean()


def masked_kl(log_p, log_q):
    """Exact per-position KL(p || q) over the vocabulary. Suppressed tokens are -inf in
    both and contribute zero; they are zero-filled before the arithmetic so no NaN
    gradient flows back through them."""
    finite = torch.isfinite(log_p)
    log_p, log_q = log_p.masked_fill(~finite, 0.0), log_q.masked_fill(~finite, 0.0)
    return (log_p.exp() * (log_p - log_q)).masked_fill(~finite, 0.0).sum(-1)
