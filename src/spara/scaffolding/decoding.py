"""Cached decoding in which the model enters its latent block by sampling BOT (§2, §3.2).

On BOT: the K query embeddings read the cache, are cropped away, and the latent
action (prior plus optional Gaussian adjustment) and EOT are appended.
"""
import copy
from dataclasses import dataclass, field

import torch
from transformers import LogitsProcessorList

from .model import suppression


def logits_processors(model, generation, device):
    """HF's own processor stack and special tokens (pinned transformers 4.51.3), so
    sampling matches the `generate` call used for the LIVR evaluation."""
    settings = copy.deepcopy(model.generation_config)
    unused = settings.update(**generation)
    if unused:
        raise ValueError(f"Unknown generation settings: {sorted(unused)}")
    settings.validate()
    model._prepare_special_tokens(settings, True, device)
    return settings, model._get_logits_processor(settings, 0, None, None, LogitsProcessorList(), device=device)


@dataclass
class Rollout:
    prompt_length: int
    ids: list = field(default_factory=list)        # generated IDs, including the forced latent block
    sampled: list = field(default_factory=list)    # indices into `ids` chosen by the policy
    log_probs: list = field(default_factory=list)  # sampling-time log-probabilities (diagnostic)
    block: int | None = None                       # index into `ids` of BOT
    latent: torch.Tensor | None = None             # latent action a = z + delta, (K, d)
    sigma: torch.Tensor | None = None              # predicted scale, (K, d)
    mean: torch.Tensor | None = None


class LatentDecoder:
    """`latent_mode`: "sample" draws delta ~ N(mu, sigma^2) (paper inference and rollouts),
    "mean" uses mu, "none" uses delta = 0 (Stage 1 models and the text-only RL control)."""

    def __init__(self, model, generation, latent_mode="sample"):
        if latent_mode not in {"sample", "mean", "none"}:
            raise ValueError(f"Unknown latent mode: {latent_mode}")
        self.model, self.mode = model, latent_mode
        self.settings, self.processors = logits_processors(model, generation, model.device)
        eos = self.settings._eos_token_tensor
        self.eos = set() if eos is None else set(eos.reshape(-1).tolist())
        pad = self.settings._pad_token_tensor
        self.pad = int(pad) if pad is not None else (min(self.eos) if self.eos else 0)

    def choose(self, logits, sequence, after_block):
        scores = self.processors(sequence, suppression(logits.float(), self.model.config, after_block))
        if self.settings.do_sample:
            token = torch.multinomial(scores.softmax(-1), num_samples=1).squeeze(1)
        else:
            token = scores.argmax(-1)
        return token, scores.log_softmax(-1).gather(-1, token[:, None]).squeeze(1)

    def action(self, prior, rows):
        """Latent actions (rows, K, d) around the prior (K, d)."""
        prior = prior.float()
        heads = getattr(self.model, "latent_heads", None)
        if self.mode == "none" or heads is None:
            return prior.expand(rows, -1, -1).clone(), None, None
        mean, sigma = heads(prior)
        if self.mode == "mean":
            return (prior + mean).expand(rows, -1, -1).clone(), mean, sigma
        noise = torch.randn((rows, *mean.shape), device=mean.device, dtype=mean.dtype)
        return prior + mean + sigma * noise, mean, sigma

    def _block_ids(self):
        c = self.model.config
        return [c.scaffold_latent_id] * c.scaffold_k + [c.scaffold_eot_id]

    @torch.no_grad()
    def generate(self, prompt, max_new_tokens=None, state=None, first=None):
        """Decode one sequence. `state`/`first` continue from a shared prefill and an
        already sampled first token (used by `sample_group`)."""
        model = self.model
        max_new_tokens = max_new_tokens or self.settings.max_new_tokens
        ids = prompt["input_ids"]
        if state is None:
            hidden, cache, delta = model.prefill(ids, prompt.get("pixel_values"), prompt.get("image_grid_thw"))
            logits = model.lm_head(hidden[:, -1])
        else:
            logits, cache, delta = state
        rollout, sequence = Rollout(prompt_length=ids.shape[1]), ids.clone()
        for step in range(max_new_tokens):
            if step == 0 and first is not None:
                token, log_prob = first
            else:
                token, log_prob = self.choose(logits, sequence, rollout.block is not None)
            value = int(token)
            rollout.sampled.append(len(rollout.ids))
            rollout.ids.append(value)
            rollout.log_probs.append(float(log_prob))
            sequence = torch.cat([sequence, token.view(1, 1).to(sequence)], dim=1)
            if value == model.config.scaffold_bot_id:
                prior = model.latent_prior(cache, delta)
                actions, rollout.mean, rollout.sigma = self.action(prior, 1)
                rollout.block, rollout.latent = len(rollout.ids) - 1, actions[0]
                logits = model.lm_head(model.append_latent(cache, delta, actions)[:, -1])
                block = self._block_ids()
                rollout.ids += block
                sequence = torch.cat([sequence, torch.tensor([block], device=sequence.device)], dim=1)
                continue
            if value in self.eos:
                break
            logits = model.lm_head(model.extend(token.view(1, 1).to(sequence), cache, delta)[:, -1])
        return rollout

    @torch.no_grad()
    def sample_group(self, prompt, size, max_new_tokens=None):
        """G rollouts sharing one prefill. Rows that open with BOT (the trained layout)
        share one latent prior and decode in lockstep; other rows decode one by one."""
        model, c = self.model, self.model.config
        max_new_tokens = max_new_tokens or self.settings.max_new_tokens
        ids = prompt["input_ids"]
        hidden, cache, delta = model.prefill(ids, prompt.get("pixel_values"), prompt.get("image_grid_thw"))
        first_logits = model.lm_head(hidden[:, -1])
        first, first_log_probs = self.choose(first_logits.expand(size, -1), ids.expand(size, -1), False)
        rollouts = [None] * size
        bot_rows = [i for i in range(size) if int(first[i]) == c.scaffold_bot_id]
        for i in range(size):
            if i not in bot_rows:
                state = (first_logits, copy.deepcopy(cache), delta)
                rollouts[i] = self.generate(prompt, max_new_tokens, state, (first[i], first_log_probs[i]))
        if not bot_rows:
            return rollouts
        n = len(bot_rows)
        prior = model.latent_prior(cache, delta)
        actions, mean, sigma = self.action(prior, n)
        cache.batch_repeat_interleave(n)
        logits = model.lm_head(model.append_latent(cache, delta, actions)[:, -1])
        block = [c.scaffold_bot_id] + self._block_ids()
        sequence = torch.cat([ids.expand(n, -1), torch.tensor([block], device=ids.device).expand(n, -1)], dim=1)
        rows = [Rollout(ids.shape[1], list(block), [0], [float(first_log_probs[i])], 0, actions[r], sigma, mean)
                for r, i in enumerate(bot_rows)]
        finished = torch.zeros(n, dtype=torch.bool, device=ids.device)
        eos = torch.tensor(sorted(self.eos), device=ids.device, dtype=torch.long)
        for _ in range(1, max_new_tokens):
            token, log_prob = self.choose(logits, sequence, True)
            token = torch.where(finished, torch.full_like(token, self.pad), token)
            for r in (~finished).nonzero().flatten().tolist():
                rows[r].sampled.append(len(rows[r].ids))
                rows[r].ids.append(int(token[r]))
                rows[r].log_probs.append(float(log_prob[r]))
            sequence = torch.cat([sequence, token[:, None]], dim=1)
            finished |= torch.isin(token, eos)
            if bool(finished.all()):
                break
            logits = model.lm_head(model.extend(token[:, None], cache, delta)[:, -1])
        for r, i in enumerate(bot_rows):
            rollouts[i] = rows[r]
        return rollouts
