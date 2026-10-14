import copy

import numpy as np
import torch
from spara.scaffolding import policy
from spara.scaffolding.decoding import LatentDecoder
from spara.scaffolding.helper import RAMP, value_heatmap, value_iteration
from spara.scaffolding.model import (ScaffoldingEncoder, ScaffoldingQwen, add_heads, answer_sequence,
                                     predictor_sequence, query_mask, suppression)
from spara.scaffolding.training import answer_loss, latent_matching, rl_update, scaffold_loss

RL = {"variant": "scaffolding", "head_hidden_size": 16, "sigma_init": 0.05, "log_sigma_bounds": [-9.21, 0.0],
      "latent_logprob_reduction": "sum", "clip_epsilon": 0.2, "kl_beta": 0.001,
      "rollout": {"temperature": 1.0}}
BOT, EOT, SLOT, QUERIES = 20, 21, 22, [23, 24, 25, 26]


def prior_of(model, prompt):
    queries = predictor_sequence(prompt["input_ids"], model.config)
    return model.project(model.hidden_states(queries, prompt["pixel_values"], prompt["image_grid_thw"],
                                             query_start=queries.shape[1] - 4)[0, -4:])


def rescore(model, prompt, rollout, indices, block_bot=False):
    """Training-path log-probabilities of the rollout's sampled tokens at `indices`."""
    ids = torch.cat([prompt["input_ids"], torch.tensor([rollout.ids])], dim=1)
    length = prompt["input_ids"].shape[1]
    positions = length + torch.tensor([rollout.sampled[i] for i in indices])
    latents = None if rollout.block is None else rollout.latent[None]
    hidden = model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], latents)
    after = (positions > length + rollout.block) if rollout.block is not None else block_bot
    logits = suppression(model.lm_head(hidden[0, positions - 1]).float(), model.config, after)
    return logits.log_softmax(-1).gather(-1, ids[0, positions][:, None]).squeeze(1)


def test_value_iteration_and_heatmap():
    np.testing.assert_allclose(value_iteration([[1, 0, 2]])[0], [0.85, 1.0, 1.0])
    walled = value_iteration([[1, -1, 2]])[0]
    assert walled.tolist() == [0.0, 0.0, 1.0]
    image = value_heatmap([[1, 0], [-1, 2]])
    assert image.size == (320, 320)
    assert image.getpixel((319, 319)) == tuple(RAMP[-1].astype(int))  # goal
    assert image.getpixel((0, 319)) == tuple(RAMP[0].astype(int))     # hole


def test_query_mask_and_bidirectional_queries(tiny_scaffold, tiny_sample):
    allowed = query_mask(5, 3, torch.float32, "cpu", past=2)[0, 0].eq(0)
    assert allowed.shape == (5, 7)
    assert allowed[0, :3].all() and not allowed[0, 3:].any()
    assert allowed[1, :4].all() and not allowed[1, 4:].any()
    assert allowed[3:].all()
    model, prompt = tiny_scaffold.eval(), tiny_sample["prompt"]
    ids = predictor_sequence(prompt["input_ids"], model.config)
    with torch.no_grad():
        base = model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], query_start=ids.shape[1] - 4)
        model.model.embed_tokens.weight[QUERIES[-1]] += 1.0
        changed = model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], query_start=ids.shape[1] - 4)
    # The last query is visible to the first (bidirectional) but to no earlier row.
    assert torch.equal(base[:, :-4], changed[:, :-4])
    assert not torch.allclose(base[:, -4], changed[:, -4])


def test_cached_latent_decoding_matches_uncached_scoring(tiny_scaffold, tiny_sample):
    model, prompt = tiny_scaffold.eval(), tiny_sample["prompt"]
    add_heads(model, RL)
    decoder = LatentDecoder(model, {"do_sample": True, "top_k": 0, "max_new_tokens": 6}, "sample")
    with torch.no_grad():
        hidden, cache, delta = model.prefill(prompt["input_ids"], prompt["pixel_values"], prompt["image_grid_thw"])
        cached_prior = model.latent_prior(cache, delta)
        assert cache.get_seq_length() == prompt["input_ids"].shape[1] + 1  # queries cropped, BOT kept
        torch.testing.assert_close(cached_prior, prior_of(model, prompt), atol=1e-5, rtol=1e-5)
        hidden, cache, delta = model.prefill(prompt["input_ids"], prompt["pixel_values"], prompt["image_grid_thw"])
        state = (model.lm_head(hidden[:, -1]), cache, delta)
        rollout = decoder.generate(prompt, 6, state, (torch.tensor(BOT), torch.tensor(0.0)))
        assert rollout.block == 0 and rollout.ids[1:6] == [SLOT] * 4 + [EOT]
        indices = list(range(1, len(rollout.sampled)))
        torch.testing.assert_close(rescore(model, prompt, rollout, indices),
                                   torch.tensor(rollout.log_probs[1:]), atol=1e-5, rtol=1e-5)
    # SDPA follows the same explicit query mask as eager attention.
    config = copy.deepcopy(model.config)
    config._attn_implementation = "sdpa"
    other = ScaffoldingQwen(config).eval()
    other.load_state_dict(model.state_dict())
    with torch.no_grad():
        torch.testing.assert_close(prior_of(other, prompt), prior_of(model, prompt), atol=1e-5, rtol=1e-5)


def test_group_rollouts_share_one_prior_and_rescore_exactly(tiny_scaffold, tiny_sample):
    model, prompt = tiny_scaffold.eval(), tiny_sample["prompt"]
    add_heads(model, RL)
    # A test-only bias makes every row open with BOT; it never touches later tokens.
    opening = LatentDecoder(model, {"do_sample": True, "top_k": 0, "max_new_tokens": 5,
                                    "sequence_bias": [[[BOT], 50.0]]}, "sample")
    with torch.no_grad():
        rollouts = opening.sample_group(prompt, 4, 5)
        assert all(r.block == 0 and r.ids[1:6] == [SLOT] * 4 + [EOT] for r in rollouts)
        assert len({tuple(r.latent.flatten().tolist()) for r in rollouts}) == 4  # distinct samples
        assert all(torch.equal(r.sigma, rollouts[0].sigma) for r in rollouts)
        torch.testing.assert_close(rollouts[0].sigma, torch.full_like(rollouts[0].sigma, 0.05))
        for r in rollouts:
            indices = list(range(1, len(r.sampled)))
            if indices:
                torch.testing.assert_close(rescore(model, prompt, r, indices), torch.tensor(r.log_probs[1:]),
                                           atol=1e-5, rtol=1e-5)
        never = LatentDecoder(model, {"do_sample": True, "top_k": 0, "max_new_tokens": 5,
                                      "sequence_bias": [[[BOT], -1e4]]}, "sample")
        for r in never.sample_group(prompt, 3, 5):
            assert r.block is None and SLOT not in r.ids
            # The -1e4 bias removes BOT's mass from sampling; score without BOT to match.
            indices = list(range(len(r.sampled)))
            torch.testing.assert_close(rescore(model, prompt, r, indices, block_bot=True), torch.tensor(r.log_probs),
                                       atol=1e-5, rtol=1e-5)


def test_zero_adjustment_reproduces_stage1_and_nonzero_changes_answer(tiny_scaffold, tiny_sample):
    model, prompt = tiny_scaffold.eval(), tiny_sample["prompt"]
    ids, bot = answer_sequence(prompt["input_ids"], tiny_sample["answer"], model.config)
    with torch.no_grad():
        prior = prior_of(model, prompt)
        stage1 = model.lm_head(model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], prior[None]))
        add_heads(model, RL)
        mean, sigma = model.latent_heads(prior)
        assert mean.eq(0).all()
        torch.testing.assert_close(sigma, torch.full_like(sigma, 0.05))
        actions, _, _ = LatentDecoder(model, {"do_sample": False}, "mean").action(prior, 1)
        assert torch.equal(actions[0], prior.float())
        zero = model.lm_head(model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], actions))
        nudged = model.lm_head(model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], actions + 0.5))
    torch.testing.assert_close(zero, stage1, atol=0, rtol=0)
    assert torch.equal(nudged[:, :bot + 1], stage1[:, :bot + 1])
    assert not torch.allclose(nudged[:, bot + 5:], stage1[:, bot + 5:])


def test_policy_terms():
    generator = torch.Generator().manual_seed(0)
    prior, mean = torch.randn(4, 8, generator=generator), 0.1 * torch.randn(4, 8, generator=generator)
    sigma = 0.01 + 0.2 * torch.rand(4, 8, generator=generator)
    action = prior + mean + sigma * torch.randn(4, 8, generator=generator)
    expected = torch.distributions.Normal(prior + mean, sigma).log_prob(action).sum()
    torch.testing.assert_close(policy.gaussian_log_prob(action, prior, mean, sigma), expected)
    prior.requires_grad_()
    log_prob = policy.gaussian_log_prob(action, prior, mean, sigma)
    log_ratio = log_prob - log_prob.detach()
    assert log_ratio.item() == 0.0  # theta == theta_old: ratio exactly one
    surrogate = policy.clipped_surrogate(log_ratio, torch.tensor(2.0), 0.2)
    surrogate.backward()
    assert surrogate.item() == 2.0 and torch.isfinite(prior.grad).all() and prior.grad.abs().sum() > 0
    assert policy.group_advantages([1, 1, 1]).eq(0).all()
    advantages = policy.group_advantages([1, 0, 0, 1])
    assert advantages.sum().abs() < 1e-6 and advantages[0] > 0 > advantages[1]
    assert abs(policy.clipped_fraction(torch.tensor([0.0, 0.5, -0.5]), 0.2).item() - 2 / 3) < 1e-6
    logits = torch.randn(3, 10, generator=generator, requires_grad=True)
    masked = logits.masked_fill(torch.arange(10) < 2, float("-inf"))
    log_p = masked.log_softmax(-1)
    log_q = (masked.detach() + 0.1 * torch.randn(3, 10, generator=generator)).log_softmax(-1)
    kl = policy.masked_kl(log_p, log_q)
    kl.sum().backward()
    assert (kl >= 0).all() and torch.isfinite(logits.grad).all()
    assert policy.masked_kl(log_p, log_p.detach()).abs().max() < 1e-6


def test_scaffold_phase_trains_only_the_encoder(tiny_scaffold, tiny_sample, tmp_path):
    vlm = tiny_scaffold.train()
    vlm.requires_grad_(False)
    encoder = ScaffoldingEncoder.from_vlm(vlm, 4, 4)
    frozen = {name: p.detach().clone() for name, p in vlm.named_parameters()}
    optimizer = torch.optim.AdamW(encoder.parameters(), lr=1e-3)
    losses = []
    for _ in range(12):
        optimizer.zero_grad()
        loss, _ = scaffold_loss(vlm, encoder, tiny_sample)
        loss.backward()
        losses.append(loss.item())
        assert encoder.pooling.queries.grad.abs().sum() > 0
        assert next(encoder.visual.parameters()).grad.abs().sum() > 0  # psi includes the vision copy
        optimizer.step()
    assert losses[-1] < losses[0]
    for name, parameter in vlm.named_parameters():
        assert parameter.grad is None and torch.equal(parameter, frozen[name])
    encoder.eval().save_pretrained(tmp_path)
    reloaded = ScaffoldingEncoder.from_pretrained(tmp_path).eval()
    with torch.no_grad():
        torch.testing.assert_close(encoder(**tiny_sample["helper"]), reloaded(**tiny_sample["helper"]), atol=0, rtol=0)
    assert reloaded.context["ids"] == [BOT, EOT, SLOT, *QUERIES]


def test_latent_prediction_trains_queries_and_language_not_vision(tiny_scaffold, tiny_sample):
    from spara.scaffolding.model import train_language
    model = tiny_scaffold.train()
    train_language(model)
    target = 0.5 * torch.randn(4, 32, generator=torch.Generator().manual_seed(3))
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=5e-3)
    first = None
    for _ in range(20):
        optimizer.zero_grad()
        latent, stats = latent_matching(model, tiny_sample, target)
        latent.backward()
        assert model.model.embed_tokens.weight.grad[QUERIES].abs().sum(-1).gt(0).all()
        ce, _ = answer_loss(model, tiny_sample, target)
        (0.5 * ce).backward()
        assert model.lm_head.weight.grad[BOT].abs().sum() > 0  # BOT is supervised
        assert all(p.grad is None for p in model.visual.parameters())
        optimizer.step()
        first = first or stats["latent_loss"]
    assert stats["latent_loss"] < 0.5 * first


def test_rl_update_starts_at_ratio_one_and_reaches_heads(tiny_scaffold, tiny_sample):
    from spara.scaffolding.model import train_language
    model, prompt = tiny_scaffold, tiny_sample["prompt"]
    add_heads(model, RL)
    reference = copy.deepcopy(model).eval()
    train_language(model)
    decoder = LatentDecoder(model.eval(), {"do_sample": True, "top_k": 0, "max_new_tokens": 5,
                                           "sequence_bias": [[[BOT], 50.0]]}, "sample")
    with torch.no_grad():
        rollouts = decoder.sample_group(prompt, 4, 5)
    group = {"prompt": prompt, "rollouts": rollouts, "rewards": [1, 0, 0, 1],
             "advantages": policy.group_advantages([1, 0, 0, 1])}
    model.train()
    stats = rl_update(model, reference, [group], RL, {})
    assert stats["latent_abs_log_ratio"] == 0 and stats["text_abs_log_ratio"] == 0
    assert stats["kl"] < 1e-6 and stats["latent_grad_norm"] > 0
    assert model.latent_heads.mean[2].weight.grad.abs().sum() > 0
    assert model.latent_heads.log_sigma[2].bias.grad.abs().sum() > 0
    assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
    model.zero_grad(set_to_none=True)
    text_only = rl_update(model, reference, [group], RL | {"variant": "text_only"}, {})
    assert "latent_grad_norm" not in text_only and model.latent_heads.mean[2].weight.grad is None


def test_checkpoint_roundtrip_keeps_heads_and_token_ids(tiny_scaffold, tiny_sample, tmp_path):
    model, prompt = tiny_scaffold.eval(), tiny_sample["prompt"]
    add_heads(model, RL)
    with torch.no_grad():
        model.latent_heads.mean[2].weight.normal_()
    model.save_pretrained(tmp_path)
    reloaded = ScaffoldingQwen.from_pretrained(tmp_path, attn_implementation="eager").eval()
    assert reloaded.config.scaffold_query_ids == QUERIES and reloaded.config.scaffold_k == 4
    for a, b in zip(model.latent_heads.state_dict().values(), reloaded.latent_heads.state_dict().values()):
        assert torch.equal(a, b)
    ids, _ = answer_sequence(prompt["input_ids"], tiny_sample["answer"], model.config)
    with torch.no_grad():
        prior = prior_of(model, prompt)
        torch.testing.assert_close(model.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], prior[None]),
                                   reloaded.hidden_states(ids, prompt["pixel_values"], prompt["image_grid_thw"], prior[None]),
                                   atol=0, rtol=0)


def test_latent_projection_starts_at_zero_and_shields_the_language_model(tiny_scaffold, tiny_sample, tmp_path):
    from spara.scaffolding.model import add_latent_projection, train_language
    model, prompt = tiny_scaffold.train(), tiny_sample["prompt"]
    target = torch.randn(4, 32, generator=torch.Generator().manual_seed(9))
    add_latent_projection(model, "linear")
    train_language(model)
    loss, stats = latent_matching(model, tiny_sample, target, "mean")
    assert abs(stats["latent_loss_normalized"] - 1.0) < 1e-6  # z = 0 at initialization
    loss.backward()
    assert model.latent_projection.weight.grad.abs().sum() > 0
    # Zero weights pass no latent gradient back into the language model yet.
    assert all(p.grad is None or p.grad.abs().sum() == 0 for name, p in model.named_parameters()
               if not name.startswith("latent_projection"))
    with torch.no_grad():
        model.latent_projection.weight.normal_(std=0.1)
    summed, _ = latent_matching(model, tiny_sample, target, "sum")
    mean, _ = latent_matching(model, tiny_sample, target, "mean")
    torch.testing.assert_close(mean, summed / 32)
    model.eval()
    with torch.no_grad():
        hidden, cache, delta = model.prefill(prompt["input_ids"], prompt["pixel_values"], prompt["image_grid_thw"])
        torch.testing.assert_close(model.latent_prior(cache, delta), prior_of(model, prompt), atol=1e-5, rtol=1e-5)
    model.save_pretrained(tmp_path)
    reloaded = ScaffoldingQwen.from_pretrained(tmp_path, attn_implementation="eager").eval()
    assert torch.equal(reloaded.latent_projection.weight, model.latent_projection.weight)
    with torch.no_grad():
        torch.testing.assert_close(prior_of(reloaded, prompt), prior_of(model, prompt), atol=0, rtol=0)
