import copy
import torch
from spara.model import LivrQwen, bottleneck_mask


def test_mask_causality_padding_and_generated_latent():
    ids = torch.tensor([[0, 1, 3, 7, 10, 11, 12, 13, 8, 14, 10]])
    padding = ids.ne(0).long()
    allowed = bottleneck_mask(ids, padding, [10, 11, 12, 13], 3, torch.float32)[0, 0].eq(0)
    assert not allowed[1:, 0].any()
    assert not allowed.triu(1).any()
    assert allowed[4:8, 2].all()
    assert not allowed[[1, 3, 8, 9, 10], 2].any()
    assert allowed[8:, 4:8].all()
    cached = bottleneck_mask(ids, padding, [10, 11, 12, 13], 3, torch.float32, 10)
    torch.testing.assert_close(cached[0, 0, 0], bottleneck_mask(ids, padding, [10, 11, 12, 13], 3, torch.float32)[0, 0, -1])


def test_no_prompt_leak_and_first_answer_must_use_latents(tiny_model, tiny_batch):
    model = tiny_model.eval()
    changed = dict(tiny_batch, pixel_values=tiny_batch["pixel_values"] + 10)
    with torch.no_grad():
        original = model(**tiny_batch, output_hidden_states=True, output_attentions=True)
        altered = model(**changed, output_hidden_states=True)
    # Text/control positions before the latent region are image-independent at every layer.
    for a, b in zip(original.hidden_states, altered.hidden_states):
        torch.testing.assert_close(a[:, [0, 1, 3, 4]], b[:, [0, 1, 3, 4]], atol=0, rtol=0)
    for attn in original.attentions:
        assert attn[:, :, [0, 1, 3, 4, 9, 10, 11, 12], 2].eq(0).all()
    assert not torch.equal(original.hidden_states[-1][:, 5:9], altered.hidden_states[-1][:, 5:9])
    assert not torch.equal(original.logits[:, 10], altered.logits[:, 10])
    first_target = (tiny_batch["labels"][0] != -100).nonzero()[0].item()
    assert first_target == 11 and tiny_batch["input_ids"][0, first_target - 1].item() not in [10, 11, 12, 13]
    # Remove visual access from all four latents: now the answer is image-independent too.
    model.config.livr_latent_ids = [30, 31, 32, 33]
    with torch.no_grad():
        a = model(**tiny_batch).logits[:, 10:]
        b = model(**changed).logits[:, 10:]
    torch.testing.assert_close(a, b, atol=0, rtol=0)
    model.config.livr_stage = 2
    with torch.no_grad():
        stage2 = model(**tiny_batch, output_attentions=True)
    assert stage2.attentions[0][..., 10, 2].gt(0).all()


def test_gradients_tiny_learning_and_checkpoint(tiny_model, tiny_batch, tmp_path):
    model = tiny_model.train()
    frozen = {n: p.detach().clone() for n, p in model.named_parameters() if not p.requires_grad}
    initial = model.model.embed_tokens.weight.detach().clone()
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=0.005)
    losses = []
    for _ in range(16):
        optimizer.zero_grad()
        loss = model(**tiny_batch, use_cache=False).loss
        losses.append(loss.item()); loss.backward()
        assert model.model.embed_tokens.weight.grad[10:14].abs().sum(dim=-1).gt(0).all()
        optimizer.step()
    assert losses[-1] < losses[0] * 0.5
    assert not torch.equal(initial[10:14], model.model.embed_tokens.weight[10:14])
    for name, value in frozen.items():
        torch.testing.assert_close(value, dict(model.named_parameters())[name], atol=0, rtol=0)
    model.eval().save_pretrained(tmp_path)
    reloaded = LivrQwen.from_pretrained(tmp_path, attn_implementation="eager").eval()
    assert reloaded.config.livr_latent_ids == [10, 11, 12, 13]
    with torch.no_grad():
        torch.testing.assert_close(model(**tiny_batch).logits, reloaded(**tiny_batch).logits, atol=0, rtol=0)


def test_cached_uncached_and_sdpa(tiny_model, tiny_batch):
    model = tiny_model.eval()
    prefix = {k: v for k, v in tiny_batch.items() if k != "labels"}
    prefix["input_ids"] = prefix["input_ids"][:, :11]
    prefix["attention_mask"] = prefix["attention_mask"][:, :11]
    for stage in [1, 2]:
        model.config.livr_stage = stage
        with torch.no_grad():
            first = model(**prefix, use_cache=True)
            full = torch.cat([prefix["input_ids"], torch.tensor([[14]])], dim=1)
            cached = model(input_ids=full[:, -1:], livr_full_input_ids=full, attention_mask=torch.ones_like(full),
                           past_key_values=first.past_key_values, cache_position=torch.tensor([11]), use_cache=True)
            uncached = model(**dict(prefix, input_ids=full, attention_mask=torch.ones_like(full)), use_cache=False)
        torch.testing.assert_close(cached.logits[:, -1], uncached.logits[:, -1], atol=1e-6, rtol=1e-5)
        with torch.no_grad():
            # Random tiny weights can emit image markers, unsupported as generated
            # image inputs by upstream Qwen. Restrict this synthetic fixture only.
            a = model.generate(**prefix, max_new_tokens=3, do_sample=False, use_cache=True, suppress_tokens=[2, 3, 4, 5])
            b = model.generate(**prefix, max_new_tokens=3, do_sample=False, use_cache=False, suppress_tokens=[2, 3, 4, 5])
        assert torch.equal(a, b)
    # Separate model construction is needed to select each attention implementation.
    config = copy.deepcopy(model.config); config._attn_implementation = "sdpa"; config.livr_stage = 1
    other = LivrQwen(config).eval(); other.load_state_dict(model.state_dict())
    model.config.livr_stage = 1
    with torch.no_grad():
        torch.testing.assert_close(other(**tiny_batch).logits, model(**tiny_batch).logits, atol=1e-6, rtol=1e-5)


def test_cache_validation_contract(tiny_model, tiny_batch):
    from spara.validation import cache_agreement
    prompt = {k: v for k, v in tiny_batch.items() if k != "labels"}
    prompt["input_ids"] = prompt["input_ids"][:, :11]
    prompt["attention_mask"] = prompt["attention_mask"][:, :11]
    for stage in (1, 2):
        tiny_model.config.livr_stage = stage
        result = cache_agreement(tiny_model.eval(), prompt)
        assert result["fp32"]["max_abs_logit_error"] < 0.0002
        assert result["bf16"]["argmax_agrees"]
        if stage == 2: assert result["bf16"]["stock_qwen_exact_match"]
