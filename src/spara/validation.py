"""Cache correctness separately from reduced-precision kernel roundoff."""
import contextlib
import torch
from transformers import Qwen2_5_VLForConditionalGeneration
from .training import amp


@torch.no_grad()
def cache_agreement(model, prompt):
    results = {}
    for precision in ("bf16", "fp32"):
        context = amp(model.device) if precision == "bf16" else contextlib.nullcontext()
        with context:
            first = model(**prompt, use_cache=True)
            token = first.logits[:, -1].argmax(dim=-1, keepdim=True)
            full = torch.cat([prompt["input_ids"], token], dim=-1)
            cached = model(input_ids=token, attention_mask=torch.ones_like(full), livr_full_input_ids=full,
                           cache_position=torch.tensor([prompt["input_ids"].shape[-1]], device=model.device),
                           past_key_values=first.past_key_values, use_cache=True).logits[:, -1].float()
            uncached = model(**dict(prompt, input_ids=full, attention_mask=torch.ones_like(full)), use_cache=False).logits[:, -1].float()
        difference = (cached - uncached).abs()
        # Double precision avoids cancellation for almost-identical distributions.
        p, q = cached.double().log_softmax(-1), uncached.double().log_softmax(-1)
        kl = float((p.exp() * (p - q)).sum())
        result = {"max_abs_logit_error": float(difference.max()),
                  "rms_logit_error": float(difference.square().mean().sqrt()),
                  "kl_cached_to_uncached": kl, "argmax_agrees": bool(cached.argmax() == uncached.argmax())}
        if precision == "fp32":
            # Stage 2 is additionally required to equal upstream exactly below.
            # Use a fixed 1e-3 bound for full-size FP32 prefill/decode reductions;
            # tiny-model tests retain their independent 1e-6 bound.
            result["tolerance"] = {"atol": 0.001, "rtol": 0.0001}
        else:
            # Reduced-precision prefill and one-token GEMMs need not round alike.
            # Report their difference; correctness is checked in FP32, and the
            # normal Stage 2 path must also match stock Qwen exactly in BF16.
            result["diagnostic_only"] = True
        if model.config.livr_stage == 2:
            forward = Qwen2_5_VLForConditionalGeneration.forward
            with amp(model.device) if precision == "bf16" else contextlib.nullcontext():
                stock_first = forward(model, **prompt, use_cache=True)
                stock_cached = forward(model, input_ids=token, attention_mask=torch.ones_like(full),
                    cache_position=torch.tensor([prompt["input_ids"].shape[-1]], device=model.device),
                    past_key_values=stock_first.past_key_values, use_cache=True).logits[:, -1].float()
                stock_uncached = forward(model, **dict(prompt, input_ids=full, attention_mask=torch.ones_like(full)), use_cache=False).logits[:, -1].float()
            torch.testing.assert_close(cached, stock_cached, atol=0, rtol=0)
            torch.testing.assert_close(uncached, stock_uncached, atol=0, rtol=0)
            result["stock_qwen_exact_match"] = True
        if precision == "fp32":
            torch.testing.assert_close(cached, uncached, atol=0.001, rtol=0.0001)
        results[precision] = result
    return results
