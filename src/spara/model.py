"""LIVR on stock Qwen2.5-VL; no Mirage model. `livr_loops` adds fed-back latents."""
import contextlib

import torch
from transformers import Qwen2_5_VLForConditionalGeneration

LATENT_TOKENS = [f"<|livr_{i}|>" for i in range(4)]


def latent_queries(ids, latent_ids):
    # Only the four prompt occurrences get visual access. A generated special
    # token must not reopen the image path during Stage 1 decoding.
    queries = torch.zeros_like(ids, dtype=torch.bool)
    for token in latent_ids:
        matches = ids == token
        queries |= matches & (matches.cumsum(-1) == 1)
    return queries


def bottleneck_mask(ids, padding, latent_ids, image_id, dtype, query_start=0, image_block=True):
    """The additive attention mask: causal and padding-aware; with `image_block` (Stage 1), only the
    latents' prompt occurrences and image tokens may attend to image tokens. Returns [B, 1, Q, K];
    `allowed_mask` returns the boolean form."""
    allowed = allowed_mask(ids, padding, latent_ids, image_id, query_start, image_block)
    mask = torch.zeros(allowed.shape, dtype=dtype, device=ids.device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[:, None, :, :]


def allowed_mask(ids, padding, latent_ids, image_id, query_start=0, image_block=True):
    length = ids.shape[-1]
    key = torch.arange(length, device=ids.device)
    query = key[query_start:]
    allowed = (key[None, :] <= query[:, None])[None, :, :] & padding[:, None, :].bool()
    image = ids == image_id
    if image_block:
        permitted = latent_queries(ids, latent_ids) | image
        allowed = allowed & ~(~permitted[:, query_start:, None] & image[:, None, :])
    # Padded queries are discarded; give them a self edge to avoid NaN rows.
    padded = ~padding[:, query_start:].bool()
    return torch.where(padded[:, :, None], key[None, None, :] == query[None, :, None], allowed)


class LivrQwen(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        # Latent RL (condition C) checkpoints carry Gaussian heads; rebuild them so
        # their saved weights load instead of being dropped.
        heads = getattr(config, "livr_latent_heads", None)
        if heads and heads.get("basis_k"):  # C′'s structured heads; the basis loads with the weights
            from .latent_rl import BasisHeads
            self.latent_heads = BasisHeads(config.hidden_size, **heads)
        elif heads:
            from .scaffolding.model import GaussianHeads
            self.latent_heads = GaussianHeads(config.hidden_size, **heads)
        if getattr(config, "livr_loops", 0):  # fed-back latents: one gate per extra pass
            self.latent_gate = torch.nn.Parameter(torch.zeros(config.livr_loops))

    def prepare_inputs_for_generation(self, input_ids, **kwargs):
        result = super().prepare_inputs_for_generation(input_ids, **kwargs)
        result["livr_full_input_ids"] = input_ids
        return result

    @contextlib.contextmanager
    def _fed_back(self, positions, state, gate):
        """Inside, the latents' input embeddings get `gate` times `state` (each latent's last-layer
        state from the pass before, rescaled to its embedding's RMS) added: e + gate * s. With no
        `state` the forward is unchanged."""
        def hook(module, args, output):
            embedded = output[positions]
            fed = state.to(output.dtype)
            scale = embedded.pow(2).mean(-1, keepdim=True).sqrt() / fed.pow(2).mean(-1, keepdim=True).sqrt().clamp_min(1e-6)
            output = output.clone()
            output[positions] = embedded + gate.to(output.dtype) * fed * scale
            return output
        handle = self.model.embed_tokens.register_forward_hook(hook) if state is not None else None
        try:
            yield
        finally:
            if handle is not None:
                handle.remove()

    def forward(self, input_ids=None, attention_mask=None, position_ids=None,
                past_key_values=None, cache_position=None, livr_full_input_ids=None,
                pixel_values=None, image_grid_thw=None, labels=None, livr_grid_size=None, livr_hide_globals=None,
                **kwargs):
        loops = getattr(self.config, "livr_loops", 0)
        positions = (latent_queries(input_ids, self.config.livr_latent_ids)
                     if loops and input_ids is not None and kwargs.get("inputs_embeds") is None else None)
        if positions is not None and positions.any():
            # Fed-back latents: `loops` extra passes over the prompt before the one that answers (and
            # fills the cache). Each pass feeds the latents' last-layer states into the next pass's
            # latent inputs through a gate; at gate 0 every pass is the plain one.
            if getattr(self.config, "livr_stage", 2) == 1 or getattr(self.config, "livr_region_latents", None) \
                    or getattr(self.config, "livr_glimpse", None):
                raise ValueError("Fed-back latents run in Stage 2 with LIVR's four latents only")
            if not (positions.sum(-1) == len(self.config.livr_latent_ids)).all():
                raise ValueError("Every row of a fed-back pass carries its four latents")
            state = None
            for loop in range(loops):
                with self._fed_back(positions, state, self.latent_gate[loop - 1] if loop else None):
                    out = super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                          pixel_values=pixel_values, image_grid_thw=image_grid_thw,
                                          use_cache=False, output_hidden_states=True, return_dict=True)
                state = out.hidden_states[-1][positions]
            with self._fed_back(positions, state, self.latent_gate[loops - 1]):
                return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                                       position_ids=position_ids, past_key_values=past_key_values,
                                       cache_position=cache_position, pixel_values=pixel_values,
                                       image_grid_thw=image_grid_thw, labels=labels, **kwargs)
        # Region latents with a mask (spara.region_latents) need the explicit mask in both stages.
        region = getattr(self.config, "livr_region_latents", None) or {}
        stage_one = getattr(self.config, "livr_stage", 2) == 1 and bool(self.config.livr_latent_ids)
        glimpse = getattr(self.config, "livr_glimpse", None)  # spara.glimpse: the mask is explicit in both stages
        if stage_one or region.get("mask") or glimpse:
            if kwargs.get("inputs_embeds") is not None:
                raise ValueError("Stage 1 requires token IDs to identify visual and latent positions")
            if attention_mask is not None and attention_mask.ndim != 2:
                raise ValueError("Pass original 2-D padding to LIVR; it constructs the bottleneck")
            full_ids = input_ids if livr_full_input_ids is None else livr_full_input_ids
            padding = attention_mask if attention_mask is not None else torch.ones_like(full_ids)
            start = full_ids.shape[-1] - input_ids.shape[-1]
            if start == 0:
                position_ids, self.rope_deltas = self.get_rope_index(
                    full_ids, image_grid_thw=image_grid_thw, attention_mask=padding)
            else:
                if self.rope_deltas is None:
                    raise ValueError("Cached decode requires the matching LIVR prefill")
                positions = torch.arange(start, full_ids.shape[-1], device=input_ids.device)
                position_ids = (positions[None, :] + self.rope_deltas).unsqueeze(0).expand(3, -1, -1)
            # Under mixed precision the decoder's Q/K/V use the autocast dtype.
            dtype = torch.get_autocast_dtype(input_ids.device.type) if torch.is_autocast_enabled(input_ids.device.type) else self.dtype
            allowed = allowed_mask(full_ids, padding, self.config.livr_latent_ids, self.config.image_token_id,
                                   start, image_block=stage_one)
            if region.get("mask"):
                from .region_latents import hide_globals, restrict_patches, restrict_regions
                if region.get("layout") == "global+patch":  # tiles of each image; no map grid needed
                    allowed = restrict_patches(allowed, full_ids, start, self.config, image_grid_thw)
                else:
                    if livr_grid_size is None:
                        raise ValueError("Masked region latents need each map's grid size (livr_grid_size)")
                    allowed = restrict_regions(allowed, full_ids, start, self.config, livr_grid_size,
                                               image_grid_thw)
                if livr_hide_globals is not None:
                    allowed = hide_globals(allowed, full_ids, start, self.config, livr_hide_globals)
            if glimpse:
                from .glimpse import restrict_glimpses
                allowed = restrict_glimpses(allowed, full_ids, start, self.config)
            attention_mask = torch.zeros(allowed.shape, dtype=dtype, device=full_ids.device).masked_fill(
                ~allowed, torch.finfo(dtype).min)[:, None, :, :]
        return super().forward(input_ids=input_ids, attention_mask=attention_mask,
                               position_ids=position_ids, past_key_values=past_key_values,
                               cache_position=cache_position, pixel_values=pixel_values,
                               image_grid_thw=image_grid_thw, labels=labels, **kwargs)


def initialize_latents(model, processor):
    tokenizer = processor.tokenizer
    if any(t in tokenizer.get_vocab() for t in LATENT_TOKENS):
        raise ValueError("Initialize only a pristine base checkpoint")
    tokenizer.add_tokens(LATENT_TOKENS, special_tokens=True)
    # Qwen has spare embedding rows beyond tokenizer length. Do not shrink them.
    size = max(model.get_input_embeddings().num_embeddings, len(tokenizer))
    model.resize_token_embeddings(size, mean_resizing=False)
    ids = tokenizer.convert_tokens_to_ids(LATENT_TOKENS)
    with torch.no_grad():
        embeddings = model.get_input_embeddings().weight
        embeddings[ids] = torch.randn((4, embeddings.shape[1]), device=embeddings.device, dtype=embeddings.dtype) * model.config.initializer_range
        head = model.get_output_embeddings().weight
        if head.data_ptr() != embeddings.data_ptr():
            head[ids] = torch.randn((4, head.shape[1]), device=head.device, dtype=head.dtype) * model.config.initializer_range
    model.config.livr_latent_ids = ids
    model.config.livr_latent_tokens = LATENT_TOKENS
    model.config.livr_stage = 1
    model.config.livr_serialization = "user-end,4-latents,assistant-header-no-newline,answer,im-end"
    return ids


def enable_loops(model, loops):
    """Fed-back latents on a trained LIVR model: `loops` extra passes, each gate at 0, so the
    model answers exactly as before until training moves the gates."""
    model.config.livr_loops = loops
    model.latent_gate = torch.nn.Parameter(torch.zeros(loops, device=model.device))
    return model


def freeze_vision(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("visual."))
    return {name: p.numel() for name, p in model.named_parameters() if p.requires_grad}
