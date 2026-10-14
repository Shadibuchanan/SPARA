"""Scaffolding Minds on stock Qwen2.5-VL: latent slots, scaffolding encoder, Gaussian heads.

One latent block per trace, as in the paper's FrozenLake setting:
[x, <BOT>, z_1..z_K, <EOT>, y]. Latent vectors are scattered into placeholder
slots right after the image merge, so M-RoPE positions come from token IDs.
"""
import json
import math
from pathlib import Path

import torch
from safetensors.torch import load_file, save_file
from torch import nn
from transformers import Qwen2_5_VLForConditionalGeneration
from transformers.cache_utils import DynamicCache
from transformers.models.qwen2_5_vl.configuration_qwen2_5_vl import Qwen2_5_VLVisionConfig
from transformers.models.qwen2_5_vl.modeling_qwen2_5_vl import Qwen2_5_VisionTransformerPretrainedModel

BOT, EOT, LATENT = "<|scaffold_bot|>", "<|scaffold_eot|>", "<|scaffold_latent|>"


def query_tokens(k):
    return [f"<|scaffold_query_{i}|>" for i in range(k)]


def scaffold_ids(config):
    return [config.scaffold_bot_id, config.scaffold_eot_id, config.scaffold_latent_id, *config.scaffold_query_ids]


def mask_dtype(model, device):
    # Under mixed precision the decoder's Q/K/V use the autocast dtype.
    return torch.get_autocast_dtype(device.type) if torch.is_autocast_enabled(device.type) else model.dtype


def query_mask(length, query_start, dtype, device, past=0):
    """Causal rows, except the trailing query rows from `query_start` on, which
    read every cached/earlier key and each other bidirectionally (App. D.1)."""
    keys = torch.arange(past + length, device=device)
    rows = torch.arange(past, past + length, device=device)
    allowed = keys[None, :] <= rows[:, None]
    allowed[query_start:] = True
    mask = torch.zeros(allowed.shape, dtype=dtype, device=device)
    return mask.masked_fill(~allowed, torch.finfo(dtype).min)[None, None]


def never_sampled(config):
    """Scaffold structure plus image/video markers, which an uncached scoring pass would
    otherwise read as visual inputs."""
    markers = [getattr(config, name, None) for name in
               ("vision_start_token_id", "vision_end_token_id", "image_token_id", "video_token_id")]
    return [config.scaffold_eot_id, config.scaffold_latent_id, *config.scaffold_query_ids,
            *[marker for marker in markers if marker is not None]]


def suppression(logits, config, after_block):
    """Never-sampled tokens are blocked, and BOT once the single block is used.
    Applied identically when sampling and when scoring, so ratios stay exact."""
    blocked = torch.zeros(logits.shape[-1], dtype=torch.bool, device=logits.device)
    blocked[never_sampled(config)] = True
    blocked = blocked.expand(logits.shape).clone()
    blocked[..., config.scaffold_bot_id] = torch.as_tensor(after_block, device=logits.device)
    return logits.masked_fill(blocked, float("-inf"))


class GaussianHeads(nn.Module):
    """Input-adaptive mean and scale of residual latent actions (eq. 3)."""

    def __init__(self, size, hidden_size, sigma_init, log_sigma_bounds):
        super().__init__()
        self.mean = nn.Sequential(nn.Linear(size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, size))
        self.log_sigma = nn.Sequential(nn.Linear(size, hidden_size), nn.SiLU(), nn.Linear(hidden_size, size))
        self.sigma_init, self.bounds = sigma_init, tuple(log_sigma_bounds)

    def reset_parameters(self):
        for head in (self.mean, self.log_sigma):
            nn.init.normal_(head[0].weight, std=0.02)
            nn.init.zeros_(head[0].bias)
            nn.init.zeros_(head[2].weight)
        # Zero mean centres exploration on the latent prior; sigma starts small (App. D.1).
        nn.init.zeros_(self.mean[2].bias)
        nn.init.constant_(self.log_sigma[2].bias, math.log(self.sigma_init))

    def forward(self, prior):
        prior = prior.float()
        with torch.autocast(prior.device.type, enabled=False):
            return self.mean(prior), self.log_sigma(prior).clamp(*self.bounds).exp()


class ScaffoldingQwen(Qwen2_5_VLForConditionalGeneration):
    def __init__(self, config):
        super().__init__(config)
        if getattr(config, "scaffold_heads", None):
            self.latent_heads = GaussianHeads(config.hidden_size, **config.scaffold_heads)

    def embed(self, input_ids, pixel_values=None, image_grid_thw=None, latents=None):
        embeds = self.model.embed_tokens(input_ids)
        if pixel_values is not None:
            image = input_ids == self.config.image_token_id
            with torch.no_grad():  # the VLM's own vision tower is frozen in every phase
                features = self.visual(pixel_values.type(self.visual.dtype), grid_thw=image_grid_thw)
            if features.shape[0] != int(image.sum()):
                raise ValueError("Image features and image tokens do not match")
            embeds = embeds.masked_scatter(image[..., None], features.to(embeds.dtype))
        slots = input_ids == self.config.scaffold_latent_id
        count = int(slots.sum())
        vectors = None if latents is None else latents.reshape(-1, embeds.shape[-1])
        if (vectors is None and count) or (vectors is not None and vectors.shape[0] != count):
            raise ValueError("Latent vectors must fill every latent slot exactly")
        if count:
            embeds = embeds.masked_scatter(slots[..., None], vectors.to(embeds.dtype))
        return embeds

    def hidden_states(self, input_ids, pixel_values=None, image_grid_thw=None, latents=None, query_start=None):
        """Uncached pass over complete, unpadded sequences (training and scoring)."""
        position_ids, _ = self.get_rope_index(input_ids, image_grid_thw)
        embeds = self.embed(input_ids, pixel_values, image_grid_thw, latents)
        mask = None if query_start is None else query_mask(
            input_ids.shape[1], query_start, mask_dtype(self, embeds.device), embeds.device)
        return self.model(inputs_embeds=embeds, position_ids=position_ids, attention_mask=mask,
                          use_cache=False).last_hidden_state

    @torch.no_grad()
    def prefill(self, input_ids, pixel_values=None, image_grid_thw=None):
        position_ids, delta = self.get_rope_index(input_ids, image_grid_thw)
        cache = DynamicCache()
        embeds = self.embed(input_ids, pixel_values, image_grid_thw)
        hidden = self.model(inputs_embeds=embeds, position_ids=position_ids, past_key_values=cache,
                            use_cache=True).last_hidden_state
        return hidden, cache, delta

    @torch.no_grad()
    def extend(self, input_ids, cache, delta, latents=None, query_start=None):
        """Append tokens after a prefill; later positions continue as text."""
        if delta.numel() != 1:
            raise ValueError("Cached rows must share one prompt")
        past, length = cache.get_seq_length(), input_ids.shape[1]
        cache_position = torch.arange(past, past + length, device=input_ids.device)
        position_ids = (cache_position + delta.reshape(()).to(input_ids.device)).view(1, 1, -1)
        position_ids = position_ids.expand(3, input_ids.shape[0], -1)
        embeds = self.embed(input_ids, latents=latents)
        mask = query_mask(length, length if query_start is None else query_start,
                          mask_dtype(self, embeds.device), embeds.device, past)
        return self.model(inputs_embeds=embeds, position_ids=position_ids, attention_mask=mask,
                          past_key_values=cache, cache_position=cache_position, use_cache=True).last_hidden_state

    @torch.no_grad()
    def latent_prior(self, cache, delta):
        """Feed the sampled BOT and the K query embeddings, then drop the queries from
        the cache: no later token attends to them. Returns z (K, d)."""
        c = self.config
        past = cache.get_seq_length()
        ids = torch.tensor([[c.scaffold_bot_id, *c.scaffold_query_ids]], device=self.device)
        hidden = self.extend(ids, cache, delta, query_start=1)
        cache.crop(past + 1)
        return hidden[0, 1:]

    @torch.no_grad()
    def append_latent(self, cache, delta, latents):
        """Write latent actions (n, K, d) into the slots after BOT, then EOT."""
        c = self.config
        ids = torch.tensor([[c.scaffold_latent_id] * c.scaffold_k + [c.scaffold_eot_id]], device=self.device)
        return self.extend(ids.expand(latents.shape[0], -1), cache, delta, latents=latents)


def answer_sequence(prompt_ids, answer_ids, config):
    """[x, BOT, latent slots, EOT, y]; returns the IDs and the index of BOT."""
    block = [config.scaffold_bot_id] + [config.scaffold_latent_id] * config.scaffold_k + [config.scaffold_eot_id]
    block = torch.tensor([block], device=prompt_ids.device)
    return torch.cat([prompt_ids, block, answer_ids.to(prompt_ids.device)], dim=1), prompt_ids.shape[1]


def predictor_sequence(prefix_ids, config):
    """[x, BOT, e_1..e_K]: the queries occupy the positions the latent slots will take."""
    queries = torch.tensor([[config.scaffold_bot_id, *config.scaffold_query_ids]], device=prefix_ids.device)
    return torch.cat([prefix_ids, queries], dim=1)


def token_tables(model):
    tables = [model.get_input_embeddings().weight]
    head = model.get_output_embeddings().weight
    if head.data_ptr() != tables[0].data_ptr():
        tables.append(head)
    return tables


def initialize_tokens(model, ids, k):
    """BOT/EOT copy the vision start/end rows so the frozen VLM reads z* between familiar
    delimiters (assumption; the paper is silent). Slot and query rows are random, as in LIVR."""
    bot, eot, latent, *queries = ids
    if len(queries) != k:
        raise ValueError("Expected one query token per latent")
    with torch.no_grad():
        for table in token_tables(model):
            table[bot] = table[model.config.vision_start_token_id]
            table[eot] = table[model.config.vision_end_token_id]
            table[[latent, *queries]] = torch.randn((k + 1, table.shape[1]), device=table.device,
                                                    dtype=table.dtype) * model.config.initializer_range
    model.config.scaffold_bot_id, model.config.scaffold_eot_id, model.config.scaffold_latent_id = bot, eot, latent
    model.config.scaffold_query_ids = list(queries)
    model.config.scaffold_k = k


def add_scaffold_tokens(model, processor, k):
    tokenizer = processor.tokenizer
    names = [BOT, EOT, LATENT, *query_tokens(k)]
    if any(name in tokenizer.get_vocab() for name in names):
        raise ValueError("Initialize only a pristine base checkpoint")
    tokenizer.add_tokens(names, special_tokens=True)
    # Qwen has spare embedding rows beyond tokenizer length. Do not shrink them.
    model.resize_token_embeddings(max(model.get_input_embeddings().num_embeddings, len(tokenizer)), mean_resizing=False)
    initialize_tokens(model, tokenizer.convert_tokens_to_ids(names), k)
    model.config.scaffold_tokens = names


def token_rows(model):
    ids = scaffold_ids(model.config)
    return {"ids": ids, "rows": [table[ids].detach().cpu().clone() for table in token_tables(model)]}


def load_token_rows(model, saved):
    """Reuse the exact rows the frozen VLM saw while the encoder was trained."""
    if saved["ids"] != scaffold_ids(model.config):
        raise ValueError("Scaffold token IDs differ from the Scaffolding Phase")
    with torch.no_grad():
        for table, rows in zip(token_tables(model), saved["rows"], strict=True):
            table[saved["ids"]] = rows.to(table)


def train_language(model):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(not name.startswith("visual."))


def add_heads(model, rl):
    settings = {"hidden_size": rl["head_hidden_size"] or model.config.hidden_size,
                "sigma_init": rl["sigma_init"], "log_sigma_bounds": list(rl["log_sigma_bounds"])}
    model.config.scaffold_heads = settings
    model.latent_heads = GaussianHeads(model.config.hidden_size, **settings).to(
        device=next(model.parameters()).device, dtype=model.dtype)
    model.latent_heads.reset_parameters()


class AttentionPooling(nn.Module):
    """K learnable embeddings read one helper image's tokens and each other (App. D.1)."""

    def __init__(self, size, k, heads):
        super().__init__()
        self.queries = nn.Parameter(torch.randn(k, size) * 0.02)
        self.norm_image = nn.LayerNorm(size)
        self.norm_query = nn.LayerNorm(size)
        self.attention = nn.MultiheadAttention(size, heads, batch_first=True)
        self.norm_mlp = nn.LayerNorm(size)
        self.mlp = nn.Sequential(nn.Linear(size, size), nn.GELU(), nn.Linear(size, size))

    def forward(self, image_tokens):
        queries = self.queries[None]
        normed = self.norm_query(queries)
        keys = torch.cat([self.norm_image(image_tokens[None]), normed], dim=1)
        hidden = queries + self.attention(normed, keys, keys, need_weights=False)[0]
        return (hidden + self.mlp(self.norm_mlp(hidden)))[0]


class ScaffoldingEncoder(nn.Module):
    """psi: a trainable copy of the VLM vision tower plus attention pooling (§3.2).
    Used only in Stage 1; `context` keeps the token rows the frozen VLM saw."""

    def __init__(self, vision_config, size, k, heads, attention="sdpa"):
        super().__init__()
        self.settings = {"vision_config": vision_config.to_dict(), "size": size, "k": k, "heads": heads,
                         "attention": attention}
        self.visual = Qwen2_5_VisionTransformerPretrainedModel._from_config(
            vision_config, torch_dtype=torch.float32, attn_implementation=attention)
        self.pooling = AttentionPooling(size, k, heads)
        self.context = None

    @classmethod
    def from_vlm(cls, vlm, k, heads):
        encoder = cls(vlm.config.vision_config, vlm.config.hidden_size, k, heads,
                      vlm.config.vision_config._attn_implementation)
        encoder.visual.load_state_dict(vlm.visual.state_dict())
        encoder.context = token_rows(vlm)
        return encoder.to(device=vlm.device, dtype=vlm.dtype)

    def forward(self, pixel_values, image_grid_thw):
        if image_grid_thw.shape[0] != 1:
            raise ValueError("Expected one helper image (one latent block) per trace")
        return self.pooling(self.visual(pixel_values.type(self.visual.dtype), grid_thw=image_grid_thw))

    def save_pretrained(self, path):
        path = Path(path)
        save_file({k: v.detach().contiguous().cpu() for k, v in self.state_dict().items()},
                  str(path / "encoder.safetensors"))
        (path / "encoder.json").write_text(json.dumps(self.settings, indent=2) + "\n", encoding="utf-8")
        torch.save(self.context, path / "scaffold_tokens.pt")

    @classmethod
    def from_pretrained(cls, path, device="cpu"):
        settings = json.loads((Path(path) / "encoder.json").read_text(encoding="utf-8"))
        encoder = cls(Qwen2_5_VLVisionConfig(**settings.pop("vision_config")), **settings)
        encoder.load_state_dict(load_file(str(Path(path) / "encoder.safetensors")))
        encoder.context = torch.load(Path(path) / "scaffold_tokens.pt", map_location="cpu", weights_only=True)
        return encoder.to(device)
