"""SPARA latent RL (condition C): condition B's text GRPO plus Gaussian adjustments
to LIVR's latent states.

The four latent states at the output of decoder layer `layer` (one-based) are
replaced by a = z + delta with delta ~ N(mu(z), sigma(z)^2), where z is that
layer's output there and mu, sigma come from Scaffolding's Gaussian heads. Later
layers, and every token after the latents, read a; zero adjustment is LIVR. A
final-layer adjustment would change nothing, since later tokens read a position
only through the layers before the last.
"""
import contextlib
import hashlib
import json
import math
import random
from pathlib import Path

import torch

from .data import records
from .evaluation import ReferenceEvaluator
from .model import latent_queries
from .runtime import json_write
from .scaffolding.model import GaussianHeads
from .scaffolding.policy import clipped_fraction, clipped_surrogate, gaussian_log_prob
from .serialization import encode
from .training import amp, to_device

PHASE = "latent-rl"


def head_input(prior):
    """Heads read each latent RMS-normalised. Mid-layer channels span about 1 to 280,
    whereas Scaffolding's heads read final-norm states."""
    prior = prior.float()
    return prior * prior.square().mean(-1, keepdim=True).add(1e-6).rsqrt()


def sigma_bounds(sigma_init, ceiling_ratio=4.0):
    """log(sigma) bounds: Scaffolding's floor, and a ceiling at `ceiling_ratio` x sigma_init
    (the Scaffolding rerun's 1.0 init with a ceiling of 4)."""
    return [-9.21, round(math.log(ceiling_ratio * sigma_init), 4)]


class BasisHeads(torch.nn.Module):
    """C′'s Gaussian heads (docs/STRUCTURED_EXPLORATION_SPEC.md, 2026-11-17 amendment): for
    each latent slot, the mean and scale of `basis_k` coefficients in that slot's principal
    basis, in units of the across-map spread along each direction. The adjustment is
    delta = sum_j u_j s_j c_j with c ~ N(mu, sigma^2), so mu = 0 and sigma = c reproduce the
    structured noise of explore_probe.structured_offsets at scale c."""

    def __init__(self, size, hidden_size, sigma_init, log_sigma_bounds, slots, basis_k):
        super().__init__()
        self.sigma_init, self.bounds, self.basis_k = sigma_init, tuple(log_sigma_bounds), basis_k
        self.register_buffer("directions", torch.zeros(slots, size, basis_k))
        self.register_buffer("scales", torch.ones(slots, basis_k))
        self.mean_in, self.log_sigma_in = torch.nn.Linear(size, hidden_size), torch.nn.Linear(size, hidden_size)
        self.mean_out = torch.nn.Parameter(torch.zeros(slots, hidden_size, basis_k))
        self.mean_bias = torch.nn.Parameter(torch.zeros(slots, basis_k))
        self.log_sigma_out = torch.nn.Parameter(torch.zeros(slots, hidden_size, basis_k))
        self.log_sigma_bias = torch.nn.Parameter(torch.zeros(slots, basis_k))

    def reset_parameters(self):
        for layer in (self.mean_in, self.log_sigma_in):
            torch.nn.init.normal_(layer.weight, std=0.02)
            torch.nn.init.zeros_(layer.bias)
        for weight in (self.mean_out, self.mean_bias, self.log_sigma_out):
            torch.nn.init.zeros_(weight)
        # Zero mean centres exploration on LIVR's latents; sigma starts at the chosen scale.
        torch.nn.init.constant_(self.log_sigma_bias, math.log(self.sigma_init))

    def set_basis(self, basis):
        directions, scales = basis["directions"], basis["scales"]
        if directions.shape != self.directions.shape or scales.shape != self.scales.shape:
            raise ValueError(f"Basis shapes {tuple(directions.shape)}, {tuple(scales.shape)} do not match the heads'")
        self.directions.copy_(directions)
        self.scales.copy_(scales)

    def forward(self, prior):
        prior = prior.float()
        with torch.autocast(prior.device.type, enabled=False):
            hidden = torch.nn.functional.silu(self.mean_in(prior))
            mean = torch.einsum("nsh,shk->nsk", hidden, self.mean_out) + self.mean_bias
            hidden = torch.nn.functional.silu(self.log_sigma_in(prior))
            log_sigma = torch.einsum("nsh,shk->nsk", hidden, self.log_sigma_out) + self.log_sigma_bias
            return mean, log_sigma.clamp(*self.bounds).exp()

    def offset(self, coefficients):
        """Coefficients (n, K, k) to a latent adjustment (n, K, d)."""
        return torch.einsum("sdk,sk,nsk->nsd", self.directions, self.scales, coefficients.float())

    def coefficients(self, delta):
        """The adjustment's coordinates in the basis: the inverse of `offset` on the subspace."""
        return torch.einsum("sdk,nsd->nsk", self.directions, delta.float()) / self.scales


def add_latent_heads(model, latent, basis=None):
    """Attach freshly initialised heads and record them in the config, so saved
    checkpoints rebuild them on load (LivrQwen.__init__). With `basis_k` in `latent`
    (C′), the heads act in `basis` (explore_probe.principal_basis)."""
    settings = {"hidden_size": latent["head_hidden_size"] or model.config.hidden_size,
                "sigma_init": latent["sigma_init"], "log_sigma_bounds": list(latent["log_sigma_bounds"])}
    if latent.get("basis_k"):
        if basis is None:
            raise ValueError("Structured latent heads need the pilot's basis")
        settings |= {"slots": len(model.config.livr_latent_ids), "basis_k": latent["basis_k"]}
        heads = BasisHeads(model.config.hidden_size, **settings)
    else:
        heads = GaussianHeads(model.config.hidden_size, **settings)
    model.config.livr_latent_heads = settings
    model.config.livr_latent_layer = latent["layer"]
    model.latent_heads = heads.to(device=next(model.parameters()).device, dtype=torch.float32)
    model.latent_heads.reset_parameters()
    if basis is not None:
        model.latent_heads.set_basis({k: basis[k].to(model.latent_heads.directions.device) for k in ("directions", "scales")})
    return model.latent_heads


class LatentAdjuster:
    """Hooks that replace the latent states at one layer's output. Inactive outside
    the `sample`, `mean`, `offset` and `replay` contexts; `capture` only records them.

    Latent positions are found from each forward's input IDs, so one context can
    span many prompts (evaluation). A cached decoding step has no latents and is
    left alone. Under gradient checkpointing the layer hook runs again during
    backward; it repeats the same replacement, so keep `replay` open until
    `backward` returns.
    """

    def __init__(self, model, layer=None):
        layer = layer or model.config.livr_latent_layer
        layers = model.model.layers
        if not 1 <= layer < len(layers):
            raise ValueError("The adjusted layer must be followed by another decoder layer")
        self.model, self.layer, self.state, self.positions = model, layer, None, None
        self.handles = [model.register_forward_pre_hook(self._capture, with_kwargs=True),
                        layers[layer - 1].register_forward_hook(self._hook)]

    def _capture(self, module, args, kwargs):
        ids = kwargs.get("input_ids", args[0] if args else None)
        self.positions = None
        if self.state is None or ids is None:
            return
        latent_ids = self.model.config.livr_latent_ids
        found = latent_queries(ids[:1], latent_ids)[0].nonzero().squeeze(-1)
        if len(found) == len(latent_ids):
            self.positions = found

    def _hook(self, module, args, output):
        state, positions = self.state, self.positions
        hidden = output[0] if isinstance(output, tuple) else output
        if state is None or positions is None or hidden.shape[1] <= int(positions[-1]):
            return None
        positions = positions.to(hidden.device)
        prior = hidden[:, positions]
        mode = state["mode"]
        if mode == "capture":
            state["priors"].append(prior.detach())
            return None
        if mode == "replay":
            state["prior"] = prior
            actions = state["actions"].to(hidden.device)
        elif mode == "offset":
            actions = prior.float() + state["offsets"].to(hidden.device)
        else:
            heads = self.model.latent_heads
            mean, sigma = heads(head_input(prior))
            drawn = mean + sigma * torch.randn_like(mean) if mode == "sample" else mean
            # Structured heads (C′) draw basis coefficients; Scaffolding's draw the adjustment itself.
            actions = prior.float() + (heads.offset(drawn) if isinstance(heads, BasisHeads) else drawn)
            state["records"].append({"actions": actions.detach(), "mean": mean.detach(), "sigma": sigma.detach()})
        if actions.shape != prior.shape:
            raise ValueError(f"Expected latent actions of shape {tuple(prior.shape)}, got {tuple(actions.shape)}")
        replaced = hidden.index_copy(1, positions, actions.to(hidden.dtype))
        return (replaced, *output[1:]) if isinstance(output, tuple) else replaced

    @contextlib.contextmanager
    def _active(self, **state):
        self.state = state
        try:
            yield state
        finally:
            self.state = self.positions = None

    def sample(self):
        """a = z + mu + sigma * eps for every batch row (rollouts, sampled evaluation)."""
        return self._active(mode="sample", records=[])

    def mean(self):
        """a = z + mu (deterministic evaluation)."""
        return self._active(mode="mean", records=[])

    def capture(self):
        """Record z (rows, K, d) without changing it (structured-noise basis)."""
        return self._active(mode="capture", priors=[])

    def offset(self, offsets):
        """a = z + offsets, offsets (rows, K, d): fixed adjustments without heads (probe)."""
        return self._active(mode="offset", offsets=offsets)

    def replay(self, actions):
        """a = actions (rows, K, d), e.g. recorded at rollout; records z as `prior`."""
        return self._active(mode="replay", actions=actions)

    def remove(self):
        for handle in self.handles:
            handle.remove()


def tensor_digest(family):
    """sha256 of a family's tensors (float32, sorted keys): shows two runs drew from the same basis."""
    digest = hashlib.sha256()
    for key in sorted(family):
        if torch.is_tensor(family[key]):
            digest.update(key.encode())
            digest.update(family[key].detach().float().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def noise_offsets(family, scale, count, generator, device):
    """C″'s fixed latent noise: `count` offsets (count, K, d) = scale * sum_j u_j w_j eps_j per
    slot, eps ~ N(0, 1), in a norm-matched principal basis (latent_diagnostics.matched); also
    eps (count, K, k). The same draw as latent_diagnostics.draw for the "slots" family."""
    if family["kind"] != "slots":
        raise ValueError("C″ draws in a per-slot principal basis")
    directions, weights = family["directions"].to(device), family["weights"].to(device)
    eps = torch.randn((count, *weights.shape), generator=generator, device=device)
    return scale * torch.einsum("sdk,sk,nsk->nsd", directions, weights, eps), eps


def replaying(adjuster, actions):
    return adjuster.replay(actions) if adjuster is not None else contextlib.nullcontext({})


def latent_objective(model, prior, action, advantage, latent, epsilon):
    """Scaffolding's latent term (eq. 6): clipped ratio of the Gaussian log-probability
    of the rollout's action given the current z, mu and sigma."""
    heads = model.latent_heads
    mean, sigma = heads(head_input(prior))
    if isinstance(heads, BasisHeads):
        # The rollout's action relative to the current prior, in the basis (C′): the same
        # residual policy on 64 coordinates. Out-of-basis drift of the prior is not scored.
        coefficients = heads.coefficients(action.float() - prior.float())
        log_prob = gaussian_log_prob(coefficients, torch.zeros_like(coefficients), mean, sigma,
                                     latent["logprob_reduction"])
    else:
        log_prob = gaussian_log_prob(action, prior, mean, sigma, latent["logprob_reduction"])
    log_ratio = log_prob - log_prob.detach()
    surrogate = clipped_surrogate(log_ratio, torch.as_tensor(advantage, device=log_prob.device), epsilon)
    stats = {"sigma_mean": sigma.mean().item(), "mean_abs_mu": mean.abs().mean().item(),
             "latent_clipped": clipped_fraction(log_ratio, epsilon).item()}
    return -surrogate, stats


def moves(answer):
    """The reference parser's view of a route: every U, D, R or L character."""
    return "".join(ch for ch in answer.upper() if ch in "UDRL")


def choose_sigma(summary, change=0.10, drop=0.05):
    """The smallest sigma that changes at least `change` of greedy routes without
    lowering success by more than `drop` (Scaffolding's rule, fixed before its probe)."""
    base = summary["base_success_rate"]
    for sigma, cell in sorted(summary["sigmas"].items(), key=lambda item: float(item[0])):
        # The tolerance keeps exact thresholds (e.g. a 5-point drop) from failing on rounding.
        if cell["route_change_rate"] >= change - 1e-9 and base - cell["success_rate"] <= drop + 1e-9:
            return float(sigma)
    return None


@torch.no_grad()
def latent_probe(model, processor, data_root, output, sigmas, rows, draws, seed, run_manifest, max_new_tokens=64):
    """Greedy routes on `rows` random training maps with no adjustment and with `draws`
    isotropic adjustments of each scale sigma at the latent layer. Greedy decoding
    makes every route change come from the adjustment."""
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    train = records(data_root, "train")
    chosen = sorted(random.Random(seed).sample(range(len(train)), rows))
    json_write(output / "manifest.json", run_manifest | {"sigmas": sigmas, "rows": chosen, "draws": draws,
                                                         "layer": model.config.livr_latent_layer})
    evaluator, adjuster = ReferenceEvaluator(data_root), LatentAdjuster(model)
    generator = torch.Generator(device=model.device).manual_seed(seed)
    settings = {"do_sample": False, "temperature": None, "top_p": None, "top_k": None,
                "max_new_tokens": max_new_tokens}
    model.eval()
    results = []
    with (output / "rows.jsonl").open("w", encoding="utf-8") as log:
        for count, index in enumerate(chosen):
            row = train[index]
            prompt = to_device(encode(processor, data_root, "train", row, training=False), model.device)
            size = model.config.hidden_size
            k = len(model.config.livr_latent_ids)
            scale = torch.tensor([0.0] + [s for s in sigmas for _ in range(draws)], device=model.device)
            offsets = scale[:, None, None] * torch.randn((len(scale), k, size), generator=generator, device=model.device)
            batch = {key: value.repeat(len(scale), 1) for key, value in prompt.items()}
            with adjuster.offset(offsets), amp(model.device):
                generated = model.generate(**batch, **settings, use_cache=True)
            scored = []
            for sequence in generated:
                result = evaluator.score(row, processor.tokenizer.decode(sequence, skip_special_tokens=False))
                scored.append((moves(result["extracted_answer"]), bool(result["success"])))
            (route, success), adjusted = scored[0], scored[1:]
            record = {"row_index": index, "grid_size": len(row["map_desc"]), "route": route, "success": success,
                      "sigmas": {str(s): {"route_changed": [adjusted[i * draws + j][0] != route for j in range(draws)],
                                          "success": [adjusted[i * draws + j][1] for j in range(draws)]}
                                 for i, s in enumerate(sigmas)}}
            results.append(record)
            log.write(json.dumps(record) + "\n"); log.flush()
            if count % 25 == 0:
                print(f"probe {count + 1}/{len(chosen)}", flush=True)
    adjuster.remove()
    summary = {"rows": len(results), "draws": draws, "layer": model.config.livr_latent_layer,
               "base_success_rate": sum(r["success"] for r in results) / len(results), "sigmas": {}}
    for sigma in map(str, sigmas):
        changed = [c for r in results for c in r["sigmas"][sigma]["route_changed"]]
        successes = [s for r in results for s in r["sigmas"][sigma]["success"]]
        summary["sigmas"][sigma] = {
            "route_change_rate": sum(changed) / len(changed),
            "rows_with_any_change": sum(any(r["sigmas"][sigma]["route_changed"]) for r in results),
            "success_rate": sum(successes) / len(successes)}
    summary["chosen_sigma"] = choose_sigma(summary)
    json_write(output / "summary.json", summary)
    print(json.dumps(summary, indent=1), flush=True)
    return summary
