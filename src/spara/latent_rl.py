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


def add_latent_heads(model, latent):
    """Attach freshly initialised heads and record them in the config, so saved
    checkpoints rebuild them on load (LivrQwen.__init__)."""
    settings = {"hidden_size": latent["head_hidden_size"] or model.config.hidden_size,
                "sigma_init": latent["sigma_init"], "log_sigma_bounds": list(latent["log_sigma_bounds"])}
    model.config.livr_latent_heads = settings
    model.config.livr_latent_layer = latent["layer"]
    model.latent_heads = GaussianHeads(model.config.hidden_size, **settings).to(
        device=next(model.parameters()).device, dtype=torch.float32)
    model.latent_heads.reset_parameters()
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
            mean, sigma = self.model.latent_heads(head_input(prior))
            actions = prior.float() + mean
            if mode == "sample":
                actions = actions + sigma * torch.randn_like(mean)
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


def replaying(adjuster, actions):
    return adjuster.replay(actions) if adjuster is not None else contextlib.nullcontext({})


def latent_objective(model, prior, action, advantage, latent, epsilon):
    """Scaffolding's latent term (eq. 6): clipped ratio of the Gaussian log-probability
    of the rollout's action given the current z, mu and sigma."""
    mean, sigma = model.latent_heads(head_input(prior))
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
