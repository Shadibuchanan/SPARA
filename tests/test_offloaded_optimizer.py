import torch

from spara.runtime import make_optimizer, offloaded_optimizer

CONFIG = {"learning_rate": 1e-2, "weight_decay": 0.01, "betas": [0.9, 0.999], "epsilon": 1e-8}


def tiny(seed):
    torch.manual_seed(seed)
    model = torch.nn.Sequential(torch.nn.Linear(4, 8), torch.nn.LayerNorm(8), torch.nn.Linear(8, 1))
    model[2].weight.requires_grad_(False)  # frozen parameters stay out of both optimizers
    return model


def train(model, optimizer, steps=5):
    torch.manual_seed(0)
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda _: 1.0)
    for _ in range(steps):
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(16, 4)).pow(2).mean().backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()


def test_offloaded_matches_make_optimizer():
    plain, offloaded = tiny(1), tiny(1)
    train(plain, make_optimizer(plain, CONFIG))
    optimizer = offloaded_optimizer(offloaded, CONFIG)
    train(offloaded, optimizer)
    for (name, a), (_, b) in zip(plain.named_parameters(), offloaded.named_parameters()):
        assert torch.allclose(a, b, atol=1e-7), name
    # The model's own gradients are freed after each step; weight decay groups match make_optimizer's.
    assert all(p.grad is None for p in offloaded.parameters())
    assert [g["weight_decay"] for g in optimizer.param_groups] == [0.01, 0.0]
