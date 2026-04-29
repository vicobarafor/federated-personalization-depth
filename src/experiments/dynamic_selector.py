
import copy
import torch
import numpy as np
from torch.utils.data import DataLoader, Subset


def one_step_probe(model, dataloader, device, train_layers, lr=0.01, max_batches=1):
    criterion = torch.nn.CrossEntropyLoss()

    m = copy.deepcopy(model).to(device)
    m.train()

    for p in m.parameters():
        p.requires_grad = False

    for name, module in m.named_children():
        if name in train_layers:
            for p in module.parameters():
                p.requires_grad = True

    opt = torch.optim.SGD(
        filter(lambda p: p.requires_grad, m.parameters()),
        lr=lr
    )

    loss_before = 0.0
    loss_after = 0.0
    batches = 0

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            loss_before += criterion(m(x), y).item()

        opt.zero_grad()
        loss = criterion(m(x), y)
        loss.backward()
        opt.step()

        with torch.no_grad():
            loss_after += criterion(m(x), y).item()

        batches += 1

    return {
        "before": loss_before / batches,
        "after": loss_after / batches,
        "delta": (loss_before - loss_after) / batches,
    }
