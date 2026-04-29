
import copy
import torch


def probe_policy_selection(model, dataloader, device, max_batches=1, lr=0.01):
    """
    Compare head vs conv2+head using 1-step probe updates.
    Returns selected policy.
    """

    criterion = torch.nn.CrossEntropyLoss()

    # ===== HEAD PROBE =====
    head_model = copy.deepcopy(model).to(device)
    head_model.train()

    for p in head_model.parameters():
        p.requires_grad = False
    for p in head_model.fc1.parameters():
        p.requires_grad = True
    for p in head_model.fc2.parameters():
        p.requires_grad = True

    optimizer_head = torch.optim.SGD(
        filter(lambda p: p.requires_grad, head_model.parameters()),
        lr=lr
    )

    head_loss_before = 0.0
    head_loss_after = 0.0

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        # before
        with torch.no_grad():
            head_loss_before += criterion(head_model(x), y).item()

        # one step
        optimizer_head.zero_grad()
        loss = criterion(head_model(x), y)
        loss.backward()
        optimizer_head.step()

        # after
        with torch.no_grad():
            head_loss_after += criterion(head_model(x), y).item()

    head_improvement = head_loss_before - head_loss_after

    # ===== CONV2 + HEAD PROBE =====
    conv2_model = copy.deepcopy(model).to(device)
    conv2_model.train()

    for p in conv2_model.parameters():
        p.requires_grad = False
    for p in conv2_model.conv2.parameters():
        p.requires_grad = True
    for p in conv2_model.fc1.parameters():
        p.requires_grad = True
    for p in conv2_model.fc2.parameters():
        p.requires_grad = True

    optimizer_conv2 = torch.optim.SGD(
        filter(lambda p: p.requires_grad, conv2_model.parameters()),
        lr=lr
    )

    conv2_loss_before = 0.0
    conv2_loss_after = 0.0

    for i, (x, y) in enumerate(dataloader):
        if i >= max_batches:
            break

        x, y = x.to(device), y.to(device)

        with torch.no_grad():
            conv2_loss_before += criterion(conv2_model(x), y).item()

        optimizer_conv2.zero_grad()
        loss = criterion(conv2_model(x), y)
        loss.backward()
        optimizer_conv2.step()

        with torch.no_grad():
            conv2_loss_after += criterion(conv2_model(x), y).item()

    conv2_improvement = conv2_loss_before - conv2_loss_after

    if conv2_improvement > head_improvement:
        return "conv2_head", conv2_improvement, head_improvement
    else:
        return "head_only", conv2_improvement, head_improvement
