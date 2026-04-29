
import copy
import torch
import torch.nn.functional as F


def compute_client_utility_features(model, dataloader, device, max_batches=5):
    """
    Returns client utility features:
      - avg loss
      - avg entropy
      - grad norms for conv2 / fc1 / fc2
    """
    local_model = copy.deepcopy(model)
    local_model.to(device)
    local_model.train()

    criterion = torch.nn.CrossEntropyLoss()

    total_loss = 0.0
    total_entropy = 0.0
    total_samples = 0

    grad_norm_sums = {
        "conv2_grad_norm": 0.0,
        "fc1_grad_norm": 0.0,
        "fc2_grad_norm": 0.0,
    }
    grad_batches = 0

    for batch_idx, (x, y) in enumerate(dataloader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        x = x.to(device)
        y = y.to(device)

        local_model.zero_grad()
        logits = local_model(x)
        loss = criterion(logits, y)
        loss.backward()

        probs = F.softmax(logits, dim=1)
        entropy = -(probs * torch.log(probs + 1e-8)).sum(dim=1).mean()

        total_loss += loss.item() * x.size(0)
        total_entropy += entropy.item() * x.size(0)
        total_samples += x.size(0)

        def module_grad_norm(module):
            sq = 0.0
            for p in module.parameters():
                if p.grad is not None:
                    sq += p.grad.norm(2).item() ** 2
            return sq ** 0.5

        grad_norm_sums["conv2_grad_norm"] += module_grad_norm(local_model.conv2)
        grad_norm_sums["fc1_grad_norm"] += module_grad_norm(local_model.fc1)
        grad_norm_sums["fc2_grad_norm"] += module_grad_norm(local_model.fc2)
        grad_batches += 1

    features = {
        "pre_loss": total_loss / total_samples,
        "entropy": total_entropy / total_samples,
        "conv2_grad_norm": grad_norm_sums["conv2_grad_norm"] / max(grad_batches, 1),
        "fc1_grad_norm": grad_norm_sums["fc1_grad_norm"] / max(grad_batches, 1),
        "fc2_grad_norm": grad_norm_sums["fc2_grad_norm"] / max(grad_batches, 1),
    }

    return features
