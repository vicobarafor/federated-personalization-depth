
import torch

def compute_variance_shift(client_vars, global_vars):
    shift = {}

    for layer in client_vars:
        c = client_vars[layer]
        g = global_vars[layer]

        # relative variance mismatch
        rel_diff = torch.abs(c - g) / (g + 1e-8)
        shift[layer] = rel_diff.mean().item()

    return shift
