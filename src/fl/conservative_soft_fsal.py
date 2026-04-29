
import torch
from torch.utils.data import DataLoader, Subset

from src.fl.server import fedavg_aggregate
from src.hooks.feature_extractor import FeatureExtractor
from analysis.feature_variance import compute_feature_variances
from analysis.variance_shift import compute_variance_shift

def make_conservative_weights(shift, min_scale=0.9, max_scale=1.1):
    c1 = shift["conv1"]
    c2 = shift["conv2"]

    low = min(c1, c2)
    high = max(c1, c2)

    if abs(high - low) < 1e-12:
        conv1_scale = 1.0
        conv2_scale = 1.0
    else:
        conv1_scale = min_scale + (c1 - low) * (max_scale - min_scale) / (high - low)
        conv2_scale = min_scale + (c2 - low) * (max_scale - min_scale) / (high - low)

    return {
        "conv1": float(conv1_scale),
        "conv2": float(conv2_scale),
        "fc1": 1.0,
        "fc2": 1.0,
    }


def run_conservative_soft_fsal(global_model, clients, full_dataset, client_indices, device,
                               rounds=3, local_epochs=1, batch_size=128,
                               min_scale=0.9, max_scale=1.1):
    log = {}

    for r in range(rounds):
        print(f"\n🌍 Conservative Soft-FSAL Round {r+1} | range=({min_scale}, {max_scale})")

        layers = {
            "conv1": global_model.conv1,
            "conv2": global_model.conv2,
            "fc1": global_model.fc1,
        }

        extractor = FeatureExtractor(global_model, layers)

        global_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        global_vars = compute_feature_variances(global_model, extractor, global_loader, device)

        client_states = []
        log[f"round_{r+1}"] = {}

        for client in clients:
            subset = Subset(full_dataset, client_indices[client.client_id])
            client_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

            client_vars = compute_feature_variances(global_model, extractor, client_loader, device)
            shift = compute_variance_shift(client_vars, global_vars)
            weights = make_conservative_weights(shift, min_scale=min_scale, max_scale=max_scale)

            log[f"round_{r+1}"][str(client.client_id)] = {
                "shift": shift,
                "weights": weights,
            }

            print(f"Client {client.client_id} weights: {weights}")

            state = client.train_with_layer_weights(
                global_model,
                layer_weights=weights,
                epochs=local_epochs
            )
            client_states.append(state)

        extractor.remove()
        global_model = fedavg_aggregate(global_model, client_states)

    return global_model, log
