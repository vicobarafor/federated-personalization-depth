
import torch
from torch.utils.data import DataLoader, Subset

from src.fl.server import fedavg_aggregate
from src.hooks.feature_extractor import FeatureExtractor
from analysis.feature_variance import compute_feature_variances
from analysis.variance_shift import compute_variance_shift

def normalize_weights(shift, alpha=0.2):
    total = sum(shift.values()) + 1e-8
    weights = {}

    for k, v in shift.items():
        norm = v / total
        weights[k] = alpha + (1 - alpha) * norm

    return weights


def run_soft_fsal(global_model, clients, full_dataset, client_indices, device,
                  rounds=3, local_epochs=1, batch_size=128):

    for r in range(rounds):
        print(f"\n🌍 Soft-FSAL Round {r+1}")

        layers = {
            "conv1": global_model.conv1,
            "conv2": global_model.conv2,
            "fc1": global_model.fc1,
        }

        extractor = FeatureExtractor(global_model, layers)

        global_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        global_vars = compute_feature_variances(global_model, extractor, global_loader, device)

        client_states = []

        for client in clients:
            subset = Subset(full_dataset, client_indices[client.client_id])
            client_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

            client_vars = compute_feature_variances(global_model, extractor, client_loader, device)
            shift = compute_variance_shift(client_vars, global_vars)

            weights = normalize_weights(shift)

            print(f"Client {client.client_id} shift: {shift}")
            print(f"Client {client.client_id} weights: {weights}")

            state = client.train_with_layer_weights(
                global_model,
                layer_weights=weights,
                epochs=local_epochs
            )

            client_states.append(state)

        extractor.remove()
        global_model = fedavg_aggregate(global_model, client_states)

    return global_model
