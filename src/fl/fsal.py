
import torch
from torch.utils.data import DataLoader, Subset

from src.fl.server import fedavg_aggregate
from src.fl.layer_select import select_top_k_layers
from src.hooks.feature_extractor import FeatureExtractor
from analysis.feature_variance import compute_feature_variances
from analysis.variance_shift import compute_variance_shift

def run_fsal(global_model, clients, full_dataset, client_indices, device, rounds=3, local_epochs=1, top_k=1, batch_size=128):
    """
    FSAL training:
    - compute per-client layer shift from current global model
    - select top-k layers
    - train only selected layers
    - aggregate updated models
    """
    for r in range(rounds):
        print(f"\n🌍 FSAL Round {r+1}")

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
            selected_layers = select_top_k_layers(shift, k=top_k)

            print(f"Client {client.client_id} shift: {shift}")
            print(f"Client {client.client_id} selected: {selected_layers}")

            state = client.train_selected_layers(
                global_model,
                selected_layers=selected_layers,
                epochs=local_epochs
            )
            client_states.append(state)

        extractor.remove()
        global_model = fedavg_aggregate(global_model, client_states)

    return global_model
