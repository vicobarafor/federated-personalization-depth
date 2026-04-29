
import torch
from torch.utils.data import DataLoader, Subset

from src.fl.server import fedavg_aggregate
from src.fl.layer_select import select_top_feature_layers
from src.hooks.feature_extractor import FeatureExtractor
from analysis.feature_variance import compute_feature_variances
from analysis.variance_shift import compute_variance_shift

def run_fsal_head(global_model, clients, full_dataset, client_indices, device,
                  rounds=3, local_epochs=1, top_k_feature=1, batch_size=128):
    """
    FSAL-Head:
    - compute per-client layer shift from current global model
    - select top-k feature layers from {conv1, conv2}
    - always train fc1 + fc2
    - aggregate client models
    """
    selection_log = {}

    for r in range(rounds):
        print(f"\n🌍 FSAL-Head Round {r+1}")

        layers = {
            "conv1": global_model.conv1,
            "conv2": global_model.conv2,
            "fc1": global_model.fc1,
        }

        extractor = FeatureExtractor(global_model, layers)

        global_loader = DataLoader(full_dataset, batch_size=batch_size, shuffle=False)
        global_vars = compute_feature_variances(global_model, extractor, global_loader, device)

        client_states = []
        selection_log[f"round_{r+1}"] = {}

        for client in clients:
            subset = Subset(full_dataset, client_indices[client.client_id])
            client_loader = DataLoader(subset, batch_size=batch_size, shuffle=False)

            client_vars = compute_feature_variances(global_model, extractor, client_loader, device)
            shift = compute_variance_shift(client_vars, global_vars)

            selected_feature_layers = select_top_feature_layers(
                shift,
                k=top_k_feature,
                feature_layers=["conv1", "conv2"]
            )

            selection_log[f"round_{r+1}"][str(client.client_id)] = {
                "shift": shift,
                "selected_feature_layers": selected_feature_layers,
                "always_train": ["fc1", "fc2"],
            }

            print(f"Client {client.client_id} shift: {shift}")
            print(f"Client {client.client_id} selected feature layers: {selected_feature_layers}")
            print(f"Client {client.client_id} always trains: ['fc1', 'fc2']")

            state = client.train_selected_layers(
                global_model,
                selected_feature_layers=selected_feature_layers,
                epochs=local_epochs
            )
            client_states.append(state)

        extractor.remove()
        global_model = fedavg_aggregate(global_model, client_states)

    return global_model, selection_log
