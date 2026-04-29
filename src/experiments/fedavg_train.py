
import torch
from torch.utils.data import DataLoader

from src.fl.client import Client
from src.fl.fedavg import run_fedavg
from src.models.cnn import SimpleCNN
from src.utils.eval import evaluate_model


def train_fedavg_experiment(
    train_dataset,
    test_dataset,
    client_indices,
    num_clients,
    num_classes,
    device,
    batch_size=32,
    rounds=5,
    local_epochs=1,
):
    clients = [
        Client(
            client_id=i,
            dataset=train_dataset,
            indices=client_indices[i],
            device=device,
            batch_size=batch_size,
        )
        for i in range(num_clients)
    ]

    global_model = SimpleCNN(num_classes=num_classes).to(device)

    global_model = run_fedavg(
        global_model=global_model,
        clients=clients,
        rounds=rounds,
        local_epochs=local_epochs,
    )

    test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False)
    metrics = evaluate_model(global_model, test_loader, device)

    return global_model, metrics
