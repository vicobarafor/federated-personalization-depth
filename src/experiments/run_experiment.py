
import json
import os
import random

import numpy as np
import torch

from src.data.split import dirichlet_split_noniid
from src.experiments.fedavg_train import train_fedavg_experiment
from src.experiments.personalization_oracle import run_personalization_oracle


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def run_full_experiment(
    experiment_name,
    train_dataset,
    test_dataset,
    num_clients,
    alpha,
    seed,
    num_classes,
    device,
    rounds=5,
    local_epochs=1,
    batch_size=32,
    results_root="/content/drive/MyDrive/fsal-fl/results",
):
    set_seed(seed)

    client_indices = dirichlet_split_noniid(
        train_dataset,
        num_clients=num_clients,
        alpha=alpha,
        seed=seed,
    )

    model, fedavg_metrics = train_fedavg_experiment(
        train_dataset=train_dataset,
        test_dataset=test_dataset,
        client_indices=client_indices,
        num_clients=num_clients,
        num_classes=num_classes,
        device=device,
        batch_size=batch_size,
        rounds=rounds,
        local_epochs=local_epochs,
    )

    oracle_results, oracle_summary = run_personalization_oracle(
        model=model,
        train_dataset=train_dataset,
        client_indices=client_indices,
        num_clients=num_clients,
        device=device,
        max_batches=20,
        epochs=1,
    )

    out_dir = os.path.join(results_root, experiment_name)
    os.makedirs(out_dir, exist_ok=True)

    config = {
        "experiment_name": experiment_name,
        "num_clients": num_clients,
        "alpha": alpha,
        "seed": seed,
        "num_classes": num_classes,
        "rounds": rounds,
        "local_epochs": local_epochs,
        "batch_size": batch_size,
    }

    with open(os.path.join(out_dir, "config.json"), "w") as f:
        json.dump(config, f, indent=2)

    with open(os.path.join(out_dir, "fedavg_metrics.json"), "w") as f:
        json.dump(fedavg_metrics, f, indent=2)

    with open(os.path.join(out_dir, "oracle_results.json"), "w") as f:
        json.dump(oracle_results, f, indent=2)

    with open(os.path.join(out_dir, "oracle_summary.json"), "w") as f:
        json.dump(oracle_summary, f, indent=2)

    torch.save(model.state_dict(), os.path.join(out_dir, "model.pt"))

    summary = {
        "fedavg_metrics": fedavg_metrics,
        "oracle_summary": oracle_summary,
    }

    with open(os.path.join(out_dir, "summary.json"), "w") as f:
        json.dump(summary, f, indent=2)

    return summary
