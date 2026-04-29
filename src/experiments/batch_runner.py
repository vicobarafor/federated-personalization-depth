
import os
import json
import torch

from src.experiments.run_experiment import run_full_experiment
from src.data.cifar import get_cifar10_datasets
from src.data.cifar100 import get_cifar100_datasets
from src.data.svhn import get_svhn_datasets
from src.data.fashionmnist import get_fashionmnist_datasets
from src.data.emnist import get_emnist_balanced_datasets


def run_batch(
    device,
    experiment_specs,
    results_root="/content/drive/MyDrive/fsal-fl/results",
):
    batch_summary = {}

    for spec in experiment_specs:
        dataset_name = spec["dataset"]
        alpha = spec["alpha"]
        seed = spec["seed"]
        num_clients = spec.get("num_clients", 20)
        rounds = spec.get("rounds", 5)
        local_epochs = spec.get("local_epochs", 1)
        batch_size = spec.get("batch_size", 32)

        if dataset_name == "cifar10":
            train_dataset, test_dataset = get_cifar10_datasets()
            num_classes = 10
        elif dataset_name == "cifar100":
            train_dataset, test_dataset = get_cifar100_datasets()
            num_classes = 100
        elif dataset_name == "svhn":
            train_dataset, test_dataset = get_svhn_datasets()
            num_classes = 10
        elif dataset_name == "fashionmnist":
            train_dataset, test_dataset = get_fashionmnist_datasets()
            num_classes = 10
        elif dataset_name == "emnist_balanced":
            train_dataset, test_dataset = get_emnist_balanced_datasets()
            num_classes = 47
        else:
            raise ValueError(f"Unsupported dataset: {dataset_name}")

        experiment_name = f"{dataset_name}_alpha{str(alpha).replace('.', '')}_seed{seed}"

        print(f"\n==============================")
        print(f"Running: {experiment_name}")
        print(f"==============================")

        summary = run_full_experiment(
            experiment_name=experiment_name,
            train_dataset=train_dataset,
            test_dataset=test_dataset,
            num_clients=num_clients,
            alpha=alpha,
            seed=seed,
            num_classes=num_classes,
            device=device,
            rounds=rounds,
            local_epochs=local_epochs,
            batch_size=batch_size,
            results_root=results_root,
        )

        batch_summary[experiment_name] = summary

    batch_path = os.path.join(results_root, "batch_summary.json")
    with open(batch_path, "w") as f:
        json.dump(batch_summary, f, indent=2)

    return batch_summary
