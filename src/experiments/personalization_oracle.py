
import numpy as np

from src.fl.personalization_probe import PersonalizationProbeClient


def run_personalization_oracle(
    model,
    train_dataset,
    client_indices,
    num_clients,
    device,
    max_batches=20,
    epochs=1,
):
    clients = [
        PersonalizationProbeClient(
            i,
            train_dataset,
            client_indices[i],
            device
        )
        for i in range(num_clients)
    ]

    oracle_results = {}
    head_accs = []
    conv2_accs = []
    full_accs = []

    for client in clients:
        cid = client.client_id

        model_head = client.finetune_selected_layers(
            model,
            ["fc1", "fc2"],
            epochs=epochs,
            max_batches=max_batches
        )
        eval_head = client.evaluate(model_head, max_batches=max_batches)

        model_conv2 = client.finetune_selected_layers(
            model,
            ["conv2", "fc1", "fc2"],
            epochs=epochs,
            max_batches=max_batches
        )
        eval_conv2 = client.evaluate(model_conv2, max_batches=max_batches)

        model_full = client.finetune_selected_layers(
            model,
            ["conv1", "conv2", "fc1", "fc2"],
            epochs=epochs,
            max_batches=max_batches
        )
        eval_full = client.evaluate(model_full, max_batches=max_batches)

        oracle_results[cid] = {
            "head_only": eval_head,
            "conv2_head": eval_conv2,
            "full_ft": eval_full,
        }

        head_accs.append(eval_head["accuracy"])
        conv2_accs.append(eval_conv2["accuracy"])
        full_accs.append(eval_full["accuracy"])

    wins = {
        "head_only": 0,
        "conv2_head": 0,
        "full_ft": 0,
    }

    for cid, res in oracle_results.items():
        scores = {
            "head_only": res["head_only"]["accuracy"],
            "conv2_head": res["conv2_head"]["accuracy"],
            "full_ft": res["full_ft"]["accuracy"],
        }
        winner = max(scores.items(), key=lambda x: x[1])[0]
        wins[winner] += 1

    summary = {
        "head_only_mean_acc": float(np.mean(head_accs)),
        "conv2_head_mean_acc": float(np.mean(conv2_accs)),
        "full_ft_mean_acc": float(np.mean(full_accs)),
        "wins": wins,
    }

    return oracle_results, summary
