
import os
import json
import numpy as np
from collections import Counter

from src.experiments.learned_selector import (
    LABEL_MAP,
    INV_LABEL_MAP,
    oracle_label,
    train_loocv_selector,
)
from src.experiments.oracle_regret import compute_oracle_regret


def build_metadata_feature_bank(train_dataset, client_indices):
    feature_bank = {}

    for cid, idxs in client_indices.items():
        labels = [train_dataset[i][1] for i in idxs]

        counts = Counter(labels)
        total = len(labels)

        probs = np.array(list(counts.values()), dtype=np.float32) / total

        entropy = -np.sum(probs * np.log(probs + 1e-12))
        max_prop = float(np.max(probs))
        num_classes = len(counts)

        feature_bank[cid] = {
            "size": total,
            "entropy": float(entropy),
            "max_prop": max_prop,
            "num_classes": num_classes,
        }

    return feature_bank


def build_selector_dataset(oracle_results, feature_bank):
    X = []
    y = []
    client_ids = []

    for cid in sorted(oracle_results.keys()):
        feats = feature_bank[cid]

        row = [
            feats["size"],
            feats["entropy"],
            feats["max_prop"],
            feats["num_classes"],
        ]
        X.append(row)

        best = oracle_label(oracle_results[cid])
        y.append(LABEL_MAP[best])
        client_ids.append(cid)

    X = np.array(X, dtype=np.float32)
    y = np.array(y, dtype=np.int64)

    return X, y, client_ids


def evaluate_selector_from_oracle(train_dataset, client_indices, oracle_results):
    feature_bank = build_metadata_feature_bank(train_dataset, client_indices)
    X, y, client_ids = build_selector_dataset(oracle_results, feature_bank)

    acc, preds, truth = train_loocv_selector(X, y)

    selector_choices = {
        cid: INV_LABEL_MAP[preds[i]]
        for i, cid in enumerate(client_ids)
    }

    regret_summary = compute_oracle_regret(
        client_results=oracle_results,
        selector_choices=selector_choices
    )

    out = {
        "selector_loocv_acc": float(acc),
        "selector_summary": regret_summary["selector"],
        "head_only_summary": regret_summary["head_only"],
        "conv2_head_summary": regret_summary["conv2_head"],
        "full_ft_summary": regret_summary["full_ft"],
        "preds": [int(p) for p in preds],
        "truth": [int(t) for t in truth],
        "client_ids": client_ids,
    }

    return out
