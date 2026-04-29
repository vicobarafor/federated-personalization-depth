
import numpy as np


def compute_oracle_regret(client_results, selector_choices):
    """
    client_results:
      dict[client_id] = {
         "head_only": {"accuracy": ...},
         "conv2_head": {"accuracy": ...},
         "full_ft": {"accuracy": ...}
      }

    selector_choices:
      dict[client_id] = chosen_method
    """

    methods = ["head_only", "conv2_head", "full_ft"]

    stats = {m: {"acc": [], "regret": []} for m in methods}
    stats["selector"] = {"acc": [], "regret": []}

    for cid, res in client_results.items():

        scores = {
            "head_only": res["head_only"]["accuracy"],
            "conv2_head": res["conv2_head"]["accuracy"],
            "full_ft": res["full_ft"]["accuracy"],
        }

        oracle_best = max(scores.values())

        # fixed baselines
        for m in methods:
            acc = scores[m]
            regret = oracle_best - acc
            stats[m]["acc"].append(acc)
            stats[m]["regret"].append(regret)

        # selector
        chosen = selector_choices[cid]
        chosen_acc = scores[chosen]
        chosen_regret = oracle_best - chosen_acc

        stats["selector"]["acc"].append(chosen_acc)
        stats["selector"]["regret"].append(chosen_regret)

    summary = {}

    for method, vals in stats.items():
        summary[method] = {
            "mean_acc": float(np.mean(vals["acc"])),
            "mean_regret": float(np.mean(vals["regret"])),
        }

    return summary
