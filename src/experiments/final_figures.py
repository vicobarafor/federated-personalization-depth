
import os
import json
import matplotlib.pyplot as plt


def load_summary(summary_json_path):
    with open(summary_json_path, "r") as f:
        return json.load(f)


def alpha_tag_to_float(alpha_tag):
    # "alpha01" -> 0.1, "alpha03" -> 0.3, "alpha10" -> 1.0
    if alpha_tag == "alpha01":
        return 0.1
    elif alpha_tag == "alpha03":
        return 0.3
    elif alpha_tag == "alpha10":
        return 1.0
    else:
        raise ValueError(f"Unknown alpha tag: {alpha_tag}")


def group_by_dataset(summary_rows):
    grouped = {}
    for row in summary_rows:
        dataset = row["dataset"]
        grouped.setdefault(dataset, []).append(row)

    for dataset in grouped:
        grouped[dataset] = sorted(
            grouped[dataset],
            key=lambda x: alpha_tag_to_float(x["alpha"])
        )

    return grouped


def plot_accuracy_vs_alpha(summary_rows, save_dir):
    grouped = group_by_dataset(summary_rows)

    for dataset, rows in grouped.items():
        alphas = [alpha_tag_to_float(r["alpha"]) for r in rows]

        fed = [r["fed_mean"] for r in rows]
        head = [r["head_mean"] for r in rows]
        conv2 = [r["conv2_mean"] for r in rows]
        full = [r["full_mean"] for r in rows]

        fed_std = [r["fed_std"] for r in rows]
        head_std = [r["head_std"] for r in rows]
        conv2_std = [r["conv2_std"] for r in rows]
        full_std = [r["full_std"] for r in rows]

        plt.figure(figsize=(8, 5))
        plt.errorbar(alphas, fed, yerr=fed_std, marker='o', capsize=4, label='FedAvg')
        plt.errorbar(alphas, head, yerr=head_std, marker='o', capsize=4, label='Head-only')
        plt.errorbar(alphas, conv2, yerr=conv2_std, marker='o', capsize=4, label='Conv2+Head')
        plt.errorbar(alphas, full, yerr=full_std, marker='o', capsize=4, label='Full FT')

        plt.xlabel("Dirichlet alpha")
        plt.ylabel("Mean personalized accuracy")
        plt.title(f"{dataset.upper()}: Accuracy vs Heterogeneity")
        plt.xticks(alphas, [str(a) for a in alphas])
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"{dataset}_accuracy_vs_alpha.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_gain_vs_fedavg(summary_rows, save_dir):
    grouped = group_by_dataset(summary_rows)

    for dataset, rows in grouped.items():
        alphas = [alpha_tag_to_float(r["alpha"]) for r in rows]

        gain_head = [r["head_mean"] - r["fed_mean"] for r in rows]
        gain_conv2 = [r["conv2_mean"] - r["fed_mean"] for r in rows]
        gain_full = [r["full_mean"] - r["fed_mean"] for r in rows]

        plt.figure(figsize=(8, 5))
        plt.plot(alphas, gain_head, marker='o', label='Head-only gain')
        plt.plot(alphas, gain_conv2, marker='o', label='Conv2+Head gain')
        plt.plot(alphas, gain_full, marker='o', label='Full FT gain')

        plt.xlabel("Dirichlet alpha")
        plt.ylabel("Gain over FedAvg")
        plt.title(f"{dataset.upper()}: Personalization Gain over FedAvg")
        plt.xticks(alphas, [str(a) for a in alphas])
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"{dataset}_gain_vs_fedavg.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


def plot_win_counts(summary_rows, save_dir):
    grouped = group_by_dataset(summary_rows)

    for dataset, rows in grouped.items():
        labels = [r["alpha"] for r in rows]

        head = [r["wins_head"] for r in rows]
        conv2 = [r["wins_conv2"] for r in rows]
        full = [r["wins_full"] for r in rows]

        x = list(range(len(labels)))

        plt.figure(figsize=(8, 5))
        plt.bar(x, head, label="Head-only")
        plt.bar(x, conv2, bottom=head, label="Conv2+Head")
        bottom2 = [head[i] + conv2[i] for i in range(len(head))]
        plt.bar(x, full, bottom=bottom2, label="Full FT")

        plt.xticks(x, labels)
        plt.xlabel("Dirichlet alpha")
        plt.ylabel("Total oracle wins")
        plt.title(f"{dataset.upper()}: Oracle Win Counts by Policy")
        plt.legend()
        plt.tight_layout()

        out_path = os.path.join(save_dir, f"{dataset}_win_counts.png")
        plt.savefig(out_path, dpi=300, bbox_inches="tight")
        plt.close()


def generate_all_figures(summary_json_path, save_dir):
    os.makedirs(save_dir, exist_ok=True)

    summary_rows = load_summary(summary_json_path)

    plot_accuracy_vs_alpha(summary_rows, save_dir)
    plot_gain_vs_fedavg(summary_rows, save_dir)
    plot_win_counts(summary_rows, save_dir)

    return [
        os.path.join(save_dir, name)
        for name in sorted(os.listdir(save_dir))
    ]
