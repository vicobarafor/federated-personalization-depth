
def shift_rank_candidates(shift):
    """
    Convert layer shift into ranking over candidate personalization choices.
    We use simple score aggregation over selected layers.
    """
    scores = {
        "head_only": shift["fc1"],
        "conv1_head": shift["conv1"] + shift["fc1"],
        "conv2_head": shift["conv2"] + shift["fc1"],
        "full_ft": shift["conv1"] + shift["conv2"] + shift["fc1"],
    }

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return scores, [name for name, _ in ranked]


def oracle_rank_candidates(metrics_by_choice):
    """
    Rank candidates by achieved personalized accuracy.
    """
    ranked = sorted(
        metrics_by_choice.items(),
        key=lambda x: x[1]["accuracy"],
        reverse=True
    )
    return [name for name, _ in ranked]
