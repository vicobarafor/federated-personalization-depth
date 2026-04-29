
def htap_select_policy(shift, ratio_threshold=0.5):
    """
    Choose between:
      - head_only: ["fc1", "fc2"]
      - conv2_head: ["conv2", "fc1", "fc2"]

    Rule:
      if conv2_shift / fc1_shift >= ratio_threshold -> conv2_head
      else -> head_only
    """
    fc1_shift = shift["fc1"] + 1e-8
    ratio = shift["conv2"] / fc1_shift

    if ratio >= ratio_threshold:
        return {
            "policy_name": "conv2_head",
            "selected_layers": ["conv2", "fc1", "fc2"],
            "ratio": ratio,
        }
    else:
        return {
            "policy_name": "head_only",
            "selected_layers": ["fc1", "fc2"],
            "ratio": ratio,
        }
