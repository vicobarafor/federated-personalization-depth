
def select_layers_by_threshold(shift, tau=0.8):
    """
    Always include head (fc1, fc2).
    Include feature layers if shift >= tau * max_shift.
    """
    max_shift = max(shift.values())

    selected = ["fc1", "fc2"]

    for layer in ["conv1", "conv2"]:
        if shift[layer] >= tau * max_shift:
            selected.append(layer)

    return selected
