
import random

def select_layers_by_threshold(shift, tau=0.8):
    max_shift = max(shift.values())
    selected = ["fc1", "fc2"]

    for layer in ["conv1", "conv2"]:
        if shift[layer] >= tau * max_shift:
            selected.append(layer)

    return selected


def select_random_layers_matching_count(selected_layers, seed=None):
    """
    Keep fc1/fc2 always.
    Randomly choose the same number of feature layers as in selected_layers.
    """
    if seed is not None:
        random.seed(seed)

    selected_feature_count = sum(1 for x in selected_layers if x in ["conv1", "conv2"])
    all_feature_layers = ["conv1", "conv2"]

    random_features = random.sample(all_feature_layers, k=selected_feature_count)
    return ["fc1", "fc2"] + random_features


def count_trainable_params(model, selected_layers):
    count = 0
    for layer_name in selected_layers:
        module = getattr(model, layer_name)
        count += sum(p.numel() for p in module.parameters())
    return count


def count_full_model_params(model):
    return sum(p.numel() for p in model.parameters())
