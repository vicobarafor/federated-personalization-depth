
def select_top_k_layers(layer_shift_dict, k=1):
    sorted_layers = sorted(layer_shift_dict.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_layers[:k]]


def select_top_feature_layers(layer_shift_dict, k=1, feature_layers=None):
    if feature_layers is None:
        feature_layers = ["conv1", "conv2"]

    filtered = {
        name: value
        for name, value in layer_shift_dict.items()
        if name in feature_layers
    }

    sorted_layers = sorted(filtered.items(), key=lambda x: x[1], reverse=True)
    return [name for name, _ in sorted_layers[:k]]
