
import torch

def compute_feature_means(model, extractor, dataloader, device):
    model.eval()
    extractor.clear()

    feature_sums = {}
    counts = {}

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)

            _ = model(x)

            for layer, feat in extractor.features.items():
                feat = feat.view(feat.size(0), -1)

                if layer not in feature_sums:
                    feature_sums[layer] = feat.sum(dim=0)
                    counts[layer] = feat.size(0)
                else:
                    feature_sums[layer] += feat.sum(dim=0)
                    counts[layer] += feat.size(0)

    feature_means = {
        layer: feature_sums[layer] / counts[layer]
        for layer in feature_sums
    }

    return feature_means
