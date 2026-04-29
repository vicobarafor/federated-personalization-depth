
import torch

def compute_feature_variances(model, extractor, dataloader, device):
    model.eval()

    feature_sum = {}
    feature_sq_sum = {}
    counts = {}

    with torch.no_grad():
        for x, _ in dataloader:
            x = x.to(device)

            extractor.clear()
            _ = model(x)

            for layer, feat in extractor.features.items():
                feat = feat.view(feat.size(0), -1)  # [B, D]

                batch_sum = feat.sum(dim=0)
                batch_sq_sum = (feat ** 2).sum(dim=0)
                batch_count = feat.size(0)

                if layer not in feature_sum:
                    feature_sum[layer] = batch_sum
                    feature_sq_sum[layer] = batch_sq_sum
                    counts[layer] = batch_count
                else:
                    feature_sum[layer] += batch_sum
                    feature_sq_sum[layer] += batch_sq_sum
                    counts[layer] += batch_count

    feature_vars = {}
    for layer in feature_sum:
        mean = feature_sum[layer] / counts[layer]
        mean_sq = feature_sq_sum[layer] / counts[layer]
        var = mean_sq - mean ** 2
        feature_vars[layer] = torch.clamp(var, min=1e-12)

    return feature_vars
