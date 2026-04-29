
import torch.nn.functional as F

def compute_feature_shift(client_means, global_means):
    shift = {}

    for layer in client_means:
        c = client_means[layer]
        g = global_means[layer]

        sim = F.cosine_similarity(c.unsqueeze(0), g.unsqueeze(0))
        shift[layer] = 1 - sim.item()

    return shift
