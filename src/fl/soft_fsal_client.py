
import copy
import torch
from torch.utils.data import DataLoader, Subset

class SoftFSALClient:
    def __init__(self, client_id, dataset, indices, device, batch_size=32):
        self.client_id = client_id
        self.device = device

        self.loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=True
        )

    def train_with_layer_weights(self, model, layer_weights, epochs=1, lr=0.01):
        """
        layer_weights: dict like {"conv1": w1, "conv2": w2, "fc1": w3}
        """
        local_model = copy.deepcopy(model)
        local_model.to(self.device)
        local_model.train()

        optimizer = torch.optim.SGD(local_model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                out = local_model(x)
                loss = criterion(out, y)
                loss.backward()

                # apply layer-wise scaling
                for name, module in local_model.named_children():
                    if name in layer_weights:
                        w = layer_weights[name]
                        for param in module.parameters():
                            if param.grad is not None:
                                param.grad *= w

                optimizer.step()

        return local_model.state_dict()
