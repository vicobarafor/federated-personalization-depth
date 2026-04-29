
import copy
import torch
from torch.utils.data import DataLoader, Subset

class ConservativeSoftFSALClient:
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
        layer_weights example:
        {
            "conv1": 0.95,
            "conv2": 1.05,
            "fc1": 1.0,
            "fc2": 1.0
        }
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

                for name, module in local_model.named_children():
                    if name in layer_weights:
                        scale = layer_weights[name]
                        for param in module.parameters():
                            if param.grad is not None:
                                param.grad *= scale

                optimizer.step()

        return local_model.state_dict()
