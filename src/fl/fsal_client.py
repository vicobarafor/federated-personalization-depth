
import copy
import torch
from torch.utils.data import DataLoader, Subset

class FSALClient:
    def __init__(self, client_id, dataset, indices, device, batch_size=32):
        self.client_id = client_id
        self.device = device
        self.indices = indices

        self.loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=True
        )

    def train_selected_layers(self, model, selected_layers, epochs=1, lr=0.01):
        """
        selected_layers: list like ["fc1"] or ["conv2", "fc1"]
        """
        local_model = copy.deepcopy(model)
        local_model.to(self.device)
        local_model.train()

        # freeze everything first
        for param in local_model.parameters():
            param.requires_grad = False

        # unfreeze only selected modules
        for layer_name in selected_layers:
            module = getattr(local_model, layer_name)
            for param in module.parameters():
                param.requires_grad = True

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, local_model.parameters()),
            lr=lr
        )
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x, y in self.loader:
                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                out = local_model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

        return local_model.state_dict()
