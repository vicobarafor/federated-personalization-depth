
import copy
import torch
from torch.utils.data import DataLoader, Subset

class FSALHeadClient:
    def __init__(self, client_id, dataset, indices, device, batch_size=32):
        self.client_id = client_id
        self.device = device
        self.indices = indices

        self.loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=True
        )

    def train_selected_layers(self, model, selected_feature_layers, epochs=1, lr=0.01):
        """
        Always train fc1 + fc2, and additionally train selected feature layers.
        selected_feature_layers: list like ["conv1"] or ["conv2"]
        """
        local_model = copy.deepcopy(model)
        local_model.to(self.device)
        local_model.train()

        # freeze everything first
        for param in local_model.parameters():
            param.requires_grad = False

        # always train classifier/head
        always_train = ["fc1", "fc2"]

        for layer_name in always_train + selected_feature_layers:
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
