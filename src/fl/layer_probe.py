
import copy
import torch
from torch.utils.data import DataLoader, Subset


class LayerProbeClient:
    def __init__(self, client_id, dataset, indices, device, batch_size=32):
        self.client_id = client_id
        self.device = device
        self.loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=True
        )

    def evaluate_loss(self, model, max_batches=None):
        model = copy.deepcopy(model)
        model.to(self.device)
        model.eval()

        criterion = torch.nn.CrossEntropyLoss()
        total_loss = 0.0
        total_samples = 0

        with torch.no_grad():
            for batch_idx, (x, y) in enumerate(self.loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                out = model(x)
                loss = criterion(out, y)

                total_loss += loss.item() * x.size(0)
                total_samples += x.size(0)

        return total_loss / total_samples

    def train_only_layers(self, model, train_layers, epochs=1, lr=0.01, max_batches=None):
        local_model = copy.deepcopy(model)
        local_model.to(self.device)
        local_model.train()

        for p in local_model.parameters():
            p.requires_grad = False

        for layer_name in train_layers:
            module = getattr(local_model, layer_name)
            for p in module.parameters():
                p.requires_grad = True

        optimizer = torch.optim.SGD(
            filter(lambda p: p.requires_grad, local_model.parameters()),
            lr=lr
        )
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for batch_idx, (x, y) in enumerate(self.loader):
                if max_batches is not None and batch_idx >= max_batches:
                    break

                x = x.to(self.device)
                y = y.to(self.device)

                optimizer.zero_grad()
                out = local_model(x)
                loss = criterion(out, y)
                loss.backward()
                optimizer.step()

        return local_model
