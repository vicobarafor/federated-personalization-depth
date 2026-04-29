
import torch
from torch.utils.data import DataLoader, Subset
import copy

class Client:
    def __init__(self, client_id, dataset, indices, device, batch_size=32):
        self.client_id = client_id
        self.device = device

        self.loader = DataLoader(
            Subset(dataset, indices),
            batch_size=batch_size,
            shuffle=True
        )

    def train(self, model, epochs=1, lr=0.01):
        model = copy.deepcopy(model)
        model.to(self.device)
        model.train()

        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        criterion = torch.nn.CrossEntropyLoss()

        for _ in range(epochs):
            for x, y in self.loader:
                x, y = x.to(self.device), y.to(self.device)

                optimizer.zero_grad()
                output = model(x)
                loss = criterion(output, y)
                loss.backward()
                optimizer.step()

        return model.state_dict()
