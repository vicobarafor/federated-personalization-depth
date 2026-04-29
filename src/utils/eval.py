
import torch

def evaluate_model(model, dataloader, device):
    model.eval()

    correct = 0
    total = 0
    loss_sum = 0.0

    criterion = torch.nn.CrossEntropyLoss()

    with torch.no_grad():
        for x, y in dataloader:
            x = x.to(device)
            y = y.to(device)

            out = model(x)
            loss = criterion(out, y)

            loss_sum += loss.item() * x.size(0)

            preds = out.argmax(dim=1)
            correct += (preds == y).sum().item()
            total += y.size(0)

    avg_loss = loss_sum / total
    acc = correct / total

    return {
        "loss": avg_loss,
        "accuracy": acc,
    }
