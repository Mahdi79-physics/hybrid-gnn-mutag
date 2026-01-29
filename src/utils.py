import torch
import torch.nn.functional as F

def train_epoch(model, loader, optimizer, criterion):
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for data in loader:
        optimizer.zero_grad()
        logits, _ = model(data.x, data.edge_index, data.batch)
        loss = criterion(logits.squeeze(), data.y.float())
        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = torch.round(torch.sigmoid(logits.squeeze()))
        correct += (preds == data.y).sum().item()
        total += data.y.size(0)

    return total_loss / len(loader), correct / total

def eval_epoch(model, loader, criterion):
    model.eval()
    total_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in loader:
            logits, _ = model(data.x, data.edge_index, data.batch)
            loss = criterion(logits.squeeze(), data.y.float())

            total_loss += loss.item()
            preds = torch.round(torch.sigmoid(logits.squeeze()))
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

    return total_loss / len(loader), correct / total
