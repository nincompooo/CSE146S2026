import time
import torch
import torch.nn as nn

from torchvision.transforms import v2

from model import build_model, freeze_backbone
from data import get_data


def train_model(
    model,
    loaders,
    sizes,
    device,
    epochs=30,
    batch_mix=None
):

    criterion = nn.CrossEntropyLoss(label_smoothing=0.1)

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=1e-4,
        weight_decay=0.05
    )

    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=epochs
    )

    best_acc = 0.0

    for epoch in range(epochs):

        print(f"\nEpoch {epoch+1}/{epochs}")

        for phase in ["train", "val"]:

            if phase not in loaders:
                continue

            model.train() if phase == "train" else model.eval()

            total_loss = 0
            correct = 0

            for x, y in loaders[phase]:

                x, y = x.to(device), y.to(device)

                optimizer.zero_grad()

                use_mix = (
                    phase == "train"
                    and batch_mix is not None
                )

                with torch.set_grad_enabled(phase == "train"):

                    if use_mix:
                        x, y_soft = batch_mix(x, y)
                        out = model(x)
                        loss = criterion(out, y_soft)
                        preds = out.argmax(1)
                        labels = y_soft.argmax(1)
                    else:
                        out = model(x)
                        loss = criterion(out, y)
                        preds = out.argmax(1)
                        labels = y

                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                total_loss += loss.item() * x.size(0)
                correct += (preds == labels).sum().item()

            if phase == "train":
                scheduler.step()

            acc = correct / sizes[phase]
            print(f"{phase} acc: {acc:.4f}")

            if phase == "val" and acc > best_acc:
                best_acc = acc
                torch.save(model.state_dict(), "model.pth")

    print(f"\nBest val acc: {best_acc:.4f}")
    return model