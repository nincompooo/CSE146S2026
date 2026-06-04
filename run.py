import torch
from torchvision.transforms import v2

from model import build_model, freeze_backbone
from data import get_data
from train import train_model


def set_seed(seed=42):
    import random
    import numpy as np
    import os

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)


def main():

    set_seed(42)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    loaders, sizes = get_data(val_fraction=0.0)

    model = build_model(num_classes=100, pretrained=True)
    freeze_backbone(model, unfrozen_layers=1)
    model = model.to(device)

    batch_mix = v2.RandomChoice([
        v2.MixUp(alpha=0.2, num_classes=100),
        v2.CutMix(alpha=1.0, num_classes=100)
    ])

    model = train_model(
        model=model,
        loaders=loaders,
        sizes=sizes,
        device=device,
        epochs=30,
        batch_mix=batch_mix
    )

    torch.save(model.state_dict(), "model.pth")


if __name__ == "__main__":
    main()