import os
import csv
import torch
from torchvision import datasets

from model import build_model
from data import get_data


def predict():

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    model = build_model(pretrained=False)
    model.load_state_dict(
        torch.load("model.pth", map_location=device)
    )

    model.to(device)
    model.eval()

    loaders, _ = get_data()
    test_loader = loaders["test"]

    class_names = datasets.ImageFolder(
        "data/train"
    ).classes

    results = []

    with torch.no_grad():
        for x, paths in test_loader:

            x = x.to(device)
            out = model(x)
            preds = out.argmax(1)

            for p, pred in zip(paths, preds):
                results.append([
                    os.path.basename(p),
                    class_names[pred.item()]
                ])

    with open("submission.csv", "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["ID", "Label"])
        writer.writerows(results)

    print("Saved submission.csv")


if __name__ == "__main__":
    predict()