import os
from PIL import Image

import torch
from torchvision import datasets, transforms


class TestDataset(torch.utils.data.Dataset):

    def __init__(self, folder, transform=None):

        self.paths = sorted(
            [
                os.path.join(folder, f)
                for f in os.listdir(folder)
                if f.endswith(".jpg")
            ]
        )

        self.transform = transform

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, idx):

        path = self.paths[idx]
        img = Image.open(path).convert("RGB")

        if self.transform:
            img = self.transform(img)

        return img, path


def get_data(val_fraction=0.0):

    bicubic = transforms.InterpolationMode.BICUBIC

    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(
            384, scale=(0.7, 1.0), interpolation=bicubic
        ),
        transforms.RandomHorizontalFlip(),
        transforms.ColorJitter(0.3, 0.3, 0.3, 0.1),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    eval_tf = transforms.Compose([
        transforms.Resize((384, 384), interpolation=bicubic),
        transforms.ToTensor(),
        transforms.Normalize(
            [0.485, 0.456, 0.406],
            [0.229, 0.224, 0.225]
        )
    ])

    root = "."

    full_train = datasets.ImageFolder(
        os.path.join(root, "train"),
        transform=train_tf
    )

    n = len(full_train)
    indices = torch.randperm(n)

    val_size = int(val_fraction * n)

    val_idx = indices[:val_size]
    train_idx = indices[val_size:]

    train_set = torch.utils.data.Subset(full_train, train_idx)

    val_set = None
    if val_fraction > 0:
        full_val = datasets.ImageFolder(
            os.path.join(root, "train"),
            transform=eval_tf
        )
        val_set = torch.utils.data.Subset(full_val, val_idx)

    test_set = TestDataset(
        os.path.join(root, "test"),
        eval_tf
    )

    loaders = {
        "train": torch.utils.data.DataLoader(
            train_set,
            batch_size=16,
            shuffle=True,
            num_workers=4
        ),
        "test": torch.utils.data.DataLoader(
            test_set,
            batch_size=16,
            shuffle=False,
            num_workers=4
        )
    }

    if val_set is not None:
        loaders["val"] = torch.utils.data.DataLoader(
            val_set,
            batch_size=16,
            shuffle=False,
            num_workers=4
        )

    sizes = {
        k: len(v.dataset)
        for k, v in loaders.items()
    }

    return loaders, sizes