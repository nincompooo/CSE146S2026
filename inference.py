import torch
import pandas as pd

from torch.utils.data import DataLoader

from data import (
    TestDataset,
    eval_tf
)

from model import build_model

# =====================================================
# DEVICE
# =====================================================

if torch.cuda.is_available():
    device = torch.device("cuda")

elif torch.backends.mps.is_available():
    device = torch.device("mps")

else:
    device = torch.device("cpu")

# =====================================================
# MODEL
# =====================================================

model = build_model()

model.load_state_dict(
    torch.load(
        "best_vit.pt",
        map_location=device
    )
)

model.to(device)
model.eval()

# =====================================================
# TEST DATA
# =====================================================

test_dataset = TestDataset(
    "test",
    eval_tf
)

test_loader = DataLoader(
    test_dataset,
    batch_size=16,
    shuffle=False,
    num_workers=0
)

# =====================================================
# PREDICTIONS
# =====================================================

predictions = []

with torch.no_grad():

    for x, names in test_loader:

        x = x.to(device)

        logits = model(x)

        pred = logits.argmax(1)

        pred = pred.cpu().numpy()

        for fname, label in zip(
            names,
            pred
        ):
            predictions.append(
                [fname, int(label)]
            )

df = pd.DataFrame(
    predictions,
    columns=["ID", "Label"]
)

df.to_csv(
    "submission.csv",
    index=False
)

print("Saved submission.csv")