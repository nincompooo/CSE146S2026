import torch.nn as nn
import torchvision


def build_model(num_classes=100, pretrained=True):

    weights = (
        torchvision.models.ViT_B_16_Weights.IMAGENET1K_SWAG_E2E_V1
        if pretrained
        else None
    )

    model = torchvision.models.vit_b_16(
        weights=weights,
        image_size=384
    )

    in_features = model.heads.head.in_features

    model.heads.head = nn.Sequential(
        nn.Dropout(0.4),
        nn.Linear(in_features, num_classes)
    )

    return model


def freeze_backbone(model, unfrozen_layers=0):

    # freeze everything first
    for p in model.parameters():
        p.requires_grad = False

    # always train head
    for p in model.heads.parameters():
        p.requires_grad = True

    # optionally unfreeze last transformer blocks
    if unfrozen_layers > 0:
        for block in model.encoder.layers[-unfrozen_layers:]:
            for p in block.parameters():
                p.requires_grad = True

    return model