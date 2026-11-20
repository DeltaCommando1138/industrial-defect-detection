"""ResNet50 single-view model for defect classification and segmentation."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F
from torchvision.models import ResNet50_Weights, resnet50


class ConvBlock(nn.Sequential):
    """Two convolution layers used to refine each decoded feature map."""

    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.ReLU(inplace=True),
        )


class DecoderBlock(nn.Module):
    """Upsample a feature map and combine it with an encoder skip connection."""

    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.upsample = nn.ConvTranspose2d(
            in_channels, out_channels, kernel_size=2, stride=2
        )
        self.refine = ConvBlock(out_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = self.upsample(x)
        if x.shape[-2:] != skip.shape[-2:]:
            x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.refine(torch.cat((x, skip), dim=1))


class RealIADSingleViewModel(nn.Module):
    """Predict one image-level defect logit and one pixel-level mask logit map."""

    def __init__(self, pretrained: bool = True) -> None:
        super().__init__()
        weights = ResNet50_Weights.DEFAULT if pretrained else None
        backbone = resnet50(weights=weights)

        self.stem = nn.Sequential(backbone.conv1, backbone.bn1, backbone.relu)
        self.maxpool = backbone.maxpool
        self.layer1 = backbone.layer1
        self.layer2 = backbone.layer2
        self.layer3 = backbone.layer3
        self.layer4 = backbone.layer4

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(p=0.2),
            nn.Linear(2048, 1),
        )

        self.decode4 = DecoderBlock(2048, 1024, 512)
        self.decode3 = DecoderBlock(512, 512, 256)
        self.decode2 = DecoderBlock(256, 256, 128)
        self.decode1 = DecoderBlock(128, 64, 64)
        self.final_upsample = nn.ConvTranspose2d(64, 32, kernel_size=2, stride=2)
        self.mask_head = nn.Sequential(ConvBlock(32, 32), nn.Conv2d(32, 1, 1))

    def forward(self, image: torch.Tensor) -> dict[str, torch.Tensor]:
        stem = self.stem(image)            # [B, 64, 128, 128]
        layer1 = self.layer1(self.maxpool(stem))  # [B, 256, 64, 64]
        layer2 = self.layer2(layer1)       # [B, 512, 32, 32]
        layer3 = self.layer3(layer2)       # [B, 1024, 16, 16]
        layer4 = self.layer4(layer3)       # [B, 2048, 8, 8]

        classification_logits = self.classifier(layer4).squeeze(1)

        decoded = self.decode4(layer4, layer3)
        decoded = self.decode3(decoded, layer2)
        decoded = self.decode2(decoded, layer1)
        decoded = self.decode1(decoded, stem)
        decoded = self.final_upsample(decoded)
        mask_logits = self.mask_head(decoded)

        return {
            "classification_logits": classification_logits,
            "mask_logits": mask_logits,
        }


if __name__ == "__main__":
    model = RealIADSingleViewModel(pretrained=False)
    outputs = model(torch.randn(1, 3, 256, 256))
    print("classification:", tuple(outputs["classification_logits"].shape))
    print("mask:          ", tuple(outputs["mask_logits"].shape))
