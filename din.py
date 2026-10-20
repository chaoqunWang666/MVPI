"""Small command-line smoke entry point for loading the local DINOv3 model."""

import torch

from models.dinov3_loader import DinoV3Loader


if __name__ == "__main__":
    loader = DinoV3Loader()
    model = loader.load_model()
    inputs = torch.randn(1, 3, 224, 224, device=loader.device)
    with torch.no_grad():
        features = model.forward_features(inputs)
    print("Patch token shape:", features["x_norm_patchtokens"].shape)