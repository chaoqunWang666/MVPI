import os

import torch


class DinoV3Loader:
    def __init__(self, model_name="dinov3_vitb16", hub_dir=None, checkpoint_dir=None, device="cuda"):
        self.model_name = model_name
        repo_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.hub_dir = os.path.abspath(
            hub_dir or os.environ.get("DINOV3_HUB_DIR", os.path.join(repo_root, "dinov3-main"))
        )
        checkpoint_dir = checkpoint_dir or os.environ.get(
            "DINOV3_CHECKPOINT_DIR", os.path.join(torch.hub.get_dir(), "checkpoints")
        )
        self.weights_path = os.path.join(checkpoint_dir, f"{model_name}_pretrain.pth")
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")

    def load_model(self):
        print("===================================")
        print(f"Loading DINOv3: {self.model_name}")
        print(f"Hub directory: {self.hub_dir}")
        print(f"Weights: {self.weights_path}")
        print("===================================")

        if not os.path.exists(self.hub_dir):
            raise FileNotFoundError(f"DINOv3 source directory not found: {self.hub_dir}")
        if not os.path.exists(self.weights_path):
            raise FileNotFoundError(f"DINOv3 weights not found: {self.weights_path}")

        model = torch.hub.load(
            self.hub_dir,
            self.model_name,
            source="local",
            weights=self.weights_path,
        )
        model.to(self.device)
        model.eval()
        print("DINOv3 loaded successfully")
        return model


def load_dinov3_from_local(model_name="dinov3_vitb16", hub_dir=None, checkpoint_dir=None, device="cuda"):
    """Load a local DINOv3 model and its pretrained checkpoint."""
    return DinoV3Loader(
        model_name=model_name,
        hub_dir=hub_dir,
        checkpoint_dir=checkpoint_dir,
        device=device,
    ).load_model()


if __name__ == "__main__":
    loader = DinoV3Loader()
    model = loader.load_model()
    inputs = torch.randn(1, 3, 224, 224, device=loader.device)
    with torch.no_grad():
        outputs = model.forward_features(inputs)
    print("Patch tokens:", outputs["x_norm_patchtokens"].shape)