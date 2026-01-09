import os
os.environ['WANDB_API_KEY'] = "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"
import torch
from mlopscookie.model import SimpleCNN
import wandb



api = wandb.Api()

artifact = api.artifact(
    "s181487-danmarks-tekniske-universitet-dtu/corrupt_mnist/corrupt_mnist_model:v0"
)

artifact_dir = artifact.download("corrupt_mnist_model")

print("Downloaded files:", os.listdir(artifact_dir))

model = SimpleCNN(num_classes=10)
state_dict = torch.load(
    os.path.join(artifact_dir, "model1.pth"),
    map_location="cpu",
)
model.load_state_dict(state_dict)
model.eval()