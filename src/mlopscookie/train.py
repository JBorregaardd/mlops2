import os
import hydra
import matplotlib.pyplot as plt
from omegaconf import DictConfig

import torch
import wandb
from sklearn.metrics import (
    RocCurveDisplay,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

from .data import corrupt_mnist
from .model import SimpleCNN

DEVICE = torch.device(
    "cuda"
    if torch.cuda.is_available()
    else "mps"
    if torch.backends.mps.is_available()
    else "cpu"
)

# Pick ONE of these:
# ENTITY = "s181487-danmarks-tekniske-universitet-dtu"
# ENTITY = "s181487"
ENTITY = "s181487-danmarks-tekniske-universitet-dtu"


@hydra.main(version_base=None, config_path="../../configs", config_name="training_conf.yaml")
def train(cfg: DictConfig) -> None:
    hp = cfg.hyperparameters
    lr = float(hp.lr)
    batch_size = int(hp.batch_size)
    epochs = int(hp.num_epochs)
    model_name = str(hp.model_name)

    print("Training day and night")
    print(f"{lr=}, {batch_size=}, {epochs=}, {model_name=}")

    # Hydra changes cwd; use project root for saving/logging files
    project_root = hydra.utils.get_original_cwd()

    # Initialize W&B
    run = wandb.init(
        project="corrupt_mnist",
        entity=ENTITY,  # remove this line if you want W&B to use the API key default
        config={"lr": lr, "batch_size": batch_size, "epochs": epochs, "model_name": model_name},
        dir=project_root,  # keeps wandb/ logs in your repo root, not hydra run dir
    )

    try:
        model = SimpleCNN(num_classes=10).to(DEVICE)
        train_set, _ = corrupt_mnist()
        train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=batch_size, shuffle=True)

        loss_fn = torch.nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)

        statistics = {"train_loss": [], "train_accuracy": []}

        global_step = 0

        for epoch in range(epochs):
            model.train()

            epoch_preds = []
            epoch_targets = []

            for i, (img, target) in enumerate(train_dataloader):
                img, target = img.to(DEVICE), target.to(DEVICE)

                optimizer.zero_grad(set_to_none=True)
                y_pred = model(img)
                loss = loss_fn(y_pred, target)
                loss.backward()
                optimizer.step()

                acc = (y_pred.argmax(dim=1) == target).float().mean().item()

                statistics["train_loss"].append(loss.item())
                statistics["train_accuracy"].append(acc)

                wandb.log(
                    {
                        "train_loss": loss.item(),
                        "train_accuracy": acc,
                        "epoch": epoch,
                        "iter": i,
                    },
                    step=global_step,
                )

                epoch_preds.append(y_pred.detach().cpu())
                epoch_targets.append(target.detach().cpu())

                if i % 100 == 0:
                    print(f"Epoch {epoch}, iter {i}, loss: {loss.item()}")

    

                    # Log gradient histogram
                    grads = torch.cat(
                        [p.grad.detach().flatten().cpu() for p in model.parameters() if p.grad is not None],
                        dim=0,
                    )
                    wandb.log({"gradients": wandb.Histogram(grads.numpy())}, step=global_step)

                global_step += 1

            # ROC curves at end of epoch (log as a matplotlib figure)
            preds = torch.cat(epoch_preds, dim=0)
            targets = torch.cat(epoch_targets, dim=0)

            fig = plt.figure(figsize=(10, 8))
            ax = plt.gca()

            for class_id in range(10):
                one_hot = torch.zeros_like(targets, dtype=torch.int32)
                one_hot[targets == class_id] = 1

                RocCurveDisplay.from_predictions(
                    one_hot.numpy(),
                    preds[:, class_id].numpy(),
                    name=f"class {class_id}",
                    ax=ax,
                )

            ax.set_title(f"ROC curves (epoch {epoch})")
            wandb.log({"roc_curves": wandb.Image(fig)}, step=global_step)
            plt.close(fig)

        print("Training complete")

        # Save model + plot into repo folders (not hydra run dir)
        models_dir = os.path.join(project_root, "models")
        reports_dir = os.path.join(project_root, "reports", "figures")
        os.makedirs(models_dir, exist_ok=True)
        os.makedirs(reports_dir, exist_ok=True)

        model_path = os.path.join(models_dir, model_name)
        torch.save(model.state_dict(), model_path)
        print(f"Saved model to: {model_path}")

        fig, axs = plt.subplots(1, 2, figsize=(15, 5))
        axs[0].plot(statistics["train_loss"])
        axs[0].set_title("Train loss")
        axs[1].plot(statistics["train_accuracy"])
        axs[1].set_title("Train accuracy")

        fig_path = os.path.join(reports_dir, "training_statistics.png")
        fig.savefig(fig_path)
        plt.close(fig)
        print(f"Saved plot to: {fig_path}")

        # Final metrics (over last epoch’s collected preds/targets)
        final_pred_labels = preds.argmax(dim=1).numpy()
        final_targets = targets.numpy()

        final_accuracy = accuracy_score(final_targets, final_pred_labels)
        final_precision = precision_score(final_targets, final_pred_labels, average="weighted", zero_division=0)
        final_recall = recall_score(final_targets, final_pred_labels, average="weighted", zero_division=0)
        final_f1 = f1_score(final_targets, final_pred_labels, average="weighted", zero_division=0)

        wandb.log(
            {
                "final_accuracy": final_accuracy,
                "final_precision": final_precision,
                "final_recall": final_recall,
                "final_f1": final_f1,
            }
        )

        # Log model artifact
        artifact = wandb.Artifact(
            name="corrupt_mnist_model",
            type="model",
            description="A model trained to classify corrupt MNIST images",
            metadata={
                "accuracy": final_accuracy,
                "precision": final_precision,
                "recall": final_recall,
                "f1": final_f1,
                "model_name": model_name,
            },
        )
        artifact.add_file(model_path)
        run.log_artifact(artifact)

    finally:
        # Always close the run cleanly
        wandb.finish()


if __name__ == "__main__":
    train()
