from dataset import create_wall_dataloader
from evaluator import ProbingEvaluator
import torch
from models import JEPA_Model
import glob


def get_device():
    """Check for GPU availability."""
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Using device:", device)
    return device


def load_data(device):
    """Load training and validation datasets."""
    data_path = "/scratch/DL24FA"

    # Create training dataloader
    probe_train_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/train",
        probing=True,
        device=device,
        batch_size=64,  # Specify batch size here
        train=True,
    )

    # Create validation dataloaders
    probe_val_normal_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_normal/val",
        probing=True,
        device=device,
        batch_size=64,  # Specify batch size for validation
        train=False,
    )

    probe_val_wall_ds = create_wall_dataloader(
        data_path=f"{data_path}/probe_wall/val",
        probing=True,
        device=device,
        batch_size=64,  # Specify batch size for validation
        train=False,
    )

    probe_val_ds = {"normal": probe_val_normal_ds, "wall": probe_val_wall_ds}

    return probe_train_ds, probe_val_ds


def load_model():
    """Load or initialize the model."""
    # TODO: Replace MockModel with your trained model
    # Example: Define a simple model or load your pre-trained model
    class JEPA_Model(torch.nn.Module):
        def __init__(self):
            super(JEPA_Model, self).__init__()
            self.layer = torch.nn.Linear(10, 1)  # Example layer

        def forward(self, x):
            return self.layer(x)

    model = JEPA_Model().to(torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    return model


def evaluate_model(device, model, probe_train_ds, probe_val_ds):
    """Evaluate the model using the provided datasets."""
    evaluator = ProbingEvaluator(
        device=device,
        model=model,
        probe_train_ds=probe_train_ds,
        probe_val_ds=probe_val_ds,
        quick_debug=False,
    )

    # Train the prober
    prober = evaluator.train_pred_prober()

    # Evaluate the model
    avg_losses = evaluator.evaluate_all(prober=prober)

    for probe_attr, loss in avg_losses.items():
        print(f"{probe_attr} loss: {loss}")


if __name__ == "__main__":
    device = get_device()  # Get the device
    probe_train_ds, probe_val_ds = load_data(device)  # Load datasets
    model = load_model()  # Load the model
    evaluate_model(device, model, probe_train_ds, probe_val_ds)  # Evaluate the model
