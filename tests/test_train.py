import os
import hydra.utils
from omegaconf import OmegaConf
from pathlib import Path
import pytest

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DATA_FILE = PROJECT_ROOT / "data" / "processed" / "train_images.pt"

pytestmark = pytest.mark.skipif(not DATA_FILE.exists(), reason="Training data not available")



def test_train(capsys, monkeypatch):
    
    monkeypatch.setattr(hydra.utils, "get_original_cwd", lambda: os.getcwd())


    import mlopscookie.train as train_mod

    # 3) Also patch the local reference in your module (covers "from hydra.utils import get_original_cwd")
    if hasattr(train_mod, "get_original_cwd"):
        monkeypatch.setattr(train_mod, "get_original_cwd", lambda: os.getcwd())

    cfg = OmegaConf.create({
        "hyperparameters": {
            "lr": 0.001,
            "batch_size": 32,
            "num_epochs": 1,
            "model_name": "SimpleCNN",
        }
    })

    train_mod.train(cfg)

    out = capsys.readouterr().out
    assert "Training day and night" in out