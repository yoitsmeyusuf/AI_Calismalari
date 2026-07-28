"""
4. Hafta: Identity LoRA eğitimi.
build_identity_dataset.py çalıştırılıp IDENTITY_DATASET_REPO_ID push edildikten
sonra çalıştırın.

Çalıştırma:
    .venv/bin/python hafta4_kimlik/train_identity_lora.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from common.lora_trainer import train_lora

load_dotenv()

if __name__ == "__main__":
    train_lora(
        dataset_repo=os.environ["IDENTITY_DATASET_REPO_ID"],
        output_repo=os.environ["IDENTITY_LORA_REPO_ID"],
        base_model=os.environ.get("BASE_MODEL", "unsloth/Qwen3.5-4B"),
    )
