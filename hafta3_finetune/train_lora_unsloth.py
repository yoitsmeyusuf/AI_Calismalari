"""
3. Hafta: unsloth ile Model Fine-Tune Etme (LoRA)

1. haftada hazırlanan veri setini (DATASET_REPO_ID) kullanarak BASE_MODEL'i
LoRA ile eğitir ve adaptörü Hugging Face Hub'a (LORA_REPO_ID) yükler.

Çalıştırma:
    .venv/bin/python hafta3_finetune/train_lora_unsloth.py
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
        dataset_repo=os.environ["DATASET_REPO_ID"],
        output_repo=os.environ["LORA_REPO_ID"],
        base_model=os.environ.get("BASE_MODEL", "unsloth/Qwen3.5-4B"),
    )
