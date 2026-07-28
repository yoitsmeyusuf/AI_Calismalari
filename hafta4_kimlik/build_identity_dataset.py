"""
4. Hafta: Identity veri setini oluşturur ve Hugging Face Hub'a push eder.
Ortak mesaj şeması kullanılır (bkz. common/hf_dataset_schema.py).

Çalıştırma:
    .venv/bin/python hafta4_kimlik/build_identity_dataset.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from common.hf_dataset_schema import build_dataset_dict, make_conversation, make_message
from identity_seeds import ENGLISH_SEEDS, TURKISH_SEEDS

load_dotenv()


def seeds_to_conversations(seeds: list[dict]) -> list:
    conversations = []
    for item in seeds:
        conversations.append(
            make_conversation(
                make_message("user", item["soru"]),
                make_message("assistant", item["cevap"], thinking=item.get("dusunce", "")),
            )
        )
    return conversations


def main():
    if not TURKISH_SEEDS or not ENGLISH_SEEDS:
        raise SystemExit(
            "identity_seeds.py içine en az 10-20 Türkçe ve İngilizce kimlik örneği "
            "eklemeden veri seti oluşturulamaz."
        )

    dataset_dict = build_dataset_dict(
        {
            "turkish": seeds_to_conversations(TURKISH_SEEDS),
            "english": seeds_to_conversations(ENGLISH_SEEDS),
        }
    )
    print(dataset_dict)

    repo_id = os.environ["IDENTITY_DATASET_REPO_ID"]
    token = os.environ.get("HF_TOKEN")
    dataset_dict.push_to_hub(repo_id, token=token)
    print(f"Yüklendi: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
