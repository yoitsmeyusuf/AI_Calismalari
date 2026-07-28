"""
Sohbet (conversation) formatında veri seti oluşturmak için ortak yardımcılar.
1. Hafta (domain dataset) ve 4. Hafta (identity dataset) burayı paylaşır
çünkü ikisi de aynı formatı kullanmak zorunda.

Şema:
    train: list<{content: str, images: null, role: str, thinking: str, tool_calls: null}>
splits: "turkish", "english"
"""
from datasets import Dataset, DatasetDict


def make_message(role: str, content: str, thinking: str = "") -> dict:
    """Referans veri setindeki tek bir mesaj (turn) sözlüğünü üretir."""
    return {
        "content": content,
        "images": None,
        "role": role,
        "thinking": thinking,
        "tool_calls": None,
    }


def make_conversation(*turns: dict) -> list:
    """Birden fazla make_message() çağrısını tek bir konuşmaya (satıra) dönüştürür."""
    return list(turns)


def build_dataset_dict(splits: dict) -> DatasetDict:
    """Verilen split'lerden bir DatasetDict kurar.

    splits: {split_adı: conversations} — her conversations, make_conversation()
    çıktılarından oluşan bir liste (liste-of-liste-of-mesaj). Referans identity
    veri seti "turkish"/"english" split'lerini kullanıyor, ama tek dilli bir
    domain veri seti için tek split (ör. sadece "turkish") de geçerli — asıl
    "standart" olan mesaj şeması (content/images/role/thinking/tool_calls),
    split isimleri değil.
    """
    return DatasetDict({name: Dataset.from_dict({"train": conversations}) for name, conversations in splits.items()})


def push(dataset_dict: DatasetDict, repo_id: str, token: str | None = None, private: bool = False) -> None:
    dataset_dict.push_to_hub(repo_id, token=token, private=private)
