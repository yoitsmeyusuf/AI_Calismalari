"""
9. Hafta: Turkce cumle benzerligi veri setini (sentence1/sentence2/score)
Hugging Face Hub'a Dataset olarak pushlar.

Repo id ve token bilerek .env'den okunmuyor; repo id asagida sabit, token ise
`hf auth login` cache'inden geliyor.

Calistirma (ust klasordeki venv'de datasets kurulu):
    ../.venv/bin/python hafta9_benzerlik_veri_seti/push_dataset.py
"""
import csv
from pathlib import Path

from datasets import Dataset

REPO_ID = "yoitsmeyusuf/turkce-cumle-benzerligi"

HERE = Path(__file__).resolve().parent
CSV_YOLU = HERE / "veri" / "ciftler.csv"
PUAN_MIN, PUAN_MAX = 0.0, 5.0


def satirlari_oku():
    with open(CSV_YOLU, encoding="utf-8", newline="") as f:
        satirlar = list(csv.DictReader(f))

    temiz = []
    for i, satir in enumerate(satirlar, start=2):  # 1. satir baslik
        s1 = (satir["sentence1"] or "").strip()
        s2 = (satir["sentence2"] or "").strip()
        ham_puan = (satir["score"] or "").strip()
        if not s1 or not s2:
            raise ValueError(f"satır {i}: boş cümle")
        try:
            puan = float(ham_puan)  # "nan" -> float('nan'); puanlar henüz verilmedi
        except ValueError:
            raise ValueError(f"satır {i}: score sayı değil ({ham_puan!r})") from None
        if puan == puan and not PUAN_MIN <= puan <= PUAN_MAX:  # nan != nan
            raise ValueError(f"satır {i}: score {puan} aralık dışı [{PUAN_MIN}, {PUAN_MAX}]")
        temiz.append({"sentence1": s1, "sentence2": s2, "score": puan})
    return temiz


KART = """---
license: mit
language:
- tr
task_categories:
- sentence-similarity
tags:
- semantic-textual-similarity
- turkish
size_categories:
- n<1K
---

# Türkçe Cümle Benzerliği (STS)

Elle yazılmış Türkçe cümle çiftleri ve anlamsal benzerlik puanları.

## Sütunlar

| sütun | tip | açıklama |
|---|---|---|
| `sentence1` | string | Birinci cümle |
| `sentence2` | string | İkinci cümle |
| `score` | float | Anlamsal benzerlik, 0.0–5.0 (STS-B ölçeği) — **şu an tüm satırlarda `NaN`, puanlama yapılmadı** |

`score` ölçeği STS-B / `stsb_multi_mt` ile aynı olacak: 0 = tamamen alakasız,
5 = aynı anlam. Puanlar henüz girilmedi; veri seti şimdilik sadece cümle
çiftlerini içeriyor.

## Kullanım

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Notlar

- {n} çift, tek `train` bölümü.
- Çiftler kolay/orta/zor karışık: parafrazlar, kısmi örtüşen ve tamamen
  ilgisiz çiftler bilinçli olarak bir arada.
- `score` sütunu henüz doldurulmadı (`NaN`); anotasyon sonraki adımda eklenecek.
"""


def main():
    satirlar = satirlari_oku()
    ds = Dataset.from_list(satirlar)
    print(f"{len(ds)} satır okundu: {ds.column_names}")

    ds.push_to_hub(REPO_ID, private=False)
    print(f"Veri seti pushlandı: https://huggingface.co/datasets/{REPO_ID}")

    from huggingface_hub import HfApi

    kart = KART.format(repo_id=REPO_ID, n=len(ds))
    HfApi().upload_file(
        path_or_fileobj=kart.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Dataset kartı",
    )
    print("Dataset kartı yüklendi.")


if __name__ == "__main__":
    main()
