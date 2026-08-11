"""
9. Hafta (ek): Mer1Alii/turkish-sts-dataset icindeki birlestirilmis veriyi
kendi repomuza aynen aktarir.

Kaynak repoda iki odevin ciftleri zaten birlestirilmis ve puanlanmis durumda
(40 satir). Burada puan uretilmiyor, degistirilmiyor: score sutunu kaynakta
neyse o kaliyor (bos/NaN varsa NaN olarak gecer).

Repo id'ler asagida sabit, .env okunmuyor.

Calistirma:
    ../.venv/bin/python hafta9_benzerlik_veri_seti/birlestir_ve_push.py
"""
import csv
from pathlib import Path

from datasets import Value, load_dataset

KAYNAK_REPO = "Mer1Alii/turkish-sts-dataset"
REPO_ID = "yoitsmeyusuf/turkce-sts-birlesik"

HERE = Path(__file__).resolve().parent
CIKTI_CSV = HERE / "veri" / "birlesik.csv"
SUTUNLAR = ["sentence1", "sentence2", "score"]


def yukle():
    # force_redownload: kaynak repo guncellenince yerel cache eski surumu vermesin.
    ds = load_dataset(KAYNAK_REPO, split="train", download_mode="force_redownload")
    if list(ds.column_names) != SUTUNLAR:
        raise SystemExit(f"{KAYNAK_REPO}: sütunlar {ds.column_names}, beklenen {SUTUNLAR}")
    ds = ds.cast_column("score", Value("float64"))

    for i, puan in enumerate(ds["score"]):
        if puan == puan and not 0.0 <= puan <= 5.0:  # NaN != NaN
            raise SystemExit(f"satır {i}: score {puan} aralık dışı [0, 5]")
    return ds


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
source_datasets:
- {kaynak}
---

# Türkçe STS — Birleşik

[{kaynak}](https://huggingface.co/datasets/{kaynak}) veri setinin aynası:
iki ödevin cümle çiftleri tek yerde birleştirilmiş ve puanlanmış hali,
{n} Türkçe cümle çifti. Cümleler ve puanlar kaynaktan aynen alındı,
burada hiçbir puan üretilmedi veya değiştirilmedi.

## Sütunlar

| sütun | tip | açıklama |
|---|---|---|
| `sentence1` | string | Birinci cümle |
| `sentence2` | string | İkinci cümle |
| `score` | float | Anlamsal benzerlik, 0.0–5.0 (STS-B ölçeği) |

Ölçek STS-B ile aynı: 5 = aynı anlam, 4 = küçük detay farkı, 3 = aynı konu /
kısmi örtüşme, 2 = aynı alan farklı olay, 1 = zayıf bağlantı, 0 = alakasız.
`sentence-transformers` + `CosineSimilarityLoss` için 5'e bölüp 0–1'e indirin.

## Kullanım

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Notlar

- {n} çift, tek `train` bölümü; satır sırası kaynakla aynı.
- Puanlar tek anotatörün yargısı; çoklu anotasyon ortalaması değil.
"""


def main():
    ds = yukle()
    puanli = sum(1 for s in ds["score"] if s == s)
    print(f"{KAYNAK_REPO}: {len(ds)} satır ({puanli} puanlı, {len(ds) - puanli} NaN)")

    with open(CIKTI_CSV, "w", encoding="utf-8", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=SUTUNLAR)
        yazici.writeheader()
        yazici.writerows(ds)
    print(f"Yerel kopya: {CIKTI_CSV}")

    ds.push_to_hub(REPO_ID, private=False)
    print(f"Veri seti pushlandı: https://huggingface.co/datasets/{REPO_ID}")

    from huggingface_hub import HfApi

    HfApi().upload_file(
        path_or_fileobj=KART.format(kaynak=KAYNAK_REPO, repo_id=REPO_ID, n=len(ds)).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=REPO_ID,
        repo_type="dataset",
        commit_message="Dataset kartı",
    )
    print("Dataset kartı yüklendi.")


if __name__ == "__main__":
    main()
