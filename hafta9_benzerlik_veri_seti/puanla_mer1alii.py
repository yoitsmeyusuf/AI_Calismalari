"""
9. Hafta (ek): Mer1Alii/turkish-sts-dataset icindeki 20 cumle ciftini puanlayip
ayri bir repoya pushlar. Kaynak veri setinde score sutunu bos (None).

Puanlar asagida, kaynak veri setiyle ayni sirada. 0-5 STS-B olcegi:
  5 = ayni anlam, 4 = kucuk detay farki, 3 = ayni konu/kismen ortusuyor,
  2 = ayni alan ama farkli olay, 1 = zayif baglanti, 0 = alakasiz.

Repo id ve token bilerek .env'den okunmuyor; repo id asagida sabit, token
`hf auth login` cache'inden geliyor.

Calistirma:
    ../.venv/bin/python hafta9_benzerlik_veri_seti/puanla_mer1alii.py
"""
import csv
from pathlib import Path

from datasets import Dataset, load_dataset

KAYNAK_REPO = "Mer1Alii/turkish-sts-dataset"
REPO_ID = "yoitsmeyusuf/turkish-sts-dataset-puanli"

HERE = Path(__file__).resolve().parent
CIKTI_CSV = HERE / "veri" / "mer1alii_puanli.csv"

# Kaynak veri setiyle ayni sirada, 20 puan. Gerekce icin README'deki tabloya bak.
PUANLAR = [
    4.6,  # 0  yagmur -> piknik iptal / hava kotu -> plan suya dustu
    4.2,  # 1  Istanbul trafigi kotulesiyor / arac yogunlugu artiyor
    4.8,  # 2  kedi her sabah pencerede / kedi her sabah camda sokagi seyrediyor
    4.8,  # 3  toplanti yarin 10'da / baslama saati yarin sabah 10
    4.2,  # 4  cocuklar parkta futbol / cocuklar disarida top pesinde
    4.4,  # 5  kitap beni etkiledi / roman derin iz birakti
    3.2,  # 6  anneannem pazar sarma yapar / pazarlari geleneksel yemek pisirilir
    1.2,  # 7  bilgisayar yavas, format / telefon ekrani kirik, servis
    4.6,  # 8  Ankara kisin cok soguk / baskentte sicaklik sifirin altinda
    4.0,  # 9  Magibu STS odevi / yapay zeka kursu odevi cumle ciftleri
    4.8,  # 10 nufus 85 milyonu gecti / insan sayisi 85 milyonun uzerinde
    3.8,  # 11 gece gec saate kadar ders / sinav icin gece boyunca kitap
    4.6,  # 12 elektrikli araclar cevreye az zarar / elektrikliler daha temiz
    1.8,  # 13 son dakika goluyle kazandik / Besiktas Fenerbahce'yi yendi
    4.0,  # 14 denize girmek icin Antalya / yaz tatili Akdeniz kiyisi
    3.6,  # 15 adam marketten ekmek sut aldi / bir kisi bakkaldan alisveris
    4.8,  # 16 deprem bolgesine yardim / afet sonrasi insani yardim
    2.4,  # 17 bugun 38 derece bunaltici / yazin Samsun'da nem yuksek
    0.6,  # 18 ogrenciler sinifta sessizce sinav / cocuklar bahcede gurultulu
    1.6,  # 19 restoran lezzetli ama pahali / kafe kahve guzel ama kalabalik
]


def puanla():
    kaynak = load_dataset(KAYNAK_REPO, split="train")
    if len(kaynak) != len(PUANLAR):
        raise SystemExit(
            f"Kaynak {len(kaynak)} satir, elimizde {len(PUANLAR)} puan var — "
            "veri seti degismis, puanlari gozden gecir."
        )

    satirlar = []
    for satir, puan in zip(kaynak, PUANLAR):
        if not 0.0 <= puan <= 5.0:
            raise ValueError(f"aralik disi puan: {puan}")
        satirlar.append(
            {
                "sentence1": satir["sentence1"].strip(),
                "sentence2": satir["sentence2"].strip(),
                "score": float(puan),
            }
        )
    return satirlar


def csv_yaz(satirlar):
    with open(CIKTI_CSV, "w", encoding="utf-8", newline="") as f:
        yazici = csv.DictWriter(f, fieldnames=["sentence1", "sentence2", "score"])
        yazici.writeheader()
        yazici.writerows(satirlar)
    print(f"Yerel kopya: {CIKTI_CSV}")


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

# Türkçe STS — Puanlanmış

[{kaynak}](https://huggingface.co/datasets/{kaynak}) veri setindeki {n} Türkçe
cümle çiftinin anlamsal benzerlik puanlarıyla birlikte sürümü. Kaynak veri
setinde cümleler var ama `score` sütunu boş (`None`); bu repoda o sütun
dolduruldu. Cümlelere dokunulmadı.

## Sütunlar

| sütun | tip | açıklama |
|---|---|---|
| `sentence1` | string | Birinci cümle (kaynaktan aynen) |
| `sentence2` | string | İkinci cümle (kaynaktan aynen) |
| `score` | float | Anlamsal benzerlik, 0.0–5.0 |

## Puanlama ölçeği (STS-B)

| puan | anlamı |
|---|---|
| 5 | Aynı anlam, fark yok |
| 4 | Aynı olay, küçük detay/vurgu farkı |
| 3 | Aynı konu, kısmen örtüşen bilgi |
| 2 | Aynı alan ama farklı olay |
| 1 | Zayıf bağlantı (ortak tema yok denecek kadar az) |
| 0 | Alakasız |

`sentence-transformers` ile `CosineSimilarityLoss` kullanırken 5'e bölüp 0–1
aralığına indirmek yeterli.

## Kullanım

```python
from datasets import load_dataset

ds = load_dataset("{repo_id}", split="train")
print(ds[0])
```

## Notlar

- {n} çift, tek `train` bölümü.
- Puanlar tek bir anotatörün yargısı; çoklu anotasyon ortalaması değil,
  bu yüzden ±0.5 civarı bir belirsizlik payı var.
- Puanlama kodu ve satır bazlı gerekçeler:
  `hafta9_benzerlik_veri_seti/puanla_mer1alii.py`
"""


def main():
    satirlar = puanla()
    csv_yaz(satirlar)

    ds = Dataset.from_list(satirlar)
    print(f"{len(ds)} satır puanlandı: {ds.column_names}")

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
