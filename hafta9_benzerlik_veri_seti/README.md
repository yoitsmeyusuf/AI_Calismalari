# 9. Hafta — Cümle Benzerliği Veri Seti (sentence1 / sentence2 / score)

Türkçe cümle çiftlerinden oluşan HF Dataset:
**https://huggingface.co/datasets/yoitsmeyusuf/turkce-cumle-benzerligi**

19 çift, tek `train` bölümü. `score` sütunu **şu an tüm satırlarda `NaN`** —
puanlama sonraki adımda elle yapılacak.

## Şema

| sütun | tip | açıklama |
|---|---|---|
| `sentence1` | string | Birinci cümle |
| `sentence2` | string | İkinci cümle |
| `score` | float | Anlamsal benzerlik, 0.0–5.0 (STS-B ölçeği); henüz `NaN` |

Ölçek 0–5 seçildi çünkü STS-B / `stsb_multi_mt` ve `sentence-transformers`'ın
`CosineSimilarityLoss` örnekleriyle aynı ölçek — eğitimde 5'e bölüp 0–1'e
indirmek yeterli.

## Ortam

Bu hafta **üst klasördeki venv** kullanılıyor (`magibu/.venv`, Python 3.13):
`datasets` orada zaten kurulu, ek kurulum gerekmiyor. Repo kökündeki
`odevler/.venv`'e dokunulmadı.

```bash
../.venv/bin/python hafta9_benzerlik_veri_seti/check_env.py     # kurulum kontrolü
../.venv/bin/python hafta9_benzerlik_veri_seti/push_dataset.py  # Hub'a push
```

`requirements.txt` sadece referans; başka bir makinede kurmak gerekirse:

```bash
uv pip install --python <venv>/bin/python -r hafta9_benzerlik_veri_seti/requirements.txt
```

Hub'a yazma için `hf auth login` cache'indeki token kullanılıyor (write izni
gerekli). Repo id `push_dataset.py` içinde **sabit** — bilinçli olarak `.env`'e
bakılmıyor.

## Ek: başka bir veri setini puanlama

[Mer1Alii/turkish-sts-dataset](https://huggingface.co/datasets/Mer1Alii/turkish-sts-dataset)
20 cümle çifti içeriyor ama `score` sütunu boş (`None`). Bu çiftler 0–5 STS-B
ölçeğinde puanlanıp ayrı bir repoya pushlandı (kaynak veri seti değiştirilmedi,
cümlelere dokunulmadı):

**https://huggingface.co/datasets/yoitsmeyusuf/turkish-sts-dataset-puanli**

```bash
../.venv/bin/python hafta9_benzerlik_veri_seti/puanla_mer1alii.py
```

Puanlar `puanla_mer1alii.py` içindeki `PUANLAR` listesinde, kaynakla aynı sırada
ve her satırın yanında hangi çift olduğu yazıyor — değiştirip script'i tekrar
çalıştırmak yeterli. Script kaynak satır sayısını kontrol ediyor, kayarsa
hata verip duruyor. Puanlanmış hali `veri/mer1alii_puanli.csv` olarak da
yerele yazılıyor.

Kullanılan ölçek: 5 = aynı anlam, 4 = küçük detay farkı, 3 = aynı konu/kısmi
örtüşme, 2 = aynı alan farklı olay, 1 = zayıf bağlantı, 0 = alakasız.

## Ek: birleştirilmiş sürüm (tek repo)

`Mer1Alii/turkish-sts-dataset` sonradan güncellendi: iki ödevin çiftleri orada
birleştirilip puanlandı (**40 satır, hepsi puanlı**). O hali kendi repomuza
aynen aktarıldı:

**https://huggingface.co/datasets/yoitsmeyusuf/turkce-sts-birlesik**

```bash
../.venv/bin/python hafta9_benzerlik_veri_seti/birlestir_ve_push.py
```

Script kaynağı `force_redownload` ile çekiyor (cache eski sürümü vermesin),
sütunları ve puan aralığını doğruluyor, sonucu `veri/birlesik.csv` olarak da
yazıyor. Puan üretmiyor/değiştirmiyor — `score` kaynakta neyse o; boş olan
satır varsa `NaN` olarak geçer. Kaynak güncellenirse tekrar çalıştırmak yeterli.

## Klasör

```
hafta9_benzerlik_veri_seti/
├── README.md
├── requirements.txt
├── check_env.py               # kurulum + CSV kontrolü (Hub'a yazmaz)
├── push_dataset.py            # kendi CSV'miz -> HF Dataset + dataset kartı
├── puanla_mer1alii.py         # Mer1Alii veri setini puanlayıp ayrı repoya push
├── birlestir_ve_push.py       # iki repoyu birleştirip üçüncü repoya push
└── veri/
    ├── ciftler.csv            # kendi çiftlerimiz (score = nan)
    ├── mer1alii_puanli.csv    # puanlanmış Mer1Alii çiftleri
    └── birlesik.csv           # birleşik hali
```

## Sonraki adım

`veri/ciftler.csv` içindeki `nan` değerlerini 0–5 arası puanlarla doldurup
`push_dataset.py`'yi tekrar çalıştırmak yeterli; script aralık kontrolü yapıyor
ve `nan`'a da izin veriyor (kısmi puanlama mümkün).
