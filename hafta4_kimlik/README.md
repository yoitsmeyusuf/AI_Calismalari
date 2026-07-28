# 4. Hafta — Yapay Zeka Kimlik Eğitimi (Identity Fine-Tuning)

1/2/3. haftalardan bağımsız, ayrı bir çalışma: modele bir domain bilgisi değil,
**kendi kimliğini** (isim, geliştirici, köken, yetenekler, sınırlamalar, ton)
öğretmeyi deniyoruz.

## Yaklaşım

`identity_seeds.py` içinde elle yazılmış Türkçe/İngilizce soru-cevap
tohumları var ("Sen kimsin?", "Seni kim eğitti?", "Nasıl bir modelsin?" vb.).
Kimlik bilgileri tek yerden, dosyanın başındaki sabitlerden geliyor:

```python
AI_NAME = "FelsefeAI"
CREATOR_NAME = "Yusuf"
BASE_MODEL_FAMILY = "qwen"
```

Böylece isim/geliştirici değiştiğinde tüm cevaplar tutarlı şekilde güncelleniyor.
Bu kadar küçük bir veri setinde tutarlılık kritik — çelişen cevaplar modelin
kimliği ezberlemesini engelliyor.

## Çalıştırma

```bash
.venv/bin/python hafta4_kimlik/build_identity_dataset.py   # dataset'i kurup HF'ye push eder
.venv/bin/python hafta4_kimlik/train_identity_lora.py      # LoRA adaptörünü eğitip push eder
```

`.env` içinde `IDENTITY_DATASET_REPO_ID` ve `IDENTITY_LORA_REPO_ID` ayarlı olmalı.

Veri şeması `../common/hf_dataset_schema.py`'de (1. hafta ile ortak), eğitim
mantığı `../common/lora_trainer.py`'de (3. hafta ile ortak) — ikisi de sadece
farklı dataset/repo değerleriyle çağrılıyor.
