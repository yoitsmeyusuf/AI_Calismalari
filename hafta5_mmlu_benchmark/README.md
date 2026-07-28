# 5. Hafta — MMLU Benchmark Karşılaştırması

**Amaç:** 3. haftada fine-tune edilen `yoitsmeyusuf/felsefe-lora`'yı taban model
`unsloth/Qwen3.5-4B` ile Türkçe MMLU benchmark'ında karşılaştırıp sonucu HF
model kartına eklemek. Soru bankası olarak hazır Türkçe MMLU veri seti
(6200 soru / 62 bölüm) kullanıldı.

## Değerlendirme yöntemi

Model HF'de LoRA adaptörü olarak duruyor (GGUF değil), bu yüzden Ollama gibi
bir ara katman kullanılmadı: `mmlu_benchmark.py` modeli doğrudan
`unsloth.FastModel` + `transformers` ile yükleyip çalıştırıyor. Doğruluk
kontrolü iki aşamalı — önce harf eşleşmesi, cevap belirsizse
`sentence-transformers/paraphrase-multilingual-mpnet-base-v2` ile anlamsal
benzerlik.

## Çalıştırma

```bash
.venv/bin/python hafta5_mmlu_benchmark/mmlu_benchmark.py
```

- `.env`'de `LORA_REPO_ID` ve `BASE_MODEL` okunur.
- Varsayılan olarak 62 bölümün her birinden rastgele (seed=42) **15 soru**
  örneklenir (`SAMPLE_PER_BOLUM` ile değiştirilebilir; `0` verilirse tüm 6200
  soru çalışır — hız testine göre ~0.4s/soru, tam koşum 2 model için ~85 dk
  sürer, bölüm başına 15 soru ile ~15 dk).
- Enable_thinking=False + greedy decoding (`do_sample=False`) kullanılır —
  taban model bir reasoning modeli olduğundan thinking kapatılmazsa cevap
  üretimi çok yavaşlar ve MCQ formatına uymayan uzun çıktılar üretebilir.
- Sonuçlar `sonuclar/` altına JSON (genel + bölüm bazlı başarı) ve CSV (ham
  cevaplar) olarak kaydedilir; herhangi bir dış liderlik tablosuna push
  edilmez.

## Sonuç (bölüm başına 15 soru, 930 soru/model)

| Model | Başarı |
|---|---|
| Taban — `unsloth/Qwen3.5-4B` | %63.23 |
| Fine-tune (LoRA) — `yoitsmeyusuf/felsefe-lora` | %62.15 |

Tam bölüm bazlı tablo ve yorum: [`model_karti_taslak.md`](model_karti_taslak.md)
(HF model kartına eklenecek taslak, henüz push edilmedi).
