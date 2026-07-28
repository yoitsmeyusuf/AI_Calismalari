# 3. Hafta — Model Fine-Tune Etme (unsloth)

**Sonuç:** `unsloth/Qwen3.5-4B` + felsefe verisi (`yoitsmeyusuf/felsefe_finetune`,
527 satır), 1 epoch, LoRA r=16, loss ~3.53 → ~1.5-2.9 →
**https://huggingface.co/yoitsmeyusuf/felsefe-lora**

1. Önce 1. hafta tamamlanmış olmalı (DATASET_REPO_ID push edilmiş).
2. `.env` içinde `LORA_REPO_ID` ve gerekirse `BASE_MODEL`'i ayarlayın.
   Varsayılan `unsloth/Qwen3.5-4B` — hybrid (gated-deltanet + full attention) dense
   model, post-trained/chat-ready, 8GB VRAM'lik RTX 4060 Laptop GPU'nuza 4bit + LoRA
   ile rahat sığar. `common/lora_trainer.py` bu modeli `unsloth.FastModel` ile yükler
   (`FastLanguageModel` değil — Qwen3.5 image-text-to-text olarak paketli, biz sadece
   metinle eğitiyoruz). Değiştirmek isterseniz ör. `unsloth/Qwen2.5-3B-Instruct-bnb-4bit`
   gibi klasik bir metin-only modele dönüp `FastModel`'i `FastLanguageModel` ile
   değiştirmeniz gerekir.
3. Çalıştırın:
   ```bash
   .venv/bin/python hafta3_finetune/train_lora_unsloth.py
   ```
4. Eğitim bitince LoRA adaptörü otomatik olarak `LORA_REPO_ID`'ye push edilir
   (tam model değil, sadece adaptör).

Eğitim mantığı `../common/lora_trainer.py` içinde; 4. hafta (identity) de aynı
fonksiyonu kullanır, sadece dataset/repo farklıdır.

## Test etme

```bash
.venv/bin/python hafta3_finetune/test_inference.py
```

Adaptörü yükleyip birkaç felsefe sorusuna cevap üretir (`QUESTIONS` listesi
script içinde). Taban model çok-modlu (image-text-to-text) işlemci olarak
paketlendiği için mesaj `content`'i `apply_chat_template(..., tokenize=True)`
çağrısında düz string değil `[{"type": "text", "text": ...}]` formatında
verilmeli — `tokenize=False` (eğitimde kullanılan yol) bu kontrolü atladığı
için orada sorun çıkmıyor, sadece inference'ta (`tokenize=True`) gerekiyor.

Sonuçlar model kartında (yukarıdaki link): üç test sorusuna da (Nietzsche/
üst-insan, varoluşçuluk, Descartes/cogito) tutarlı, konuya hakim Türkçe
cevaplar üretti — LoRA'nın taban modeli felsefe domain'ine yönlendirdiği
gözlemlendi. Model bir reasoning modeli olduğu için önce İngilizce bir
"thinking" bölümü üretiyor; kısa `max_new_tokens` ile test edilirse Türkçe
cevap kesilebilir, gerekirse artırın.
