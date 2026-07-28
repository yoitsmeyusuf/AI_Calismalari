# Türkçe Felsefe LLM — Uçtan Uca Çalışma Serisi

Türkçe felsefe alanında sıfırdan bir veri seti toplayıp, kendi tokenizer'ımı
eğitip, 4B'lik bir taban modeli bu veriyle ince ayarlayıp (LoRA), sonucu iki
ayrı benchmark ile ölçtüğüm 6 haftalık bir çalışma serisi. Her hafta bir önceki
haftanın çıktısı üzerine kuruluyor.

**Alan:** Felsefe (düşünürler, akımlar, temel kavramlar) — Türkçe, tek dilli
**Taban model:** `unsloth/Qwen3.5-4B` (4bit + LoRA)
**Donanım:** RTX 4060 Laptop, 8 GB VRAM

## Haftalar

| # | Klasör | Konu | Çıktı |
|---|---|---|---|
| 1 | [`hafta1_veri_seti/`](hafta1_veri_seti/) | Web scraping ile veri seti hazırlama (Ekşi Sözlük + r/felsefe) | 527 satırlık Türkçe soru-cevap veri seti |
| 2 | [`hafta2_tokenizer/`](hafta2_tokenizer/) | Sıfırdan byte-level BPE tokenizer eğitimi | `vocab_size=8000` tokenizer |
| 3 | [`hafta3_finetune/`](hafta3_finetune/) | unsloth ile LoRA fine-tune | LoRA adaptörü (`r=16`, 1 epoch) |
| 4 | [`hafta4_kimlik/`](hafta4_kimlik/) | Kimlik (identity) fine-tune — bağımsız deneme | Identity dataset + LoRA |
| 5 | [`hafta5_mmlu_benchmark/`](hafta5_mmlu_benchmark/) | Türkçe MMLU ile taban model vs. fine-tune karşılaştırması | Bölüm bazlı sonuç tabloları |
| 6 | [`hafta6_felsefe_benchmark/`](hafta6_felsefe_benchmark/) | Sıfırdan yazılmış 100 soruluk felsefe benchmark'ı, 5 model | Benchmark veri seti + karşılaştırma |

`common/` iki haftanın paylaştığı kodu tutuyor: `hf_dataset_schema.py` (1 ve 4)
ve `lora_trainer.py` (3 ve 4).

Her klasörün kendi `README.md`'si var — neyi neden o şekilde yaptığımı,
karşılaştığım sorunları ve sonuçları orada anlattım.

## Öne çıkan sonuçlar

**Türkçe MMLU (bölüm başına 15 soru, 930 soru/model):**

| Model | Başarı |
|---|---|
| Taban — `unsloth/Qwen3.5-4B` | %63.23 |
| Fine-tune (LoRA) | %62.15 |

Fine-tune, genel amaçlı MMLU'da taban modelin bir tık altında kaldı — 527
satırlık dar bir domain verisiyle 1 epoch eğitim modelin genel yeteneklerini
iyileştirmiyor. Beklenen bir sonuç; asıl kazanç domain içi ton ve terminolojide.

**Kendi felsefe benchmark'ım (100 soru, 5 model):**

| Model | Başarı |
|---|---|
| `unsloth/Qwen3.5-4B` (taban) | %100.0 |
| Fine-tune (LoRA) | %100.0 |
| `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | %99.0 |
| `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | %91.0 |
| `unsloth/gemma-2-2b-it-bnb-4bit` | %87.0 |

Sorular net ve tek doğru cevaplı ders kitabı tarzı olduğu için güçlü modellerde
bir tavan etkisi (ceiling effect) oluştu; benchmark asıl küçük modelleri ayırt
etmekte işe yarıyor. Bunu 6. haftanın README'sinde ayrıca tartıştım.

## Kurulum

```bash
python3 -m venv .venv
uv pip install --python .venv/bin/python -r requirements.txt
.venv/bin/python -m playwright install chromium   # sadece 1. haftanın Reddit scraper'ı için
```

Hugging Face girişi (token'ı dosyaya yazmadan, cache'e kaydeder):

```bash
.venv/bin/hf auth login
```

Alternatif olarak `.env.example`'ı `.env` olarak kopyalayıp `HF_TOKEN`'ı
doldurabilirsiniz (push_to_hub için write yetkili token gerekir). `.env`
içindeki `*_REPO_ID` alanlarını da kendi HF kullanıcı adınızla güncelleyin —
`.env` git'e girmez, `.env.example` şablondur.

Her şeyin yerinde olduğunu doğrulamak için:

```bash
.venv/bin/python check_env.py
```

## Çalıştırma

Bütün script'ler ilgili klasörden değil, **proje kökünden** çalıştırılacak
şekilde yazıldı:

```bash
.venv/bin/python hafta1_veri_seti/build_dataset.py
```

Sıra önemli — her hafta bir öncekinin çıktısını (HF'ye push edilmiş dataset /
adaptör) kullanıyor.

## Notlar

- **Reddit scraping:** Reddit artık login'siz `.json` ve `old.reddit.com`
  isteklerini 403'lüyor, yeni arayüz ise JS ile çözülen bir bot-doğrulama
  duvarına sahip. Bu yüzden `scrape_reddit.py` Playwright (headless Chromium)
  kullanıyor, gerçek bir tarayıcı gibi JS'i çalıştırıyor. Önce genel r/Turkey
  denendi ama ağırlıklı siyasi gündem çıktı, r/AskTurkey ise ağırlıklı
  İngilizceydi; r/felsefe'ye geçildi. Detaylar ve etik notlar 1. haftanın
  README'sinde.
- **Model seçimi:** `unsloth/Qwen3.5-4B` image-text-to-text olarak paketli
  olduğundan `common/lora_trainer.py` `FastLanguageModel` yerine
  `unsloth.FastModel` kullanıyor — biz sadece metinle eğitiyoruz. Bu ayrıntı
  inference tarafında da mesaj formatını değiştiriyor (3. haftanın README'si).
- **Veri kalitesi:** Ham scraping çıktısı doğrudan kullanılmadı; konu uygunluğu,
  uzunluk ve güvenlik (küfür/PII/nefret söylemi) taramalarından geçirildi.
  Elenen satırların id'leri kalıcı olarak tutuluyor, böylece scraper tekrar
  çalıştırıldığında elenenler geri gelmiyor. Süreç 1. haftanın README'sinde.
