# Türkçe Felsefe LLM — Uçtan Uca Çalışma Serisi

Türkçe felsefe alanında sıfırdan bir veri seti toplayıp, kendi tokenizer'ımı
eğitip, 4B'lik bir taban modeli bu veriyle ince ayarlayıp (LoRA), sonucu iki
ayrı benchmark ile ölçtüğüm ve son olarak bir modeli önce dış API'lerle, sonra
gerçek bir veritabanıyla konuşturup (tool calling) canlıya aldığım 8 haftalık bir
çalışma serisi. Her hafta bir önceki haftanın çıktısı üzerine kuruluyor.

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
| 7 | [`hafta7_tool_calling/`](hafta7_tool_calling/) | Tool calling (function calling) + Open-Meteo API + Gradio, HF Spaces'te (ZeroGPU) canlı | [4 araçlı şeffaf hava durumu ajanı](https://huggingface.co/spaces/yoitsmeyusuf/tool-calling-hava-durumu) |
| 8 | [`hafta8_veritabani_ajani/`](hafta8_veritabani_ajani/) | Veritabanına **okuyup yazan** tool calling ajanı (SQLite) + halüsinasyon guardrail'i | [4 araçlı felsefe kitapçısı sipariş asistanı](https://huggingface.co/spaces/yoitsmeyusuf/kitapci-siparis-ajani) |

`common/` iki haftanın paylaştığı kodu tutuyor: `hf_dataset_schema.py` (1 ve 4)
ve `lora_trainer.py` (3 ve 4). 7. ve 8. hafta serinin diğer haftalarından
bağımsız çalışır (eğitim gerektirmez; modeli ya Space'in ZeroGPU'sunda ya da
uzaktan Inference Providers üzerinden çalıştırır). 8. hafta model katmanını
(`modeller.py`) 7. haftadan olduğu gibi kullanıyor.

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

**Tool calling (7. hafta, ödevin örnek sorusu — iki turlu zincir gerekiyor):**

| Yaklaşım | Sonuç |
|---|---|
| `Qwen2.5-7B-Instruct` + sistem promptu kuralı | ❌ çevirim aracını atlayıp kendi hesapladı |
| + araç çıktısına not, + promptta örnek, + `Qwen2.5-14B-Instruct` | ❌ üçü de değiştirmedi |
| + döngüde guardrail: ihlali yakala, araç çağrısını zorunlu tut | ✅ zincir kuruluyor |
| `Qwen3-Coder-30B-A3B-Instruct` (Inference Providers) | ✅ kendiliğinden doğru zincir |

Küçük modellerde "aracı kullan" talimatı tek başına yetmiyor; kuralı prompt'tan
harness'a taşımak gerekti. Detay 7. haftanın README'sinde.

**Halüsinasyon engelleme (8. hafta — modelin veritabanına yazdığı senaryo):**

| Model | Ne yaptı | Guardrail |
|---|---|---|
| `Qwen2.5-7B-Instruct` | Stoğu tükenmiş kitap yerine **katalogda hiç olmayan bir kitap** önerdi | ✅ yakalandı (tırnak içindeki kitap adı kontrolü) |
| `Qwen2.5-7B-Instruct` | `create_order`'ı atlayıp **sipariş kodunu kendisi uydurdu** | ✅ yakalandı (kod kontrolü) |
| `Qwen2.5-0.5B-Instruct` | Hiç sipariş vermeden "2 adet sipariş verildi" dedi | ✅ yakalandı; düzeltmede ısrar etti, cevap "Doğrulanmadı" diye işaretlendi |
| `Qwen3-Coder-30B-A3B-Instruct` | Zinciri kendiliğinden doğru kurdu (ara → sipariş ver) | — ihlal yok |

Prompt ve araç çıktısındaki uyarılar tek başına yetmiyor; yanıt yayınlanmadan
önce fiyatların, sipariş kodlarının ve kitap adlarının araç çıktısından geldiğini
harness doğruluyor. Detay 8. haftanın README'sinde.

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

Sıra önemli — 1-6. haftalar bir öncekinin çıktısını (HF'ye push edilmiş dataset /
adaptör) kullanıyor. 7. hafta bağımsızdır:

```bash
uv pip install --python .venv/bin/python -r hafta7_tool_calling/requirements.txt
.venv/bin/python hafta7_tool_calling/app.py                        # yerel Gradio arayüzü
.venv/bin/python hafta7_tool_calling/deploy_space.py               # ZeroGPU Space'e yayınla
```

8. hafta da bağımsız; ek olarak SQLite veritabanını ilk çalıştırmada kendisi
kuruyor (elle bir adım gerekmiyor):

```bash
uv pip install --python .venv/bin/python -r hafta8_veritabani_ajani/requirements.txt
.venv/bin/python hafta8_veritabani_ajani/veritabani.py --sifirla    # DB'yi kur + içeriğini dök
.venv/bin/python hafta8_veritabani_ajani/araclar.py                 # 4 aracı + hata yollarını dene
.venv/bin/python hafta8_veritabani_ajani/ajan.py --guardrail        # halüsinasyon guardrail testi
.venv/bin/python hafta8_veritabani_ajani/app.py                     # yerel Gradio arayüzü
.venv/bin/python hafta8_veritabani_ajani/deploy_space.py            # ZeroGPU Space'e yayınla
```

Yerelde 7B'yi çalıştırmak GPU istiyor; GPU'suz makinede küçük bir modelle
(`TOOL_MODEL=Qwen/Qwen2.5-0.5B-Instruct`) ya da `TOOL_BACKEND=api` ile
denenebilir. Detay 7. ve 8. haftanın README'lerinde.

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
- **Veritabanı kalıcılığı (8. hafta):** Space'te kalıcı disk yoksa SQLite dosyası
  geçicidir — Space yeniden başlatıldığında siparişler sıfırlanır, katalog tohum
  veriden yeniden kurulur. `veritabani.py` kalıcı disk (`/data`) varsa oraya
  yazıyor; yol `KITAPCI_DB` ile de verilebilir.
- **Space barındırma (7. hafta):** Hugging Face ücretsiz `cpu-basic` donanımda
  Gradio Space barındırmayı kaldırdı (yalnızca static Space'ler ücretsiz), ama
  **ZeroGPU** (`zero-a10g`) PRO olmayan hesapta da açılıyor — üstüne ücretsiz GPU
  veriyor. Bu yüzden 7. haftanın Space'i ZeroGPU'da koşuyor ve model
  (`Qwen2.5-7B-Instruct`) `transformers` ile Space'in **içinde** çalışıyor;
  Inference Providers kredisine ihtiyaç duymuyor. Ajan katmanı arka uçtan bağımsız:
  `TOOL_BACKEND=api` ile aynı kod Inference Providers üzerinden de çalışıyor.
- **Veri kalitesi:** Ham scraping çıktısı doğrudan kullanılmadı; konu uygunluğu,
  uzunluk ve güvenlik (küfür/PII/nefret söylemi) taramalarından geçirildi.
  Elenen satırların id'leri kalıcı olarak tutuluyor, böylece scraper tekrar
  çalıştırıldığında elenenler geri gelmiyor. Süreç 1. haftanın README'sinde.
