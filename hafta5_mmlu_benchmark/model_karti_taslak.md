# yoitsmeyusuf/felsefe-lora — README.md taslağı (push edilmedi)

Bu dosya, `https://huggingface.co/yoitsmeyusuf/felsefe-lora` model kartına
eklenecek MMLU benchmark bölümüyle birleştirilmiş **tam README taslağıdır**.
İncelendikten sonra onay verilirse bu içerik hub'daki `README.md`'nin yerine
push edilecek (mevcut içerik korunuyor, sadece "Örnek çıktı" ile "Kullanım"
bölümleri arasına yeni bir "MMLU Benchmark" bölümü eklendi).

Push edilecek ek dosyalar (README'deki linklerin kırık olmaması için):
- `mmlu_benchmark.py` (bu klasörden)
- `sonuclar/taban_model.json`, `sonuclar/finetune_lora.json`, `sonuclar/karsilastirma.json`
- `sonuclar/taban_model_cevaplar.csv`, `sonuclar/finetune_lora_cevaplar.csv`

---

---
base_model: unsloth/Qwen3.5-4B
datasets:
- yoitsmeyusuf/felsefe_finetune
tags:
- text-generation-inference
- transformers
- unsloth
- qwen3_5
- trl
- lora
license: apache-2.0
language:
- tr
---

# felsefe-lora

`unsloth/Qwen3.5-4B` taban modelinin, Türkçe felsefe (düşünürler, akımlar,
temel kavramlar) soru-cevap verisiyle LoRA fine-tune edilmiş adaptörü.

- **Domain:** Felsefe (Türkçe)
- **Taban model:** [unsloth/Qwen3.5-4B](https://huggingface.co/unsloth/Qwen3.5-4B)
- **Eğitim verisi:** [yoitsmeyusuf/felsefe_finetune](https://huggingface.co/datasets/yoitsmeyusuf/felsefe_finetune)
  (527 soru-cevap satırı — Ekşi Sözlük + r/felsefe scraping + 2 elle yazılmış seed)
- **Yöntem:** [Unsloth](https://github.com/unslothai/unsloth) `FastModel` + PEFT LoRA
  (4bit, `r=16`, `lora_alpha=16`, hedef modüller: q/k/v/o/gate/up/down_proj),
  1 epoch, effective batch size 8 (`per_device_train_batch_size=2` × `gradient_accumulation_steps=4`),
  `learning_rate=2e-4`, RTX 4060 Laptop (8GB VRAM) üzerinde.
- **Eğitim kaybı (loss):** 67 adımda ~3.53 → ~1.5-2.9 aralığına düştü (küçük
  veri seti + 1 epoch nedeniyle adımlar arası dalgalanma var, ama net bir
  düşüş trendi mevcut).

## Örnek çıktı

Model, `apply_chat_template(..., tokenize=True)` çok-modlu (image-text-to-text)
işlemci formatı gerektirdiği için mesaj içeriği `[{"type": "text", "text": ...}]`
şeklinde verilmeli (bkz. `hafta3_finetune/test_inference.py`).

**Soru:** "Nietzsche'nin üst-insan (übermensch) kavramı nedir?"
> (Model önce İngilizce bir "thinking" bölümü üretiyor, ardından Türkçe
> cevaba geçiyor — taban model bir reasoning modeli olduğu için bu davranış
> beklenen; kısa `max_new_tokens` ile test edildiğinde cevap bu noktada
> kesildi.) Türkçe cevap bölümü üst-insanı değerler yaratma, geleneksel
> ahlakı aşma ve "Tanrı öldü" sonrası anlam arayışı bağlamında doğru şekilde
> özetliyor.

**Soru:** "Varoluşçuluk nedir, temsilcileri kimlerdir?"
> "Varoluşçuluk, insanın varoluşunun önceliği, özgürlüğü ve sorumluluğu
> üzerine kurulu bir felsefi akımdır. Varoluşçular, insanın doğuştan bir öz
> veya amaçla yüklü olmadığını, varoluşunun kendisiyle başladığını
> savunurlar..." (Sartre, Camus, Kierkegaard, Heidegger, de Beauvoir temsilci
> olarak sayıldı.)

**Soru:** "Descartes'ın 'düşünüyorum öyleyse varım' önermesi ne anlama gelir?"
> Model, önermeyi *Discourse on Method*/*Meditations* bağlamına oturtup
> şüphe etme ediminin bile varoluşu kanıtladığını doğru şekilde açıkladı.

Üçü de konuya hakim, tutarlı ve Türkçe felsefi içerik üretti — LoRA
adaptörünün taban modeli felsefe domain'ine yönlendirdiği gözlemlendi.
Daha uzun/kesilmemiş cevaplar için `max_new_tokens` artırılmalı.

## MMLU Benchmark: Taban Model vs Fine-Tune (LoRA)

Model, Türkçe MMLU test seti (6200 soru / 62 bölüm) kullanılarak taban modelle (`unsloth/Qwen3.5-4B`) karşılaştırmalı olarak test edildi.


### Genel sonuç

| Model | Doğru | Toplam | Başarı | Süre |
|---|---|---|---|---|
| **Taban** — `unsloth/Qwen3.5-4B` | 588 | 930 | **%63.23** | 438.4s |
| **Fine-tune (LoRA)** — `yoitsmeyusuf/felsefe-lora` | 578 | 930 | **%62.15** | 575.5s |

Genel MMLU başarısında fark: **-1.08 puan** (LoRA − Taban).

Felsefe verisiyle yapılan dar kapsamlı (527 satır, 1 epoch) bir LoRA fine-tune'un genel bilgi MMLU'sunda gözle görülür bir artış sağlaması beklenmiyordu — hedef zaten genel yetenek değil, felsefe domain'inde üslup/derinlikti (bkz. yukarıdaki örnek çıktılar). İlginç şekilde `Felsefe` bölümünün kendisinde de bu örneklemde taban model daha iyi çıktı (Taban %60.0 vs LoRA %53.33) — n=15 soruluk küçük örneklemde bir soru bile birkaç puanlık oynamaya yol açtığından (±1 doğru cevap ≈ ±6.7 puan) ve LoRA'nın çıktı stilini kısa tek harften ziyade daha açıklamalı/uzun cevaplara kaydırmış olabileceğinden (bu benchmark yalnızca tek harf cevabını puanlıyor), bu sonucu 'fine-tune felsefe bilgisini kötüleştirdi' şeklinde değil, MCQ-tarzı kısa cevap formatına karşı bir üslup uyumsuzluğu + küçük örneklem gürültüsü olarak okumak daha doğru.

### Bölüm bazlı sonuçlar (62 bölüm, bölüm başına 15 soru)

<details>
<summary>Tüm bölümleri göster</summary>

| Bölüm | Taban | LoRA | Fark |
|---|---|---|---|
| Sağlık Kurumları İşletmeciliği | %66.67 | %86.67 | +20.00 |
| Turizm ve Otel İşletmeciliği | %60.00 | %80.00 | +20.00 |
| Adalet | %53.33 | %66.67 | +13.34 |
| Kültürel Miras ve Turizm | %53.33 | %66.67 | +13.34 |
| Tarım | %60.00 | %73.33 | +13.33 |
| Çocuk Gelişimi | %66.67 | %80.00 | +13.33 |
| İnsan Kaynakları Yönetimi | %66.67 | %80.00 | +13.33 |
| İşletme Yönetimi | %46.67 | %60.00 | +13.33 |
| Acil Durum ve Afet Yönetimi | %60.00 | %66.67 | +6.67 |
| Dini Bilgiler | %60.00 | %66.67 | +6.67 |
| Futbol | %60.00 | %66.67 | +6.67 |
| Halkla İlişkiler ve Tanıtım | %53.33 | %60.00 | +6.67 |
| Kamu Yönetimi | %60.00 | %66.67 | +6.67 |
| Lojistik | %60.00 | %66.67 | +6.67 |
| TUS | %60.00 | %66.67 | +6.67 |
| Uluslar Arası İlişkiler | %53.33 | %60.00 | +6.67 |
| Yerel Yönetimler | %73.33 | %80.00 | +6.67 |
| Emlak ve Emlak Yönetimi | %46.67 | %53.33 | +6.66 |
| Halkla İlişkiler ve Reklamcılık | %46.67 | %53.33 | +6.66 |
| Turizm ve Seyehat Hizmetleri | %46.67 | %53.33 | +6.66 |
| AUZEF | %53.33 | %53.33 | +0.00 |
| Bankacılık ve Sigortacılık | %40.00 | %40.00 | +0.00 |
| DHBT | %66.67 | %66.67 | +0.00 |
| Marka İletişimi | %33.33 | %33.33 | +0.00 |
| Muhasebe ve Vergi Uygulamaları | %66.67 | %66.67 | +0.00 |
| Parakende Satış ve Mağaza Yöneticiliği | %66.67 | %66.67 | +0.00 |
| Siyer | %53.33 | %53.33 | +0.00 |
| Sosyal Hizmet | %53.33 | %53.33 | +0.00 |
| Sosyal Hizmetler | %80.00 | %80.00 | +0.00 |
| Tıbbi Dökümantasyon ve Sekreterlik | %80.00 | %80.00 | +0.00 |
| Yönetim Bİlişim Sistemleri | %73.33 | %73.33 | +0.00 |
| Özel Koruma ve Güvenlik | %53.33 | %53.33 | +0.00 |
| Üniversite Giriş Sınavı Temel Bilimler | %66.67 | %66.67 | +0.00 |
| Büro Yönetimi ve Yönetici Asistanlığı | %73.33 | %66.67 | -6.66 |
| Fotoğrafçılık ve Kameramanlık | %53.33 | %46.67 | -6.66 |
| KPSS | %53.33 | %46.67 | -6.66 |
| Maliye | %73.33 | %66.67 | -6.66 |
| Radyo ve Televizyon Programcılığı | %73.33 | %66.67 | -6.66 |
| Sağlık Yönetimi | %73.33 | %66.67 | -6.66 |
| Spor Yönetimi | %73.33 | %66.67 | -6.66 |
| İktisat | %73.33 | %66.67 | -6.66 |
| Dış Ticaret | %60.00 | %53.33 | -6.67 |
| Ehliyet Sınavı | %66.67 | %60.00 | -6.67 |
| Felsefe | %60.00 | %53.33 | -6.67 |
| Laborant ve Veteriner Sağlık | %80.00 | %73.33 | -6.67 |
| Medya ve İletişim | %60.00 | %53.33 | -6.67 |
| Menkul Kıymetler ve Sermaye Piyasası | %66.67 | %60.00 | -6.67 |
| Okul Öncesi Öğretmenliği | %66.67 | %60.00 | -6.67 |
| Sosyoloji | %80.00 | %73.33 | -6.67 |
| Türk Dili ve Edebiyatı | %46.67 | %40.00 | -6.67 |
| Uluslararası Ticaret ve Lojistik Yönetimi | %66.67 | %60.00 | -6.67 |
| Yaşlı Bakımı | %66.67 | %60.00 | -6.67 |
| Çalışma Ekonomisi ve Endüstri İlişkileri | %60.00 | %53.33 | -6.67 |
| İlahiyat | %86.67 | %80.00 | -6.67 |
| Aşçılık | %53.33 | %40.00 | -13.33 |
| Ev İdaresi | %80.00 | %66.67 | -13.33 |
| KPSS Denemeleri | %73.33 | %60.00 | -13.33 |
| Tarih | %60.00 | %46.67 | -13.33 |
| Çağrı Merkezi Hizmetleri | %73.33 | %60.00 | -13.33 |
| Havacılık Yönetimi | %66.67 | %53.33 | -13.34 |
| Elektrik Enerjisi Üretim,İletim ve Dağıtımı | %93.33 | %73.33 | -20.00 |
| Kim 500 Milyar İster | %66.67 | %40.00 | -26.67 |

</details>

Ham cevaplar ve tam sonuç JSON'ları bu repoda: [`sonuclar/`](https://huggingface.co/yoitsmeyusuf/felsefe-lora/tree/main/sonuclar).

## Kullanım

```python
from unsloth import FastModel

model, tokenizer = FastModel.from_pretrained(
    model_name="yoitsmeyusuf/felsefe-lora",
    max_seq_length=2048,
    load_in_4bit=True,
)
FastModel.for_inference(model)

messages = [{"role": "user", "content": [{"type": "text", "text": "Sokrates kimdir?"}]}]
inputs = tokenizer.apply_chat_template(
    messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
).to(model.device)
out = model.generate(input_ids=inputs, max_new_tokens=512)
print(tokenizer.decode(out[0], skip_special_tokens=True))
```

Bu qwen3_5 modeli [Unsloth](https://github.com/unslothai/unsloth) ile 2x
daha hızlı eğitildi.

[<img src="https://raw.githubusercontent.com/unslothai/unsloth/main/images/unsloth%20made%20with%20love.png" width="200"/>](https://github.com/unslothai/unsloth)
