# 6. Hafta — Felsefeye Özel Benchmark (100 Soru, 5 Model)

**Amaç:** 5. haftadaki genel Türkçe MMLU testinin aksine, sıfırdan yazılmış
ve yalnızca felsefe konularına odaklanan 100 soruluk çoktan seçmeli (A-E) bir
benchmark hazırlayıp, 3. haftada fine-tune edilen
[`yoitsmeyusuf/felsefe-lora`](https://huggingface.co/yoitsmeyusuf/felsefe-lora)'yı
hem taban modeliyle (`unsloth/Qwen3.5-4B`) hem de 3 farklı model
ailesinden referans modelle (Qwen2.5, Gemma-2, Llama-3.2) karşılaştırmak.

## Dosyalar

| Dosya | Ne işe yarar |
|---|---|
| `felsefe_sorulari.py` | 100 soru (13 kategori), her biri 5 seçenekli (A-E) |
| `felsefe_benchmark.py` | 5 modeli sırayla yükleyip tüm soruları çalıştırır, sonuçları kaydeder |
| `push_dataset.py` | Soru setini + sonuçları HF Hub'a **Dataset** olarak pushlar (kart dahil) |
| `update_model_card.py` | `yoitsmeyusuf/felsefe-lora` model kartına yeni bir "Felsefe Benchmark (Özel, 100 Soru)" bölümü ekler |
| `sonuclar/` | Her modelin JSON özeti + ham cevap CSV'si + `karsilastirma.json` |

## Kategoriler (13, 100 soru)

İlk 90 soru 10 kategoride dengeli (8-10'ar soru) elle yazıldı: Antik Yunan
Felsefesi, Ortaçağ ve İslam Felsefesi, Modern Felsefe (Rönesans–Aydınlanma),
Epistemoloji, Metafizik ve Ontoloji, Etik, Siyaset Felsefesi, Estetik,
Mantık, Zihin Felsefesi ve Çağdaş Felsefe. Sonradan 10 soru daha eklendi
(19. Yüzyıl Felsefesi, 20. Yüzyıl Felsefesi, Çağdaş Felsefe gibi yeni/ek
kategorilerle) — toplam 13 kategori, 100 soru.

Doğru cevabın konumu ilk 90 soruda pozisyon önyargısını (position bias)
önlemek için A-E arasında dengeli dağıtıldı (`idx % 5`, 18'er soru/harf);
sonradan eklenen 10 soru dağılımı hafifçe kaydırdı ama istismara açık
değil (`felsefe_sorulari.py` çalıştırıldığında harf dağılımı basılır).

## Test edilen 5 model

1. **Taban model** — `unsloth/Qwen3.5-4B` (`.env`'deki `BASE_MODEL`)
2. **Fine-tune (LoRA)** — `yoitsmeyusuf/felsefe-lora` (`.env`'deki `LORA_REPO_ID`, 3. haftanın çıktısı)
3. `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
4. `unsloth/gemma-2-2b-it-bnb-4bit`
5. `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`

İlk iki model Qwen3.5'in çok-modlu (image-text-to-text) işlemci formatı
gerektirdiği için mesaj içeriği `[{"type": "text", "text": ...}]` olarak
gönderiliyor (bkz. `hafta3_finetune/test_inference.py`); diğer 3 model
düz metin (`content` string) formatını kullanıyor. Değerlendirme mantığı
(harf eşleşmesi + belirsiz durumda `sentence-transformers` ile anlamsal
benzerlik) `hafta5_mmlu_benchmark/mmlu_benchmark.py` ile birebir aynı.

## Çalıştırma sırası

```bash
.venv/bin/python hafta6_felsefe_benchmark/felsefe_benchmark.py   # 5 model x 100 soru, sonuclar/ doldurur
.venv/bin/python hafta6_felsefe_benchmark/push_dataset.py         # soru seti + sonuçları HF Dataset olarak pushlar
.venv/bin/python hafta6_felsefe_benchmark/update_model_card.py    # felsefe-lora model kartına sonuç bölümü ekler
```

`.env`'de `FELSEFE_BENCHMARK_REPO_ID` (yeni dataset reposu) ve mevcut
`BASE_MODEL` / `LORA_REPO_ID` / `HF_TOKEN` okunur.

## Sonuç (100 soru)

| Model | Doğru | Başarı |
|---|---|---|
| `unsloth/Qwen3.5-4B` (taban) | 100/100 | **%100.0** |
| `yoitsmeyusuf/felsefe-lora` (fine-tune) | 100/100 | **%100.0** |
| `unsloth/Qwen2.5-7B-Instruct-bnb-4bit` | 99/100 | %99.0 |
| `unsloth/Llama-3.2-3B-Instruct-bnb-4bit` | 91/100 | %91.0 |
| `unsloth/gemma-2-2b-it-bnb-4bit` | 87/100 | %87.0 |

Taban model ve fine-tune (LoRA) her ikisi de %100 aldı — sorular net, tek
doğru cevaplı ders kitabı tarzı MCQ olduğundan güçlü modeller için bir tavan
etkisi (ceiling effect) var; benchmark asıl daha küçük modelleri (gemma-2-2b,
Llama-3.2-3B) ayırt etmekte işe yarıyor. Tam kategori bazlı tablo ve ham
cevaplar: [`sonuclar/karsilastirma.json`](sonuclar/karsilastirma.json).

Pushlandıktan sonra: [`FELSEFE_BENCHMARK_REPO_ID` deposu](https://huggingface.co)
/ güncellenmiş [model kartı](https://huggingface.co/yoitsmeyusuf/felsefe-lora).
