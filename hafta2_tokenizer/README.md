# 2. Hafta — BPE Tokenizer Oluşturma

Sıfırdan, kendi korpusumuz üzerinde eğitilmiş bir byte-level BPE tokenizer.
**Sonuç:** vocab_size=8000 →
**https://huggingface.co/yoitsmeyusuf/felsefe-bpe-tokenizer**

Bu tokenizer bağımsız bir çıktı — 3. haftadaki fine-tune, Qwen3.5'in kendi
tokenizer'ıyla eğitiliyor (`common/lora_trainer.py`, `FastModel.from_pretrained`).
Kendi BPE tokenizer'ımızla modeli baştan eğitmek (embedding matrisini
sıfırdan öğrenmek anlamına geldiği için) bu kadar küçük bir korpusla iyi
sonuç vermez; bu yüzden burası sadece "BPE nasıl eğitilir" çalışması olarak
bağımsız kalıyor.

## Korpus

`data/corpus.txt`, 1. haftanın temizlenmiş veri setinden
(`hafta1_veri_seti/data/raw/scraped_turkish_qa.jsonl`, 527 satır) üretildi: her
satırın soru ve cevap metni ayrı satırlar halinde yazıldı (~382K karakter,
~1050 satır). Aynı domain metinlerinin kullanılması, felsefe terimlerine
(kavram adları, düşünür isimleri, Türkçe'nin eklemeli yapısındaki sık ekler)
daha duyarlı bir tokenizer verir.

```bash
.venv/bin/python - <<'EOF'
import json
rows = [json.loads(l) for l in open("hafta1_veri_seti/data/raw/scraped_turkish_qa.jsonl", encoding="utf-8")]
with open("hafta2_tokenizer/data/corpus.txt", "w", encoding="utf-8") as out:
    for r in rows:
        out.write(r["soru"].replace("\n", " ").strip() + "\n")
        out.write(r["cevap"].replace("\n", " ").strip() + "\n")
EOF
```

## Neden `VOCAB_SIZE = 8000` (32000 değil)

İlk denemede 32000 kullanıldı, ama korpusumuz (~382K karakter, ~47K
kelime, ~16K benzersiz kelime formu) production tokenizer'ların (GPT-2,
Llama) eğitildiği gigabayt'larca metnin yanında çok küçük. 32000 hedefine
zorlarsak iki olası sonuç var: BPE ya anlamsız/korpusa-özgü kalıpları
ezberler (ör. "kimdir?" sürekli tekrar ettiği için tek bir merge'e döner),
ya da yeterli sıklıkta tekrarlanan çift bulamayıp hedefe hiç ulaşamaz. 16K
benzersiz kelimeye yakın bir vocab da BPE'yi neredeyse kelime-düzeyi bir
tokenizer'a indirger — asıl amaç olan alt-kelime (subword) genellemesi
kaybolur.

8000, Türkçe'nin eklemeli yapısındaki sık ekleri (-ler, -dir, -nin, -lik
vb.) ve alan-spesifik kelimeleri (felsefe terimleri) öğrenmeye yetecek kadar
büyük, korpusu ezberlemeyecek kadar küçük bir orta nokta.

## Çalıştırma

```bash
.venv/bin/python hafta2_tokenizer/train_bpe_tokenizer.py
```

`CORPUS_PATHS` ve `VOCAB_SIZE` script içinde zaten ayarlı (yukarıdaki
korpusu ve 8000'i kullanıyor); `.env`'deki `TOKENIZER_REPO_ID` doldurulmuş
olmalı.

## Doğrulama

```python
from transformers import AutoTokenizer
tok = AutoTokenizer.from_pretrained("yoitsmeyusuf/felsefe-bpe-tokenizer")
ids = tok.encode("Nietzsche'nin varoluşçuluk üzerine düşünceleri özgür irade kavramıyla ilişkilidir.")
print(tok.convert_ids_to_tokens(ids))
print(tok.decode(ids))  # orijinal metinle birebir eşleşmeli
```

Sonuç: "Nietzsche" ve "irade" gibi sık geçen kelimeler tek token, "varoluşçuluk"/
"düşünceleri"/"kavramıyla" gibi daha uzun/nadir kelimeler kök+ek şeklinde
alt-parçalara ayrılıyor — beklenen BPE davranışı. Decode, orijinal metinle
birebir (boşluk/noktalama dahil) eşleşiyor.
