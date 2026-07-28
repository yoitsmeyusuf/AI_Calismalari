# 1. Hafta — Veri Seti Hazırlama (hesapsız web scraping)

**Domain:** Felsefe (düşünürler, akımlar, temel kavramlar) — Türkçe, tek dilli.
**Kaynaklar:** Ekşi Sözlük + r/felsefe (Reddit). İkisi de hesap/API anahtarı
gerektirmiyor.
**Sonuç:** 527 soru-cevap satırı (462 Ekşi Sözlük + 65 Reddit) + 2 elle
yazılmış seed →
**https://huggingface.co/datasets/yoitsmeyusuf/felsefe_finetune**

## Neden bu iki kaynak (ve neden r/Turkey değil)

- İlk denemede genel **r/Turkey** kullanıldı ama "top" listesi ağırlıklı
  siyasi gündem/tepki-avı çıktı (hakaret içeren yorumlar dahil) — soru-cevap
  formatına uymuyordu. r/AskTurkey de denendi ama ağırlıklı İngilizce çıktı.
  **r/felsefe**'ye geçildi: gerçek Türkçe felsefi tartışma var, siyasi çamur yok.
- **Reddit**: Ne OAuth'suz `.json` uç noktaları (403) ne de old.reddit.com
  (403) çalışıyor; yeni `www.reddit.com` ise JS ile çözülen bir "please wait
  for verification" duvarına sahip — plain `requests` bunu geçemez. Bu yüzden
  `scrape_reddit.py` **Playwright (headless Chromium)** kullanıyor, gerçek bir
  tarayıcı gibi JS'i çalıştırıp duvarı geçiyor. Hesap/app kaydı yok.
- **Ekşi Sözlük**: Login istemiyor, düz `requests`+`BeautifulSoup` yeterli.
  Başlık = soru (düşünür/akım/kavram adı), en çok favorilenen entry'ler = cevap.

## Kurulum

```bash
uv pip install --python ../.venv/bin/python playwright
../.venv/bin/python -m playwright install chromium
```
(`check_env.py` Chromium'un çalışıp çalışmadığını kontrol eder.)

## Çalıştırma

```bash
.venv/bin/python hafta1_veri_seti/scrape_reddit.py       # r/felsefe, Playwright
.venv/bin/python hafta1_veri_seti/scrape_eksisozluk.py    # felsefe başlıkları
.venv/bin/python hafta1_veri_seti/build_dataset.py         # ikisini birleştirip push et
```

İkisi de aynı dosyaya (`data/raw/scraped_turkish_qa.jsonl`) satır ekliyor,
`build_dataset.py` bunları (+ `seed_examples.py`'deki elle yazılmış 2 örneği)
birleştirip HF'ye push ediyor. Script'ler idempotent — tekrar
çalıştırıldığında zaten toplanmış `id`'leri (`data/raw/rejected_ids.txt`
dahil, aşağıya bakın) atlar.

Format, ortak mesaj şemasını kullanır (bkz. `../common/hf_dataset_schema.py`)
ama tek dilli — sadece `"turkish"` split'i:
her satır `[{role: "user", content: soru, ...}, {role: "assistant", content: cevap,
thinking: "", ...}]` şeklinde bir konuşma.

## Reddit kaynakları

`scrape_reddit.py`, `SOURCES` listesinde tanımlı **12 kaynaktan** çekiyor:
genel `/top/?t=all`, popüler olmayanları da yakalamak için `/new/`, ve
r/felsefe'nin flair'lerinin tamamı (mizah/"güldürü" hariç — çok küçük ve
şaka ağırlıklı, bilerek dışarıda bırakıldı):

`bilim • philosophy of science`, `düşünürler, düşünceler, düşünmeler`,
`eseme • logic`, `bilgi • epistemology`, `yaşamın içinden • axiology`,
`inanç • philosophy of religion`, `yönetim • philosophy of politics`,
`«iyilik» üzerine • ethics`, `varlık • ontology`,
`«güzellik» üzerine • aesthetics`, `/r/felsefe'ye değgin` (subreddit-geneli).

Bunlar community tarafından zaten konuya göre etiketlenmiş, genel `/top/`
listesinden daha az gürültülü çıkıyor. Flair sayfaları sanal liste kullandığı
için düz listeden daha yavaş/dengesiz hidrate oluyor ve çok sayıda kaynağı
art arda çekince Reddit tarafında geçici bir throttling'e giriliyor — bu
yüzden kaynaklar arası `SOURCE_DELAY_MS` beklemesi var, ve bir kaynak
timeout'a girerse script çökmeden diğerlerine devam edip uyarı basıyor. Tek
çalıştırmada tüm 12 kaynağın başarılı olacağı garanti değil; script'i
birkaç kez arka arkaya çalıştırmak gerekebilir (idempotent olduğu için
güvenli).

## Kalite ve güvenlik süreci

Veri, ham haliyle kullanılmadı — iki ayrı geçişte manuel/programatik olarak
temizlendi (orijinal ham veri `data/raw/scraped_turkish_qa.raw_backup.jsonl`
içinde yedekli):

1. **Konu uygunluğu:** Reddit'in gevşek soru-tespit mantığı (bir yorumun
   "?" ile bitip bitmediğine değil, gevşek anahtar-kelime eşleşmesine
   bakması) uzun rant paragraflarını "soru" sanıyordu; bu ve konu dışı
   (siyaset, flört sohbeti, oyun fiyatı) içerik `is_strict_question` ve
   `OFFTOPIC_WORDS` ile sıkılaştırıldı. Toplamda ~100 düşük kaliteli Reddit
   satırı manuel incelemeyle elendi (siyasi kışkırtıcı içerik, meme/pop-kültür
   referansları, argo, alakasız/eşleşmeyen cevaplar, bağlamsız başlıklar).
2. **Uzunluk:** `common/lora_trainer.py`'deki `MAX_SEQ_LENGTH` (2048 token)
   sınırını zorlayacak aşırı uzun (>6000 karakter) cevaplar her iki
   scraper'da da kaynağında son cümle sınırında kesiliyor
   (`MAX_ANSWER_CHARS`/`MAX_ENTRY_LEN`).
3. **Güvenlik taraması:** tüm veri seti küfür/argo, kişisel bilgi (PII) ve
   nefret söylemi/aşırıcılık açısından programatik olarak tarandı. Ekşi
   Sözlük'ün kendi "(bkz: konu/@yazar)" iç link söz diziminden sızan birkaç
   `@kullanıcıadı` referansı temizlendi; ırksal imalı/magazinsel bir entry ve
   içeriksiz/bozuk bir entry elendi. Hassas ama meşru konulardan (savaş,
   zulüm, cinsellik felsefesi vb.) geçen satırlar tek tek okundu — hepsi
   gerçek felsefi/tarihsel bağlamda kullanıldığı için (ör. Hannah Arendt'in
   Nazi Almanyası'ndan kaçışı, Budizm tarihinde kast tacizi) dokunulmadı.

**Kalıcılık:** elenen tüm id'ler `data/raw/rejected_ids.txt`'e eklendi; her
iki scraper da bu dosyayı `seen_ids`'e dahil ediyor, yani script tekrar
çalıştırıldığında elenen satırlar geri gelmiyor. Yeni bir satırı manuel
elerseniz id'sini bu dosyaya da eklemeyi unutmayın.

## Kalite / etik notları

- `looks_turkish()` kaba bir sezgisel filtre (r/felsefe için ayrıca Reddit'in
  kendi `post-language` etiketi de kullanılıyor, daha güvenilir).
- `BLOCKLIST_WORDS` açık hakaret/küfür içeren yorumları eler (tam kapsamlı değil).
- Kullanıcı adları hiçbir yerde saklanmıyor, sadece post/entry metni ve
  kaynak URL'i tutuluyor.
- `scrape_reddit.py` Reddit'in JS bot-doğrulama duvarını Playwright ile
  aşıyor — bu, açık bir API'yi kullanmaktan daha gri bir alan (bilerek
  konulmuş bir engeli aşmak). Küçük ölçekli/kişisel/akademik kullanım için
  script istekler arasında bekliyor ve düşük hacimde çalışacak şekilde
  ayarlı (`POST_LIMIT`), agresif/yüksek hacimli çekim yapmayın.
