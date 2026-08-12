# 11. Hafta — Ollama ile Yerel Asistan + Öğrenen RAG Belleği


## Ne yapıyor?

Asistan bir soruyu ilk kez
görünce internete çıkıp cevabı üretiyor ve cevabı soruyla birlikte ChromaDB'ye
yazıyor. Aynı soru (ya da parafrazı) ikinci kez geldiğinde internete hiç
çıkmıyor, cevabı kendi belleğinden üretiyor. Yani asistanın hızı ve tutarlılığı
kullandıkça artıyor — 10. haftanın vektör araması burada bir *önbellek* olarak
çalışıyor.

Beş araç var (`tools.py`):

| Araç | Kaynak | Anahtar gerekli mi? |
|---|---|---|
| `knowledge_question` | Öğrenilmiş bellek (ChromaDB) → yoksa internet | — |
| `get_weather` | Open-Meteo | hayır |
| `get_exchange_rate` | Frankfurter | hayır |
| `internet_search` | DuckDuckGo Lite → Wikipedia (yedek) | hayır |
| `code_error_fix_search` | DDG + `site:stackoverflow.com` / `site:learn.microsoft.com` | hayır |


## Üç kapılı topraklama (grounding)

`code_rag.py`'nin tek derdi, modelin sadece elindeki metinlerde yazanı
söylemesi ve öğrendiğini bir dahaki sefere kullanması. Üç kapı:

1. **Arama kapısı** — soru bellekteki kayıtlarla karşılaştırılır. Benzerlik
   eşiği geçilirse cevap bellekten üretilir, **internete hiç çıkılmaz**. Eşik
   geçilemezse bellek "boş" sayılır ve arama sonuçlarıyla devam edilir.
2. **Üretim kapısı** — LLM'e "sadece bu metinlerden cevapla, kendi ezberini
   kullanma" talimatı verilir. Kaynak yetmezse cevap sabit:
   `Bilmiyorum — bu soru icin guvenilir bir kaynak bulamadim.`
3. **Kayıt kapısı** — internetten üretilen cevap, soruyla birlikte bellekte
   saklanır. **"Bilmiyorum" kaydedilmez**: kaydedilirse asistan o soruyu bir
   daha araştırmaz, kendi bilgisizliğini ezberler.

`chat.py`'deki sistem promptu da üstüne bir kural koyuyor: aracın döndürdüğü
metni aynen aktar, üzerine ekleme yapma. Aksi halde dış model "Bilmiyorum"
cevabını kendi bilgisiyle "düzeltmeye" kalkıyor ve topraklama boşa gidiyor.

### Guardrail: modeli belleği atlamaktan alıkoymak

**Sorun.** Modelin elinde beş araç var ve bunların ikisi (`internet_search`,
`code_error_fix_search`) belleği hiç görmeden doğrudan internete çıkıyor. İlk
denemede `qwen3:4b`, "python modulu bulunamiyor hatasi aliyorum" sorusunda
`knowledge_question` yerine `code_error_fix_search`'ü seçti. İki şey birden
bozuldu:

1. **Cevap topraklanmadı.** Ham arama sadece başlık + link listesi döndürüyor;
   üretim kapısından geçmediği için modelin cevabı ne kadarını o listeden, ne
   kadarını kendi ezberinden yazdığı belirsiz.
2. **Asistan hiçbir şey öğrenmedi.** Kayıt kapısı `knowledge_question`'ın
   içinde; ham arama onu atladığı için bellek boş kaldı.

İkinci madde bu haftanın tüm fikrini sessizce iptal ediyor — ekranda her şey
yolunda görünüyor, cevaplar makul, ama `code_rag.py --liste` boş dönüyor ve
asistan aynı soruyu ertesi gün yine internette arıyor. Hatanın görünür bir
belirtisi yok; bu yüzden `📚 kaynak:` satırını ekledim.

**Neden prompt'la çözmedim.** Sistem promptunda "bilgi soruları için
`knowledge_question` kullan" yazıyordu, araç açıklamalarında da "hazır cevap
için `knowledge_question`'ı tercih et" yazıyor. Model yine de ham aramayı
seçti. 7. ve 8. haftanın dersi bu: küçük modelde araç seçimi kuralını prompt'ta
bırakırsan tutmuyor, **harness'a taşıyacaksın**. `chat.py`'de araç çağrılarını
zaten biz çalıştırdığımız için kural oraya bir `if` olarak giriyor:

```python
HAM_ARAMA_ARACLARI = {"internet_search", "code_error_fix_search"}
...
if name in HAM_ARAMA_ARACLARI:
    arguments = {"question": arguments.get("query") or arguments.get("question") or ""}
    name = "knowledge_question"
```

Ekranda böyle görünüyor:

```
🔧 code_error_fix_search({'query': 'python module not found error'})
⚠️  guardrail: knowledge_question'a yonlendirildi (bellek atlanmasin)
```

**Neden araçları listeden tamamen silmedim?** Silmek de bir seçenekti ve daha
basit olurdu. Ama modelin "bunu aramam lazım" niyeti işe yarar bir bilgi:
çağrıyı yakalayıp yönlendirince niyet korunuyor, sadece hangi yoldan
gideceğine biz karar veriyoruz. Ayrıca bu iki fonksiyon zaten silinemezdi —
`web_lookup` onları boru hattının **içinde** kullanıyor: kod hatası gibi duran
sorularda `code_error_fix_search` (sorguya `site:stackoverflow.com` /
`site:learn.microsoft.com` ekliyor), diğerlerinde `internet_search`.

**Yan etkisi: bayatlama.** Artık internete her çıkış belleğe yazıldığı için
güncel bilgi soruları da kaydediliyor ("son sürüm hangisi", bir haber). Bir
soru-cevap belleği için bu gerçek bir risk: cevap doğru kaydedildi ama zamanla
yanlışa dönüşüyor. Bu yüzden kayıtlara raf ömrü koydum — `code_rag.MAX_AGE_DAYS`
(varsayılan 30 gün). `remember()` her kayda `created_ts` yazıyor, arama kapısı
da eşiği geçen kayıtları bir de yaştan süzüyor:

```python
relevant = [
    h for h in hits
    if h["similarity"] >= threshold
    and (MAX_AGE_DAYS <= 0 or h["age_days"] <= MAX_AGE_DAYS)
]
```

Bayat kayıt yok sayılınca soru yeniden araştırılıyor ve `upsert` aynı kimliğin
üstüne yazıyor — yani kayıt çoğalmıyor, tazeleniyor. Saatlik değişen veriler
(hava durumu, döviz) zaten bu yoldan hiç geçmiyor: onların kendi deterministik
araçları var ve sonuçları hiç önbelleğe alınmıyor.

*

## Eşik neden model başına değişir?

`ollama_client.EMBED_MODELS` içinde her modelin kendi `min_similarity` değeri
var. Sebep basit: her modelin benzerlik skorları farklı bir aralığa yayılıyor,
birinde 0.50 "alakasız" demekken diğerinde "çok alakalı" olabiliyor. Eşiği tek
bir sabit yazmak, model değiştirince sessizce bozuluyor.

Ayrıca `embeddinggemma` **önek (prefix)** ile eğitilmiş bir model: metnin belge
mi soru mu olduğunu belirten kısa bir ön ek bekliyor. Koymazsanız kod çalışır
ama isabet belirgin şekilde düşer — sessiz bir hata sınıfı.

| anahtar | model | boyut | önek | eşik |
|---|---|---|---|---|
| `gemma` | `embeddinggemma:latest` | 768 | var | 0.40 |
| `bge` | `bge-m3:latest` | 1024 | yok | 0.66 |
| `magibu` | `alibayram/embeddingmagibu-200m` | 768 | yok | 0.77 |

`magibu` bu görevde iyi ayrım yapamıyor; "kötü retriever nasıl anlaşılır?"
örneği olarak bilerek bırakıldı (aşağıdaki tabloya bakın).

## Ölçüm: "hissetmek" yerine saymak

`olcum_karsilastirma.py` belleğe 6 sabit soru-cevap tohumluyor (gerçek belleği
kirletmemek için ayrı bir koleksiyona) ve iki soru soruyor:

- **İsabet@1** — bellekteki bir kaydın *parafrazı* sorulunca doğru kayıt ilk
  sırada mı geliyor?
- **AYRIM** — parafrazların *en düşük* skoru, bellekte olmaması gereken
  soruların *en yüksek* skorundan büyük mü?

Sonuçlar (6 parafraz, 9 negatif):

| model | İsabet@1 | parafraz en düşük | negatif en yüksek | AYRIM | güvenli eşik |
|---|---|---|---|---|---|
| `gemma` | 6/6 | 0.358 | 0.376 | −0.018 | 0.40 (2/6 parafraz kaçar) |
| `bge` | 5/6 | 0.418 | 0.643 | −0.224 | 0.66 (3/6 parafraz kaçar) |
| `magibu` | 1/6 | 0.639 | 0.752 | −0.114 | 0.77 (6/6 parafraz kaçar) |


## Çalıştırma

Ollama kurulu ve modeller çekilmiş olmalı. Bu makinede root yetkisi olmadığı
için resmî `install.sh` yerine tarball'ı ev dizinine açtım:

```bash
curl -fL -o ollama.tar.zst \
  https://github.com/ollama/ollama/releases/latest/download/ollama-linux-amd64.tar.zst
mkdir -p ~/.local/ollama && tar --use-compress-program=unzstd -xf ollama.tar.zst -C ~/.local/ollama
~/.local/ollama/bin/ollama serve &

~/.local/ollama/bin/ollama pull qwen3:4b
~/.local/ollama/bin/ollama pull embeddinggemma
~/.local/ollama/bin/ollama pull bge-m3
```

Sohbet modeli `OLLAMA_CHAT_MODEL` ile seçiliyor (varsayılan `qwen3:4b` — 8 GB
VRAM'e sığan, araç çağırabilen en küçük model; ders reposundaki varsayılan
`muse-glimmer:30b-mlx` Apple Silicon derlemesiydi, burada çalışmıyor).

Bu hafta da **üst klasördeki venv** kullanılıyor (`magibu/.venv`, Python 3.13) —
10. hafta ve bitirme projesiyle aynı. İki bağımlılık (`chromadb`, `requests`)
orada zaten kurulu; `odevler/.venv` (Python 3.10) `chromadb` içermiyor, onunla
çalışmaz.

```bash
# 1) retriever'i olc (gercek bellegi kirletmez)
../.venv/bin/python hafta11_ollama_asistan/olcum_karsilastirma.py

# 2) sohbet et
../.venv/bin/python hafta11_ollama_asistan/chat.py
../.venv/bin/python hafta11_ollama_asistan/chat.py --embed-model bge

# 3) bellegi elle yokla
../.venv/bin/python hafta11_ollama_asistan/code_rag.py --liste
../.venv/bin/python hafta11_ollama_asistan/code_rag.py --sor "npm EACCES hatasi"
../.venv/bin/python hafta11_ollama_asistan/code_rag.py --sifirla
```

Kurulum adımı yok: `chroma_db/` ilk soruyla birlikte kendiliğinden oluşuyor,
git'e girmiyor. Silmek asistanın öğrendiklerini sıfırlar.

## Örnek oturum

Boş bellekle başlayan gerçek bir `chat.py` oturumu (kısaltılmadı, sadece
cevapların uzun kısmı atıldı):

```
  ogrenilmis soru : 0

Siz > Docker container neden hemen kapaniyor?
  🔧 knowledge_question({'question': 'Docker container neden hemen kapaniyor?'})
  📚 kaynak: internet (bellekteki en yakin soru 0.000) — bellege kaydedildi
Asistan > Docker container hemen kapanır çünkü Dockerfile'da belirtilen komut
          geçersizdir veya entrypoint eksik. (...)

Siz > docker konteynerim aninda duruyor sebebi ne olabilir
  🔧 knowledge_question({'question': 'docker konteynerim aninda duruyor sebebi ne olabilir'})
  📚 kaynak: bellek (bellekteki en yakin soru 0.643)
Asistan > Docker konteyneriniz aninda duruyor olabilir çünkü Dockerfile'da
          belirtilen komut geçersizdir veya entrypoint eksik.
          Bellekten geldi — eslesen kayitlar:
          - "Docker container neden hemen kapaniyor?" (benzerlik 0.643)

Siz > Rust borrow checker hatasi nasil cozulur?
  🔧 knowledge_question({'question': 'Rust borrow checker hatasi nasil cozulur?'})
  📚 kaynak: internet (bellekteki en yakin soru 0.161) — bellege kaydedildi
Asistan > Rust borrow checker hatası için hata mesajını inceleyin (...)
```

Üç satırın üçü de ayrı bir davranış: **soğuk soru** internete çıkıp öğreniyor,
**parafraz** 0.643 ile eşiği geçtiği için internete hiç çıkmıyor, **yakın ama
farklı konu** (Rust) 0.161'de kalıp doğru şekilde yeniden araştırılıyor.

Dördüncü davranış, 2. kapının belleği reddedip internete düşürmesi. Aynı konuda
ama farklı **niyette** iki soru sorunca çıkıyor:

```
Siz > C# NullReferenceException neden alinir?
  📚 kaynak: internet — bellege kaydedildi      # "neden" sorusu kaydedildi

Siz > c sharp null referans istisnasi nasil duzeltilir
  📚 kaynak: internet — bellege kaydedildi      # eşiği geçti ama 2. kapı reddetti:
                                                # "neden" cevabı "nasıl"ı karşılamıyor
```

İkisi bellekte ayrı kayıt olarak duruyor; asistan artık hem sebebi hem çözümü
biliyor. Bellek dosyaya yazıldığı için bu birikim oturumlar arasında kalıcı:

```
$ code_rag.py --liste
[gemma] bellekte 2 soru var:
  - C# NullReferenceException neden alinir?
    2026-08-13 01:31 / internet : C#'da NullReferenceException, null değerine...
  - c sharp null referans istisnasi nasil duzeltilir
    2026-08-13 01:32 / internet : Null değer kontrolü yapın veya C# 8.0+ için...
```

Diğer üç araç (`get_weather`, `get_exchange_rate`, ham arama) 7. haftadaki gibi
çalışmaya devam ediyor:

```
Siz > 100 dolar kac TL?
  🔧 get_exchange_rate({'from_currency': 'USD', 'to_currency': 'TRY', 'amount': 100})
Asistan > 100 USD = 4775.7 TL (2026-08-12 tarihli kur).
```

## Not: `think: False` her modelde tutmuyor

`qwen3:4b`, Ollama'ya `"think": False` verilse bile düşünme monologunu
`content`'in içine yazabiliyor. İlk denemede bellekteki cevabın tamamı model
monoloğuydu — üstelik sessizce, çünkü bir hata oluşmuyor. `ollama_client.chat`
artık son `</think>` etiketinden sonrasını alıyor. Kaydedilen veriyi üreten
katmanda bu tür temizliği yapmak şart: yanlış kaydedilen cevap kalıcı oluyor.

## Veri

Bu hafta dışarıdan veri seti indirilmiyor (10. haftanın tıbbi indeksi
kaldırıldı). Bellek tamamen kullanım sırasında oluşuyor; ölçüm betiğindeki 6
tohum soru-cevap çifti de repoda, `olcum_karsilastirma.py` içinde elle yazılı.
