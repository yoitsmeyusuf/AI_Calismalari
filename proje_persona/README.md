# Proje: Kendi Karakteristik AI'ni Yarat

Sagopa Kajmer korpusu üzerine kurulu dört parçalı çalışma: bir FastAPI öğrenme
görevi, sıfırdan yazılmış Transformer bileşenleri, bu bileşenlerden inşa edilen
karakter düzeyinde bir MiniGPT ve aynı korpusu vektörleştiren bir RAG persona
chatbot'u.

| Klasör | Ödev | Öne çıkan sonuç |
|---|---|---|
| [`1_fastapi/`](1_fastapi/) | FastAPI öğrenme görevi (Agent ekibi ön koşulu) | 35/35 kontrol; `async def` içine CPU işi koymak hafif istekleri **6.2x** geciktiriyor |
| [`2_transformer/`](2_transformer/) | Görev 4–7: Positional Encoding, Attention, Multi-Head, Block | 34/34 kontrol; PE'siz self-attention permütasyona eşdeğer (fark **4.8e-07**) |
| [`3_rag_persona/`](3_rag_persona/) | Proje / Agent kanadı: RAG Persona Chatbot | 1.127 chunk, eşik **0.38** (18/21); guardrail harness'ta |
| [`4_mini_gpt/`](4_mini_gpt/) | Proje / Model kanadı: MiniGPT (2_transformer'ı import eder) | Veriye ölçekli model, ödevin ayarlarını **%39 parametreyle** geçiyor |

Ortak korpus [`veri_topla.py`](veri_topla.py) ile üretiliyor:

```bash
../.venv/bin/python proje_persona/veri_topla.py
```

Ortak sabitler [`ayarlar.py`](ayarlar.py)'de. Klasör adları rakamla başladığı
için paket olarak import edilemiyorlar; alt projeler birbirinin modülünü
`sys.path` üzerinden alıyor.

Bu hafta **üst klasördeki venv** kullanılıyor (`magibu/.venv`, Python 3.13) —
9. ve 10. haftada olduğu gibi. `torch`, `fastapi`, `chromadb`,
`sentence-transformers` orada zaten kuruluydu.

---

## Korpus

**Sanatçı:** Sagopa Kajmer
**Kaynak:** [`metncelik/turkish-song-lyrics`](https://huggingface.co/datasets/metncelik/turkish-song-lyrics) (MIT), `songs-unprocessed.csv`

### Neden scraping değil

Ödev "web scraping **veya hazır veri setleri**" diyor. Dört kaynağı ölçtüm:

| Kaynak | Durum |
|---|---|
| genius.com | Cloudflare bot doğrulama duvarı |
| lyricstranslate.com | Cloudflare challenge |
| sarkisozlerihd.com | Cloudflare challenge |
| alternatifim.com | `robots.txt` erişime açık, **ama** `Content-Signal: ai-train=no` |

Üçü teknik olarak duvarlı; dördüncüsü erişime açık olmasına rağmen AI eğitimi
için veri toplanmasını açıkça reddediyor. 1. haftada Reddit için Playwright'a
geçmiştim, burada aynısını yapmak bu beyanı çiğnemek olurdu. MIT lisanslı hazır
veri seti hem daha temiz hem lisansen sağlam.

**Not:** MIT lisansı *derlemeye* ait; şarkı sözlerinin telifi hak sahiplerinde.
Bu yüzden korpus `.gitignore`'da ve — 9./10. haftadan bilinçli olarak ayrılarak
— HF Hub'a push **edilmiyor**. Repoda yalnızca toplama script'i ve istatistikler
duruyor.

### Hangi CSV

Veri setinde iki dosya var ve ayrım kritik:

| Dosya | Satır yapısı |
|---|---|
| `songs.csv` | işlenmiş — satır sonları **silinmiş** (medyan 1 satır) |
| `songs-unprocessed.csv` | ham — satır yapısı **korunmuş** (medyan 24 satır) |

`songs-unprocessed.csv` kullanılıyor. Hem bar sınırından chunk'lamak hem
MiniGPT'nin okunabilir metin üretmesi satır yapısına bağlı.

Bir ölçüm tuzağı: `datasets-server` API'si "default" config'de iki dosyayı
**birleştirip** sunuyor. İlk ölçümümde sanatçı başına satır sayısı iki katına
çıkmış (263 → 526) ve "şarkıların yarısında satır sonu yok" gibi sahte bir
bulgu üretmişti. İkisi de aynı şarkının iki hâliymiş — boşluk ve noktalama
atılınca 262 başlığın 262'sinde metinler birebir aynı çıkıyor. CSV'yi indirip
yerelde ölçünce düzeldi.

### Profil

| | |
|---|---|
| Şarkı | **262** (tekrar yok) |
| Karakter | **268.183 — 262 KB** |
| Satır / şarkı | medyan 24, p10 20, max 82 |
| Satır uzunluğu | medyan 39, p90 54 karakter |
| `vocab_size` | 115 — bunun **29'u** 10'dan az geçiyor |
| Yapısal etiket | `[Nakarat]`, `(x2)`, boş satırlı kıta ayrımı: **hiçbirinde yok** |

İki sonuç:

**Chunking (3_rag_persona):** Kıta ayrımı olmadığı için sınır belgeden
okunamıyor, kurulmak zorunda. Bar medyanı 39 karakter olduğundan 6 barlık grup
~230 karaktere denk geliyor — ödevin istediği 200–400 aralığına doğal olarak
oturuyor.

**Hacim (4_mini_gpt):** 262 KB, ödevin önerdiği 500 KB–1 MB'ın yarısı.
`5000 iter × 64 batch × 128 block` ≈ 41M token, 236K karakterlik eğitim verisi
üzerinde ~174 epoch demek. Ezber bekleniyor. Korpusu başka sanatçılarla
şişirmek yerine (persona bulanırdı) iki koşu yapılacak: ödevin
hiperparametreleriyle bir taban koşu ve veriye ölçekli ikinci bir koşu; fark
val loss eğrisinde gösterilecek.
