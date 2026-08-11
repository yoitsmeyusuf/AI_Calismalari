# FastAPI Öğrenme Görevi

RAG persona projesinin ön koşulu. Ödevin beş başlığının her biri burada
çalışan bir uca bağlandı; README'deki bütün tablolar `dene.py` ve `kiyas.py`
çıktısından geliyor, elle yazılmadı.

```bash
# proje_persona/1_fastapi içinden
../../.venv/bin/python dene.py     # 33 kontrol, API'yi kendisi ayağa kaldırır
../../.venv/bin/python kiyas.py    # WSGI/ASGI farkının ölçümü

# elle kurcalamak için
../../.venv/bin/python -m uvicorn main:app --reload --port 8000
# -> http://127.0.0.1:8000/docs
```

| Dosya | İş |
|---|---|
| `semalar.py` | Pydantic modelleri — gelen/giden veri sözleşmesi |
| `main.py` | API: 6 yol, 9 uç, CORS middleware |
| `dene.py` | Kanıt script'i — sunucuyu iki kez (CORS açık/kapalı) kaldırıp ölçer |
| `kiyas.py` | "FastAPI hızlıdır" iddiasının ölçümü |
| `cors_demo/index.html` | Farklı origin'den gerçek tarayıcı isteği |

---

## 1. FastAPI nedir, ne işe yarar

Python için bir **web API çatısı**. Ayırt edici yanı, tip ipuçlarını (type
hints) dekoratif olmaktan çıkarıp **çalışma zamanı sözleşmesine** çevirmesi:

```python
def sarki_getir(sarki_id: int) -> SarkiYanit: ...
```

Bu tek satırdan üç şey birden türüyor — `sarki_id`'nin doğrulanması, yanıtın
biçimlenmesi ve `/docs`'taki dokümantasyon. Flask'ta bunlar üç ayrı iş
(`int()` çevirimi + try/except, elle `jsonify`, ayrıca yazılan bir OpenAPI
şeması).

Üç bileşenin üstünde duruyor: **Starlette** (ASGI, routing, middleware),
**Pydantic** (doğrulama/serileştirme), **Uvicorn** (ASGI sunucusu).

### Diğer backend'lerle kıyas

| | FastAPI | Flask | Django REST | Express | Spring Boot |
|---|---|---|---|---|---|
| Model | ASGI (async) | WSGI (senkron)¹ | WSGI¹ | Event loop | Thread-per-request² |
| Doğrulama | Tip ipucundan otomatik | Elle / eklenti | Serializer sınıfı | Elle / zod | Bean Validation |
| OpenAPI | Koddan otomatik | Eklenti + elle | Eklenti | Eklenti | Eklenti |
| Kapsam | Sadece API | Mikro | Tam yığın (ORM, admin, auth) | Mikro | Tam yığın |
| Öğrenme | Düşük | En düşük | Yüksek | Düşük | Yüksek |
| Olgunluk | 2018 | 2010 | 2005 | 2010 | 2002 |

¹ Flask 2.0+ ve Django 3.0+ `async def` kabul ediyor ama ekosistemin (ORM,
eklentiler) büyük kısmı hâlâ senkron; ASGI'a sonradan eklenmiş bir yetenek.
² Sanal iş parçacıklarıyla (Java 21+) değişiyor.

**FastAPI ne zaman doğru seçim:** JSON API'ler, I/O ağırlıklı iş yükü
(veritabanı, dış servis çağrısı, LLM inference), sözleşmenin tipten türemesinin
değerli olduğu yerler — bu projedeki RAG servisi tam örnek.

**Ne zaman değil:** Sunucu tarafı HTML render eden klasik web uygulaması
(Django'nun template + admin + ORM üçlüsü kazandırır), ağır CPU işi
(aşağıdaki ölçüme bakın), ya da ekibin Python bilmediği durumlar.

### "Hızlı" iddiasının ölçümü

Flask kurup TechEmpower tekrarı yapmak yerine farkın **kaynağını** ölçtüm:
WSGI her isteği bir iş parçacığı/süreç işgal ederek işler, ASGI I/O beklerken
event loop'u serbest bırakır. Aynı fark FastAPI'nin kendi içinde `def` ile
`async def` arasında da var (`def` uçları iş parçacığı havuzuna gider).

200 eşzamanlı istek, her biri 200 ms I/O bekliyor:

| Uç | Süre | İstek/sn |
|---|---|---|
| `async def` + `await asyncio.sleep` | 0.871 sn | 229.7 |
| `def` + `time.sleep` (40'lık havuz) | 1.229 sn | 162.7 |

Ham oran **1.4x**. Ama teorik beklenti 5x idi (async 0.20 sn, sync 5 parti ×
40 iş parçacığı = 1.00 sn). Aradaki farkı tahmin etmek yerine ölçtüm:
gecikmesiz bir uca aynı yükü basınca **0.330 sn** çıkıyor — bu tamamen istemci
tarafı ek yükü. Düşülünce **1.7x** (0.540 vs 0.899 sn).

Yani 1.4x, sunucu modelinin farkı değil, `httpx` istemcisinin tavanı. Gerçek
oran teorik 5x ile ölçülen 1.7x arasında bir yerde; bunu tam ayırmak için
istemciyi çok süreçli bir yük aracına (wrk, locust) taşımak gerekirdi.
**Ölçtüğüm şeyin sınırını olduğu gibi yazıyorum.**

Asıl öğretici sonuç ikinci testte. `async def` içine `await`siz CPU işi koyup
**bu sırada gelen hafif isteklerin gecikmesini** ölçtüm:

| Ağır uç | Hafif isteğin medyan gecikmesi |
|---|---|
| `async def` (event loop kilitli) | **1463 ms** |
| `def` (iş parçacığı havuzuna gider) | **236 ms** |

**6.2x.** Toplam iş/sn ikisinde de benzer (GIL zaten seri hâle getiriyor);
fark **cevap verebilirlikte**. `async def` yazıp içine `await` koymamak
FastAPI'nin en yaygın tuzağı: hızlanmıyorsunuz, sunucunun tamamını
kilitliyorsunuz. Doğru refleks — CPU-yoğun ucu `def` yazmak.

---

## 2. `@app.get()` vs `@app.post()`

Aynı yol, farklı metot = **iki ayrı uç**. `/ogretici/echo` ikisini de tanımlıyor.

| | GET | POST |
|---|---|---|
| Veri nerede | Query string (URL'de) | Request body |
| Amaç | Okuma | Oluşturma / yan etkili işlem |
| Idempotent | Evet | Hayır |
| Cache'lenir | Evet | Hayır |
| Tarayıcı tekrarı | Serbestçe | "Formu yeniden gönder?" uyarısı |
| Gövde | Olmamalı | Olmalı |
| Yer imi / geçmiş | Eklenebilir | Eklenemez |

Pratik sonuç: GET'te veri URL'de taşındığı için **tarayıcı geçmişine, sunucu
log'larına ve `Referer` başlığına yazılır**. Şifre veya kişisel veri GET ile
gönderilmez — sunucuya ulaşır ama üç yerde iz bırakır.

FastAPI'de ayrım parametre tipinden okunuyor: Pydantic modeli olan parametre
gövdeden, skaler olan query'den alınır. `dene.py` çıktısı:

```
GET  /ogretici/echo  ->  query string (URL'de gorunur)
POST /ogretici/echo  ->  request body (URL'de gorunmez)
POST /sarkilar/1     ->  405 Method Not Allowed
```

Son satır önemli: `/sarkilar/{id}` yolu var ama POST tanımlı değil, o yüzden
404 değil **405** dönüyor — yol var, metot yok.

---

## 3. Pydantic ile veri doğrulama

### Gelen veri

`POST /sarkilar` gövdesine 12 farklı vaka gönderildi, hepsi beklendiği gibi:

| Gövde | Beklenen | Sonuç | Pydantic hata tipi |
|---|---|---|---|
| geçerli | 201 | 201 | — |
| `album` yok | 422 | 422 | `missing` |
| `yil: "iki bin"` | 422 | 422 | `int_parsing` |
| `yil: "2020"` (sayı stringi) | **201** | 201 | — (coercion) |
| `yil: 1800` | 422 | 422 | `greater_than_equal` |
| `yil: 2099` | 422 | 422 | `value_error` (özel) |
| `sure_sn: 0` | 422 | 422 | `greater_than` |
| `baslik: ""` | 422 | 422 | `string_too_short` |
| `baslik: "   "` | 422 | 422 | `value_error` (özel) |
| `tur: "pop"` | 422 | 422 | `enum` |
| fazla alan `xx` | 422 | 422 | `extra_forbidden` |
| `basslik` (yazım hatası) | 422 | 422 | `missing` |

Üç satır özellikle anlamlı:

**`"2020"` neden 201?** Pydantic varsayılan olarak *lax* modda: JSON'dan gelen
sayı stringini int'e çevirir. Bu bilinçli bir tasarım (HTTP'de her şey
string'dir) ama sürpriz olabilir. `strict=True` ile kapatılabilir.

**`"   "` neden 422?** `min_length=1` üç boşluğu geçirir. `field_validator`
kırpıp boş kalırsa hata veriyor — kısıtların ifade edemediği kural için
doğrulayıcı gerekiyor. Aynı şekilde `yil: 2099` de: "gelecekte olamaz"
değişken bir sınır, `le=` ile yazılamaz.

**`basslik` neden `missing`?** `extra="forbid"` olmasaydı bu alan sessizce
yutulur, `baslik` boş kalır ve **hata almazdınız**. Varsayılan davranış
(`ignore`) yazım hatalarını gizliyor; bu yüzden açıkça kapattım.

422 gövdesinin şekli — hangi alan, ne tip hata, ne gönderilmişti:

```json
{"detail": [{
    "type": "int_parsing",
    "loc": ["body", "yil"],
    "msg": "Input should be a valid integer, unable to parse string as an integer",
    "input": "iki bin"
}]}
```

Query ve path parametreleri de aynı mekanizmadan geçiyor: `limit=0` → 422
(`ge=1`), `limit=999` → 422 (`le=100`), `/sarkilar/abc` → 422 (int değil).
Uç fonksiyonuna hiç girilmiyor.

### Giden veri

Asıl az bilinen taraf bu. `SarkiKayit` içeride `ic_not` ve `kaynak_ip`
tutuyor; uç bu nesneyi **olduğu gibi** döndürüyor ama
`response_model=SarkiYanit` olduğu için:

```
donen alanlar: ['album', 'baslik', 'id', 'sure_mmss', 'sure_sn', 'tur', 'yil']
ic_not yanitta yok     : True
kaynak_ip yanitta yok  : True
```

Yani `response_model` bir sızıntı bariyeri. Ayrıca `sure_mmss`, `computed_field`
ile türetilip yanıta ekleniyor — depoda böyle bir alan yok.

**Yolda düzeltilen bir hata:** `SayfaliYanit`'i uç içinde elle kurunca 500
aldım. Sebep: `response_model` üzerinden dönerken FastAPI nesne→model
dönüşümünü kendisi yapıyor, ama modeli **elle** kurduğumda düz Pydantic
doğrulaması çalışıyor ve `SarkiKayit` nesnesi dict olmadığı için reddediliyor.
Çözüm `model_config = ConfigDict(from_attributes=True)`. Ayrım şu: FastAPI'nin
serileştirme yolu ile Pydantic'in doğrulama yolu aynı şey değil.

---

## 4. Swagger UI

FastAPI koddan **OpenAPI 3.1** şeması üretiyor; `/docs` ve `/redoc` o şemayı
render eden iki ayrı arayüz.

| Yol | Ne |
|---|---|
| `/openapi.json` | Ham şema — asıl kaynak |
| `/docs` | Swagger UI — deneme yapılabilir ("Try it out") |
| `/redoc` | ReDoc — okumaya odaklı, deneme yok |

Ölçülen: 6 yol, **8 model şeması**, `/sarkilar` altında `get` + `post`.
`SarkiYanit` şeması `sure_mmss` dahil 7 alanla görünüyor — yani `computed_field`
dokümana da yansıyor.

Şemayı zenginleştiren şeyler `main.py`'de: `openapi_tags` (uçları gruplar),
`summary`/`description`, `responses={404: {"model": HataYaniti}}` (hata
gövdesini de dokümante eder), `status_code=201`. Docstring'ler doğrudan
açıklama olarak geçiyor.

Postman'e almak için `/openapi.json` → Import yeterli; ayrıca koleksiyon
yazmak gerekmiyor.

---

## 5. CORS

Tarayıcı, sayfanın origin'i (şema + host + **port**) ile isteğin gittiği origin
farklıysa yanıtı JS'e vermez. `cors_demo/index.html` `:8080`'den servis edilip
API'ye (`:8000`) istek atıyor — farklı port, dolayısıyla cross-origin.

Middleware açıkken preflight yanıtı:

```
access-control-allow-origin       : http://localhost:8080
access-control-allow-methods      : GET, POST, PUT, DELETE, OPTIONS
access-control-allow-headers      : Accept, Accept-Language, Content-Language,
                                    Content-Type, X-Istek-Kimligi
access-control-allow-credentials  : true
access-control-max-age            : 600
```

İzinsiz origin (`http://kotu-site.example`) preflight'ı **400** alıyor ve
`allow-origin` başlığı hiç dönmüyor.

Middleware kapalıyken **aynı** istekler:

| İstek | CORS açık | CORS kapalı |
|---|---|---|
| `OPTIONS /sarkilar` (preflight) | 200 | **405** |
| `GET /sarkilar` | 200 | **200** |
| `allow-origin` başlığı | `http://localhost:8080` | **(yok)** |

**En kritik satır ortadaki:** CORS kapalıyken de GET **200 dönüyor**. İstek
sunucuya ulaştı, işlendi, yanıt üretildi. Engelleyen sunucu değil, **tarayıcı**.

Bunun iki sonucu var:

1. **CORS bir güvenlik önlemi değil.** Sunucuyu korumaz — `curl`, `httpx`,
   Postman CORS uygulamaz, hepsi veriyi alır. Koruduğu şey, *kullanıcının
   tarayıcısındaki başka bir sitenin* onun kimliğiyle sizin API'nize istek
   atıp yanıtı okuması. Yetkilendirme yerine geçmez.
2. **Kanıt için `dene.py` yetmez.** httpx CORS uygulamadığı için tablo
   "engellendi" diyemez; gerçek tarayıcı gerekiyor. `cors_demo/index.html`
   bunun için var.

### Tarayıcı demosu

```bash
# 1. terminal
../../.venv/bin/python -m uvicorn main:app --port 8000
# 2. terminal
../../.venv/bin/python -m http.server 8080 --directory cors_demo
# -> http://localhost:8080
```

Üç düğme üç farklı davranışı gösteriyor:

1. **Basit GET** — `Content-Type` ve özel başlık yok, "simple request".
   Preflight atılmaz, istek doğrudan gider, tarayıcı **yanıtı** filtreler.
2. **POST + özel başlık** — simple request değil. Tarayıcı önce `OPTIONS`
   atar; reddedilirse **asıl POST hiç gönderilmez**. Ağ sekmesinde yalnızca
   `OPTIONS` satırı görünür.
3. **Özel yanıt başlığını oku** — sunucu `X-Toplam-Kayit` gönderiyor ama
   tarayıcı JS'e varsayılan olarak yalnızca birkaç "güvenli" başlığı gösterir.
   `expose_headers` olmadan `null` okunur.

Aynı sayfayı `CORS_ACIK=0` ile başlatılmış sunucuya karşı açınca üçü de
hata veriyor — uvicorn log'unda istekler 200 görünmeye devam ederken.

### `allow_origins=["*"]` tuzağı

`allow_origins=["*"]` ile `allow_credentials=True` **birlikte çalışmaz**.
Spec gereği joker origin'e kimlik bilgisi (cookie) gönderilemez; tarayıcı
credential'lı isteği reddeder. Bu yüzden origin'ler `main.py`'de açıkça
yazılı. Üretimde `["*"]` zaten istenmeyen bir varsayılan.
