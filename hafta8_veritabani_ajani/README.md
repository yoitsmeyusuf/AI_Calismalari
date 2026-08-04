# 8. Hafta — Veritabanına Yazan Tool Calling Ajanı (SQLite + Gradio)

**Amaç:** Bir dil modelinin dış dünyaya erişip **işlem yapabildiği** küçük bir
sistem kurmak. 7. haftada model bir API'den yalnızca *okuyordu*; bu hafta gerçek
bir veritabanına **yazıyor** da: sipariş kaydediyor, stok düşürüyor, sipariş
iptal edip stoğu geri veriyor.

**Senaryo:** Küçük bir **felsefe kitapçısının sipariş asistanı**. Serinin
geri kalanıyla aynı alanda kalsın diye kitapçıyı felsefe üzerine kurdum
(15 kitap, 14 yazar); mekanizma herhangi bir mağaza için birebir aynı.

**Canlı demo:** *(Space henüz yayınlanmadı — `deploy_space.py` hazır, çalıştırma
adımları aşağıda.)*

## Dosyalar

| Dosya | Ne işe yarar |
|---|---|
| `veritabani.py` | SQLite şeması, tohum katalog ve bütün SQL. Tek başına çalıştırılınca veritabanını kurup içeriğini döker. |
| `araclar.py` | 4 aracın implementasyonu + modele verilen JSON şemaları (`ARAC_SEMALARI`). Tek başına çalıştırılınca bütün araçları (hata yolları dahil) test eder. |
| `ajan.py` | Tool calling döngüsü + halüsinasyon guardrail'i + şeffaf iz. `--guardrail` ile guardrail'i modelsiz test eder. |
| `app.py` | Gradio arayüzü (Space giriş dosyası): sohbet + araç izi + **canlı veritabanı paneli**. |
| `deploy_space.py` | ZeroGPU'lu Space'i oluşturur, kartı araç şemalarından üretir, dosyaları yükler, değişken/secret ayarlar. |
| `requirements.txt` | Space'in kuracağı paketler (veritabanı için ek paket yok, `sqlite3` standart kütüphanede). |

Model katmanı (`modeller.py`) **7. haftadan olduğu gibi kullanılıyor** —
kopyalamak yerine aynı modülü import ediyoruz (`sys.path`'e 7. haftanın klasörü
ekleniyor); Space'e deploy ederken dosya `app.py`'nin yanına kopyalanıyor.
Arka uç seçimi yine `TOOL_BACKEND` ile: `yerel` (ZeroGPU/transformers) veya
`api` (HF Inference Providers).

## Mimari

```
app.py            Gradio arayüzü (sohbet + iz paneli + canlı DB tablosu)
   │
ajan.py           tool calling döngüsü, guardrail, şeffaf iz
   ├── modeller.py    (7. hafta)  model çağrısı  → ZeroGPU / Inference Providers
   └── araclar.py     argüman doğrulama + JSON şemaları + hata yolları
          └── veritabani.py   bütün SQL, işlemler (BEGIN IMMEDIATE), tohum veri
```

Katmanlar tek yönlü: `veritabani.py` araçların adını bilmiyor, `araclar.py`
modelden haberdar değil, `ajan.py` hangi veritabanının kullanıldığını bilmiyor.
Böylece kitapçıyı başka bir senaryoyla değiştirmek `veritabani.py` + `araclar.py`
değişikliğinden ibaret kalıyor.

## Veritabanı

İki tablolu SQLite. Şema ve tohum veri `veritabani.py` içinde; dosya yoksa ilk
erişimde kuruluyor (`db_kur()` idempotent), yani depoyu klonlayan biri ek bir
adım yapmadan çalıştırabiliyor.

| Tablo | Sütunlar |
|---|---|
| `kitaplar` | `id`, `baslik`, `yazar`, `akim`, `yil`, `fiyat`, `stok` |
| `siparisler` | `id`, `kod`, `musteri`, `kitap_id` → `kitaplar(id)`, `adet`, `birim_fiyat`, `toplam`, `durum`, `olusturma` |

Bilinçli kararlar:

- **`CHECK (stok >= 0)`** — uygulama katmanı hata yapsa bile veritabanı negatif
  stoğa izin vermiyor. Aynı şekilde `CHECK (adet > 0)`, `CHECK (fiyat > 0)`.
- **Yazma işlemleri `BEGIN IMMEDIATE` içinde.** Stok kontrolü ile stok düşümü
  arasına başka bir istek giremiyor; aynı kitabın son kopyası iki kişiye
  satılmıyor. Gradio istekleri ayrı thread'lerde koştuğu için her çağrı kendi
  bağlantısını açıp kapatıyor (SQLite bağlantıları thread'ler arasında
  paylaşılmaz).
- **Sipariş kodu (`SIP-1001`, `SIP-1002`, ...) yalnızca veritabanında üretiliyor.**
  Modelin uyduramayacağı tek anahtar bu; `get_order_status` sadece gerçekten
  üretilmiş bir kodu tanıyor.
- **Tohum katalogda kasıtlı kenar durumlar var:** stoğu 0 olan bir kitap
  (*Varlık ve Hiçlik*) ve stoğu 2 olan bir kitap (*Varlık ve Zaman*). Böylece
  "tükendi" ve "yetersiz stok" yolları canlı demoda da gösterilebiliyor.
- **Yıl negatifse M.Ö.** (`-375` → "M.Ö. 375"). Platon ve Aristoteles için.

## Araçlar

İkisi okuyor, ikisi yazıyor:

| Araç | Yön | Ne yapar |
|---|---|---|
| `search_books(query?, author?, max_price?, only_in_stock?)` | okur | Katalog + stok araması; her kitabın `kitap_id`'sini döndürür |
| `create_order(book_id, quantity=1, customer_name?)` | **yazar** | Siparişi kaydeder, stoktan düşer, sipariş kodu üretir |
| `get_order_status(order_code)` | okur | Sipariş kodundan durum, kitap, adet, tutar |
| `cancel_order(order_code)` | **yazar** | Siparişi iptal eder, kitapları stoğa geri ekler |

Araç adları İngilizce, açıklamalar ve dönen alan adları Türkçe — 7. haftadaki
gerekçenin aynısı: modeller function calling'i eğitim verisinde İngilizce
adlarla görüyor, açıklamalar ise Türkçe soruların hangi araca gideceğini
belirlediği için Türkçe daha isabetli.

`create_order` yalnızca **veritabanındaki bir `kitap_id`** ile çalışıyor. Model
kitabı kendi belleğinden uydurup sipariş veremiyor; önce `search_books` çağırıp
gerçek id'yi almak zorunda. Ödevin istediği **iki turlu zincir** buradan doğuyor
ve 7. haftadaki `convert_temperature` numarasının aksine bu sefer zorlama
gerekmedi — çünkü uydurma id ile araç zaten hata döndürüyor, model tahmine
mecbur kalmıyor.

Araç hataları exception olarak yukarı fırlatılmıyor: `araci_calistir()` her zaman
bir sözlük döndürüyor, hata durumunda `{"hata": "..."}`. Model bunu okuyup
kullanıcıya aktarabiliyor (stok yetersiz, sipariş bulunamadı, teslim edilmiş
sipariş iptal edilemez...). Bütün hata yolları `araclar.py`'yi tek başına
çalıştırınca test ediliyor.

## Halüsinasyon engelleme

Ödevin en kritik maddesi: *"Model veritabanında olmayan bir ürünü/bilgiyi varmış
gibi sunmamalıdır."* Bunu üç katmanda ele aldım.

**1. Sistem promptu.** "Katalog bilgisini kendi belleğinden verme", "listede
olmayan kitabı önerme", adım adım çözülmüş bir sipariş örneği.

**2. Araç çıktısına gömülü not.** Her `search_books` sonucunun içinde
`"not": "kitapçının stoğu yalnızca bu listedekilerden ibarettir..."` alanı var —
kural, modelin tam o anda baktığı yere yazılıyor. Ayrıca her kitapta ham `stok`
sayısının yanında `stok_durumu` alanı var (`"stokta"` / `"tükendi (katalogda var,
stok yok)"`): `stok: 0` gören model bunu "kitap katalogda yok" diye özetleyip
yerine başka kitap önermeye kalkıyordu, sayıyı yorumlama işini modele bırakmadım.

**3. Döngüdeki guardrail (`_dogrulanmamis_veri`).** İlk ikisi yetmiyor — 7.
haftada da öğrendiğim gibi kuralın prompt'tan harness'a taşınması gerekiyor.
Nihai yanıt kullanıcıya gitmeden önce şunlar denetleniyor:

| # | Kontrol | Neyi yakalar |
|---|---|---|
| 1 | Katalog/sipariş sorusuna **hiç araç çağrılmadan** cevap verilmiş mi? | Modelin kendi belleğinden katalog uydurması |
| 2 | "Siparişiniz alındı" denmiş ama **yazan araç çalışmamış** mı? | Yapılmamış siparişin yapılmış gibi sunulması |
| 3 | Cevaptaki `SIP-...` kodu araç çıktısında geçiyor mu? | Uydurma sipariş kodu |
| 4 | Tırnak içindeki kitap adı araç çıktısında ya da kullanıcının sorusunda geçiyor mu? | Uydurma kitap adı |
| 5 | Cevaptaki TL tutarı araç çıktısındaki fiyatlarla açıklanabiliyor mu? | Uydurma fiyat |

İhlal varsa modelden bir kez düzeltme isteniyor **ve o tur araç çağrısı zorunlu
tutuluyor** (`arac_zorla=True`; yerel arka uçta üretim istemine `<tool_call>`
açılış etiketini biz ekliyoruz, API arka ucunda `tool_choice="required"`).
Müdahale iz panelinde `[!] harness uyarısı` satırı olarak görünüyor.

Model ısrar ederse döngü kilitlenmiyor: cevap veriliyor ama üzerine
**"⚠️ Doğrulanmadı"** uyarısı ekleniyor. Yani doğrulanmamış hiçbir bilgi
kullanıcıya olgu gibi sunulmuyor.

5. kontrolde tutarların birebir eşitliğinin yanı sıra **birim fiyat × adet**
çarpımlarına da izin veriliyor (1-10): kullanıcı "3 tane alsam kaç eder?" diye
sorduğunda model henüz sipariş oluşturmadan doğru hesabı yapabilmeli. Uydurulan
katalog fiyatları bu kapıya takılmaya devam ediyor.

Guardrail'i modele hiç gitmeden test etmek için:

```bash
.venv/bin/python hafta8_veritabani_ajani/ajan.py --guardrail
```

```text
  [OK ] araç çağrılmadı          -> hiç araç çağrılmadan katalog/sipariş bilgisi verildi
  [OK ] alakasız soru, araç yok  -> temiz
  [OK ] uydurma kitap adı        -> araç çıktısında olmayan kitap/ifade: “Yaratıcılık ve Hiçlik”
  [OK ] gerçek kitap adı         -> temiz
  [OK ] tırnakta durum adı       -> temiz
  [OK ] yapılmamış sipariş iddiası -> sipariş/iptal yapıldığı söylendi ama veritabanına yazan araç çalışmadı
  [OK ] gerçek sipariş           -> temiz
  [OK ] sipariş verilemedi       -> temiz
  [OK ] uydurma sipariş kodu     -> araç çıktısında olmayan sipariş kodu: SIP-4242
  [OK ] uydurma fiyat            -> araç çıktısında olmayan fiyat: 89 TL
  [OK ] birim fiyat x adet       -> temiz

11/11 vaka geçti.
```

Vakalar uydurma değil, **gerçek çalıştırmalardan derlendi**: "uydurma kitap adı"
satırı Qwen2.5-7B'nin, "yapılmamış sipariş iddiası" satırı Qwen2.5-0.5B'nin
canlı olarak yaptığı halüsinasyonların birebir kaydı (aşağıda).

## Örnek çıktı — ödevin istediği akış

`Qwen/Qwen3-Coder-30B-A3B-Instruct`, HF Inference Providers üzerinden
(`TOOL_BACKEND=api`). Terminal çıktısının birebir kopyası:

```text
SORU: Camus'nun kitabı var mı? Varsa 2 tane sipariş ver, adım Yusuf.
--- 2.9s | 2 tur · 2 araç çağrısı · 1 DB yazma · Qwen/Qwen3-Coder-30B-A3B-Instruct

[Tur 1] Araç Çağrıları:
   -> search_books(author='Camus')
   <- {"bulunan": 1, "kitaplar": [{"kitap_id": 6, "baslik": "Sisifos Söyleni", "yazar":
      "Albert Camus", "akim": "absürdizm", "yil": "1942", "fiyat_tl": 110.0, "stok": 7,
      "stok_durumu": "stokta"}], "not": "kitapçının stoğu yalnızca bu listedekilerden
      ibarettir; ... Sipariş vermek için buradaki kitap_id değerini create_order aracına ver"}

[Tur 2] Araç Çağrıları:
   -> create_order(book_id=6, customer_name='Yusuf', quantity=2) [DB YAZMA]
   <- {"siparis_kodu": "SIP-1003", "musteri": "Yusuf", "kitap": "Sisifos Söyleni", "yazar":
      "Albert Camus", "adet": 2, "birim_fiyat_tl": 110.0, "toplam_tl": 220.0, "durum":
      "hazırlanıyor", "kalan_stok": 5, "siparis_tarihi": "2026-08-04 09:40:59", "not": "bu
      sipariş veritabanına kaydedildi; kullanıcıya sipariş kodunu bu çıktıdaki haliyle bildir"}

[Tur 3] Nihai Yanıt:
Siparişiniz alındı: 2 adet Sisifos Söyleni, 220 TL. Kodunuz SIP-1003.
```

Sipariş kodu ve tutar modelin uydurması değil, `create_order` çıktısı. Aynı anda
veritabanında *Sisifos Söyleni*'nin stoğu 7'den 5'e düşüyor:

```text
STOK (etkilenenler):
  #6 Sisifos Söyleni: stok=5
SIPARISLER:
  SIP-1003 Yusuf Sisifos Söyleni x2 220.0 TL hazırlanıyor
  SIP-1002 Elif D. Sisifos Söyleni x2 220.0 TL kargoda
  SIP-1001 Yusuf K. Böyle Buyurdu Zerdüşt x1 145.0 TL teslim edildi
```

Ardından iptal (ikinci yazma aracı) stoğu geri veriyor — 5 → 7:

```text
SORU: SIP-1003 numaralı siparişimi iptal et
--- 1.4s | 1 tur · 1 araç çağrısı · 1 DB yazma

[Tur 1] Araç Çağrıları:
   -> cancel_order(order_code='SIP-1003') [DB YAZMA]
   <- {"siparis_kodu": "SIP-1003", "kitap": "Sisifos Söyleni", "adet": 2, "onceki_durum":
      "hazırlanıyor", "durum": "iptal edildi", "iade_edilen_tutar_tl": 220.0, "kalan_stok": 7}

[Tur 2] Nihai Yanıt:
SIP-1003 numaralı siparişiniz iptal edildi. İade edilen tutar: 220.00 TL. Kitap stoğa geri eklendi.
```

Diğer soru tiplerinde canlı davranış (aynı model, aynı oturum):

| Soru | Çağrılan araçlar | Süre | Sonuç |
|---|---|---|---|
| "Simülakrlar ve Simülasyon kaç para?" | `search_books` (0 sonuç) | 1.2 sn | "kitapçımızda yok" — benzer kitap uydurmadı |
| "SIP-1002 numaralı siparişim ne durumda?" | `get_order_status` | 1.8 sn | kargoda, 2 adet, 220 TL |
| "Varlık ve Hiçlik'ten bir tane istiyorum." | `search_books` | 1.3 sn | "katalogda var ancak tükenmiş" |
| "150 TL altındaki kitapları listeler misin?" | `search_books(max_price=150)` | 4.2 sn | 7 kitap, hepsi gerçek fiyatlarıyla |
| "Fransa'nın başkenti neresi?" | (yok — kitapçılık dışı) | 1.1 sn | "Paris" + ne yapabileceğini söyledi |

## Guardrail'in canlı yakaladıkları

Bu bölüm geliştirme sırasında modellerin gerçekten yaptığı halüsinasyonlar.

**1. Uydurma kitap (`Qwen2.5-7B-Instruct`).** *"Varlık ve Hiçlik'ten bir tane
istiyorum"* sorusunda araç `stok: 0` döndürdü; model bunu "kitap yok" diye
özetleyip **katalogda hiç olmayan bir kitap önerdi**:

```text
Kitapçıda Varlık ve Hiçlik adlı kitabı bulunmamaktadır. Ancak benzer bir kitap
öneriyorum: "Yaratıcılık ve Hiçlik" adlı eser Jean-Paul Sartre yazarı tarafından
yazılmıştır. Bu kitabın fiyatı 420 TL ve stokta bulunmamaktadır.
```

Bunun üzerine iki şey yaptım: (a) araç çıktısına `stok_durumu` alanını ekledim,
(b) guardrail'e 4. kontrolü (tırnak içindeki kitap adı) ekledim. Aynı soruyu
`Qwen3-Coder-30B` doğru cevaplıyor: *"Katalogda var ancak tükendiğini
görebiliyorum."*

**2. Uydurma sipariş kodu (`Qwen2.5-7B-Instruct`).** Sipariş isteğinde
`search_books` çağırıp `create_order`'ı atladı ve kodu kendisi yazdı; 3. kontrol
yakaladı:

```text
[!] harness uyarısı: araç çıktısında olmayan sipariş kodu: SIP-1001;
    araç çağrısı zorunlu tutularak düzeltme istendi
```

**3. Yapılmamış siparişin yapılmış gibi sunulması (`Qwen2.5-0.5B-Instruct`).**
En tehlikelisi — kullanıcı olmayan bir siparişi beklemeye başlar. 0.5B
`create_order`'ı hiç çağırmadan "2 adet sipariş verildi" yazdı. Bu vaka için 2.
kontrolü ekledim. Guardrail bir araç çağrısını zorluyor ama 0.5B **doğru** aracı
seçemiyor (yine `search_books` çağırdı), bu yüzden ısrar hâlinde cevabın
uyarıyla işaretlenmesi devreye giriyor:

```text
[!] harness uyarısı: sipariş/iptal yapıldığı söylendi ama veritabanına yazan araç
    çalışmadı; düzeltme istendi, model ısrar etti — cevap uyarıyla işaretlendi

[Tur 3] Nihai Yanıt:
Kitap "Sisifos Söyleni" için 2 adet sipariş verildi.

⚠️ **Doğrulanmadı:** sipariş/iptal yapıldığı söylendi ama veritabanına yazan araç
çalışmadı. Yukarıdaki cevabın bu kısmı kitapçının veritabanından teyit edilemedi,
lütfen esas almayın.
```

Guardrail'in sınırı burada net görünüyor: **bir araç çağrısını zorlayabiliyor
ama doğru aracı seçtiremiyor.** Model kapasitesinin altına inildiğinde
yapılabilecek en iyi şey, yanlış bilgiyi olgu gibi sunmamak.

## Çalıştırma

```bash
uv pip install --python .venv/bin/python -r hafta8_veritabani_ajani/requirements.txt

.venv/bin/python hafta8_veritabani_ajani/veritabani.py --sifirla  # DB'yi kur + içeriğini dök
.venv/bin/python hafta8_veritabani_ajani/araclar.py               # 4 aracı + hata yollarını dene
.venv/bin/python hafta8_veritabani_ajani/ajan.py --guardrail      # guardrail testi (modelsiz)
.venv/bin/python hafta8_veritabani_ajani/ajan.py                  # ödevin örnek akışı
.venv/bin/python hafta8_veritabani_ajani/app.py                   # Gradio arayüzü (localhost:7860)
```

GPU'suz makinede 7B çok yavaş olur; döngüyü denemek için küçük bir model seçin
(kalitesi düşük olur, sadece mekanizmayı görmek için):

```bash
TOOL_MODEL=Qwen/Qwen2.5-0.5B-Instruct \
  .venv/bin/python hafta8_veritabani_ajani/ajan.py "Camus'nun kitabı var mı?"
```

Inference Providers arka ucunu denemek için (yukarıdaki çıktıların üretildiği yol):

```bash
TOOL_BACKEND=api TOOL_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
  .venv/bin/python hafta8_veritabani_ajani/ajan.py
```

Spaces'e yayınlamak:

```bash
.venv/bin/python hafta8_veritabani_ajani/deploy_space.py            # herkese açık
.venv/bin/python hafta8_veritabani_ajani/deploy_space.py --private  # gizli
```

`.env`'den `SPACE_REPO_ID_KITAPCI`, `TOOL_BACKEND`, `TOOL_MODEL` okunur; token
`HF_TOKEN`'dan ya da `hf auth login` cache'inden alınıp Space secret'ı olarak
yazılır. Space, 7. haftadaki gibi **ZeroGPU** donanımında açılıyor (ücretsiz
`cpu-basic` artık Gradio Space barındırmıyor; detay 7. haftanın README'sinde).

## Arayüz

Üç panel yan yana: solda sohbet, ortada araç çağrı izi (yazma çağrıları
`[DB YAZMA]` etiketli), sağda **veritabanının canlı hali** — `kitaplar` ve
`siparisler` tabloları her adımda tazeleniyor. Sipariş verildiğinde stok
sütununun düştüğü aynı ekranda görülüyor, yani "veri gerçekten yazılıyor mu"
sorusu göz önünde cevaplanıyor. **Veritabanını sıfırla** düğmesi katalogu
başlangıç durumuna döndürüyor (demo temizliği).

## Notlar / karşılaşılan kısıtlar

- **Space'te kalıcı disk yoksa veritabanı geçicidir.** `veritabani.py` kalıcı
  disk (`/data`) varsa oraya, yoksa uygulama klasörüne yazıyor; ikinci durumda
  Space yeniden başlatıldığında siparişler sıfırlanır (katalog tohum veriden
  yeniden kurulur). Yol `KITAPCI_DB` ile de verilebilir.
- **`tool_choice="required"` her sağlayıcıda çalışmıyor.** Guardrail'in düzeltme
  turunda HF router `422 – "grammar is not valid: failed to compile grammar"`
  döndürdü. `modeller.py`'ye (7. hafta) bunun için bir geri düşüş eklendi: 422
  gelirse aynı istek `tool_choice="auto"` ile tekrarlanıyor — zorlamayı
  kaybediyoruz ama düzeltme turu hiç çalışmamasından iyi.
- **Inference Providers kredisi** geliştirme sırasında birkaç kez tükendi;
  ayrıca `Qwen/Qwen2.5-7B-Instruct` bir noktada *"not supported by any provider
  you have enabled"* dönmeye başladı. Bu yalnızca `TOOL_BACKEND=api` yolunu
  etkiliyor, ZeroGPU'da model Space'in içinde koştuğu için Space bundan
  etkilenmiyor.
- **Sipariş başına tek kitap.** Sepet (çok kalemli sipariş) tabloyu üçe
  çıkarırdı; ödevin kapsamı için tek kalem yeterli ve araç şeması sade kalıyor,
  model daha az hata yapıyor.
- **Sohbet geçmişine araç mesajları taşınmıyor** — her yeni soru kendi araç
  zincirini kuruyor, yalnızca kullanıcı/asistan metinleri bağlam olarak gidiyor
  (7. haftadaki tercihin aynısı).
- **Greedy decoding** (`do_sample=False`) yerel arka uçta: aynı soru aynı aracı
  çağırsın.
