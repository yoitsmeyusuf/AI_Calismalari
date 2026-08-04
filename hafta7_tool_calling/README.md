# 7. Hafta — Tool Calling (Function Calling) + Hugging Face Spaces

**Amaç:** Bir modelin dış veri kaynağıyla konuşmasını sağlamak. Herkese açık,
anahtarsız bir API ([Open-Meteo](https://open-meteo.com)) için JSON tool şemaları
yazıp, modelin gelen soruya göre doğru aracı kendisi seçmesini, gerektiğinde
araçları zincirlemesini ve **arka planda hangi aracı hangi argümanlarla çağırdığını
kullanıcıya açıkça göstermesini** sağlamak; sonucu Gradio ile HF Spaces'te yayına
almak.

**Canlı demo:** [huggingface.co/spaces/yoitsmeyusuf/tool-calling-hava-durumu](https://huggingface.co/spaces/yoitsmeyusuf/tool-calling-hava-durumu)
— ücretsiz **ZeroGPU** (H200 dilimi) üzerinde, model Space'in içinde çalışıyor.

## Dosyalar

| Dosya | Ne işe yarar |
|---|---|
| `araclar.py` | 4 aracın implementasyonu + modele verilen JSON şemaları (`ARAC_SEMALARI`). Tek başına çalıştırılınca tüm araçları (hata yolları dahil) test eder. |
| `modeller.py` | Model katmanı: ZeroGPU/transformers arka ucu (`YerelModel`) ve HF Inference Providers arka ucu (`ApiModel`), `<tool_call>` ayrıştırıcısı. Tek başına çalıştırılınca ayrıştırıcıyı test eder. |
| `ajan.py` | Tool calling döngüsü + şeffaf iz (`TurIzi`/`AracIzi`) + `izi_metne_cevir()`. |
| `app.py` | Gradio arayüzü (Space giriş dosyası): solda sohbet, sağda canlı araç çağrı izi. Döngüyü `@spaces.GPU` ile sarar. |
| `deploy_space.py` | ZeroGPU'lu Space'i oluşturur, kartı araç şemalarından üretir, dosyaları yükler, değişken/secret ayarlar. |
| `requirements.txt` | Space'in kuracağı paketler (torch pinlenmez, ZeroGPU imajı sağlar). |

## Seçilen API ve araçlar

Open-Meteo'yu seçtim: ücretsiz, kayıt/anahtar istemiyor, hem anlık hem tahmin hem
hava kalitesi için **üç ayrı endpoint** veriyor — yani tek bir veri kaynağıyla
modelin "hangi aracı seçmeli" kararını gerçekten sınayabiliyorum.

| Araç | Ne yapar | Kaynak |
|---|---|---|
| `get_weather(city)` | Anlık sıcaklık (°C), nem, rüzgâr, gökyüzü durumu | Forecast API (`current`) |
| `get_forecast(city, days=3)` | Günlük min/max sıcaklık, yağış, gökyüzü (1-7 gün) | Forecast API (`daily`) |
| `get_air_quality(city)` | Avrupa AQI + PM2.5 / PM10 / NO2 / ozon | Air Quality API |
| `convert_temperature(value, to_unit, from_unit="C")` | C ↔ F ↔ K çevirimi | yerel, saf matematik |

Şehir adı → koordinat dönüşümünü (Geocoding API) araçlar kendi içinde yapıyor,
modele ayrı bir araç olarak açmadım: model için gereksiz bir tur demek olurdu.
Sonuçlar process içinde cache'leniyor, aynı şehir için üst üste gelen çağrılarda
API'ye tekrar gidilmiyor.

`convert_temperature` bilinçli olarak **ayrı bir araç**: sıcaklık çevirimini
modelin kafadan yapmasına izin vermek yerine, ödevdeki örnek akışta olduğu gibi
iki turlu bir zincir kurmasını istiyorum. Bunu modele kabul ettirmek bu ödevin en
uğraştıran kısmı oldu (aşağıya bakın).

Araç adları İngilizce (`get_weather`), açıklamalar ve dönen alan adları Türkçe.
Modeller function calling'i eğitim verisinde İngilizce adlarla görüyor; açıklamalar
ise Türkçe soruların hangi araca gittiğini belirlediği için Türkçe daha isabetli.

## Modeli nerede çalıştırıyoruz: ZeroGPU

Space, HF'nin ücretsiz **ZeroGPU** donanımında (`zero-a10g` etiketi, pratikte bir
H200 dilimi) koşuyor; model `transformers` ile **Space'in içinde** çalışıyor.
Böylece Inference Providers kredisine hiç ihtiyaç kalmıyor.

ZeroGPU'da GPU yalnızca istek süresince ayrılıyor. Bu yüzden:

- Model ağırlıkları **Space açılışında** (GPU ayrılmadan) indirilip belleğe
  alınıyor (`app.py` içinde `SOHBET_MODELI.yukle()`), ilk istek 15 GB indirmeye
  takılmıyor.
- Bütün ajan döngüsü **tek bir `@spaces.GPU` çağrısının içinde** dönüyor. Her
  model turu için ayrı ayrı GPU kuyruğuna girmek yerine GPU bir kez alınıyor;
  dekoratör bir generator'ı sardığı için adımlar yine oluştukça ekrana basılıyor.
- `spaces` paketi yerelde kurulu olmayabilir; `app.py` bu durumda dekoratörü
  işlevsiz bir geçişe çeviriyor, yerel çalıştırma bozulmuyor.

Model katmanı (`modeller.py`) ajandan ayrı: `TOOL_BACKEND=yerel` (varsayılan,
ZeroGPU/transformers) veya `TOOL_BACKEND=api` (HF Inference Providers). Ajan
döngüsü ikisini de aynı arayüzle (`tamamla()`) konuşuyor, mesajları arka uçtan
bağımsız kanonik biçimde tutuyor.

Yerel arka uçta araç çağrıları modelin **kendi sohbet şablonuyla** kuruluyor:
Qwen2.5 şablonu araç şemalarını `<tools>` bloğuna gömüyor, model çağrıyı
`<tool_call>{"name": ..., "arguments": {...}}</tool_call>` olarak üretiyor,
`arac_cagrilarini_ayikla()` bunu ayrıştırıyor (bozuk JSON ve kapanmamış etiket
dahil — `modeller.py`'yi tek başına çalıştırınca bu vakalar test ediliyor).

## Zincir kurdurmak: prompt yetmedi, guardrail gerekti

Ödevin örnek sorusu (*"...Fahrenheit olarak kaç eder?"*) iki turlu zincir
gerektiriyor. Modeller bunu yapmak yerine sıcaklığı çekip **çevirimi kendileri
hesaplamayı** tercih ediyor. Sırayla denediklerim:

| Deneme | Sonuç |
|---|---|
| 1. Sistem promptuna "çevirimi kafadan hesaplama" kuralı | ❌ `Qwen2.5-7B-Instruct` yine kendi hesapladı |
| 2. Araç çıktısına not eklemek (`"not": "...convert_temperature'a ver, kendin hesaplama"`) — modelin tam o anda baktığı yere | ❌ değişmedi |
| 3. Sistem promptuna adım adım çözülmüş örnek (few-shot) | ❌ değişmedi |
| 4. Daha büyük model: `Qwen2.5-14B-Instruct` | ❌ o da atladı (boyut sorunu değil) |
| 5. Kuralı döngüye taşımak: **ihlali yakala + araç çağrısını zorunlu tut** | ✅ zincir kuruluyor |

5. adımın işleyişi (`ajan.py`):

- `_cevirim_atlandi_mi()` üç koşulu birlikte arıyor: kullanıcı F/K istemiş,
  cevapta F/K sayısı geçiyor, ama `convert_temperature` hiç çağrılmamış.
- İhlal varsa modele bir düzeltme mesajı gidiyor **ve o tur araç çağrısı zorunlu
  tutuluyor** (`arac_zorla=True`).
- Zorlama, yerel arka uçta üretim istemine `<tool_call>` açılış etiketini bizim
  eklememizle oluyor: model metin yazamıyor, bir araç çağrısı tamamlamak zorunda —
  ama hangi aracı ve hangi argümanları kullanacağını yine kendisi seçiyor. Bu,
  API'lerdeki `tool_choice="required"`nin yerel karşılığı; `ApiModel` de zaten
  o parametreyi kullanıyor.
- Düzeltme **bir kez** isteniyor; model ısrar ederse döngü kilitlenmiyor, cevap
  olduğu gibi veriliyor. `MAKS_TUR` (4) ayrı bir üst sınır.
- Guardrail devreye girdiğinde iz panelinin başında `[!] harness uyarısı: ...`
  satırı görünüyor — yani müdahale de kullanıcıdan gizlenmiyor.

Sadece ricayla (zorlama olmadan) 7B'nin düzeltmeyi de yaptığı görülmedi; zorlama
şart oldu. Notlar: `Qwen3-Coder-30B-A3B-Instruct` Inference Providers üzerinden
zinciri **kendiliğinden** doğru kuruyordu, yani bu bir küçük-model davranışı.
ZeroGPU'da 7B'yi tercih ettim: soğuk açılışta indirilecek ağırlık daha az ve yanıt
süresi ~5-15 sn (14B ~25 sn), doğruluk zaten guardrail ile garanti.

## Diğer tasarım kararları

- **Araç hataları modele geri veriliyor**, exception olarak yukarı fırlatılmıyor:
  `araci_calistir()` her zaman bir sözlük döndürür, hata durumunda
  `{"hata": "..."}`. Şehir adı yanlışsa model bunu okuyup kullanıcıya aktarabiliyor.
- **`<think>` blokları ayrıştırılıyor.** Qwen3 gibi modeller muhakemeyi cevabın
  içine gömüyor; bu kullanıcıya gösterilen yanıttan çıkarılıp iz panelinde
  "model düşüncesi" satırı olarak gösteriliyor.
- **Sohbet geçmişine araç mesajları taşınmıyor** — her yeni soru kendi araç
  zincirini kuruyor, yalnızca kullanıcı/asistan metinleri bağlam olarak gidiyor.
- **Greedy decoding** (`do_sample=False`): aynı soru aynı aracı çağırsın; serinin
  benchmark haftalarındaki tercihle de tutarlı.

## Canlı çıktı

Space'e `gradio_client` ile gönderilen ödev sorusunun birebir çıktısı:

```text
--- 14.8s | 2 tur · 4 araç çağrısı · model: Qwen/Qwen2.5-7B-Instruct

[!] harness uyarısı: convert_temperature çağrılmadan Fahrenheit/Kelvin yazıldı; araç çağrısı zorunlu tutularak düzeltme istendi

[Tur 1] Araç Çağrıları:
   -> get_weather(city='Ankara')
   <- {"sehir": "Ankara", "sicaklik_c": 15.6, "nem_yuzde": 53, "ruzgar_kmh": 3.4, "gokyuzu": "açık", ...}
   -> get_weather(city='London')
   <- {"sehir": "Londra", "sicaklik_c": 23.2, "nem_yuzde": 57, "ruzgar_kmh": 11.9, "gokyuzu": "açık", ...}

[Tur 2] Araç Çağrıları:
   -> convert_temperature(value=15.6, to_unit='F')
   <- {"girdi": 15.6, "girdi_birimi": "C", "sonuc": 60.08, "birim": "F", "birim_adi": "Fahrenheit"}
   -> convert_temperature(value=23.2, to_unit='F')
   <- {"girdi": 23.2, "girdi_birimi": "C", "sonuc": 73.76, "birim": "F", "birim_adi": "Fahrenheit"}

[Tur 3] Nihai Yanıt:
Ankara'nın sıcaklığı 60.1°F ve Londra'nın sıcaklığı 73.8°F'dir.
```

Cevaptaki Fahrenheit değerleri modelin kendi hesabı değil, `convert_temperature`
çıktısı (60.08 / 73.76, yuvarlanmış).

Diğer soru tiplerinde canlı davranış:

| Soru | Çağrılan araçlar | Süre |
|---|---|---|
| "İzmir'de önümüzdeki 3 gün yağmur var mı?" | `get_forecast` | 6.2 sn |
| "Bugün İstanbul'da koşuya çıkmak sağlıklı mı?" | `get_air_quality` | 4.9 sn |
| "Erzurum kaç Kelvin?" | `get_weather` → `convert_temperature` (guardrail'e gerek kalmadan) | 5.1 sn |
| "Fransa'nın başkenti neresi?" | (yok — hava dışı soru, araç çağırmadı) | 2.4 sn |

## Çalıştırma

Yerelde:

```bash
uv pip install --python .venv/bin/python -r hafta7_tool_calling/requirements.txt
uv pip install --python .venv/bin/python torch --index-url https://download.pytorch.org/whl/cpu  # GPU'suz denemek için

.venv/bin/python hafta7_tool_calling/araclar.py     # araçları tek tek dene (canlı API)
.venv/bin/python hafta7_tool_calling/modeller.py    # <tool_call> ayrıştırıcısını dene
.venv/bin/python hafta7_tool_calling/ajan.py        # ödevin örnek sorusu
.venv/bin/python hafta7_tool_calling/app.py         # Gradio arayüzü (localhost:7860)
```

GPU'suz makinede 7B çok yavaş olur; döngüyü denemek için küçük bir model seçin
(kalitesi düşük olur, sadece mekanizmayı görmek için):

```bash
TOOL_MODEL=Qwen/Qwen2.5-0.5B-Instruct .venv/bin/python hafta7_tool_calling/ajan.py "Ankara'da hava nasıl?"
```

Inference Providers arka ucunu denemek için:

```bash
TOOL_BACKEND=api TOOL_MODEL=Qwen/Qwen3-Coder-30B-A3B-Instruct \
  .venv/bin/python hafta7_tool_calling/ajan.py
```

Spaces'e yayınlamak:

```bash
.venv/bin/python hafta7_tool_calling/deploy_space.py            # herkese açık
.venv/bin/python hafta7_tool_calling/deploy_space.py --private  # gizli
```

`.env`'den `SPACE_REPO_ID`, `TOOL_BACKEND`, `TOOL_MODEL` okunur; token
`HF_TOKEN`'dan ya da `hf auth login` cache'inden alınıp Space secret'ı olarak
yazılır (model ağırlıklarını indirmek için gerekiyor).

## Notlar / karşılaşılan kısıtlar

- **Ücretsiz `cpu-basic` artık Gradio Space barındırmıyor.** İlk denemede
  `deploy_space.py` şunu aldı: *"Static Spaces are free for everyone, but hosting
  Gradio and Docker Spaces on free cpu-basic requires a PRO subscription."*
  Çözüm ZeroGPU oldu: `create_repo(..., space_hardware="zero-a10g")` PRO
  olmayan hesapta sorunsuz çalıştı ve üstüne ücretsiz GPU verdi. Yani bu ödev
  için ZeroGPU sadece "daha hızlı" değil, **barındırmanın da tek ücretsiz yolu.**
- **Inference Providers kredisi** yalnızca `TOOL_BACKEND=api` yolunu etkiliyor;
  ücretsiz hesaplarda aylık sınırlı ve bu ödevi geliştirirken model
  karşılaştırması sırasında tükendi. Ajan `402`'yi yakalayıp ne yapılacağını
  yazıyor. Varsayılan `yerel` arka uç kredi harcamadığı için Space bu durumdan
  etkilenmiyor.
- **Bilinmeyen şehir sorulduğunda** (örn. "Xyzqwe") 7B bazen aracı hiç
  çağırmayıp "bilmiyorum" diyor; aracın hata yolu (`{"hata": "... bulunamadı"}`)
  çalışıyor ama modele hiç ulaşmıyor. Araç tarafı `araclar.py` testinde
  doğrulanıyor.
- **Soğuk açılış:** Space uzun süre kullanılmazsa uyuyor; ilk istekte ağırlıklar
  yeniden inip yükleneceği için birkaç dakika sürebilir.
