"""
7. Hafta: Tool Calling icin arac tanimlari ve implementasyonlari.

Veri kaynagi: Open-Meteo (https://open-meteo.com) - tamamen ucretsiz, API
anahtari gerektirmeyen bir public API. Uc ayri endpoint'i kullaniyoruz:
  - Geocoding  : sehir adi -> enlem/boylam  (get_weather / get_forecast /
                 get_air_quality hepsi bunu once dahili olarak cagirir)
  - Forecast   : anlik hava + gunluk tahmin
  - Air Quality: anlik hava kalitesi (European AQI, PM2.5, PM10)

Dorduncu arac (convert_temperature) hicbir agi kullanmaz, saf matematiktir;
odevdeki ornek akista oldugu gibi modelin *iki turlu* arac zinciri kurmasini
saglamak icin var: once sicakligi cekiyor, sonra birimini cevirtiyor.

Modele verilen semalar OpenAI/HF uyumlu "function calling" formatinda
(ARAC_SEMALARI). Arac adlari Ingilizce cunku modeller bu adlandirmayi egitim
verisinde bu sekilde gormus; aciklamalar ve dondurulen alan adlari Turkce
kullanicilarla calisacagi icin Turkce/karma tutuldu.

Tek basina calistirilirsa butun araclari ornek girdilerle test eder:
    .venv/bin/python hafta7_tool_calling/araclar.py
"""
import requests

GEOCODING_URL = "https://geocoding-api.open-meteo.com/v1/search"
FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
AIR_QUALITY_URL = "https://air-quality-api.open-meteo.com/v1/air-quality"

ZAMAN_ASIMI = 15

# WMO hava durumu kodlari -> Turkce karsiliklari
# https://open-meteo.com/en/docs (weather_code alani)
WMO_KODLARI = {
    0: "açık",
    1: "az bulutlu",
    2: "parçalı bulutlu",
    3: "kapalı",
    45: "puslu",
    48: "kırağılı puslu",
    51: "hafif çisenti",
    53: "çisenti",
    55: "yoğun çisenti",
    56: "hafif dondurucu çisenti",
    57: "yoğun dondurucu çisenti",
    61: "hafif yağmurlu",
    63: "yağmurlu",
    65: "şiddetli yağmurlu",
    66: "hafif dondurucu yağmur",
    67: "şiddetli dondurucu yağmur",
    71: "hafif kar",
    73: "kar",
    75: "yoğun kar",
    77: "kar taneli",
    80: "hafif sağanak",
    81: "sağanak",
    82: "şiddetli sağanak",
    85: "hafif kar sağanağı",
    86: "yoğun kar sağanağı",
    95: "gök gürültülü fırtına",
    96: "dolulu gök gürültülü fırtına",
    99: "şiddetli dolulu fırtına",
}

# European AQI bantlari (üst sınır, etiket)
AQI_BANTLARI = [
    (20, "çok iyi"),
    (40, "iyi"),
    (60, "orta"),
    (80, "kötü"),
    (100, "çok kötü"),
]

BIRIMLER = {"C": "Celsius", "F": "Fahrenheit", "K": "Kelvin"}

# Sicaklik donduren araclarin sonucuna eklenen not. Sistem promptundaki ayni
# kural tek basina yetmedi: Qwen2.5-7B-Instruct convert_temperature'i atlayip
# cevirimi kendisi yapiyordu. Notu aracin *cikisina* koymak, modelin tam o anda
# baktigi yere koymak demek ve zinciri guvenilir sekilde kurduruyor.
BIRIM_NOTU = (
    "sıcaklıklar Celsius cinsindendir; kullanıcı Fahrenheit veya Kelvin istediyse "
    "bu değeri convert_temperature aracına ver, kendin hesaplama"
)

_konum_cache: dict[str, dict] = {}


class AracHatasi(Exception):
    """Arac calistirilirken olusan, modele geri bildirilecek hata."""


def _gokyuzu(kod) -> str:
    return WMO_KODLARI.get(kod, f"bilinmeyen (WMO {kod})")


def _aqi_etiketi(aqi) -> str:
    if aqi is None:
        return "bilinmiyor"
    for ust_sinir, etiket in AQI_BANTLARI:
        if aqi <= ust_sinir:
            return etiket
    return "çok kötü (tehlikeli)"


def _istek(url: str, params: dict) -> dict:
    try:
        yanit = requests.get(url, params=params, timeout=ZAMAN_ASIMI)
        yanit.raise_for_status()
        return yanit.json()
    except requests.RequestException as e:
        raise AracHatasi(f"Open-Meteo isteği başarısız: {e}") from e


def _konum_bul(sehir: str) -> dict:
    """Sehir adini enlem/boylama cevirir (Open-Meteo Geocoding API).

    Sonuclar process icinde cache'lenir; ayni sehir icin ust uste gelen
    get_weather + get_air_quality cagrilarinda API'ye tekrar gidilmez.
    """
    anahtar = sehir.strip().lower()
    if anahtar in _konum_cache:
        return _konum_cache[anahtar]

    veri = _istek(
        GEOCODING_URL,
        {"name": sehir, "count": 1, "language": "tr", "format": "json"},
    )
    sonuclar = veri.get("results") or []
    if not sonuclar:
        raise AracHatasi(
            f"'{sehir}' adlı bir yer bulunamadı. Şehir adını kontrol edip tekrar dene."
        )

    ilk = sonuclar[0]
    konum = {
        "sehir": ilk["name"],
        "ulke": ilk.get("country", ""),
        "enlem": round(ilk["latitude"], 4),
        "boylam": round(ilk["longitude"], 4),
    }
    _konum_cache[anahtar] = konum
    return konum


# --------------------------------------------------------------------------
# Araclar
# --------------------------------------------------------------------------


def get_weather(city: str) -> dict:
    """Bir sehrin anlik hava durumu (Open-Meteo Forecast API)."""
    konum = _konum_bul(city)
    veri = _istek(
        FORECAST_URL,
        {
            "latitude": konum["enlem"],
            "longitude": konum["boylam"],
            "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
            "timezone": "auto",
        },
    )
    anlik = veri["current"]
    return {
        "sehir": konum["sehir"],
        "ulke": konum["ulke"],
        "sicaklik_c": anlik["temperature_2m"],
        "nem_yuzde": anlik["relative_humidity_2m"],
        "ruzgar_kmh": anlik["wind_speed_10m"],
        "gokyuzu": _gokyuzu(anlik["weather_code"]),
        "olcum_zamani": anlik["time"],
        "not": BIRIM_NOTU,
    }


def get_forecast(city: str, days: int = 3) -> dict:
    """Bir sehrin gunluk hava tahmini (min/max sicaklik + yagis)."""
    try:
        gun = int(days)
    except (TypeError, ValueError):
        raise AracHatasi("days tam sayı olmalı (1-7).") from None
    if not 1 <= gun <= 7:
        raise AracHatasi("days 1 ile 7 arasında olmalı.")

    konum = _konum_bul(city)
    veri = _istek(
        FORECAST_URL,
        {
            "latitude": konum["enlem"],
            "longitude": konum["boylam"],
            "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weather_code",
            "forecast_days": gun,
            "timezone": "auto",
        },
    )
    gunluk = veri["daily"]
    return {
        "sehir": konum["sehir"],
        "gunler": [
            {
                "tarih": gunluk["time"][i],
                "en_yuksek_c": gunluk["temperature_2m_max"][i],
                "en_dusuk_c": gunluk["temperature_2m_min"][i],
                "yagis_mm": gunluk["precipitation_sum"][i],
                "gokyuzu": _gokyuzu(gunluk["weather_code"][i]),
            }
            for i in range(len(gunluk["time"]))
        ],
        "not": BIRIM_NOTU,
    }


def get_air_quality(city: str) -> dict:
    """Bir sehrin anlik hava kalitesi (Open-Meteo Air Quality API)."""
    konum = _konum_bul(city)
    veri = _istek(
        AIR_QUALITY_URL,
        {
            "latitude": konum["enlem"],
            "longitude": konum["boylam"],
            "current": "european_aqi,pm10,pm2_5,nitrogen_dioxide,ozone",
            "timezone": "auto",
        },
    )
    anlik = veri["current"]
    aqi = anlik.get("european_aqi")
    return {
        "sehir": konum["sehir"],
        "avrupa_aqi": aqi,
        "aqi_yorumu": _aqi_etiketi(aqi),
        "pm2_5": anlik.get("pm2_5"),
        "pm10": anlik.get("pm10"),
        "no2": anlik.get("nitrogen_dioxide"),
        "ozon": anlik.get("ozone"),
        "olcum_zamani": anlik["time"],
    }


def convert_temperature(value: float, to_unit: str, from_unit: str = "C") -> dict:
    """Sicaklik birimi cevirir (C/F/K). Ag erisimi yok, saf matematik."""
    try:
        deger = float(value)
    except (TypeError, ValueError):
        raise AracHatasi("value sayısal olmalı.") from None

    kaynak = (from_unit or "C").strip().upper()[:1]
    hedef = (to_unit or "").strip().upper()[:1]
    for birim in (kaynak, hedef):
        if birim not in BIRIMLER:
            raise AracHatasi(f"Bilinmeyen birim: {birim!r}. Geçerli birimler: C, F, K.")

    # Once Celsius'a normalize et, sonra hedefe cevir.
    if kaynak == "C":
        celsius = deger
    elif kaynak == "F":
        celsius = (deger - 32) * 5 / 9
    else:
        celsius = deger - 273.15

    if hedef == "C":
        sonuc = celsius
    elif hedef == "F":
        sonuc = celsius * 9 / 5 + 32
    else:
        sonuc = celsius + 273.15

    return {
        "girdi": deger,
        "girdi_birimi": kaynak,
        "sonuc": round(sonuc, 2),
        "birim": hedef,
        "birim_adi": BIRIMLER[hedef],
    }


# --------------------------------------------------------------------------
# Modele verilen JSON semalari (Tool / Function Definition)
# --------------------------------------------------------------------------

ARAC_SEMALARI = [
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": (
                "Bir şehrin ŞU ANDAKİ hava durumunu Open-Meteo'dan çeker: sıcaklık "
                "(Celsius), nem, rüzgâr hızı ve gökyüzü durumu. Kullanıcı bir şehrin "
                "anlık havasını, sıcaklığını veya iki şehrin havasını karşılaştırmayı "
                "sorduğunda bu aracı çağır. Sıcaklığı yalnızca Celsius döndürür; "
                "Fahrenheit veya Kelvin isteniyorsa sonucu convert_temperature'a ver."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {
                        "type": "string",
                        "description": "Şehir adı, örn. 'Ankara', 'London', 'İzmir'.",
                    }
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_forecast",
            "description": (
                "Bir şehir için GÜNLÜK hava tahmini döndürür: her gün için en yüksek/en "
                "düşük sıcaklık (Celsius), toplam yağış ve gökyüzü durumu. 'yarın', "
                "'hafta sonu', 'önümüzdeki 3 gün', 'yağmur yağacak mı' gibi ileriye "
                "dönük sorularda bu aracı kullan, get_weather'ı değil."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Şehir adı."},
                    "days": {
                        "type": "integer",
                        "description": "Kaç günlük tahmin isteniyor (1-7). Belirtilmezse 3.",
                        "minimum": 1,
                        "maximum": 7,
                    },
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_air_quality",
            "description": (
                "Bir şehrin anlık hava kalitesini döndürür: Avrupa Hava Kalitesi "
                "İndeksi (AQI) ve PM2.5, PM10, NO2, ozon değerleri. Hava kirliliği, "
                "'dışarıda spor yapmak sağlıklı mı', polen/toz gibi sorularda çağır."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Şehir adı."},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "convert_temperature",
            "description": (
                "Bir sıcaklık değerini birimler arasında çevirir (C, F, K). Kullanıcı "
                "Fahrenheit veya Kelvin istediğinde, önce get_weather/get_forecast ile "
                "Celsius değeri al, sonra o değeri bu araca ver. Çevirimi kendi başına "
                "hesaplamaya çalışma, her zaman bu aracı kullan."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "value": {"type": "number", "description": "Çevrilecek sıcaklık değeri."},
                    "to_unit": {
                        "type": "string",
                        "enum": ["C", "F", "K"],
                        "description": "Hedef birim.",
                    },
                    "from_unit": {
                        "type": "string",
                        "enum": ["C", "F", "K"],
                        "description": "Kaynak birim. Belirtilmezse C kabul edilir.",
                    },
                },
                "required": ["value", "to_unit"],
            },
        },
    },
]

ARAC_TABLOSU = {
    "get_weather": get_weather,
    "get_forecast": get_forecast,
    "get_air_quality": get_air_quality,
    "convert_temperature": convert_temperature,
}


def araci_calistir(ad: str, argumanlar: dict) -> dict:
    """Model tarafindan istenen araci calistirir.

    Hata durumunda exception firlatmaz; modelin okuyup toparlanabilecegi bir
    {"hata": "..."} sozlugu dondurur (tool_calling'de hatayi modele geri
    beslemek, zinciri kirmaktan iyidir).
    """
    fonksiyon = ARAC_TABLOSU.get(ad)
    if fonksiyon is None:
        return {"hata": f"'{ad}' adlı bir araç yok. Mevcut araçlar: {list(ARAC_TABLOSU)}"}
    try:
        return fonksiyon(**(argumanlar or {}))
    except AracHatasi as e:
        return {"hata": str(e)}
    except TypeError as e:
        return {"hata": f"'{ad}' için geçersiz argümanlar: {e}"}
    except Exception as e:  # beklenmeyen durumda da zinciri kirmiyoruz
        return {"hata": f"'{ad}' çalıştırılırken beklenmeyen hata: {type(e).__name__}: {e}"}


if __name__ == "__main__":
    import json

    ornekler = [
        ("get_weather", {"city": "Ankara"}),
        ("get_forecast", {"city": "İzmir", "days": 3}),
        ("get_air_quality", {"city": "London"}),
        ("convert_temperature", {"value": 27, "to_unit": "F"}),
        ("convert_temperature", {"value": 300, "to_unit": "C", "from_unit": "K"}),
        ("get_weather", {"city": "Xyzqwe"}),  # hata yolu
        ("bilinmeyen_arac", {}),  # hata yolu
    ]
    for ad, argumanlar in ornekler:
        print(f"-> {ad}({argumanlar})")
        print(f"<- {json.dumps(araci_calistir(ad, argumanlar), ensure_ascii=False)}\n")
