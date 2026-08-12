"""Modelin cagirabilecegi 5 arac.

Her arac sade bir Python fonksiyonudur ve METIN dondurur; bu metin modele
geri beslenir. Hicbiri hata firlatmaz — hata olursa Turkce bir aciklama doner,
boylece sohbet dongusu cokmez.

TOOL_SCHEMAS listesi ise modele "elinde su araclar var" demenin JSON halidir.
"""

import html
import re

import requests

import code_rag
import ollama_client

TIMEOUT = 20
HEADERS = {"User-Agent": "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7)"}

# Wikimedia, tarayici taklidi User-Agent'lari 403'luyor (UA politikasi: kendini
# tanitan bir istemci istiyor). Ustteki HEADERS ile yedek arama sessizce
# "Arama yapilamadi" donuyordu; Wikipedia'ya ayri bir UA ile gidiyoruz.
WIKI_HEADERS = {"User-Agent": "hafta11-ollama-asistan/1.0 (egitim odevi; ollama + chromadb)"}

# WMO hava durumu kodlarindan Turkce aciklamaya (Open-Meteo bu kodlari kullanir).
WMO_TR = {
    0: "acik", 1: "az bulutlu", 2: "parcali bulutlu", 3: "cok bulutlu",
    45: "sisli", 48: "kirragi sisi", 51: "hafif ciseleme", 53: "ciseleme",
    55: "yogun ciseleme", 61: "hafif yagmur", 63: "yagmurlu", 65: "kuvvetli yagmur",
    71: "hafif kar", 73: "kar yagisli", 75: "yogun kar", 77: "kar taneli",
    80: "saganak", 81: "kuvvetli saganak", 82: "siddetli saganak",
    85: "kar saganagi", 86: "yogun kar saganagi",
    95: "gok gurultulu firtina", 96: "dolulu firtina", 99: "siddetli dolulu firtina",
}

# chat.py bunu ayarlar: hangi embedding modeliyle arama yapilacagi.
ACTIVE_EMBED_KEY = ollama_client.DEFAULT_EMBED

# Kod hatasi gibi gorunen sorulari ayirmak icin kaba bir isaret listesi.
KOD_ISARETLERI = (
    "error", "exception", "traceback", "hatasi", "hatası", "stack trace",
    "undefined", "nullreference", "segmentation fault", "compile", "derleme",
    "npm", "pip", "import", "syntax",
)


def internet_search(query: str, max_results: int = 5) -> str:
    """DuckDuckGo'nun sade (lite) arayuzunde arama yapar. API anahtari gerekmez."""
    return _ddg_search(query, max_results)


def code_error_fix_search(query: str, max_results: int = 5) -> str:
    """Bir kod hata mesajinin cozumunu arar.

    Sorgunun basina hangi kaynakta aranacagini soyleyen bir ipucu ekleriz:
    C# icin Microsoft Docs, digerleri icin Stack Overflow. Ham hata metnini
    aramak cok gurultulu sonuc veriyor.
    """
    lowered = query.lower()
    if any(k in lowered for k in ("c#", ".net", "csharp", "asp.net")):
        hint = "site:learn.microsoft.com"
    else:
        hint = "site:stackoverflow.com"
    return _ddg_search(f"{query} {hint}", max_results)


def _ddg_search(query: str, max_results: int) -> str:
    """Iki aramanin da ortak govdesi: DDG lite, olmazsa Wikipedia yedegi."""
    try:
        response = requests.post(
            "https://lite.duckduckgo.com/lite/",
            data={"q": query},
            headers=HEADERS,
            timeout=TIMEOUT,
        )
        # DDG'nin HTML'inde href, class'tan once gelir ve class tek tirnaklidir:
        #   <a rel="nofollow" href="https://..." class='result-link'>Baslik</a>
        pairs = re.findall(
            r"""<a[^>]*href="([^"]+)"[^>]*class=['"]result-link['"][^>]*>(.*?)</a>""",
            response.text,
            flags=re.DOTALL,
        )
        results = []
        for url, raw_title in pairs[:max_results]:
            title = html.unescape(re.sub(r"<[^>]+>", "", raw_title)).strip()
            if title:
                results.append(f"{len(results) + 1}. {title}\n   {html.unescape(url)}")
        if results:
            return f"'{query}' icin internet sonuclari:\n" + "\n".join(results)
    except requests.RequestException:
        pass  # asagidaki Wikipedia yedegine dus

    return _wikipedia_search(query, max_results)

def _wikipedia_search(query: str, max_results: int) -> str:
    """Yedek arama: Turkce Wikipedia API'si."""
    try:
        data = requests.get(
            "https://tr.wikipedia.org/w/api.php",
            params={
                "action": "query", "list": "search", "srsearch": query,
                "srlimit": max_results, "format": "json",
            },
            headers=WIKI_HEADERS,
            timeout=TIMEOUT,
        ).json()
        items = data.get("query", {}).get("search", [])
        if not items:
            return f"'{query}' icin sonuc bulunamadi."
        lines = []
        for i, item in enumerate(items, start=1):
            snippet = html.unescape(re.sub(r"<[^>]+>", "", item.get("snippet", "")))
            slug = item["title"].replace(" ", "_")
            lines.append(f"{i}. {item['title']}\n   {snippet}\n   https://tr.wikipedia.org/wiki/{slug}")
        return f"'{query}' icin Wikipedia sonuclari:\n" + "\n".join(lines)
    except requests.RequestException as exc:
        return f"Arama yapilamadi: {exc}"


def get_weather(city: str) -> str:
    """Bir sehrin guncel hava durumunu Open-Meteo'dan getirir. API anahtari gerekmez."""
    try:
        geo = requests.get(
            "https://geocoding-api.open-meteo.com/v1/search",
            params={"name": city, "count": 1, "language": "tr"},
            timeout=TIMEOUT,
        ).json()
        places = geo.get("results")
        if not places:
            return f"'{city}' adinda bir sehir bulunamadi."
        place = places[0]

        data = requests.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": place["latitude"],
                "longitude": place["longitude"],
                "current": "temperature_2m,relative_humidity_2m,wind_speed_10m,weather_code",
                "timezone": "auto",
            },
            timeout=TIMEOUT,
        ).json()
        now = data["current"]
        description = WMO_TR.get(now["weather_code"], "bilinmiyor")
        return (
            f"{place['name']} ({place.get('country', '')}) hava durumu: {description}, "
            f"{now['temperature_2m']}°C, nem %{now['relative_humidity_2m']}, "
            f"ruzgar {now['wind_speed_10m']} km/s. (Olcum: {now['time']})"
        )
    except (requests.RequestException, KeyError) as exc:
        return f"Hava durumu alinamadi: {exc}"


def get_exchange_rate(from_currency: str, to_currency: str, amount: float = 1.0) -> str:
    """Guncel doviz kurunu Frankfurter API'sinden getirir. API anahtari gerekmez."""
    source = from_currency.strip().upper()
    target = to_currency.strip().upper()
    try:
        data = requests.get(
            "https://api.frankfurter.dev/v1/latest",
            params={"base": source, "symbols": target},
            timeout=TIMEOUT,
        ).json()
        rate = data.get("rates", {}).get(target)
        if rate is None:
            return f"{source} -> {target} kuru bulunamadi. Para birimi kodlarini kontrol edin."
        return (
            f"1 {source} = {rate} {target} ({data['date']} tarihli kur). "
            f"{amount} {source} = {round(float(amount) * rate, 2)} {target}"
        )
    except (requests.RequestException, ValueError) as exc:
        return f"Kur bilgisi alinamadi: {exc}"


def web_lookup(question: str) -> str:
    """code_rag'in internete cikarken kullandigi arama. Soruya gore kaynak secer.

    Kod hatasi gibi duran sorular Stack Overflow / Microsoft Docs'a, digerleri
    genel aramaya gider.
    """
    lowered = question.lower()
    if any(k in lowered for k in KOD_ISARETLERI):
        return code_error_fix_search(question)
    return internet_search(question)


# Son knowledge_question cagrisinin ozeti. chat.py bunu okuyup ekranda
# "cevap bellekten mi internetten mi geldi" satirini basiyor.
SON_SONUC: dict = {}


def knowledge_question(question: str) -> str:
    """Bilgi sorusunu once OGRENILMIS BELLEKTEN, yoksa internetten cevaplar.

    Cevap burada bitmis haldedir; chat.py'deki model bunu aynen aktarmalidir.
    Internetten uretilen her cevap bellege yazilir, yani ayni soru bir daha
    gelirse aramaya cikilmaz (bkz. code_rag.py'deki uc kapi).
    """
    result = code_rag.answer_with_memory(
        question, embed_key=ACTIVE_EMBED_KEY, web_search=web_lookup
    )
    SON_SONUC.clear()
    SON_SONUC.update(
        {
            "answered_from": result["answered_from"],
            "learned": result["learned"],
            "en_yakin": max((h["similarity"] for h in result["hits"]), default=0.0),
        }
    )
    if not result["grounded"]:
        return result["answer"]

    if result["answered_from"] == "bellek":
        kaynaklar = "\n".join(
            f"- \"{h['question']}\" (benzerlik {h['similarity']:.3f}, {h['created']})"
            for h in result["hits"]
        )
        return f"{result['answer']}\n\nBellekten geldi — eslesen kayitlar:\n{kaynaklar}"

    return f"{result['answer']}\n\n(Internetten arandi ve bellege kaydedildi.)"


TOOLS = {
    "internet_search": internet_search,
    "code_error_fix_search": code_error_fix_search,
    "get_weather": get_weather,
    "get_exchange_rate": get_exchange_rate,
    "knowledge_question": knowledge_question,
}

TOOL_SCHEMAS = [
    {
        "type": "function",
        "function": {
            "name": "internet_search",
            "description": (
                "SADECE ham arama sonucu (baslik + link) listesi ister isen kullan. "
                "Bilgi sorulari icin once knowledge_question'i dene."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Arama sorgusu"},
                    "max_results": {"type": "integer", "description": "Sonuc sayisi (varsayilan 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_weather",
            "description": "Bir sehrin guncel hava durumunu (sicaklik, nem, ruzgar) getirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "city": {"type": "string", "description": "Sehir adi, ornegin 'Istanbul'"},
                },
                "required": ["city"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "code_error_fix_search",
            "description": (
                "Bir kod hata mesajinin cozumunu arar (C# icin Microsoft Docs, digerleri "
                "icin Stack Overflow). Ham link listesi doner; hazir cevap icin "
                "knowledge_question'i tercih et."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Hata mesajı"},
                    "max_results": {"type": "integer", "description": "Sonuc sayisi (varsayilan 5)"},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "get_exchange_rate",
            "description": "Iki para birimi arasindaki guncel kuru ve cevrilmis tutari getirir.",
            "parameters": {
                "type": "object",
                "properties": {
                    "from_currency": {"type": "string", "description": "Kaynak para birimi kodu, ornegin USD"},
                    "to_currency": {"type": "string", "description": "Hedef para birimi kodu, ornegin TRY"},
                    "amount": {"type": "number", "description": "Cevrilecek tutar (varsayilan 1)"},
                },
                "required": ["from_currency", "to_currency"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "knowledge_question",
            "description": (
                "Bilgi gerektiren TUM sorular icin (kod hatasi, teknik konu, genel bilgi) "
                "bu araci kullan. Once asistanin ogrenilmis bellegine bakar, orada yoksa "
                "internetten arayip cevabi ogrenir. "
                "Donen metni aynen kullaniciya aktar, uzerine kendi bilgini EKLEME."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "question": {"type": "string", "description": "Kullanicinin sorusu"},
                },
                "required": ["question"],
            },
        },
    },
]
