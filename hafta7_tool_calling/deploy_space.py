"""
7. Hafta: Gradio arayuzunu Hugging Face Spaces'e (ZeroGPU) yayinlar.

Yapilanlar:
  1. `.env`'deki SPACE_REPO_ID icin Space reposu olusturur; donanim olarak
     **ZeroGPU** (`zero-a10g`) ister - ucretsiz H200 dilimi, modeli Space icinde
     kendimiz calistirdigimiz icin gerekli.
  2. Space kartini (frontmatter'li README.md) arac semalarindan uretip yukler,
     boylece araclar degistiginde kart otomatik guncel kalir.
  3. app.py / ajan.py / araclar.py / modeller.py / requirements.txt yukler.
  4. TOOL_BACKEND + TOOL_MODEL'i Space degiskeni, HF_TOKEN'i secret olarak ayarlar.

Calistirma:
    .venv/bin/python hafta7_tool_calling/deploy_space.py
    .venv/bin/python hafta7_tool_calling/deploy_space.py --private   # gizli Space
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

from araclar import ARAC_SEMALARI
from modeller import VARSAYILAN_YEREL_MODEL

load_dotenv()

HERE = Path(__file__).resolve().parent
YUKLENECEK_DOSYALAR = [
    "app.py",
    "ajan.py",
    "araclar.py",
    "modeller.py",
    "requirements.txt",
]
GRADIO_SURUMU = "6.21.0"


def arac_tablosu_markdown() -> str:
    satirlar = [
        "| Araç | Ne yapar | Parametreler | Kaynak |",
        "|---|---|---|---|",
    ]
    kaynaklar = {
        "get_weather": "Open-Meteo Forecast API (`current`)",
        "get_forecast": "Open-Meteo Forecast API (`daily`)",
        "get_air_quality": "Open-Meteo Air Quality API",
        "convert_temperature": "yerel (saf matematik, ağ yok)",
    }
    for sema in ARAC_SEMALARI:
        fn = sema["function"]
        ozet = fn["description"].split(".")[0].strip() + "."
        params = fn["parameters"]
        zorunlu = set(params.get("required", []))
        param_metni = ", ".join(
            f"`{ad}`" + ("" if ad in zorunlu else " *(ops.)*")
            for ad in params.get("properties", {})
        )
        satirlar.append(
            f"| `{fn['name']}` | {ozet} | {param_metni} | {kaynaklar.get(fn['name'], '-')} |"
        )
    return "\n".join(satirlar)


def space_karti(model: str) -> str:
    return f"""---
title: Tool Calling Demo - Hava Durumu Ajani
emoji: 🌦️
colorFrom: blue
colorTo: indigo
sdk: gradio
sdk_version: {GRADIO_SURUMU}
app_file: app.py
pinned: false
license: apache-2.0
short_description: Tool calling ile seffaf Turkce hava durumu ajani
suggested_hardware: zero-a10g
tags:
- tool-calling
- function-calling
- agent
- open-meteo
- turkish
---

# 🌦️ Tool Calling Demo — Hava Durumu Ajanı

Model, kullanıcının sorusuna göre **hangi aracı çağıracağına kendisi karar eder**;
veriler [Open-Meteo](https://open-meteo.com) (ücretsiz, API anahtarı gerektirmeyen
public API) üzerinden gerçek zamanlı gelir. Arayüzün sağ panelinde her turda
çağrılan araç, gönderilen argümanlar ve dönen ham JSON görünür — yani cevabın
nereden geldiği izlenebilir.

Model [`{model}`](https://huggingface.co/{model}) bu Space'in **içinde**,
ücretsiz **ZeroGPU** (H200 dilimi) üzerinde `transformers` ile çalışıyor. Araç
çağrıları modelin kendi sohbet şablonuyla kuruluyor (`<tool_call>{{...}}</tool_call>`
blokları), üretilen çıktı bu bloklardan ayrıştırılıyor. GPU yalnızca istek
süresince ayrıldığı için bütün ajan döngüsü tek bir `@spaces.GPU` çağrısı içinde
dönüyor.

## Araçlar

{arac_tablosu_markdown()}

`convert_temperature` bilinçli olarak ayrı bir araç: sıcaklık çevirimini modelin
kafadan hesaplamasına izin vermek yerine, modelin **iki turlu** bir zincir kurması
(önce sıcaklığı çek, sonra çevir) isteniyor.

## Örnek akış

Soru: *"Ankara mı daha sıcak Londra mı, ve bu değerler Fahrenheit olarak kaç eder?"*

```text
[Tur 1] Araç Çağrıları:
   -> get_weather(city='Ankara')
   <- {{"sehir": "Ankara", "sicaklik_c": 16.0, "gokyuzu": "açık", ...}}
   -> get_weather(city='London')
   <- {{"sehir": "Londra", "sicaklik_c": 24.0, "gokyuzu": "açık", ...}}

[Tur 2] Araç Çağrıları:
   -> convert_temperature(value=16.0, to_unit='F')
   <- {{"sonuc": 60.8, "birim": "F"}}
   -> convert_temperature(value=24.0, to_unit='F')
   <- {{"sonuc": 75.2, "birim": "F"}}

[Tur 3] Nihai Yanıt:
Londra 75.2°F ile Ankara'dan (60.8°F) daha sıcaktır.
```

Cevaptaki Fahrenheit değerleri modelin kendi hesabı değil, `convert_temperature`
aracının çıktısıdır.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `araclar.py` | Araç implementasyonları + modele verilen JSON şemaları |
| `modeller.py` | Model katmanı: ZeroGPU/transformers arka ucu + `<tool_call>` ayrıştırıcı, alternatif olarak Inference Providers arka ucu |
| `ajan.py` | Tool calling döngüsü (araç çağrısı → araç çalıştır → sonucu modele geri ver) |
| `app.py` | Gradio arayüzü (sohbet + araç çağrı izi paneli), `@spaces.GPU` sarmalayıcı |

## Ortam değişkenleri

| Ad | Tip | Ne işe yarar |
|---|---|---|
| `TOOL_BACKEND` | variable | `yerel` (varsayılan, ZeroGPU'da transformers) veya `api` (HF Inference Providers). |
| `TOOL_MODEL` | variable | Kullanılacak model (varsayılan `{VARSAYILAN_YEREL_MODEL}`). Sohbet şablonu tool calling desteklemeli. |
| `HF_TOKEN` | secret | Model indirmek ve (api arka ucunda) inference çağrısı yapmak için. |
| `ZEROGPU_DURATION` | variable | `@spaces.GPU` süre limiti, saniye (varsayılan 120). |
| `TOOL_BASE_URL` / `TOOL_API_KEY` | variable / secret | `api` arka ucunda OpenAI uyumlu başka bir endpoint kullanmak için. |

Bu Space, 7 haftalık bir Türkçe LLM çalışma serisinin son ödevidir; kod ve diğer
haftalar: [github.com/yoitsmeyusuf](https://github.com/yoitsmeyusuf).
"""


def main():
    from huggingface_hub import HfApi, SpaceHardware, get_token

    repo_id = os.environ.get("SPACE_REPO_ID")
    if not repo_id or "kullanici-adi" in repo_id:
        raise SystemExit(
            "SPACE_REPO_ID tanımlı değil. .env dosyanıza ekleyin, örn:\n"
            "    SPACE_REPO_ID=kullanici-adiniz/tool-calling-hava-durumu"
        )

    arka_uc = (os.environ.get("TOOL_BACKEND") or "yerel").strip().lower()
    model = os.environ.get("TOOL_MODEL") or VARSAYILAN_YEREL_MODEL
    gizli = "--private" in sys.argv

    # Model indirmek ve (api arka ucunda) inference icin token gerekiyor.
    # Ortamda yoksa `hf auth login` ile cache'lenmis token kullanilir.
    token = os.environ.get("HF_TOKEN") or get_token()
    api = HfApi(token=token)

    try:
        api.create_repo(
            repo_id=repo_id,
            repo_type="space",
            space_sdk="gradio",
            space_hardware=SpaceHardware.ZERO_A10G,  # ücretsiz ZeroGPU
            private=gizli,
            exist_ok=True,
        )
    except Exception as e:
        metin = str(e)
        if "402" in metin:
            raise SystemExit(
                "Space oluşturulamadı (402). Hugging Face ücretsiz `cpu-basic` "
                "donanımda Gradio Space barındırmayı kaldırdı; bu script ZeroGPU "
                "(`zero-a10g`) istiyor. Hesabınızda ZeroGPU kotası yoksa Space "
                "ayarlarından donanımı elle seçmeyi ya da arayüzü yerelde "
                "çalıştırmayı deneyin:\n"
                "    .venv/bin/python hafta7_tool_calling/app.py\n\n"
                f"Ham hata: {metin}"
            ) from e
        raise

    print(f"Space hazır ({'gizli' if gizli else 'herkese açık'}, ZeroGPU): "
          f"https://huggingface.co/spaces/{repo_id}")

    api.upload_file(
        path_or_fileobj=space_karti(model).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="space",
    )
    for ad in YUKLENECEK_DOSYALAR:
        api.upload_file(
            path_or_fileobj=str(HERE / ad),
            path_in_repo=ad,
            repo_id=repo_id,
            repo_type="space",
        )
        print(f"  yüklendi: {ad}")

    api.add_space_variable(repo_id=repo_id, key="TOOL_BACKEND", value=arka_uc)
    api.add_space_variable(repo_id=repo_id, key="TOOL_MODEL", value=model)
    print(f"  değişkenler ayarlandı: TOOL_BACKEND={arka_uc}, TOOL_MODEL={model}")

    if token:
        api.add_space_secret(repo_id=repo_id, key="HF_TOKEN", value=token)
        print("  HF_TOKEN secret'ı ayarlandı.")
    else:
        print(
            "  UYARI: Ne HF_TOKEN ortam değişkeni ne de `hf auth login` cache'i "
            "bulundu. Token'ı elle girin:\n"
            f"    https://huggingface.co/spaces/{repo_id}/settings"
        )

    print(f"\nHazır: https://huggingface.co/spaces/{repo_id}")
    print("İlk açılışta model ağırlıkları indiği için build birkaç dakika sürebilir.")


if __name__ == "__main__":
    main()
