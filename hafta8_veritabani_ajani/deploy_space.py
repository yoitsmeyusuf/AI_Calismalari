"""
8. Hafta: Gradio arayuzunu Hugging Face Spaces'e (ZeroGPU) yayinlar.

Yapilanlar:
  1. `.env`'deki SPACE_REPO_ID_KITAPCI icin Space reposu olusturur; donanim
     olarak **ZeroGPU** (`zero-a10g`) ister - modeli Space icinde kendimiz
     calistirdigimiz icin gerekli (ayrica ucretsiz `cpu-basic` artik Gradio
     Space barindirmiyor).
  2. Space kartini (frontmatter'li README.md) arac semalarindan uretip yukler,
     boylece araclar degistiginde kart otomatik guncel kalir.
  3. Kod dosyalarini yukler. `modeller.py` 7. haftadan geliyor: yerelde ayni
     modulu import ediyoruz, Space'e ise app.py'nin yanina kopyalaniyor.
  4. TOOL_BACKEND + TOOL_MODEL'i Space degiskeni, HF_TOKEN'i secret olarak ayarlar.

Not: veritabani (kitapci.db) yuklenmez; Space acilirken `veritabani.db_kur()`
semayi kurup ornek katalogu yaziyor.

Calistirma:
    .venv/bin/python hafta8_veritabani_ajani/deploy_space.py
    .venv/bin/python hafta8_veritabani_ajani/deploy_space.py --private   # gizli Space
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
HAFTA7 = HERE.parent / "hafta7_tool_calling"
sys.path.insert(0, str(HAFTA7))
sys.path.insert(0, str(HERE))

from dotenv import load_dotenv

import veritabani as db
from araclar import ARAC_SEMALARI, YAZAN_ARACLAR, ilk_cumle
from modeller import VARSAYILAN_YEREL_MODEL

load_dotenv()

# (yerel yol, Space icindeki ad)
YUKLENECEK_DOSYALAR = [
    (HERE / "app.py", "app.py"),
    (HERE / "ajan.py", "ajan.py"),
    (HERE / "araclar.py", "araclar.py"),
    (HERE / "veritabani.py", "veritabani.py"),
    (HERE / "requirements.txt", "requirements.txt"),
    (HAFTA7 / "modeller.py", "modeller.py"),  # model katmani 7. haftadan
]
GRADIO_SURUMU = "6.21.0"


def arac_tablosu_markdown() -> str:
    satirlar = ["| Araç | Yön | Ne yapar | Parametreler |", "|---|---|---|---|"]
    for sema in ARAC_SEMALARI:
        fn = sema["function"]
        yon = "**yazar**" if fn["name"] in YAZAN_ARACLAR else "okur"
        ozet = ilk_cumle(fn["description"])
        params = fn["parameters"]
        zorunlu = set(params.get("required", []))
        param_metni = ", ".join(
            f"`{ad}`" + ("" if ad in zorunlu else " *(ops.)*")
            for ad in params.get("properties", {})
        ) or "-"
        satirlar.append(f"| `{fn['name']}` | {yon} | {ozet} | {param_metni} |")
    return "\n".join(satirlar)


def katalog_ozeti() -> str:
    yazarlar = sorted({y for _, y, *_ in db.KATALOG})
    return f"{len(db.KATALOG)} kitap / {len(yazarlar)} yazar"


def space_karti(model: str) -> str:
    return f"""---
title: Tool Calling - Felsefe Kitapcisi Asistani
emoji: 📚
colorFrom: purple
colorTo: pink
sdk: gradio
sdk_version: {GRADIO_SURUMU}
app_file: app.py
pinned: false
license: apache-2.0
short_description: SQLite'a okuyup yazan Turkce siparis asistani (tool calling)
suggested_hardware: zero-a10g
tags:
- tool-calling
- function-calling
- agent
- sqlite
- turkish
---

# 📚 Tool Calling — Felsefe Kitapçısı Asistanı

Küçük bir felsefe kitapçısının sipariş asistanı. Model, sorulan soruya göre
**hangi aracı çağıracağına kendisi karar eder**; bütün veriler bir **SQLite
veritabanından** okunur ve siparişler oraya **yazılır** (stok düşer). Arayüzde
solda sohbet, ortada araç çağrı izi, sağda veritabanının canlı hali var — yani
cevabın nereden geldiği ve verinin gerçekten değiştiği aynı ekranda izlenebilir.

Model [`{model}`](https://huggingface.co/{model}) bu Space'in **içinde**,
ücretsiz **ZeroGPU** (H200 dilimi) üzerinde `transformers` ile çalışıyor. GPU
yalnızca istek süresince ayrıldığı için bütün ajan döngüsü tek bir `@spaces.GPU`
çağrısının içinde dönüyor.

## Araçlar

{arac_tablosu_markdown()}

`create_order` bilinçli olarak yalnızca **veritabanındaki bir `kitap_id`** ile
çalışıyor: model kitabı kendi belleğinden uydurup sipariş veremiyor, önce
`search_books` çağırıp gerçek id'yi almak zorunda (iki turlu zincir).

## Halüsinasyon engelleme

Sistem promptundaki "uydurma" kuralı tek başına yetmiyor, bu yüzden kural
döngüye taşındı. Nihai yanıt yayınlanmadan önce şunlar denetleniyor:

1. Katalog/sipariş sorusuna **hiç araç çağrılmadan** cevap verilmiş mi?
2. "Siparişiniz alındı" denmiş ama veritabanına yazan araç hiç çalışmamış mı?
3. Cevaptaki sipariş kodları (`SIP-...`) ve tırnak içindeki kitap adları araç
   çıktısında geçiyor mu?
4. Cevaptaki TL tutarları araç çıktısındaki fiyatlarla açıklanabiliyor mu?

İhlal varsa modelden bir kez düzeltme isteniyor ve o turda araç çağrısı zorunlu
tutuluyor. Müdahale iz panelinde `[!] harness uyarısı` satırı olarak görünür.
Model ısrar ederse cevap yine de verilir ama üzerine **"Doğrulanmadı"** uyarısı
eklenir — doğrulanmamış hiçbir bilgi kullanıcıya olgu gibi sunulmuyor.

## Örnek akış

Soru: *"Camus'nun kitabı var mı? Varsa 2 tane sipariş ver, adım Yusuf."*

```text
[Tur 1] Araç Çağrıları:
   -> search_books(author='Camus')
   <- {{"bulunan": 1, "kitaplar": [{{"kitap_id": 6, "baslik": "Sisifos Söyleni",
        "fiyat_tl": 110.0, "stok": 7}}], ...}}

[Tur 2] Araç Çağrıları:
   -> create_order(book_id=6, quantity=2, customer_name='Yusuf') [DB YAZMA]
   <- {{"siparis_kodu": "SIP-1003", "toplam_tl": 220.0, "kalan_stok": 5, ...}}

[Tur 3] Nihai Yanıt:
Siparişiniz alındı: 2 adet Sisifos Söyleni, 220 TL. Kodunuz SIP-1003.
```

Sipariş kodu ve tutar modelin uydurması değil, `create_order` çıktısı; aynı anda
sağdaki tabloda stok 7'den 5'e düşüyor.

## Veritabanı

Katalog ({katalog_ozeti()}) Space her açıldığında `veritabani.py` içindeki
tohum veriden kuruluyor. Space'te kalıcı disk yoksa veritabanı geçicidir:
yeniden başlatıldığında siparişler sıfırlanır. Arayüzdeki **Veritabanını sıfırla**
düğmesi katalogu elle başlangıç durumuna döndürür.

| Tablo | İçerik |
|---|---|
| `kitaplar` | id, başlık, yazar, akım, yıl, fiyat, stok (`CHECK (stok >= 0)`) |
| `siparisler` | kod, müşteri, kitap_id, adet, birim fiyat, toplam, durum, tarih |

Yazma işlemleri `BEGIN IMMEDIATE` ile tek bir işlemde dönüyor: stok kontrolü ile
stok düşümü arasına başka bir istek giremiyor.

## Dosyalar

| Dosya | İçerik |
|---|---|
| `veritabani.py` | SQLite şeması, tohum katalog, okuma/yazma fonksiyonları |
| `araclar.py` | Araç implementasyonları + modele verilen JSON şemaları |
| `ajan.py` | Tool calling döngüsü + halüsinasyon guardrail'i + şeffaf iz |
| `modeller.py` | Model katmanı (7. haftadan): ZeroGPU/transformers veya Inference Providers |
| `app.py` | Gradio arayüzü (sohbet + araç izi + canlı DB paneli) |

## Ortam değişkenleri

| Ad | Tip | Ne işe yarar |
|---|---|---|
| `TOOL_BACKEND` | variable | `yerel` (varsayılan, ZeroGPU'da transformers) veya `api` (HF Inference Providers). |
| `TOOL_MODEL` | variable | Kullanılacak model (varsayılan `{VARSAYILAN_YEREL_MODEL}`). Sohbet şablonu tool calling desteklemeli. |
| `KITAPCI_DB` | variable | Veritabanı dosyasının yolu. Verilmezse kalıcı disk (`/data`) varsa oraya, yoksa uygulama klasörüne yazılır. |
| `HF_TOKEN` | secret | Model indirmek ve (api arka ucunda) inference çağrısı yapmak için. |
| `ZEROGPU_DURATION` | variable | `@spaces.GPU` süre limiti, saniye (varsayılan 120). |

Bu Space, 8 haftalık bir Türkçe LLM çalışma serisinin son ödevidir; kod ve diğer
haftalar: [github.com/yoitsmeyusuf](https://github.com/yoitsmeyusuf).
"""


def main():
    from huggingface_hub import HfApi, SpaceHardware, get_token

    repo_id = os.environ.get("SPACE_REPO_ID_KITAPCI")
    if not repo_id or "kullanici-adi" in repo_id:
        raise SystemExit(
            "SPACE_REPO_ID_KITAPCI tanımlı değil. .env dosyanıza ekleyin, örn:\n"
            "    SPACE_REPO_ID_KITAPCI=kullanici-adiniz/kitapci-siparis-ajani"
        )

    arka_uc = (os.environ.get("TOOL_BACKEND") or "yerel").strip().lower()
    model = os.environ.get("TOOL_MODEL") or VARSAYILAN_YEREL_MODEL
    gizli = "--private" in sys.argv

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
                "    .venv/bin/python hafta8_veritabani_ajani/app.py\n\n"
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
    for yerel, uzak in YUKLENECEK_DOSYALAR:
        api.upload_file(
            path_or_fileobj=str(yerel),
            path_in_repo=uzak,
            repo_id=repo_id,
            repo_type="space",
        )
        kaynak = "" if yerel.parent == HERE else f"  ({yerel.parent.name}/)"
        print(f"  yüklendi: {uzak}{kaynak}")

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
