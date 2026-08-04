"""
8. Hafta: Gradio arayuzu (Hugging Face Spaces giris dosyasi).

Spaces bu dosyayi otomatik calistirir; ayni dosya yerelde de calisir:
    .venv/bin/python hafta8_veritabani_ajani/app.py

Arayuz uc seyi ayni anda gosteriyor:
  - solda normal sohbet (kullanicinin gordugu nihai yanit),
  - ortada ajanin hangi araci hangi argumanlarla cagirdigi ve ne dondugu
    (yazma cagrilari `[DB YAZMA]` diye isaretli),
  - sagda **veritabaninin canli hali** - katalog/stok ve son siparisler. Model
    siparis verince stok sutununun dustugu ayni ekranda goruluyor; yani "veri
    gercekten yaziliyor mu" sorusu goz onunde cevaplaniyor.

ZeroGPU notu: GPU yalnizca istek suresince ayrilir, bu yuzden butun ajan dongusu
`@spaces.GPU` ile sarilmis tek bir generator icinde donuyor.
"""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / "hafta7_tool_calling"))  # modeller.py (7. hafta)
sys.path.insert(0, str(HERE))

import gradio as gr  # noqa: E402

import veritabani as db  # noqa: E402
from ajan import MAKS_TUR, ajan_akisi, izi_metne_cevir  # noqa: E402
from araclar import ARAC_SEMALARI, YAZAN_ARACLAR, ilk_cumle  # noqa: E402
from modeller import model_yukle  # noqa: E402

# Yerelde .env okunur; Spaces'te secret'lar dogrudan ortam degiskeni olarak gelir.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ARKA_UC = (os.environ.get("TOOL_BACKEND") or "yerel").strip().lower()
API_ARKA_UCU = ARKA_UC in ("api", "inference", "hf")
GPU_SURESI = int(os.environ.get("ZEROGPU_DURATION", "120"))

# ZeroGPU dekoratoru: Space disinda `spaces` paketi olmayabilir, o zaman
# dekorator islevsiz bir gecise donusuyor (yerel calistirmalar bozulmasin).
try:
    import spaces  # type: ignore

    def gpu_dekoratoru(fn):
        return spaces.GPU(duration=GPU_SURESI)(fn)

except ImportError:  # yerel calistirma / API arka ucu

    def gpu_dekoratoru(fn):
        return fn


db.db_kur()  # semayi kur, bossa ornek katalogu yaz

# Model bir kez kurulur. Yerel arka uctaki agirliklari Space acilisinda (GPU
# ayrilmadan once) indirip bellege aliyoruz; ilk istek indirmeyi ZeroGPU suresi
# icinde yapmaya kalkmasin.
SOHBET_MODELI = model_yukle(ARKA_UC)
YUKLEME_HATASI: str | None = None
if not API_ARKA_UCU:
    try:
        SOHBET_MODELI.yukle()
    except Exception as e:  # arayuz acilsin, hatayi sohbette gosterelim
        YUKLEME_HATASI = f"Model yüklenemedi ({SOHBET_MODELI.ad}): {type(e).__name__}: {e}"
        print(YUKLEME_HATASI)

ORNEK_SORULAR = [
    "Camus'nun kitabı var mı? Varsa 2 tane sipariş ver, adım Yusuf.",
    "150 TL altındaki kitapları listeler misin?",
    "Nietzsche'nin hangi kitapları stokta?",
    "SIP-1002 numaralı siparişim ne durumda?",
    "Varlık ve Hiçlik'ten bir tane istiyorum.",
    "Simülakrlar ve Simülasyon kaç para?",
]

ARAC_LISTESI = "\n".join(
    f"- **`{s['function']['name']}`** "
    f"({'DB yazar' if s['function']['name'] in YAZAN_ARACLAR else 'DB okur'}) — "
    f"{ilk_cumle(s['function']['description'])}"
    for s in ARAC_SEMALARI
)

ACIKLAMA = f"""Küçük bir **felsefe kitapçısının** sipariş asistanı. Model, sorulan soruya göre
aşağıdaki araçlardan gerekli olanları kendisi seçip çağırır; bütün veriler bir **SQLite
veritabanından** okunur ve siparişler oraya **yazılır**.

{ARAC_LISTESI}

Ortadaki **araç çağrı izi** panelinde her turda hangi aracın hangi argümanlarla çağrıldığı
ve veritabanının ne döndürdüğü ham haliyle görünür. Sağdaki panelde veritabanının canlı hali
var — sipariş verdiğinizde stok sütununun düştüğünü aynı ekranda görebilirsiniz.

*Model: `{SOHBET_MODELI.ad}` — {"HF Inference Providers üzerinden"
if API_ARKA_UCU else "ZeroGPU üzerinde transformers ile Space'in içinde"} çalışıyor.*"""

KATALOG_BASLIKLARI = ["id", "Kitap", "Yazar", "Akım", "Fiyat (TL)", "Stok"]
SIPARIS_BASLIKLARI = ["Kod", "Müşteri", "Kitap", "Adet", "Tutar (TL)", "Durum"]


def katalog_tablosu() -> list[list]:
    return [
        [k["kitap_id"], k["baslik"], k["yazar"], k["akim"], k["fiyat_tl"], k["stok"]]
        for k in db.katalogu_listele()
    ]


def siparis_tablosu() -> list[list]:
    return [
        [s["siparis_kodu"], s["musteri"], s["kitap"], s["adet"], s["toplam_tl"], s["durum"]]
        for s in db.siparisleri_listele(limit=12)
    ]


@gpu_dekoratoru
def yanitla(mesaj: str, sohbet: list[dict]):
    """Gradio handler'i: her adimda (sohbet, iz, ozet, girdi, katalog, siparisler)."""
    mesaj = (mesaj or "").strip()
    if not mesaj:
        yield sohbet, "", "", mesaj, katalog_tablosu(), siparis_tablosu()
        return

    if YUKLEME_HATASI:
        yield (
            sohbet + [
                {"role": "user", "content": mesaj},
                {"role": "assistant", "content": f"⚠️ {YUKLEME_HATASI}"},
            ],
            f"[HATA] {YUKLEME_HATASI}",
            "",
            "",
            katalog_tablosu(),
            siparis_tablosu(),
        )
        return

    # Ajanin gormesi gereken gecmis: sadece kullanici/asistan metinleri.
    gecmis = [
        {"role": m["role"], "content": m["content"]}
        for m in sohbet
        if m.get("role") in ("user", "assistant") and m.get("content")
    ]

    sohbet = sohbet + [
        {"role": "user", "content": mesaj},
        {"role": "assistant", "content": "_araçlar çağrılıyor..._"},
    ]
    # girdi kutusu bosaltilir
    yield sohbet, "(model çağrılıyor...)", "", "", katalog_tablosu(), siparis_tablosu()

    durum = None
    for durum in ajan_akisi(mesaj, gecmis=gecmis, model=SOHBET_MODELI, maks_tur=MAKS_TUR):
        iz = izi_metne_cevir(durum)
        if durum.nihai_yanit:
            sohbet[-1]["content"] = durum.nihai_yanit
        elif durum.hata:
            sohbet[-1]["content"] = f"⚠️ {durum.hata}"
        else:
            adim = (
                durum.turlar[-1].araclar[-1]
                if durum.turlar and durum.turlar[-1].araclar
                else None
            )
            sohbet[-1]["content"] = (
                f"_`{adim.ad}` çağrılıyor..._" if adim else "_araçlar çağrılıyor..._"
            )
        # Tablolari her adimda tazeliyoruz: yazma araci calisir calismaz stok
        # sutunundaki degisiklik ekranda goruluyor.
        yield sohbet, iz, "", "", katalog_tablosu(), siparis_tablosu()

    if durum is not None:
        ozet = (
            f"{len(durum.turlar)} tur · {durum.arac_cagri_sayisi} araç çağrısı · "
            f"{durum.yazma_sayisi} DB yazma · model: `{durum.model}`"
        )
        yield (
            sohbet,
            izi_metne_cevir(durum),
            ozet,
            "",
            katalog_tablosu(),
            siparis_tablosu(),
        )


def temizle():
    """Sohbeti temizler; veritabanina dokunmaz."""
    return [], "", "", "", katalog_tablosu(), siparis_tablosu()


def veritabanini_sifirla():
    """Katalogu ve siparisleri baslangic durumuna dondurur (demo temizligi)."""
    db.db_kur(sifirla=True)
    return (
        [],
        "",
        "Veritabanı başlangıç durumuna döndürüldü.",
        "",
        katalog_tablosu(),
        siparis_tablosu(),
    )


with gr.Blocks(title="Tool Calling — Felsefe Kitapçısı Asistanı", fill_height=True) as demo:
    gr.Markdown("# 📚 Tool Calling — Felsefe Kitapçısı Asistanı")
    gr.Markdown(ACIKLAMA)

    with gr.Row():
        with gr.Column(scale=4):
            sohbet = gr.Chatbot(
                height=430,
                label="Sohbet",
                placeholder="Kitap sorun ya da sipariş verin; model gerekli araçları kendisi seçecek.",
            )
            with gr.Row():
                girdi = gr.Textbox(
                    placeholder="Örn: Camus'nun kitabı var mı? Varsa 2 tane sipariş ver.",
                    show_label=False,
                    scale=8,
                    submit_btn=True,
                )
                temizle_btn = gr.Button("Sohbeti temizle", scale=1)
            gr.Examples(examples=ORNEK_SORULAR, inputs=girdi, label="Örnek sorular")

        with gr.Column(scale=3):
            iz_kutusu = gr.Code(
                label="Araç çağrı izi (Tool Calling adımları)",
                language=None,
                lines=22,
                interactive=False,
            )
            ozet_kutusu = gr.Markdown("")

        with gr.Column(scale=3):
            gr.Markdown("### 🗄️ Veritabanı (canlı)")
            katalog_kutusu = gr.Dataframe(
                headers=KATALOG_BASLIKLARI,
                value=katalog_tablosu(),
                label="kitaplar tablosu (stok)",
                interactive=False,
                wrap=True,
                max_height=300,
            )
            siparis_kutusu = gr.Dataframe(
                headers=SIPARIS_BASLIKLARI,
                value=siparis_tablosu(),
                label="siparisler tablosu (son 12)",
                interactive=False,
                wrap=True,
                max_height=240,
            )
            sifirla_btn = gr.Button("Veritabanını sıfırla", variant="secondary")

    ciktilar = [sohbet, iz_kutusu, ozet_kutusu, girdi, katalog_kutusu, siparis_kutusu]
    girdi.submit(yanitla, [girdi, sohbet], ciktilar)
    temizle_btn.click(temizle, None, ciktilar)
    sifirla_btn.click(veritabanini_sifirla, None, ciktilar)


if __name__ == "__main__":
    demo.queue().launch()
