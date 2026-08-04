"""
7. Hafta: Tool Calling demosu icin Gradio arayuzu (Hugging Face Spaces girisi).

Spaces bu dosyayi otomatik olarak calistirir; ayni dosya yerelde de calisir:
    .venv/bin/python hafta7_tool_calling/app.py

Arayuz iki seyi ayni anda gosterir:
  - solda normal sohbet (kullanicinin gordugu nihai yanit),
  - sagda ajanin arka planda hangi araci hangi argumanlarla cagirdigi ve ne
    donduguu (odevin "adimlarini acikca goster" sarti).

ZeroGPU notu: Space ucretsiz ZeroGPU donaniminda kosuyor. GPU yalnizca istek
suresince ayrilir, bu yuzden butun ajan dongusunu `@spaces.GPU` ile sarilmis bir
generator icinde calistiriyoruz - hem GPU tek seferde alinip birakiliyor (her
model cagrisi icin ayri ayri kuyruga girilmiyor) hem de adimlar olustukca
ekrana basilmaya devam ediyor.
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import gradio as gr  # noqa: E402

from ajan import MAKS_TUR, ajan_akisi, izi_metne_cevir  # noqa: E402
from araclar import ARAC_SEMALARI  # noqa: E402
from modeller import model_yukle  # noqa: E402

# Yerelde .env okunur; Spaces'te secret'lar dogrudan ortam degiskeni olarak gelir.
try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

ARKA_UC = (os.environ.get("TOOL_BACKEND") or "yerel").strip().lower()
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


# Model bir kez kurulur. Yerel arka uctaki agirliklari Space acilisinda (GPU
# ayrilmadan once) indirip bellege aliyoruz; ilk istek 15 GB indirmeyi ZeroGPU
# suresi icinde yapmaya kalkmasin.
SOHBET_MODELI = model_yukle(ARKA_UC)
YUKLEME_HATASI: str | None = None
if ARKA_UC not in ("api", "inference", "hf"):
    try:
        SOHBET_MODELI.yukle()
    except Exception as e:  # arayuz acilsin, hatayi sohbette gosterelim
        YUKLEME_HATASI = f"Model yüklenemedi ({SOHBET_MODELI.ad}): {type(e).__name__}: {e}"
        print(YUKLEME_HATASI)

ORNEK_SORULAR = [
    "Ankara mı daha sıcak Londra mı, ve bu değerler Fahrenheit olarak kaç eder?",
    "İzmir'de önümüzdeki 3 gün yağmur var mı?",
    "Bugün İstanbul'da koşuya çıkmak sağlıklı mı?",
    "Erzurum kaç Kelvin?",
    "Berlin ve Paris'in hava kalitesini karşılaştır.",
]

ARAC_LISTESI = "\n".join(
    f"- **`{s['function']['name']}`** — {s['function']['description'].split('.')[0]}."
    for s in ARAC_SEMALARI
)

ACIKLAMA = f"""Sorduğunuz soruya göre model, aşağıdaki araçlardan gerekli olanları kendisi
seçip çağırır; veriler [Open-Meteo](https://open-meteo.com) (ücretsiz, anahtarsız
public API) üzerinden gerçek zamanlı gelir.

{ARAC_LISTESI}

Sağdaki **Araç çağrı izi** panelinde her turda hangi aracın hangi argümanlarla
çağrıldığını ve API'nin ne döndürdüğünü ham haliyle görebilirsiniz.

*Model: `{SOHBET_MODELI.ad}` — {"ZeroGPU üzerinde transformers ile yerel çalışıyor"
if ARKA_UC not in ("api", "inference", "hf") else "HF Inference Providers üzerinden"}.*"""


@gpu_dekoratoru
def yanitla(mesaj: str, sohbet: list[dict]):
    """Gradio handler'i: her adimda (sohbet, iz, ozet, girdi) demetini yayinlar."""
    mesaj = (mesaj or "").strip()
    if not mesaj:
        yield sohbet, "", "", mesaj
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
    yield sohbet, "(model çağrılıyor...)", "", ""  # girdi kutusu bosaltilir

    durum = None
    for durum in ajan_akisi(
        mesaj, gecmis=gecmis, model=SOHBET_MODELI, maks_tur=MAKS_TUR
    ):
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
        yield sohbet, iz, "", ""

    if durum is not None:
        ozet = (
            f"{len(durum.turlar)} tur · {durum.arac_cagri_sayisi} araç çağrısı · "
            f"model: `{durum.model}`"
        )
        yield sohbet, izi_metne_cevir(durum), ozet, ""


def temizle():
    return [], "", "", ""


with gr.Blocks(title="Tool Calling Demo — Hava Durumu Ajanı", fill_height=True) as demo:
    gr.Markdown("# 🌦️ Tool Calling Demo — Hava Durumu Ajanı")
    gr.Markdown(ACIKLAMA)

    with gr.Row():
        with gr.Column(scale=3):
            sohbet = gr.Chatbot(
                height=430,
                label="Sohbet",
                placeholder="Bir hava durumu sorusu yazın; model gerekli araçları kendisi seçecek.",
            )
            with gr.Row():
                girdi = gr.Textbox(
                    placeholder="Örn: Ankara mı daha sıcak Londra mı, Fahrenheit olarak kaç eder?",
                    show_label=False,
                    scale=8,
                    submit_btn=True,
                )
                temizle_btn = gr.Button("Temizle", scale=1)
            gr.Examples(examples=ORNEK_SORULAR, inputs=girdi, label="Örnek sorular")

        with gr.Column(scale=2):
            iz_kutusu = gr.Code(
                label="Araç çağrı izi (Tool Calling adımları)",
                language=None,
                lines=22,
                interactive=False,
            )
            ozet_kutusu = gr.Markdown("")

    ciktilar = [sohbet, iz_kutusu, ozet_kutusu, girdi]
    girdi.submit(yanitla, [girdi, sohbet], ciktilar)
    temizle_btn.click(temizle, None, ciktilar)


if __name__ == "__main__":
    demo.queue().launch()
