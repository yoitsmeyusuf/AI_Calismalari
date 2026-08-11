"""
Asama 4: arayuz.

Gradio sohbet arayuzu. Modeli DOGRUDAN cagirmiyor, FastAPI sunucusuna HTTP
ile bagliyor - odev "FastAPI sunucunuza baglanan bir arayuz" istiyor ve iki
katmanin gercekten ayri oldugunu gostermenin yolu bu. Ayrica CORS'un neden
gerektigini de somutlastiriyor (arayuz :7860, API :8000).

    # 1. terminal
    ../../.venv/bin/python -m uvicorn api:app --port 8000
    # 2. terminal
    ../../.venv/bin/python arayuz.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import gradio as gr
import httpx

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI.parent))
import ayarlar  # noqa: E402

API = os.environ.get("PERSONA_API", f"http://{ayarlar.API_HOST}:{ayarlar.API_PORT}")


def _durum_rozeti(y: dict) -> str:
    d = y["status"]
    if d == "ok":
        b = y.get("benzerlikler", [])
        ek = ""
        if y.get("guardrail_denemesi", 0) > 0:
            ek = f" · guardrail {y['guardrail_denemesi']} kez yeniden ürettirdi"
        return (f"✅ {len(y['retrieved_context'])} bölüm kullanıldı · "
                f"benzerlik {', '.join(f'{s:.2f}' for s in b)}"
                f" · en uzun birebir örtüşme {y.get('en_uzun_kopya', 0)} karakter{ek}")
    if d == "baglam_yok":
        return (f"⚠️ Hiçbir bölüm {ayarlar.BENZERLIK_ESIGI} eşiğini geçemedi — "
                f"LLM çağrılmadı, sabit savuşturma döndü")
    return "🛑 Guardrail yanıtı reddetti (alıntı eşiği aşıldı)"


def cevapla(mesaj: str, gecmis: list, kullanici: str):
    if not mesaj.strip():
        return gecmis, "", ""
    try:
        y = httpx.post(f"{API}/api/v1/chat/ayrintili",
                       json={"user_id": kullanici or "anonim", "message": mesaj},
                       timeout=180.0).raise_for_status().json()
    except Exception as e:
        gecmis = gecmis + [{"role": "user", "content": mesaj},
                           {"role": "assistant",
                            "content": f"API'ye ulaşılamadı: {e}"}]
        return gecmis, "", "❌ API kapalı mı? `uvicorn api:app --port 8000`"

    gecmis = gecmis + [{"role": "user", "content": mesaj},
                       {"role": "assistant", "content": y["reply"]}]

    kaynak = ""
    if y["retrieved_context"]:
        satirlar = ["### Kullanılan bölümler\n"]
        for i, (m, b) in enumerate(zip(y["retrieved_context"],
                                       y.get("benzerlikler", []))):
            satirlar.append(f"**{i+1}. benzerlik {b:.4f}**\n\n> "
                            + m.replace("\n", "\n> ") + "\n")
        kaynak = "\n".join(satirlar)
    return gecmis, "", _durum_rozeti(y) + ("\n\n" + kaynak if kaynak else "")


with gr.Blocks(title=f"{ayarlar.SANATCI} — RAG Persona") as demo:
    gr.Markdown(
        f"# {ayarlar.SANATCI} — RAG Persona Chatbot\n"
        f"Sanatçının şarkı sözleri ChromaDB'de vektörleştirildi. Mesajınla en "
        f"alakalı bölümler çekilip LLM'e persona promptuyla veriliyor.\n\n"
        f"Bot **sözleri alıntılamıyor**, onların felsefesiyle kendi cümlelerini "
        f"kuruyor — bu bir kural değil, yayın öncesi ölçülen bir koşul "
        f"(guardrail, {ayarlar.KOPYA_ESIGI} karakter eşiği).\n\n"
        f"`{API}` üzerindeki FastAPI servisine HTTP ile bağlanır."
    )
    with gr.Row():
        with gr.Column(scale=3):
            sohbet = gr.Chatbot(type="messages", height=460, label="Sohbet")
            with gr.Row():
                girdi = gr.Textbox(placeholder="İçinden geçeni yaz…",
                                   show_label=False, scale=5, lines=2)
                gonder = gr.Button("Gönder", variant="primary", scale=1)
            gr.Examples(
                examples=["bugün kendimi çok yalnız hissediyorum",
                          "insanlara güvenmekten korkuyorum artık",
                          "çok yoruldum, pes etmek üzereyim",
                          "Fenerbahçe kaç şampiyonluk kazandı?"],
                inputs=girdi, label="Örnekler (sonuncusu eşiğin altında kalır)")
        with gr.Column(scale=2):
            kullanici = gr.Textbox(value="kullanici-1", label="user_id")
            durum = gr.Markdown("")

    for olay in (gonder.click, girdi.submit):
        olay(cevapla, [girdi, sohbet, kullanici], [sohbet, girdi, durum])

if __name__ == "__main__":
    demo.launch(server_name="127.0.0.1", server_port=7860)
