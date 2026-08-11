"""
Asama 3: FastAPI katmani (ZORUNLU).

RAG mimarisini disariya acan RESTful servis. 1_fastapi'de ogrenilen her sey
burada kullaniliyor: Pydantic ile gelen/giden dogrulama, response_model ile
ic alanlarin gizlenmesi, /docs, CORS.

    ../../.venv/bin/python -m uvicorn api:app --port 8000
    -> http://127.0.0.1:8000/docs
"""
from __future__ import annotations

import os
import sys
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Annotated, Literal

from fastapi import FastAPI, HTTPException, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, ConfigDict, Field

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI))
sys.path.insert(0, str(BURASI.parent))

try:
    from dotenv import load_dotenv
    load_dotenv(BURASI.parent.parent / ".env")
except Exception:
    pass

import ayarlar  # noqa: E402
from persona import PersonaChatbot  # noqa: E402

_bot: PersonaChatbot | None = None


@asynccontextmanager
async def yasam_dongusu(app: FastAPI):
    # Embedding modeli ve Chroma baglantisi ilk istekte degil, acilista
    # kuruluyor: ilk kullanicinin 10 saniye beklemesi gerekmesin.
    global _bot
    _bot = PersonaChatbot()
    _bot.retriever.yukle()
    yield
    _bot = None


# --- Semalar ---------------------------------------------------------------
class SohbetIstegi(BaseModel):
    model_config = ConfigDict(extra="forbid", json_schema_extra={
        "examples": [{"user_id": "kullanici-1",
                      "message": "bugün kendimi çok yalnız hissediyorum"}]})

    user_id: Annotated[str, Field(min_length=1, max_length=64)]
    message: Annotated[str, Field(min_length=1, max_length=1000)]
    # Odev sabit esik istemiyor; istemcinin deneyebilmesi icin opsiyonel.
    esik: Annotated[float, Field(ge=0.0, le=1.0)] | None = None


class SohbetYaniti(BaseModel):
    status: Literal["ok", "baglam_yok", "guardrail_reddetti"]
    persona: str
    reply: str
    retrieved_context: list[str]


class AyrintiliYanit(SohbetYaniti):
    """/api/v1/chat/ayrintili - olcum degerleri de donuyor."""

    benzerlikler: list[float]
    guardrail_denemesi: int
    en_uzun_kopya: int
    sure_ms: int


class SaglikYaniti(BaseModel):
    durum: str
    persona: str
    esik: float
    kopya_esigi: int
    chunk_sayisi: int
    llm_arka_uc: str
    llm_model: str


class HataYaniti(BaseModel):
    detay: str


# --- Uygulama --------------------------------------------------------------
app = FastAPI(
    title="Sagopa Kajmer RAG Persona API",
    description=(
        "Sanatçının şarkı sözleri ChromaDB'de vektörleştirildi. Gelen mesajla en "
        "alakalı bölümler çekilip bir LLM'e persona promptuyla veriliyor.\n\n"
        "**Guardrail:** yanıt yayınlanmadan önce getirilen sözlerle karşılaştırılıyor; "
        f"{ayarlar.KOPYA_ESIGI} karakterden uzun birebir alıntı varsa yeniden üretiliyor.\n\n"
        f"**Eşik:** {ayarlar.BENZERLIK_ESIGI} — altında kalan mesajlarda LLM hiç "
        "çağrılmıyor, sabit bir savuşturma dönüyor."
    ),
    version="1.0.0",
    lifespan=yasam_dongusu,
    openapi_tags=[
        {"name": "sohbet", "description": "Persona ile konuşma."},
        {"name": "saglik", "description": "Servis durumu ve ayarlar."},
    ],
)

app.add_middleware(
    CORSMiddleware,
    # Arayuz (Gradio/Streamlit/statik HTML) baska bir port'ta kosuyor.
    allow_origins=[
        "http://localhost:7860", "http://127.0.0.1:7860",   # Gradio
        "http://localhost:8501", "http://127.0.0.1:8501",   # Streamlit
        "http://localhost:8080", "http://127.0.0.1:8080",   # statik HTML
    ],
    allow_credentials=True,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["Content-Type"],
    max_age=600,
)


def _bot_al() -> PersonaChatbot:
    if _bot is None:
        raise HTTPException(status_code=503, detail="servis henüz hazır değil")
    return _bot


@app.get("/api/v1/saglik", tags=["saglik"], response_model=SaglikYaniti)
def saglik() -> SaglikYaniti:
    bot = _bot_al()
    return SaglikYaniti(
        durum="ayakta", persona=bot.sanatci,
        esik=ayarlar.BENZERLIK_ESIGI, kopya_esigi=ayarlar.KOPYA_ESIGI,
        chunk_sayisi=bot.retriever._koleksiyon.count(),
        llm_arka_uc=bot.llm.arka_uc,
        llm_model=bot.llm.ad if bot.llm.arka_uc == "api" else bot.llm.yerel_ad,
    )


@app.post(
    "/api/v1/chat", tags=["sohbet"], response_model=SohbetYaniti,
    status_code=status.HTTP_200_OK,
    responses={422: {"description": "Gövde doğrulaması başarısız"},
               503: {"model": HataYaniti}},
    summary="Persona ile konuş",
)
def chat(istek: SohbetIstegi) -> SohbetYaniti:
    y = _bot_al().cevapla(istek.message, esik=istek.esik)
    # Yanit nesnesi olcum alanlari da tasiyor; response_model onlari kirpiyor.
    return SohbetYaniti(status=y.status, persona=y.persona, reply=y.reply,
                        retrieved_context=y.retrieved_context)


@app.post("/api/v1/chat/ayrintili", tags=["sohbet"], response_model=AyrintiliYanit,
          summary="Aynı uç, ölçüm değerleriyle (arayüz ve hata ayıklama için)")
def chat_ayrintili(istek: SohbetIstegi) -> AyrintiliYanit:
    t0 = time.perf_counter()
    y = _bot_al().cevapla(istek.message, esik=istek.esik)
    return AyrintiliYanit(
        status=y.status, persona=y.persona, reply=y.reply,
        retrieved_context=y.retrieved_context, benzerlikler=y.benzerlikler,
        guardrail_denemesi=y.guardrail_denemesi, en_uzun_kopya=y.en_uzun_kopya,
        sure_ms=int((time.perf_counter() - t0) * 1000),
    )


@app.exception_handler(HTTPException)
async def http_hata(request: Request, exc: HTTPException) -> JSONResponse:
    return JSONResponse(status_code=exc.status_code, content={"detay": exc.detail})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host=ayarlar.API_HOST, port=int(os.getenv("PORT", ayarlar.API_PORT)))
