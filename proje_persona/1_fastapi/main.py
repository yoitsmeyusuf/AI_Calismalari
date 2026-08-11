"""
FastAPI ogrenme gorevi - calisan demo API.

Odevin istedigi bes basligin her biri burada bir uca (endpoint) baglandi:
  HTTP metotlari / routing -> @app.get vs @app.post, path ve query parametreleri
  Pydantic dogrulama       -> semalar.py, request + response_model
  Swagger UI               -> /docs (otomatik), semasi /openapi.json
  CORS                     -> CORSMiddleware, CORS_ACIK=0 ile kapatilabiliyor

Calistirma:
    ../.venv/bin/python -m uvicorn main:app --reload --port 8000
    (proje_persona/1_fastapi icinden)
veya kokten:
    ../.venv/bin/python proje_persona/1_fastapi/main.py
"""
import os
import sys
from pathlib import Path

from fastapi import FastAPI, HTTPException, Query, Request, Response, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

sys.path.insert(0, str(Path(__file__).resolve().parent))
from semalar import (  # noqa: E402
    HataYaniti,
    SarkiGuncelle,
    SarkiKayit,
    SarkiOlustur,
    SarkiYanit,
    SayfaliYanit,
    Tur,
)

app = FastAPI(
    title="Sarki Kunye API",
    description=(
        "FastAPI ogrenme gorevi. Sarki **kunyesi** (baslik/album/yil/sure) tutar; "
        "sarki sozu metni burada yok, o RAG asamasinin isi.\n\n"
        "Swagger UI: `/docs` - ReDoc: `/redoc` - ham sema: `/openapi.json`"
    ),
    version="1.0.0",
    # Uclari /docs'ta grupla; etiket aciklamalari da orada gorunur.
    openapi_tags=[
        {"name": "saglik", "description": "Servisin ayakta olup olmadigi."},
        {"name": "sarkilar", "description": "CRUD - GET/POST/PUT/DELETE farklari."},
        {"name": "ogretici", "description": "Dogrulama ve CORS davranisini gosteren uclar."},
    ],
)

# --- CORS ---------------------------------------------------------------
# Tarayici, sayfanin origin'i (semа+host+port) ile istegin gittigi origin
# farkliysa yaniti JS'e VERMEZ - istek sunucuya ulassa bile. Sunucunun
# Access-Control-Allow-Origin basligiyla izin vermesi gerekir.
#
# CORS_ACIK=0 ile middleware devre disi kalir; cors_demo/index.html ayni
# sayfayi iki durumda da gosterip farki kanitliyor.
CORS_ACIK = os.getenv("CORS_ACIK", "1") == "1"

# allow_origins=["*"] ile allow_credentials=True BIRLIKTE calismaz: spec
# geregi joker origin'e kimlik bilgisi (cookie) gonderilemez. Starlette bu
# durumda Allow-Origin'i "*" yapar ve tarayici credential'li istegi reduces.
# Bu yuzden origin'ler acikca yaziliyor.
IZINLI_ORIGINLER = [
    "http://localhost:8080",
    "http://127.0.0.1:8080",
]

if CORS_ACIK:
    app.add_middleware(
        CORSMiddleware,
        allow_origins=IZINLI_ORIGINLER,
        allow_credentials=True,
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Content-Type", "X-Istek-Kimligi"],
        # Tarayici varsayilan olarak JS'e sadece birkac "guvenli" basligi
        # gosterir; ozel basligimizi gormesi icin acikca aciga cikarmali.
        expose_headers=["X-Toplam-Kayit"],
        max_age=600,  # preflight yaniti 10 dk cache'lenir
    )


# --- Bellek ici depo ----------------------------------------------------
# Odev bir veritabani istemiyor; kalicilik 3_rag_persona'da Chroma ile geliyor.
_DEPO: dict[int, SarkiKayit] = {}
_SONRAKI_ID = 1


def _tohumla() -> None:
    global _SONRAKI_ID
    tohum = [
        ("Bir Pesimistin Gozyaslari", "Bir Pesimistin Gozyaslari", 2005, 254, Tur.RAP),
        ("Kalpsiz Bir Gencin Sozleri", "Romantizma", 2007, 231, Tur.RAP),
        ("Melankolia Girisi", "Sagopa Kajmer", 2003, 47, Tur.SKIT),
        ("Enstrumantal Ara", "Bir Pesimistin Gozyaslari", 2005, 96, Tur.ENSTRUMANTAL),
    ]
    for baslik, album, yil, sure, tur in tohum:
        _DEPO[_SONRAKI_ID] = SarkiKayit(
            id=_SONRAKI_ID,
            baslik=baslik,
            album=album,
            yil=yil,
            sure_sn=sure,
            tur=tur,
            ic_not="tohum kayit - disari cikmamali",
            kaynak_ip="127.0.0.1",
        )
        _SONRAKI_ID += 1


_tohumla()


# --- Saglik -------------------------------------------------------------
@app.get("/saglik", tags=["saglik"], summary="Servis ayakta mi")
def saglik() -> dict[str, object]:
    return {"durum": "ayakta", "kayit_sayisi": len(_DEPO), "cors_acik": CORS_ACIK}


# --- GET: okuma ---------------------------------------------------------
@app.get(
    "/sarkilar",
    tags=["sarkilar"],
    response_model=SayfaliYanit,
    summary="Sarkilari listele (query parametreleriyle filtre)",
)
def sarkilari_listele(
    response: Response,
    # Query parametreleri de Pydantic ile dogrulaniyor: limit=0 veya limit=999
    # endpoint'e hic girmeden 422 doner.
    album: str | None = Query(None, description="Albüm adina gore filtre (kismi eslesme)"),
    tur: Tur | None = Query(None, description="Ture gore filtre"),
    limit: int = Query(10, ge=1, le=100, description="Sayfa basina kayit"),
    offset: int = Query(0, ge=0, description="Atlanacak kayit sayisi"),
) -> SayfaliYanit:
    kayitlar = list(_DEPO.values())
    if album:
        kayitlar = [k for k in kayitlar if album.lower() in k.album.lower()]
    if tur:
        kayitlar = [k for k in kayitlar if k.tur == tur]

    toplam = len(kayitlar)
    dilim = kayitlar[offset : offset + limit]
    # Ozel basligi CORS'ta expose_headers ile aciga cikardik - fark oradan gorunur.
    response.headers["X-Toplam-Kayit"] = str(toplam)
    return SayfaliYanit(toplam=toplam, limit=limit, offset=offset, kayitlar=dilim)


@app.get(
    "/sarkilar/{sarki_id}",
    tags=["sarkilar"],
    response_model=SarkiYanit,
    responses={404: {"model": HataYaniti, "description": "Kayit yok"}},
    summary="Tek sarki getir (path parametresi)",
)
def sarki_getir(sarki_id: int) -> SarkiKayit:
    # sarki_id int olarak anote edildigi icin /sarkilar/abc daha endpoint'e
    # girmeden 422 doner - elle int() cevirimi ve try/except gerekmiyor.
    kayit = _DEPO.get(sarki_id)
    if kayit is None:
        raise HTTPException(status_code=404, detail=f"{sarki_id} numarali sarki yok")
    # DIKKAT: SarkiKayit donduruyoruz (ic_not, kaynak_ip iceriyor) ama
    # response_model=SarkiYanit oldugu icin FastAPI o alanlari kirpiyor.
    # Giden veri dogrulamasinin somut kaniti bu.
    return kayit


# --- POST: yazma --------------------------------------------------------
@app.post(
    "/sarkilar",
    tags=["sarkilar"],
    response_model=SarkiYanit,
    status_code=status.HTTP_201_CREATED,
    summary="Yeni sarki ekle (govde Pydantic ile dogrulanir)",
)
def sarki_ekle(govde: SarkiOlustur, request: Request) -> SarkiKayit:
    global _SONRAKI_ID
    kayit = SarkiKayit(
        id=_SONRAKI_ID,
        **govde.model_dump(),
        ic_not="API uzerinden eklendi",
        kaynak_ip=request.client.host if request.client else "?",
    )
    _DEPO[_SONRAKI_ID] = kayit
    _SONRAKI_ID += 1
    return kayit


@app.put(
    "/sarkilar/{sarki_id}",
    tags=["sarkilar"],
    response_model=SarkiYanit,
    responses={404: {"model": HataYaniti}},
    summary="Sarkiyi kismi guncelle",
)
def sarki_guncelle(sarki_id: int, govde: SarkiGuncelle) -> SarkiKayit:
    kayit = _DEPO.get(sarki_id)
    if kayit is None:
        raise HTTPException(status_code=404, detail=f"{sarki_id} numarali sarki yok")
    # exclude_unset: gonderilmeyen alan None ile ezilmesin.
    guncel = kayit.model_copy(update=govde.model_dump(exclude_unset=True))
    _DEPO[sarki_id] = guncel
    return guncel


@app.delete(
    "/sarkilar/{sarki_id}",
    tags=["sarkilar"],
    status_code=status.HTTP_204_NO_CONTENT,
    responses={404: {"model": HataYaniti}},
    summary="Sarkiyi sil",
)
def sarki_sil(sarki_id: int) -> Response:
    if sarki_id not in _DEPO:
        raise HTTPException(status_code=404, detail=f"{sarki_id} numarali sarki yok")
    del _DEPO[sarki_id]
    # 204 "govde yok" demek; JSON dondurursek spec ihlali olur.
    return Response(status_code=status.HTTP_204_NO_CONTENT)


# --- Ogretici uclar -----------------------------------------------------
@app.get("/ogretici/echo", tags=["ogretici"], summary="GET: veri query string'de")
def echo_get(mesaj: str = Query(..., min_length=1, max_length=200)) -> dict[str, str]:
    """
    GET'te veri URL'de tasinir: tarayici gecmisine, sunucu loglarina ve
    Referer basligina yazilir, cache'lenebilir, yer imine eklenebilir.
    """
    return {"metot": "GET", "mesaj": mesaj, "nerede": "query string (URL'de gorunur)"}


@app.post("/ogretici/echo", tags=["ogretici"], summary="POST: veri govdede")
def echo_post(govde: dict) -> dict[str, object]:
    """
    Ayni yol, farkli metot - FastAPI ikisini ayri uc olarak kaydeder.
    POST'ta veri govdede tasinir: URL'de gorunmez, boyut siniri pratikte cok
    daha yuksek, tarayici kendiliginden tekrar etmez (idempotent degil).
    """
    return {"metot": "POST", "govde": govde, "nerede": "request body (URL'de gorunmez)"}


@app.get("/ogretici/gizli-alan", tags=["ogretici"], response_model=SarkiYanit)
def gizli_alan() -> SarkiKayit:
    """
    Bilerek ic alanlarla dolu bir kayit donduruyoruz. Yanitta `ic_not` ve
    `kaynak_ip` YOK - cunku response_model onlari tanimiyor.
    """
    return SarkiKayit(
        id=999,
        baslik="Sizinti testi",
        album="Test",
        yil=2024,
        sure_sn=100,
        ic_not="BU DISARI CIKMAMALI",
        kaynak_ip="10.0.0.1",
    )


@app.get("/ogretici/cors-durumu", tags=["ogretici"])
def cors_durumu() -> dict[str, object]:
    return {
        "cors_acik": CORS_ACIK,
        "izinli_originler": IZINLI_ORIGINLER if CORS_ACIK else [],
        "not": "CORS_ACIK=0 ile sunucuyu yeniden baslatinca middleware kalkar.",
    }


# --- Hata sekli ---------------------------------------------------------
@app.exception_handler(HTTPException)
async def http_hata(request: Request, exc: HTTPException) -> JSONResponse:
    """HTTPException'in varsayilan {"detail": ...} sekli yerine kendi semamiz."""
    return JSONResponse(status_code=exc.status_code, content={"detay": exc.detail})


if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="127.0.0.1", port=int(os.getenv("PORT", "8000")))
