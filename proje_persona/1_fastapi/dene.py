"""
Kanit script'i: API'yi kendisi ayaga kaldirir, her iddiayi olcer, tablo doker.

README'deki tablolar bu script'in ciktisindan geliyor - elle yazilmadi.
Sunucuyu iki kez baslatiyor: CORS_ACIK=1 ve CORS_ACIK=0, cunku CORS'un ne
yaptigini gostermenin tek yolu iki durumu yan yana koymak.

    ../../.venv/bin/python dene.py
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

BURASI = Path(__file__).resolve().parent
PORT = 8011  # 8000 elle acik olabilir, catismasin
TABAN = f"http://127.0.0.1:{PORT}"
TARAYICI_ORIGIN = "http://localhost:8080"


def sunucu_baslat(cors_acik: bool) -> subprocess.Popen:
    ortam = {**os.environ, "CORS_ACIK": "1" if cors_acik else "0"}
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "main:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=BURASI,
        env=ortam,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    for _ in range(100):
        try:
            httpx.get(f"{TABAN}/saglik", timeout=1.0)
            return p
        except Exception:
            time.sleep(0.2)
    p.kill()
    raise SystemExit("sunucu ayaga kalkmadi")


def baslik(s: str) -> None:
    print(f"\n{'=' * 74}\n{s}\n{'=' * 74}")


def satir(ad: str, beklenen: object, gercek: object, not_: str = "") -> None:
    # bool'u str'e cevirmeden formatlarsak int gibi basilir (True -> "1").
    b, g = str(beklenen), str(gercek)
    isaret = "OK " if b == g else "FARK"
    print(f"  [{isaret}] {ad:<44} bekl={b:<5} gercek={g:<5} {not_}")


def main() -> None:
    p = sunucu_baslat(cors_acik=True)
    try:
        c = httpx.Client(base_url=TABAN, timeout=10.0)

        # ---------------------------------------------------------------
        baslik("1. HTTP METOTLARI VE ROUTING")
        r = c.get("/saglik")
        satir("GET /saglik", 200, r.status_code)

        r = c.get("/sarkilar")
        satir("GET /sarkilar (liste)", 200, r.status_code, f"toplam={r.json()['toplam']}")

        r = c.get("/sarkilar/1")
        satir("GET /sarkilar/1 (path param)", 200, r.status_code)

        r = c.get("/sarkilar/9999")
        satir("GET /sarkilar/9999 (yok)", 404, r.status_code, f"govde={r.json()}")

        r = c.post("/sarkilar", json={"baslik": "Dene", "album": "Dene", "yil": 2020, "sure_sn": 200})
        satir("POST /sarkilar (yeni kayit)", 201, r.status_code, f"id={r.json()['id']}")
        yeni_id = r.json()["id"]

        r = c.put(f"/sarkilar/{yeni_id}", json={"yil": 2021})
        satir("PUT (kismi guncelleme)", 200, r.status_code, f"yil={r.json()['yil']}")

        r = c.delete(f"/sarkilar/{yeni_id}")
        satir("DELETE", 204, r.status_code, f"govde uzunlugu={len(r.content)}")

        # Ayni yol, iki metot -> iki ayri uc
        g = c.get("/ogretici/echo", params={"mesaj": "merhaba"})
        po = c.post("/ogretici/echo", json={"mesaj": "merhaba"})
        satir("GET  /ogretici/echo", 200, g.status_code, g.json()["nerede"])
        satir("POST /ogretici/echo", 200, po.status_code, po.json()["nerede"])

        r = c.request("GET", "/sarkilar", params={"limit": 2})
        r2 = c.post("/sarkilar/1", json={})
        satir("POST /sarkilar/1 (metot yok)", 405, r2.status_code, "Method Not Allowed")

        # ---------------------------------------------------------------
        baslik("2. PYDANTIC - GELEN VERI DOGRULAMASI")
        vakalar: list[tuple[str, dict, int]] = [
            ("gecerli govde", {"baslik": "A", "album": "B", "yil": 2020, "sure_sn": 100}, 201),
            ("eksik alan (album yok)", {"baslik": "A", "yil": 2020, "sure_sn": 100}, 422),
            ("yanlis tip (yil='iki bin')", {"baslik": "A", "album": "B", "yil": "iki bin", "sure_sn": 100}, 422),
            ("sayi stringi (yil='2020')", {"baslik": "A", "album": "B", "yil": "2020", "sure_sn": 100}, 201),
            ("aralik disi (yil=1800)", {"baslik": "A", "album": "B", "yil": 1800, "sure_sn": 100}, 422),
            ("gelecek yil (yil=2099)", {"baslik": "A", "album": "B", "yil": 2099, "sure_sn": 100}, 422),
            ("sure_sn=0 (gt=0)", {"baslik": "A", "album": "B", "yil": 2020, "sure_sn": 0}, 422),
            ("bos baslik ('')", {"baslik": "", "album": "B", "yil": 2020, "sure_sn": 100}, 422),
            ("sadece bosluk ('   ')", {"baslik": "   ", "album": "B", "yil": 2020, "sure_sn": 100}, 422),
            ("gecersiz enum (tur='pop')", {"baslik": "A", "album": "B", "yil": 2020, "sure_sn": 100, "tur": "pop"}, 422),
            ("fazla alan (extra=forbid)", {"baslik": "A", "album": "B", "yil": 2020, "sure_sn": 100, "xx": 1}, 422),
            ("yazim hatasi (basslik)", {"basslik": "A", "album": "B", "yil": 2020, "sure_sn": 100}, 422),
        ]
        for ad, govde, bekl in vakalar:
            r = c.post("/sarkilar", json=govde)
            ek = ""
            if r.status_code == 422:
                d = r.json()["detail"][0]
                ek = f"{d['type']} @ {'.'.join(str(x) for x in d['loc'][1:])}"
            satir(ad, bekl, r.status_code, ek)

        print("\n  422 govdesinin sekli (yil='iki bin' ornegi):")
        r = c.post("/sarkilar", json={"baslik": "A", "album": "B", "yil": "iki bin", "sure_sn": 100})
        import json as _json

        for h in _json.dumps(r.json(), ensure_ascii=False, indent=2).splitlines():
            print("   ", h)

        baslik("3. PYDANTIC - GIDEN VERI DOGRULAMASI")
        r = c.get("/ogretici/gizli-alan")
        alanlar = set(r.json())
        satir("ic_not yanitta yok", True, "ic_not" not in alanlar)
        satir("kaynak_ip yanitta yok", True, "kaynak_ip" not in alanlar)
        satir("turetilmis alan sure_mmss var", True, "sure_mmss" in alanlar,
              f"deger={r.json().get('sure_mmss')}")
        print(f"    donen alanlar: {sorted(alanlar)}")

        # Query parametresi de dogrulanir
        satir("limit=0 (ge=1)", 422, c.get("/sarkilar", params={"limit": 0}).status_code)
        satir("limit=999 (le=100)", 422, c.get("/sarkilar", params={"limit": 999}).status_code)
        satir("/sarkilar/abc (int degil)", 422, c.get("/sarkilar/abc").status_code)

        # ---------------------------------------------------------------
        baslik("4. SWAGGER UI / OPENAPI")
        satir("GET /docs (Swagger UI)", 200, c.get("/docs").status_code)
        satir("GET /redoc", 200, c.get("/redoc").status_code)
        sema = c.get("/openapi.json").json()
        satir("GET /openapi.json", 200, 200)
        yollar = sorted(sema["paths"])
        print(f"    OpenAPI surumu : {sema['openapi']}")
        print(f"    tanimli yol    : {len(yollar)}")
        print(f"    sema (model)   : {len(sema['components']['schemas'])} adet")
        print(f"    /sarkilar metotlari: {sorted(sema['paths']['/sarkilar'])}")
        print(f"    SarkiYanit alanlari: {sorted(sema['components']['schemas']['SarkiYanit']['properties'])}")

        # ---------------------------------------------------------------
        baslik("5. CORS - MIDDLEWARE ACIKKEN")
        r = c.options(
            "/sarkilar",
            headers={
                "Origin": TARAYICI_ORIGIN,
                "Access-Control-Request-Method": "POST",
                "Access-Control-Request-Headers": "content-type,x-istek-kimligi",
            },
        )
        satir("preflight OPTIONS", 200, r.status_code)
        for h in ["access-control-allow-origin", "access-control-allow-methods",
                  "access-control-allow-headers", "access-control-allow-credentials",
                  "access-control-max-age"]:
            print(f"    {h:34s}: {r.headers.get(h, '(yok)')}")

        r = c.get("/sarkilar", headers={"Origin": TARAYICI_ORIGIN})
        print(f"    basit GET -> allow-origin      : {r.headers.get('access-control-allow-origin', '(yok)')}")
        print(f"    basit GET -> expose-headers    : {r.headers.get('access-control-expose-headers', '(yok)')}")

        r = c.options("/sarkilar", headers={"Origin": "http://kotu-site.example",
                                            "Access-Control-Request-Method": "POST"})
        satir("izinsiz origin preflight", 400, r.status_code,
              f"allow-origin={r.headers.get('access-control-allow-origin', '(yok)')}")
        c.close()
    finally:
        p.terminate()
        p.wait(timeout=10)

    # -------------------------------------------------------------------
    p = sunucu_baslat(cors_acik=False)
    try:
        c = httpx.Client(base_url=TABAN, timeout=10.0)
        baslik("6. CORS - MIDDLEWARE KAPALIYKEN (ayni istekler)")
        r = c.options("/sarkilar", headers={"Origin": TARAYICI_ORIGIN,
                                            "Access-Control-Request-Method": "POST"})
        satir("preflight OPTIONS", 405, r.status_code, "izin verecek kimse yok")
        r = c.get("/sarkilar", headers={"Origin": TARAYICI_ORIGIN})
        satir("GET /sarkilar sunucuya ULASTI", 200, r.status_code)
        print(f"    allow-origin basligi           : {r.headers.get('access-control-allow-origin', '(yok)')}")
        print("\n    Onemli: istek 200 dondu. Sunucu engellemedi, ENGELLEYEN TARAYICI.")
        print("    curl/httpx CORS uygulamaz; bu yuzden kanit icin cors_demo/index.html gerekiyor.")
        c.close()
    finally:
        p.terminate()
        p.wait(timeout=10)

    print("\nBitti.\n")


if __name__ == "__main__":
    main()
