"""
"FastAPI hizlidir" iddiasini olcen script.

Flask/Django kurup TechEmpower tekrari yapmak yerine farkin KAYNAGINI olcuyoruz:
WSGI (Flask, Django'nun klasik hali) her istegi bir is parcacigi/surec isgal
ederek isler; ASGI (FastAPI/Starlette) I/O beklerken event loop'u serbest
birakir. Bu fark FastAPI'nin kendi icinde de gozlenebilir:

    async def + await asyncio.sleep -> event loop serbest kalir (ASGI modeli)
    def      + time.sleep           -> is parcacigi havuzunda blokle (WSGI'a benzer)

Ayrica ucuncu bir vaka var ve asil ogretici olan o: CPU-yogun isi `async def`
icine koymak. Orada async KAZANDIRMAZ, tam tersine event loop'u kilitler.
"FastAPI her zaman hizli" degil; "I/O bekleyen is yukunde hizli".

    ../../.venv/bin/python kiyas.py
"""
from __future__ import annotations

import asyncio
import os
import subprocess
import sys
import time
from pathlib import Path

import httpx

BURASI = Path(__file__).resolve().parent
PORT = 8012
TABAN = f"http://127.0.0.1:{PORT}"

# Gecikme ve eszamanlilik bilerek yuksek: 50 ms / 100 istekte istemci tarafi
# ek yuku (baglanti kurma, event loop planlama) sunucu modelinin farkini
# bastiriyordu ve fark 1.2x gibi anlamsiz bir seye dusuyordu. 200 ms ve 200
# eszamanli istekte is parcacigi havuzunun (40) siniri net gorunuyor.
ESZAMANLI = 200
IO_GECIKME = 0.20
CPU_DONGU = 40_000_000

SUNUCU = '''
import asyncio, time
from fastapi import FastAPI

app = FastAPI()
IO_GECIKME = %f
CPU_DONGU = %d

@app.get("/bos")
async def bos():
    # Gecikmesiz uc: istemci tarafi ek yukunu olcmek icin taban cizgisi.
    return {"tip": "bos"}

@app.get("/io-async")
async def io_async():
    # I/O beklerken event loop serbest: diger istekler ayni anda ilerler.
    await asyncio.sleep(IO_GECIKME)
    return {"tip": "io-async"}

@app.get("/io-sync")
def io_sync():
    # `def` uc -> Starlette bunu is parcacigi havuzuna atar (varsayilan 40).
    # Havuz dolunca istekler sirada bekler; WSGI'in worker modeline benzer.
    time.sleep(IO_GECIKME)
    return {"tip": "io-sync"}

@app.get("/cpu-async")
async def cpu_async():
    # CPU-yogun is `async def` icinde: await yok, event loop KILITLENIR.
    # Tum sunucu bu istek bitene kadar baska hicbir sey yapamaz.
    t = 0
    for i in range(CPU_DONGU): t += i
    return {"tip": "cpu-async", "t": t}

@app.get("/cpu-sync")
def cpu_sync():
    # Ayni is `def` icinde: is parcacigi havuzuna gider, event loop serbest.
    t = 0
    for i in range(CPU_DONGU): t += i
    return {"tip": "cpu-sync", "t": t}
''' % (IO_GECIKME, CPU_DONGU)


async def yukle(yol: str, n: int) -> tuple[float, float]:
    limits = httpx.Limits(max_connections=n + 10, max_keepalive_connections=n + 10)
    async with httpx.AsyncClient(base_url=TABAN, timeout=120.0, limits=limits) as c:
        await c.get(yol)  # isinma
        t0 = time.perf_counter()
        yanitlar = await asyncio.gather(*(c.get(yol) for _ in range(n)))
        sure = time.perf_counter() - t0
    assert all(r.status_code == 200 for r in yanitlar)
    return sure, n / sure


def main() -> None:
    gecici = BURASI / "_kiyas_sunucu.py"
    gecici.write_text(SUNUCU, encoding="utf-8")
    p = subprocess.Popen(
        [sys.executable, "-m", "uvicorn", "_kiyas_sunucu:app", "--port", str(PORT), "--log-level", "warning"],
        cwd=BURASI, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
        env={**os.environ},
    )
    try:
        for _ in range(120):
            try:
                httpx.get(f"{TABAN}/io-async", timeout=1.0)
                break
            except Exception:
                time.sleep(0.25)
        else:
            raise SystemExit("sunucu kalkmadi")

        # Taban cizgisi: gecikmesiz uc. Bu sure tamamen istemci + ag ek yuku.
        taban, _ = asyncio.run(yukle("/bos", ESZAMANLI))

        print(f"\n{ESZAMANLI} eszamanli istek, her biri {IO_GECIKME*1000:.0f} ms I/O bekliyor")
        print(f"(teorik alt sinir: async {IO_GECIKME:.2f} sn | "
              f"sync {IO_GECIKME * -(-ESZAMANLI // 40):.2f} sn = {-(-ESZAMANLI // 40)} parti x 40 is parcacigi)")
        print(f"(olculen istemci ek yuku: {taban:.3f} sn - asagidaki iki sureden de dusulmeli)\n")
        print(f"  {'uc':<12} {'sure (sn)':>10} {'istek/sn':>10}  {'yorum'}")
        print(f"  {'-'*12} {'-'*10:>10} {'-'*10:>10}  {'-'*38}")

        sonuc = {}
        for yol, yorum in [
            ("/io-async", "event loop serbest - hepsi paralel"),
            ("/io-sync", "40'lik is parcacigi havuzu - partiler halinde"),
        ]:
            sure, rps = asyncio.run(yukle(yol, ESZAMANLI))
            sonuc[yol] = sure
            print(f"  {yol:<12} {sure:>10.3f} {rps:>10.1f}  {yorum}")

        kat = sonuc["/io-sync"] / sonuc["/io-async"]
        net_a = max(sonuc["/io-async"] - taban, 1e-6)
        net_s = max(sonuc["/io-sync"] - taban, 1e-6)
        print(f"\n  -> ham oran           : async {kat:.1f}x hizli")
        print(f"  -> ek yuk dusulunce   : async {net_s / net_a:.1f}x hizli "
              f"({net_a:.3f} sn vs {net_s:.3f} sn)")
        print("     Ham oran istemci ek yuku yuzunden dusuk gorunuyor; sunucu")
        print("     modelinin gercek farki ikinci satir.")
        print("     WSGI/ASGI farkinin ozu bu: bekleyen istek kaynak isgal etmiyor.")

        # --- CPU: asil zarar yavaslama degil, DIGER istekleri bloklamak ----
        # Toplam sureyi olcmek yaniltici (GIL yuzunden ikisi de serilesir).
        # Olculmesi gereken: agir bir istek KOSARKEN, hafif bir istek ne kadar
        # bekliyor? Head-of-line blocking'i gosteren sey bu.
        print("\nBaskilama testi: 1 agir istek kosarken 10 hafif istek (/io-async, 200 ms)")
        print("Hafif isteklerin gecikmesi olculuyor - dusuk = sunucu cevap verebiliyor.\n")
        print(f"  {'agir uc':<12} {'hafif istek medyan gecikme':>28}  {'yorum'}")
        print(f"  {'-'*12} {'-'*28:>28}  {'-'*34}")

        async def baskilama(agir_yol: str) -> float:
            limits = httpx.Limits(max_connections=40, max_keepalive_connections=40)
            async with httpx.AsyncClient(base_url=TABAN, timeout=180.0, limits=limits) as c:
                await c.get("/io-async")
                agir = asyncio.create_task(c.get(agir_yol))
                await asyncio.sleep(0.05)  # agir istek islenmeye baslasin

                async def hafif() -> float:
                    t0 = time.perf_counter()
                    await c.get("/io-async")
                    return time.perf_counter() - t0

                sureler = await asyncio.gather(*(hafif() for _ in range(10)))
                await agir
            return sorted(sureler)[len(sureler) // 2]

        for yol, yorum in [
            ("/cpu-async", "event loop kilitli -> hafifler bekliyor"),
            ("/cpu-sync", "havuza gitti -> event loop serbest"),
        ]:
            gec = asyncio.run(baskilama(yol))
            sonuc[yol] = gec
            print(f"  {yol:<12} {gec*1000:>25.0f} ms  {yorum}")

        print(f"\n  -> `async def` icine await'siz CPU isi koymak hafif istekleri "
              f"{sonuc['/cpu-async']/sonuc['/cpu-sync']:.1f}x geciktiriyor.")
        print("     Toplam is/sn ikisinde de benzer (GIL); fark CEVAP VEREBILIRLIKTE.")
        print("     Dogru refleks: CPU-yogun ucu `def` yazmak (havuza gitsin).\n")
    finally:
        p.terminate()
        p.wait(timeout=10)
        gecici.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
