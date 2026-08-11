"""
Sarki sozlerini bar (satir) sinirindan chunk'lara boler.

Neden bu kural: korpusta KITA AYRIMI YOK. Olculen durum (bkz. ust README):
  - bos satirla ayrilmis kita  : 260 sarkinin 1'inde
  - [Nakarat] / (x2) etiketi   : hicbirinde
  - satir/sarki                : medyan 24
  - satir uzunlugu             : medyan 39, p90 54 karakter

Yani 10. haftadaki gibi "belgenin kendi bolum sinirindan kes" mumkun degil;
sinir KURULMAK zorunda. Bar medyani 39 karakter oldugu icin 6 barlik grup
~230 karaktere denk geliyor ve odevin istedigi 200-400 araligina dogal
olarak oturuyor.

Tek basina calistirilirsa ornek doker ve dagilimi raporlar:
    ../../.venv/bin/python chunklama.py
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ayarlar  # noqa: E402


def sarkiyi_chunkla(
    baslik: str,
    sozler: str,
    bar: int = ayarlar.CHUNK_BAR,
    bar_min: int = ayarlar.CHUNK_BAR_MIN,
    ust_sinir: int = ayarlar.CHUNK_KARAKTER_HEDEFI[1],
) -> list[dict]:
    """
    Kayan degil, ARDISIK pencereler: her bar tam olarak bir chunk'a girer.

    Overlap yok. 10. haftada ogrenilen sey burada da gecerli: overlap, keyfi
    bir noktadan kesince anlamin ikiye bolunmesini telafi eden bir yamadir.
    Sarki sozunde tekrar cok (nakarat), overlap eklemek ayni dizeleri
    veritabanina uc-dort kez koyup arama sonuclarini tek bir nakaratla
    doldururdu.

    Tek istisna: bir grup ust siniri asarsa daha kucuk parcalara bolunur.
    """
    satirlar = [s.strip() for s in sozler.split("\n") if s.strip()]
    if not satirlar:
        return []

    gruplar: list[list[str]] = []
    i = 0
    while i < len(satirlar):
        grup = satirlar[i:i + bar]
        # Son grup cok kisa kalirsa oncekine yapistir (tek basina anlamsiz
        # bir-iki dizelik chunk uretmemek icin).
        if len(grup) < bar_min and gruplar:
            gruplar[-1].extend(grup)
        else:
            gruplar.append(grup)
        i += bar

    chunklar: list[dict] = []
    for g in gruplar:
        metin = "\n".join(g)
        if len(metin) <= ust_sinir:
            parcalar = [g]
        else:
            # Ust siniri asan grubu ikiye bol; nadir (uzun barlarda oluyor).
            orta = max(bar_min, len(g) // 2)
            parcalar = [g[:orta], g[orta:]]
        for p in parcalar:
            if not p:
                continue
            m = "\n".join(p)
            chunklar.append({
                "sarki": baslik,
                "bar_sayisi": len(p),
                "karakter": len(m),
                "chunk_text": m,
                # Gomulecek metin: basligi enjekte ediyoruz. Sarki sozleri
                # anafora dolu ("o", "bu", "sen"); tek basina bir chunk cogu
                # zaman neden bahsettigini soylemiyor. chunk_text ham kaliyor.
                "gomme_metni": f"title: {baslik} | text: {m}",
            })
    return chunklar


def korpusu_chunkla() -> list[dict]:
    import json

    chunklar: list[dict] = []
    with ayarlar.KORPUS_JSONL.open(encoding="utf-8") as f:
        for satir in f:
            k = json.loads(satir)
            chunklar += sarkiyi_chunkla(k["baslik"], k["sozler"])
    for i, c in enumerate(chunklar):
        c["id"] = f"chunk_{i:05d}"
    return chunklar


if __name__ == "__main__":
    chunklar = korpusu_chunkla()
    uzunluklar = sorted(c["karakter"] for c in chunklar)
    barlar = sorted(c["bar_sayisi"] for c in chunklar)
    n = len(chunklar)

    def p(dizi, yuzde):
        return dizi[min(int(len(dizi) * yuzde), len(dizi) - 1)]

    alt, ust = ayarlar.CHUNK_KARAKTER_HEDEFI
    print(f"chunk sayisi        : {n}")
    print(f"sarki sayisi        : {len({c['sarki'] for c in chunklar})}")
    print(f"karakter medyan     : {p(uzunluklar, 0.5)}")
    print(f"karakter p10 / p90  : {p(uzunluklar, 0.10)} / {p(uzunluklar, 0.90)}")
    print(f"karakter min / max  : {uzunluklar[0]} / {uzunluklar[-1]}")
    print(f"bar medyan          : {p(barlar, 0.5)}")
    icinde = sum(1 for u in uzunluklar if alt <= u <= ust)
    print(f"\nodev araligi {alt}-{ust} icinde : {icinde}/{n} (%{100*icinde/n:.0f})")
    print(f"  {alt} altinda : {sum(1 for u in uzunluklar if u < alt)}")
    print(f"  {ust} ustunde : {sum(1 for u in uzunluklar if u > ust)}")
