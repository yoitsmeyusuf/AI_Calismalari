"""
Ortak korpus: Sagopa Kajmer sarki sozleri.

Hem 3_rag_persona (chunk + vektor) hem 4_mini_gpt (karakter dizisi) bunu
kullaniyor, o yuzden proje kokunde duruyor.

Kaynak secimi ve lisans notu icin ayarlar.py ve ust README'ye bakin. Ozetle:
scrape edilebilir siteler ya Cloudflare duvarli ya da "ai-train=no" diyor;
MIT lisansli hazir veri seti tercih edildi. Lisans DERLEMEYE ait, sarki
sozlerinin telifi hak sahiplerinde - bu yuzden korpus git'e girmiyor ve
HF'ye push edilmiyor.

    ../.venv/bin/python proje_persona/veri_topla.py
    ../.venv/bin/python proje_persona/veri_topla.py --istatistik   # sadece rapor
"""
from __future__ import annotations

import argparse
import collections
import json
import re
import sys
import unicodedata
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import ayarlar  # noqa: E402


def indir() -> "list[dict[str, str]]":
    """HF Hub'dan ham CSV'yi indirip sanatciya gore filtreler."""
    import pandas as pd
    from huggingface_hub import hf_hub_download

    yol = hf_hub_download(
        ayarlar.KAYNAK_DATASET, ayarlar.KAYNAK_DOSYA, repo_type="dataset"
    )
    df = pd.read_csv(yol)
    s = df[df["artist"].astype(str).str.strip().str.lower() == ayarlar.SANATCI.lower()]
    return [
        {"baslik": str(r.title).strip(), "sozler": str(r.lyrics)}
        for r in s.itertuples()
        if isinstance(r.lyrics, str) and r.lyrics.strip()
    ]


def temizle(metin: str) -> str:
    """
    Az mudahale: korpus MiniGPT'nin uslubu ogrenecegi seyin ta kendisi.

    Kufur/argo FILTRELENMIYOR - 1. haftadaki guvenlik taramasinin aynisini
    burada yapmak uslubu silmek olurdu. Kontrol cikti tarafinda
    (3_rag_persona/persona.py guardrail'i).
    """
    metin = unicodedata.normalize("NFKC", metin)
    metin = metin.replace("\r\n", "\n").replace("\r", "\n")
    # Satir ici fazla bosluk; satir yapisi korunuyor (chunk'lama ona dayaniyor)
    satirlar = [re.sub(r"[ \t]+", " ", s).strip() for s in metin.split("\n")]
    satirlar = [s for s in satirlar if s]
    return "\n".join(satirlar)


def tekille(kayitlar: "list[dict[str, str]]") -> "list[dict[str, str]]":
    """Basliga gore tekilleme; ayni baslikta en uzun metin kalir."""
    def anahtar(b: str) -> str:
        return re.sub(r"\s+", " ", re.sub(r"[^\w\s]", "", b.lower())).strip()

    en_iyi: dict[str, dict[str, str]] = {}
    for k in kayitlar:
        a = anahtar(k["baslik"])
        if a not in en_iyi or len(k["sozler"]) > len(en_iyi[a]["sozler"]):
            en_iyi[a] = k
    return list(en_iyi.values())


def istatistik(kayitlar: "list[dict[str, str]]") -> dict:
    metinler = [k["sozler"] for k in kayitlar]
    toplam = sum(len(m) for m in metinler)
    satir_sayilari = sorted(m.count("\n") + 1 for m in metinler)
    satir_uzunluklari = sorted(len(s) for m in metinler for s in m.split("\n"))
    karakterler = collections.Counter()
    for m in metinler:
        karakterler.update(m)

    def p(dizi: list[int], yuzde: float) -> int:
        return dizi[min(int(len(dizi) * yuzde), len(dizi) - 1)]

    return {
        "sarki": len(kayitlar),
        "karakter": toplam,
        "kb": round(toplam / 1024, 1),
        "satir_medyan": p(satir_sayilari, 0.5),
        "satir_p10": p(satir_sayilari, 0.10),
        "satir_max": satir_sayilari[-1],
        "satir_uzunluk_medyan": p(satir_uzunluklari, 0.5),
        "satir_uzunluk_p90": p(satir_uzunluklari, 0.90),
        "vocab_size": len(karakterler),
        "nadir_karakter": sum(1 for _, n in karakterler.items() if n < ayarlar.GPT_NADIR_ESIGI),
        "en_sik_40_kapsam": round(
            100 * sum(n for _, n in karakterler.most_common(40)) / sum(karakterler.values()), 2
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--istatistik", action="store_true", help="sadece raporla, yazma")
    args = ap.parse_args()

    print(f"Kaynak : {ayarlar.KAYNAK_DATASET} / {ayarlar.KAYNAK_DOSYA}")
    print(f"Sanatci: {ayarlar.SANATCI}\n")

    ham = indir()
    print(f"  ham kayit          : {len(ham)}")
    kayitlar = tekille(ham)
    print(f"  basliga gore tekil : {len(kayitlar)}")
    for k in kayitlar:
        k["sozler"] = temizle(k["sozler"])
    kayitlar = [k for k in kayitlar if len(k["sozler"]) >= 100]
    print(f"  temizlik sonrasi   : {len(kayitlar)}  (100 karakterden kisalar elendi)")

    ist = istatistik(kayitlar)
    print("\n--- KORPUS PROFILI ---")
    for ad, deger in ist.items():
        print(f"  {ad:<24}: {deger}")

    if args.istatistik:
        print("\n(--istatistik verildi, dosya yazilmadi)")
        return

    ayarlar.VERI.mkdir(parents=True, exist_ok=True)
    # MiniGPT icin: sarkilar arasi cift newline ayraci. Model sarki sinirini
    # da ogrensin diye; korpusta zaten bos satir yok, bu yuzden "\n\n" gercek
    # bir sinir isareti oluyor.
    ayarlar.KORPUS_TXT.write_text(
        "\n\n".join(k["sozler"] for k in kayitlar), encoding="utf-8"
    )
    with ayarlar.KORPUS_JSONL.open("w", encoding="utf-8") as f:
        for k in kayitlar:
            f.write(json.dumps(k, ensure_ascii=False) + "\n")
    (ayarlar.VERI / "korpus_istatistik.json").write_text(
        json.dumps(ist, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    print(f"\n  -> {ayarlar.KORPUS_TXT.relative_to(ayarlar.KOK)} "
          f"({ayarlar.KORPUS_TXT.stat().st_size / 1024:.0f} KB)")
    print(f"  -> {ayarlar.KORPUS_JSONL.relative_to(ayarlar.KOK)}")
    print(f"  -> veri/korpus_istatistik.json")


if __name__ == "__main__":
    main()
