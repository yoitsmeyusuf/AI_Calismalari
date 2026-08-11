"""
Asama 4: uretim (inference) + EZBER OLCUMU.

Taban kosu 170 epoch'ta train loss 0.35'e indi; boyle bir modelin "urettigi"
metnin aslinda korpustan birebir kopya olma ihtimali yuksek. Bu yuzden her
ornek icin korpusla en uzun ortak alt dizi (LCS) olculuyor.

Olcum iki isi birden goruyor:
  1. Teknik: ezberin val loss'tan bagimsiz, dogrudan kaniti.
  2. Pratik: telifli metnin birebir yeniden uretilmesini yakalayip
     raporlanacak orneklerden eliyor.

    ../../.venv/bin/python 3_uret.py
    ../../.venv/bin/python 3_uret.py --model olcekli --temperature 0.8
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import torch

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI))
sys.path.insert(0, str(BURASI.parent))

import ayarlar  # noqa: E402
from model import MiniGPT  # noqa: E402

KAYITLAR = BURASI / "checkpoints"
CIKTI_DIZINI = BURASI / "uretimler"
# Bu esigin ustunde ortak alt dizi = birebir kopya sayilir ve rapora girmez.
# 40 karakter Sagopa'nin bar medyaninin (39) hemen ustu, yani "tam bir dize"
# uzunlugu. Ilk deneme 60'ti; taban kosunun T=0.8'de urettigi 50 karakterlik
# birebir alinti esigi geciyor ve rapora giriyordu. Uretilen metin bir bari
# butun olarak kopyaliyorsa o uretim degil, hatirlama.
KOPYA_ESIGI = 40


def yukle(ad: str, aygit: str):
    kayit = torch.load(KAYITLAR / f"{ad}.pt", weights_only=False)
    model = MiniGPT(vocab_size=kayit["vocab_size"], **kayit["hp"]).to(aygit)
    model.load_state_dict(kayit["model"])
    model.eval()
    return model, kayit["stoi"], kayit["itos"]


def en_uzun_ortak_altdizi(uretim: str, korpus: str, tavan: int = 400) -> tuple[int, str]:
    """
    Uretilen metnin korpusta BIREBIR gecen en uzun parcasi.

    Ikili arama + dilim kumesi: her uzunluk icin uretimin butun alt
    dizilerini bir kumeye alip korpusta arıyoruz. Naif O(n*m) yerine
    O(log(tavan) * n) civari.
    """
    def var_mi(uzunluk: int) -> str | None:
        if uzunluk <= 0 or uzunluk > len(uretim):
            return None
        for i in range(len(uretim) - uzunluk + 1):
            parca = uretim[i:i + uzunluk]
            if parca in korpus:
                return parca
        return None

    alt, ust, en_iyi = 1, min(tavan, len(uretim)), ""
    while alt <= ust:
        orta = (alt + ust) // 2
        bulunan = var_mi(orta)
        if bulunan is not None:
            en_iyi = bulunan
            alt = orta + 1
        else:
            ust = orta - 1
    return len(en_iyi), en_iyi


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=None, choices=["taban", "olcekli"])
    ap.add_argument("--uzunluk", type=int, default=500)
    ap.add_argument("--temperature", type=float, default=None)
    ap.add_argument("--top_k", type=int, default=None)
    ap.add_argument("--adet", type=int, default=3)
    args = ap.parse_args()

    aygit = "cuda" if torch.cuda.is_available() else "cpu"
    korpus = ayarlar.KORPUS_TXT.read_text(encoding="utf-8")
    CIKTI_DIZINI.mkdir(exist_ok=True)

    modeller = [args.model] if args.model else ["taban", "olcekli"]
    sicakliklar = [args.temperature] if args.temperature else [0.6, 0.8, 1.0]

    rapor: dict = {}
    for ad in modeller:
        model, stoi, itos = yukle(ad, aygit)
        kodla = lambda s: torch.tensor([[stoi[c] for c in s]], dtype=torch.long, device=aygit)
        coz = lambda t: "".join(itos[int(i)] for i in t)

        print(f"\n{'='*72}\n{ad.upper()}  ({model.parametre_sayisi():,} parametre)\n{'='*72}")
        rapor[ad] = []
        for sic in sicakliklar:
            torch.manual_seed(42)
            baslangic = kodla("\n")
            for n in range(args.adet):
                cikti = model.generate(baslangic, args.uzunluk,
                                       temperature=sic, top_k=args.top_k)
                metin = coz(cikti[0])[1:]
                uzunluk, parca = en_uzun_ortak_altdizi(metin, korpus)
                kopya = uzunluk >= KOPYA_ESIGI
                rapor[ad].append({
                    "temperature": sic,
                    "ornek": n,
                    "uzunluk": len(metin),
                    "en_uzun_kopya": uzunluk,
                    "kopya_orani": round(uzunluk / len(metin), 3),
                    "birebir_kopya": kopya,
                    "metin": metin,
                })
            grup = [r for r in rapor[ad] if r["temperature"] == sic]
            ort = sum(r["en_uzun_kopya"] for r in grup) / len(grup)
            enb = max(r["en_uzun_kopya"] for r in grup)
            print(f"  T={sic}: en uzun birebir kopya  ort {ort:5.1f}  max {enb:3d} karakter"
                  f"  ({'KOPYA VAR' if enb >= KOPYA_ESIGI else 'esik alti'})")

    # --- Ozet ---
    print(f"\n{'='*72}\nEZBER KARSILASTIRMASI (esik {KOPYA_ESIGI} karakter)\n{'='*72}")
    print(f"  {'model':<10}{'T':>6}{'ort kopya':>12}{'max kopya':>12}{'kopya/uzunluk':>16}")
    print(f"  {'-'*10}{'-'*6:>6}{'-'*12:>12}{'-'*12:>12}{'-'*16:>16}")
    for ad in modeller:
        for sic in sicakliklar:
            grup = [r for r in rapor[ad] if r["temperature"] == sic]
            print(f"  {ad:<10}{sic:>6}"
                  f"{sum(r['en_uzun_kopya'] for r in grup)/len(grup):>12.1f}"
                  f"{max(r['en_uzun_kopya'] for r in grup):>12}"
                  f"{max(r['kopya_orani'] for r in grup):>16.1%}")

    (CIKTI_DIZINI / "uretimler.json").write_text(
        json.dumps(rapor, ensure_ascii=False, indent=2), encoding="utf-8")

    # Rapora yalnizca esigin ALTINDA kalan ornekler yaziliyor.
    for ad in modeller:
        temiz = [r for r in rapor[ad] if not r["birebir_kopya"]]
        yol = CIKTI_DIZINI / f"{ad}_ornekler.txt"
        with yol.open("w", encoding="utf-8") as f:
            f.write(f"# {ad} - korpusla birebir ortusmesi {KOPYA_ESIGI} karakterin\n"
                    f"# ALTINDA kalan ornekler ({len(temiz)}/{len(rapor[ad])})\n\n")
            for r in temiz:
                f.write(f"--- T={r['temperature']} ornek {r['ornek']} "
                        f"(en uzun kopya {r['en_uzun_kopya']} karakter) ---\n")
                f.write(r["metin"] + "\n\n")
        print(f"  -> uretimler/{yol.name}  ({len(temiz)}/{len(rapor[ad])} ornek esik alti)")
    print(f"  -> uretimler/uretimler.json  (hepsi, olcumleriyle)")


if __name__ == "__main__":
    main()
