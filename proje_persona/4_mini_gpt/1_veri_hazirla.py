"""
Asama 1: karakter duzeyinde veri hazirligi.

    korpus .txt -> vocab -> stoi/itos -> torch.tensor -> %90 train / %10 val

    ../../.venv/bin/python 1_veri_hazirla.py
"""
from __future__ import annotations

import collections
import json
import sys
from pathlib import Path

import torch

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI.parent))
import ayarlar  # noqa: E402

CIKTI = BURASI / "veri_hazir.pt"
SOZLUK_JSON = BURASI / "sozluk.json"
NADIR = "�"  # nadir karakterlerin toplandigi tek sembol


def main() -> None:
    if not ayarlar.KORPUS_TXT.exists():
        raise SystemExit(f"korpus yok: {ayarlar.KORPUS_TXT}\n"
                         f"once: ../.venv/bin/python proje_persona/veri_topla.py")

    metin = ayarlar.KORPUS_TXT.read_text(encoding="utf-8")
    print(f"korpus: {len(metin):,} karakter ({len(metin)/1024:.0f} KB)")

    sayim = collections.Counter(metin)
    print(f"ham vocab_size: {len(sayim)}")

    # Nadir karakterleri tek sembolde birlestir. 10'dan az gecen bir karakter
    # icin model anlamli bir gomme ogrenemez; o satirlar olu agirliga doner.
    # Ayrica generate() sirasinda dusuk olasilikla ortaya cikip cikti kalitesini
    # bozarlar.
    nadirler = {c for c, n in sayim.items() if n < ayarlar.GPT_NADIR_ESIGI}
    print(f"nadir karakter (<{ayarlar.GPT_NADIR_ESIGI} kez): {len(nadirler)}"
          f"  toplam gecis: {sum(sayim[c] for c in nadirler)}"
          f"  (korpusun %{100*sum(sayim[c] for c in nadirler)/len(metin):.3f}'u)")

    if nadirler:
        metin = "".join(NADIR if c in nadirler else c for c in metin)

    karakterler = sorted(set(metin))
    vocab_size = len(karakterler)
    stoi = {c: i for i, c in enumerate(karakterler)}
    itos = {i: c for c, i in stoi.items()}
    print(f"son vocab_size: {vocab_size}")

    veri = torch.tensor([stoi[c] for c in metin], dtype=torch.long)
    n = int(ayarlar.GPT_EGITIM_ORANI * len(veri))
    train, val = veri[:n], veri[n:]

    print(f"\ntrain: {len(train):,} token  ({100*ayarlar.GPT_EGITIM_ORANI:.0f}%)")
    print(f"val  : {len(val):,} token")

    # Odev bilgisi: bir epoch kac iterasyon eder?
    token_per_iter = ayarlar.GPT_BATCH * ayarlar.GPT_TABAN["block_size"]
    epoch = ayarlar.GPT_ITER * token_per_iter / len(train)
    print(f"\n{ayarlar.GPT_ITER} iter x {ayarlar.GPT_BATCH} batch x "
          f"{ayarlar.GPT_TABAN['block_size']} block = "
          f"{ayarlar.GPT_ITER*token_per_iter:,} token")
    print(f"  -> egitim verisi uzerinde ~{epoch:.0f} EPOCH")
    print("  Odevin onerdigi 500 KB-1 MB yerine 261 KB oldugu icin ezber")
    print("  bekleniyor; 2_egit.py bunu val loss egrisinde olcuyor.")

    torch.save({"train": train, "val": val, "stoi": stoi, "itos": itos,
                "vocab_size": vocab_size}, CIKTI)
    SOZLUK_JSON.write_text(
        json.dumps({"vocab_size": vocab_size,
                    "karakterler": karakterler,
                    "nadir_birlestirilen": sorted(nadirler)},
                   ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"\n  -> {CIKTI.name}")
    print(f"  -> {SOZLUK_JSON.name}")


if __name__ == "__main__":
    main()
