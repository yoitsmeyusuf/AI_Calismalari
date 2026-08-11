"""
Loss egrisini egitim_sonuclari.json'dan yeniden cizer.

Ayri bir dosya olmasinin sebebi pratik: gorselde bir etiket yerini
duzeltmek icin 5 dakikalik egitimi tekrar kosmak gerekmesin. Butun loss
gecmisi zaten JSON'da.

    ../../.venv/bin/python cizim.py
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BURASI = Path(__file__).resolve().parent
GORSELLER = BURASI / "gorseller"
GORSELLER.mkdir(exist_ok=True)

YUZEY, MURK, IKINCIL = "#fcfcfb", "#0b0b0b", "#52514e"
SERI = ["#2a78d6", "#eb6834"]


def main() -> None:
    sonuclar = json.loads((BURASI / "egitim_sonuclari.json").read_text(encoding="utf-8"))
    bigram = sonuclar["_taban_cizgileri"]["bigram"]
    kosular = [a for a in sonuclar if not a.startswith("_")]

    fig, eksenler = plt.subplots(1, len(kosular), figsize=(5.8 * len(kosular), 4.4),
                                 facecolor=YUZEY, squeeze=False)
    for ax, ad in zip(eksenler[0], kosular):
        g = sonuclar[ad]
        son = {b: g[b][-1] for b in ("train", "val")}
        for i, bolum in enumerate(("train", "val")):
            ax.plot(g["iter"], g[bolum], marker="o", markersize=5, linewidth=2,
                    color=SERI[i], label=bolum, zorder=3)
            # Etiketler egrinin SAGINDAKI bosluga yaziliyor. Egri ucunun
            # altina/ustune konunca ya x-ekseni etiketiyle ("5000") ya da
            # olcekli kosuda birbirleriyle cakisiyorlardi.
            ustte = son[bolum] >= max(son.values())
            ax.annotate(bolum, xy=(g["iter"][-1], son[bolum]),
                        xytext=(8, 5 if ustte else -5), textcoords="offset points",
                        color=SERI[i], fontsize=9, ha="left", va="center",
                        weight="bold", annotation_clip=False)
        ax.set_xlim(-g["iter"][-1] * 0.03, g["iter"][-1] * 1.14)

        ax.axhline(bigram, color=IKINCIL, linewidth=1, linestyle=":", zorder=2)
        ax.annotate("bigram taban cizgisi", xy=(0, bigram), xytext=(4, 5),
                    textcoords="offset points", color=IKINCIL, fontsize=8)
        ax.scatter([g["en_iyi_iter"]], [g["en_iyi_val"]], s=150, facecolors="none",
                   edgecolors=SERI[1], linewidths=2, zorder=4)

        ax.set_facecolor(YUZEY)
        ax.set_title(f"{ad}  ({g['parametre']:,} parametre)\n"
                     f"en dusuk val {g['en_iyi_val']:.3f} @ iter {g['en_iyi_iter']}"
                     f"   |   son fark {g['son_val'] - g['son_train']:+.2f}",
                     color=MURK, fontsize=10.5, loc="left", pad=10)
        ax.set_xlabel("iterasyon", color=IKINCIL, fontsize=9)
        ax.set_ylabel("cross-entropy kayip (nat)", color=IKINCIL, fontsize=9)
        ax.tick_params(colors=IKINCIL, labelsize=8, length=3)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        for k in ("left", "bottom"):
            ax.spines[k].set_color("#d8d7d2")
        ax.grid(True, color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, labelcolor=IKINCIL, loc="center right")

    fig.tight_layout()
    yol = GORSELLER / "loss_egrisi.png"
    fig.savefig(yol, dpi=150, facecolor=YUZEY)
    plt.close(fig)
    print(f"  -> {yol.relative_to(BURASI)}")
    for ad in kosular:
        g = sonuclar[ad]
        print(f"  {ad:<10} en dusuk val {g['en_iyi_val']:.4f} @ {g['en_iyi_iter']:>5}"
              f"  son train {g['son_train']:.4f}  son val {g['son_val']:.4f}")


if __name__ == "__main__":
    main()
