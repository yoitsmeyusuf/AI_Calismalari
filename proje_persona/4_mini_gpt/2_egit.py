"""
Asama 3: egitim dongusu.

Iki kosu yapiliyor:
  taban   -> odevin onerdigi hiperparametreler (n_embd=128, n_layer=3, dropout=0)
  olcekli -> veriye gore kucultulmus + dropout (n_embd=96, n_layer=2, dropout=0.2)

Korpus 261 KB, yani odevin onerdigi 500 KB-1 MB'in yarisi; 5000 iter ~170
epoch demek. Taban kosunun ezberlemesi bekleniyor - amac bunu gizlemek degil
val loss egrisinde GOSTERMEK ve ikinci kosuyla karsilastirmak.

    ../../.venv/bin/python 2_egit.py              # ikisi de
    ../../.venv/bin/python 2_egit.py --sadece taban
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI))
sys.path.insert(0, str(BURASI.parent))

import ayarlar  # noqa: E402
from model import MiniGPT  # noqa: E402

GORSELLER = BURASI / "gorseller"
KAYITLAR = BURASI / "checkpoints"
GORSELLER.mkdir(exist_ok=True)
KAYITLAR.mkdir(exist_ok=True)

YUZEY, MURK, IKINCIL = "#fcfcfb", "#0b0b0b", "#52514e"
SERI = ["#2a78d6", "#eb6834", "#1baf7a"]


def parti(veri: torch.Tensor, block_size: int, batch: int, aygit: str):
    ix = torch.randint(len(veri) - block_size - 1, (batch,))
    x = torch.stack([veri[i:i + block_size] for i in ix])
    y = torch.stack([veri[i + 1:i + 1 + block_size] for i in ix])
    return x.to(aygit), y.to(aygit)


@torch.no_grad()
def kayip_olc(model, bolumler, block_size, batch, aygit, tekrar=50):
    model.eval()
    sonuc = {}
    for ad, veri in bolumler.items():
        kayiplar = torch.zeros(tekrar)
        for i in range(tekrar):
            x, y = parti(veri, block_size, batch, aygit)
            _, k = model(x, y)
            kayiplar[i] = k.item()
        sonuc[ad] = kayiplar.mean().item()
    model.train()
    return sonuc


def bigram_taban_cizgisi(train: torch.Tensor, val: torch.Tensor, vocab_size: int) -> float:
    """
    Karsilastirma cizgisi: sadece onceki karakteri kullanan bigram sayaci.
    MiniGPT'nin ogrendigi seyin "sik harf" istatistiginden fazlasi oldugunu
    gostermek icin gerekli - yoksa 1.5 loss'un iyi mi kotu mu oldugu belirsiz.
    """
    sayim = torch.ones(vocab_size, vocab_size)  # Laplace duzeltmesi
    sayim.index_put_((train[:-1], train[1:]), torch.ones(len(train) - 1), accumulate=True)
    olasilik = sayim / sayim.sum(dim=1, keepdim=True)
    log_p = olasilik[val[:-1], val[1:]].log()
    return float(-log_p.mean())


def egit(ad: str, hp: dict, veri: dict, aygit: str) -> dict:
    torch.manual_seed(1337)
    model = MiniGPT(vocab_size=veri["vocab_size"], **hp).to(aygit)
    opt = torch.optim.AdamW(model.parameters(), lr=ayarlar.GPT_LR)
    bolumler = {"train": veri["train"], "val": veri["val"]}

    print(f"\n{'='*70}\n{ad.upper()}  {hp}\n{'='*70}")
    print(f"parametre: {model.parametre_sayisi():,}")

    gecmis = {"iter": [], "train": [], "val": []}
    t0 = time.time()
    for it in range(ayarlar.GPT_ITER + 1):
        if it % ayarlar.GPT_DEGERLENDIRME_ARALIGI == 0 or it == ayarlar.GPT_ITER:
            k = kayip_olc(model, bolumler, hp["block_size"], ayarlar.GPT_BATCH, aygit)
            gecmis["iter"].append(it)
            gecmis["train"].append(k["train"])
            gecmis["val"].append(k["val"])
            print(f"  iter {it:>5}  train {k['train']:.4f}  val {k['val']:.4f}"
                  f"  fark {k['val']-k['train']:+.4f}  ({time.time()-t0:.0f} sn)")
        if it == ayarlar.GPT_ITER:
            break
        x, y = parti(veri["train"], hp["block_size"], ayarlar.GPT_BATCH, aygit)
        _, kayip = model(x, y)
        opt.zero_grad(set_to_none=True)
        kayip.backward()
        opt.step()

    en_iyi = min(range(len(gecmis["val"])), key=lambda i: gecmis["val"][i])
    gecmis["en_iyi_iter"] = gecmis["iter"][en_iyi]
    gecmis["en_iyi_val"] = gecmis["val"][en_iyi]
    gecmis["son_train"] = gecmis["train"][-1]
    gecmis["son_val"] = gecmis["val"][-1]
    gecmis["parametre"] = model.parametre_sayisi()
    gecmis["sure_sn"] = round(time.time() - t0, 1)
    gecmis["hp"] = hp

    torch.save({"model": model.state_dict(), "hp": hp,
                "stoi": veri["stoi"], "itos": veri["itos"],
                "vocab_size": veri["vocab_size"]}, KAYITLAR / f"{ad}.pt")
    print(f"  en dusuk val: {gecmis['en_iyi_val']:.4f} @ iter {gecmis['en_iyi_iter']}")
    print(f"  -> checkpoints/{ad}.pt")
    return gecmis


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sadece", choices=["taban", "olcekli"], default=None)
    args = ap.parse_args()

    veri = torch.load(BURASI / "veri_hazir.pt", weights_only=False)
    aygit = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"aygit: {aygit}"
          + (f" ({torch.cuda.get_device_name(0)})" if aygit == "cuda" else ""))
    print(f"vocab_size: {veri['vocab_size']}  train: {len(veri['train']):,} token")

    bigram = bigram_taban_cizgisi(veri["train"], veri["val"], veri["vocab_size"])
    tekduze = float(torch.tensor(float(veri["vocab_size"])).log())
    print(f"\nTaban cizgileri (val, dogal log):")
    print(f"  tekduze (rastgele tahmin) : {tekduze:.4f}")
    print(f"  bigram sayaci             : {bigram:.4f}")

    kosular = {"taban": ayarlar.GPT_TABAN, "olcekli": ayarlar.GPT_OLCEKLI}
    if args.sadece:
        kosular = {args.sadece: kosular[args.sadece]}

    sonuclar = {ad: egit(ad, hp, veri, aygit) for ad, hp in kosular.items()}
    sonuclar["_taban_cizgileri"] = {"tekduze": tekduze, "bigram": bigram}
    (BURASI / "egitim_sonuclari.json").write_text(
        json.dumps(sonuclar, ensure_ascii=False, indent=2), encoding="utf-8")

    # --- Loss egrisi ---
    fig, eksenler = plt.subplots(1, len(kosular), figsize=(5.6 * len(kosular), 4.2),
                                 facecolor=YUZEY, squeeze=False)
    for ax, (ad, g) in zip(eksenler[0], ((a, sonuclar[a]) for a in kosular)):
        # Etiket yonu son degere gore: olcekli kosuda iki egri neredeyse
        # ust uste bitiyor ve sabit yon (train yukari / val asagi) etiketleri
        # capraz getirip cakistiriyordu.
        son = {b: g[b][-1] for b in ("train", "val")}
        for i, bolum in enumerate(("train", "val")):
            ax.plot(g["iter"], g[bolum], marker="o", markersize=5, linewidth=2,
                    color=SERI[i], label=bolum, zorder=3)
            ustte = son[bolum] >= max(son.values())
            ax.annotate(bolum, xy=(g["iter"][-1], son[bolum]),
                        xytext=(-4, 12 if ustte else -20), textcoords="offset points",
                        color=SERI[i], fontsize=9, ha="right", weight="bold")
        ax.axhline(bigram, color=IKINCIL, linewidth=1, linestyle=":", zorder=2)
        ax.annotate("bigram taban cizgisi", xy=(0, bigram), xytext=(4, 5),
                    textcoords="offset points", color=IKINCIL, fontsize=8)
        # Sadece isaretci; "en dusuk val" metni basliga tasindi cunku grafik
        # icinde nereye konsa (ust: bigram cizgisi, alt: train etiketi)
        # bir seyin uzerine biniyordu.
        ax.scatter([g["en_iyi_iter"]], [g["en_iyi_val"]], s=140, facecolors="none",
                   edgecolors=SERI[1], linewidths=2, zorder=4)
        ax.set_facecolor(YUZEY)
        ax.set_title(f"{ad}  ({g['parametre']:,} parametre)\n"
                     f"en dusuk val {g['en_iyi_val']:.3f} @ iter {g['en_iyi_iter']}",
                     color=MURK, fontsize=11, loc="left", pad=10)
        ax.set_xlabel("iterasyon", color=IKINCIL, fontsize=9)
        ax.set_ylabel("cross-entropy kayip (nat)", color=IKINCIL, fontsize=9)
        ax.tick_params(colors=IKINCIL, labelsize=8, length=3)
        for k in ("top", "right"):
            ax.spines[k].set_visible(False)
        for k in ("left", "bottom"):
            ax.spines[k].set_color("#d8d7d2")
        ax.grid(True, color="#e8e7e3", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.legend(frameon=False, fontsize=8, labelcolor=IKINCIL, loc="best")
    fig.tight_layout()
    fig.savefig(GORSELLER / "loss_egrisi.png", dpi=150, facecolor=YUZEY)
    plt.close(fig)
    print(f"\n  -> gorseller/loss_egrisi.png")
    print(f"  -> egitim_sonuclari.json")


if __name__ == "__main__":
    main()
