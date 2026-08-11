"""
Gorev 4-7'nin hepsini kosar, README'deki tablolari ve gorseller/ altindaki
figurleri uretir.

    ../../.venv/bin/python test_gorevler.py
"""
from __future__ import annotations

import math
import sys
from pathlib import Path

import matplotlib
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI))

from cok_kafali_dikkat import MultiHeadAttention  # noqa: E402
from dikkat import ScaledDotProductAttention, nedensel_mask, olcekleme_testi  # noqa: E402
from pozisyon_kodlama import GommeVePozisyon, PositionalEncoding, permutasyon_testi  # noqa: E402
from transformer_blok import BlokYigini, TransformerBlock, gradyan_olcumu  # noqa: E402

GORSELLER = BURASI / "gorseller"
GORSELLER.mkdir(exist_ok=True)

# --- Palet -----------------------------------------------------------------
# dataviz referans paletinin ilk uc kategorik slotu. Bu ucu all-pairs modunda
# dogrulandi (CVD dE 9.2, normal gorus dE 24.0, light yuzey #fcfcfb).
# Aqua'nin yuzeye karsi kontrasti 3:1'in altinda -> dogrudan etiket zorunlu,
# o yuzden her seri legend'in yani sira ucunda etiketleniyor.
YUZEY = "#fcfcfb"
MURK = "#0b0b0b"
IKINCIL = "#52514e"
SERI = ["#2a78d6", "#eb6834", "#1baf7a"]

# Sequential: tek hue (mavi), acik -> koyu. Buyukluk kodlar.
ARDISIK = LinearSegmentedColormap.from_list(
    "mavi", ["#cde2fb", "#9ec5f4", "#5598e7", "#2a78d6", "#256abf", "#184f95", "#0d366b"]
)
# Diverging: mavi <-> kirmizi, notr gri orta nokta. Isaretli buyukluk kodlar.
IKIYONLU = LinearSegmentedColormap.from_list("mavi_kirmizi", ["#184f95", "#2a78d6", "#f0efec", "#e34948", "#a02020"])


def eksen_duzenle(ax: plt.Axes, baslik: str = "", x: str = "", y: str = "") -> None:
    """Izgara ve eksenler geri planda; veri onde."""
    ax.set_facecolor(YUZEY)
    if baslik:
        ax.set_title(baslik, color=MURK, fontsize=11, pad=10, loc="left")
    ax.set_xlabel(x, color=IKINCIL, fontsize=9)
    ax.set_ylabel(y, color=IKINCIL, fontsize=9)
    ax.tick_params(colors=IKINCIL, labelsize=8, length=3)
    for kenar in ("top", "right"):
        ax.spines[kenar].set_visible(False)
    for kenar in ("left", "bottom"):
        ax.spines[kenar].set_color("#d8d7d2")
    ax.grid(True, color="#e8e7e3", linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)


def baslik(s: str) -> None:
    print(f"\n{'=' * 74}\n{s}\n{'=' * 74}")


def kontrol(ad: str, kosul: bool, ek: str = "") -> bool:
    print(f"  [{'OK ' if kosul else 'HATA'}] {ad:<52} {ek}")
    return kosul


# ===========================================================================
def gorev4() -> list[bool]:
    baslik("GOREV 4: Positional Encoding + Embedding")
    s = []
    d_model = 128
    pe = PositionalEncoding(d_model, max_len=512)

    s.append(kontrol("PE sekli (1, max_len, d_model)", tuple(pe.pe.shape) == (1, 512, 128)))
    s.append(kontrol("ogrenilebilir parametre yok", sum(p.numel() for p in pe.parameters()) == 0))
    s.append(kontrol("state_dict'te buffer var", "pe" in pe.state_dict()))
    s.append(kontrol("pos=0 cift indisler sin(0)=0", bool(pe.pe[0, 0, 0::2].abs().max() < 1e-6)))
    s.append(kontrol("pos=0 tek indisler cos(0)=1", bool((pe.pe[0, 0, 1::2] - 1).abs().max() < 1e-6)))
    s.append(kontrol("degerler [-1, 1] araliginda", bool(pe.pe.abs().max() <= 1.0)))
    normlar = pe.pe[0].norm(dim=-1)
    s.append(kontrol("pozisyon vektoru normu sabit", bool((normlar - normlar[0]).abs().max() < 1e-4),
                     f"|pe| = {normlar[0]:.4f} = sqrt(d_model/2)"))

    kat = GommeVePozisyon(vocab_size=50, d_model=d_model)
    idx = torch.randint(0, 50, (2, 10))
    s.append(kontrol("embedding + PE sekli korur", kat(idx).shape == (2, 10, d_model)))

    p = permutasyon_testi()
    s.append(kontrol("PE YOK -> permutasyona esdeger", p["pe_yok_fark"] < 1e-5,
                     f"max fark {p['pe_yok_fark']:.2e}"))
    s.append(kontrol("PE VAR -> sira gorunuyor", p["pe_var_fark"] > 0.1,
                     f"max fark {p['pe_var_fark']:.4f}"))

    # Olcek uyusmazligi - README'ye not
    g = kat.gomme(idx) * math.sqrt(d_model)
    print(f"\n  Olcek notu: gomme std={g.std():.2f}, PE std={pe.pe[0, :10].std():.2f}"
          f"  -> oran {g.std() / pe.pe[0, :10].std():.1f}x")
    print("  N(0,1) baslatma + sqrt(d_model) carpani PE'yi bastiriyor. MiniGPT")
    print("  ogrenilebilir pozisyon gommesi kullanacak (nanoGPT gibi, std=0.02).")

    # --- Gorsel: PE matrisi ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=YUZEY,
                                   gridspec_kw={"width_ratios": [1.25, 1]})
    im = ax1.imshow(pe.pe[0, :64].numpy().T, aspect="auto", cmap=IKIYONLU,
                    vmin=-1, vmax=1, interpolation="nearest")
    eksen_duzenle(ax1, "PE matrisi (ilk 64 pozisyon)", "pozisyon", "boyut (d_model)")
    ax1.grid(False)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    cb.ax.tick_params(colors=IKINCIL, labelsize=8)
    cb.outline.set_visible(False)

    # Boyutlar bilerek uzak secildi: dalga boyu boyut indisiyle USSEL buyuyor
    # (period ~ 2*pi*10000^(i/d_model)). Yakin indisler (0,1,8,20) ekranda
    # ust uste binip okunamiyordu.
    N = 128
    for i, boyut in enumerate([0, 24, 56]):
        periyot = 2 * math.pi * 10000 ** (boyut / d_model)
        ax2.plot(pe.pe[0, :N, boyut].numpy(), linewidth=2, color=SERI[i],
                 label=f"boyut {boyut}  (periyot ~{periyot:.0f})", zorder=3)
        ax2.annotate(f"d{boyut}", xy=(N - 1, pe.pe[0, N - 1, boyut].item()),
                     xytext=(5, 0), textcoords="offset points",
                     color=SERI[i], fontsize=9, va="center", weight="bold")
    eksen_duzenle(ax2, "Boyut indisi buyudukce dalga boyu ussel artiyor", "pozisyon", "deger")
    # Legend'e yer acmak icin ust bosluk; aksi halde egrilerin uzerine biniyordu.
    ax2.set_ylim(-1.1, 2.0)
    ax2.set_yticks([-1.0, -0.5, 0.0, 0.5, 1.0])
    ax2.legend(frameon=False, fontsize=8, labelcolor=IKINCIL, loc="upper left", ncol=1)
    ax2.set_xlim(0, N + 8)
    fig.tight_layout()
    fig.savefig(GORSELLER / "gorev4_pozisyon_kodlama.png", dpi=150, facecolor=YUZEY)
    plt.close(fig)
    print(f"\n  -> gorseller/gorev4_pozisyon_kodlama.png")
    return s


# ===========================================================================
def gorev5() -> list[bool]:
    baslik("GOREV 5: Scaled Dot-Product Attention")
    s = []
    torch.manual_seed(0)
    dikkat = ScaledDotProductAttention()
    B, S, d_k = 2, 6, 16
    q, k, v = torch.randn(B, S, d_k), torch.randn(B, S, d_k), torch.randn(B, S, d_k)

    cikti, agirlik = dikkat(q, k, v)
    s.append(kontrol("cikti sekli (B, S, d_v)", cikti.shape == (B, S, d_k)))
    s.append(kontrol("agirlik sekli (B, S, S)", agirlik.shape == (B, S, S)))
    s.append(kontrol("agirlik satir toplami = 1", torch.allclose(agirlik.sum(-1), torch.ones(B, S))))

    m = nedensel_mask(S)
    _, a_mask = dikkat(q, k, v, mask=m)
    s.append(kontrol("causal mask: ust ucgen tam 0", float(a_mask.triu(diagonal=1).abs().max()) == 0.0))
    s.append(kontrol("maskeliyken satir toplami hala 1",
                     torch.allclose(a_mask.sum(-1), torch.ones(B, S))))
    s.append(kontrol("ilk token yalniz kendini gorur", bool((a_mask[:, 0, 0] - 1).abs().max() < 1e-6)))

    olc = olcekleme_testi()
    e_ham = [r["entropi_ham"] for r in olc]
    e_olc = [r["entropi_olcekli"] for r in olc]
    s.append(kontrol("olceksiz: d_k buyudukce entropi duser", all(a > b for a, b in zip(e_ham, e_ham[1:])),
                     f"{e_ham[0]:.2f} -> {e_ham[-1]:.2f} nat"))
    s.append(kontrol("olcekli: entropi d_k'dan bagimsiz", max(e_olc) - min(e_olc) < 0.05,
                     f"{min(e_olc):.3f} - {max(e_olc):.3f} nat"))

    # --- Gorsel ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=YUZEY)
    im = ax1.imshow(a_mask[0].numpy(), cmap=ARDISIK, vmin=0, vmax=1, interpolation="nearest")
    eksen_duzenle(ax1, "Nedensel maskeli dikkat agirliklari", "anahtar (bakilan)", "sorgu (bakan)")
    ax1.grid(False)
    ax1.set_xticks(range(S)); ax1.set_yticks(range(S))
    for i in range(S):
        for j in range(S):
            d = a_mask[0, i, j].item()
            if j <= i:
                ax1.text(j, i, f"{d:.2f}", ha="center", va="center", fontsize=7,
                         color="#ffffff" if d > 0.5 else MURK)
    cb = fig.colorbar(im, ax=ax1, fraction=0.046, pad=0.02)
    cb.ax.tick_params(colors=IKINCIL, labelsize=8); cb.outline.set_visible(False)

    d_ks = [r["d_k"] for r in olc]
    for seri, (deg, etiket) in enumerate([(e_ham, "olceksiz"), (e_olc, "sqrt(d_k) ile olcekli")]):
        ax2.plot(d_ks, deg, marker="o", markersize=8, linewidth=2, color=SERI[seri],
                 label=etiket, zorder=3)
        # "olceksiz" etiketi son noktanin altina konunca 512 tick'iyle
        # cakisiyordu; ustune alindi.
        ax2.annotate(etiket, xy=(d_ks[-1], deg[-1]), xytext=(-6, 12),
                     textcoords="offset points", color=SERI[seri], fontsize=9,
                     ha="right", weight="bold")
    ax2.axhline(math.log(16), color=IKINCIL, linewidth=1, linestyle=":", zorder=2)
    ax2.annotate("duzgun dagilim (ln 16)", xy=(8, math.log(16)), xytext=(0, 5),
                 textcoords="offset points", color=IKINCIL, fontsize=8)
    ax2.set_xscale("log", base=2); ax2.set_xticks(d_ks)
    ax2.set_xticklabels([str(d) for d in d_ks])
    eksen_duzenle(ax2, "sqrt(d_k) bolmesi softmax'in doymasini engelliyor",
                  "d_k", "dikkat entropisi (nat)")
    ax2.set_ylim(0, 3.1)
    fig.tight_layout()
    fig.savefig(GORSELLER / "gorev5_dikkat.png", dpi=150, facecolor=YUZEY)
    plt.close(fig)
    print(f"\n  -> gorseller/gorev5_dikkat.png")
    return s


# ===========================================================================
def gorev6() -> list[bool]:
    baslik("GOREV 6: Multi-Head Attention (n_head=4, d_model=128)")
    s = []
    torch.manual_seed(0)
    B, S, d_model, n_head = 2, 10, 128, 4
    mha = MultiHeadAttention(d_model, n_head)
    x = torch.randn(B, S, d_model)
    cikti, agirlik = mha(x)

    s.append(kontrol("d_k = d_model / n_head", mha.d_k == 32, f"{d_model}/{n_head} = {mha.d_k}"))
    s.append(kontrol("cikti sekli girdiyle ayni", cikti.shape == x.shape, str(tuple(cikti.shape))))
    s.append(kontrol("agirlik sekli (B, n_head, S, S)", agirlik.shape == (B, n_head, S, S),
                     str(tuple(agirlik.shape))))
    s.append(kontrol("agirlik satir toplami = 1",
                     torch.allclose(agirlik.sum(-1), torch.ones(B, n_head, S), atol=1e-6)))

    p_sayilari = {h: sum(p.numel() for p in MultiHeadAttention(d_model, h).parameters())
                  for h in (1, 2, 4, 8, 16)}
    s.append(kontrol("parametre sayisi n_head'den bagimsiz", len(set(p_sayilari.values())) == 1,
                     f"{list(p_sayilari.values())[0]:,}"))

    torch.manual_seed(1)
    tek = MultiHeadAttention(32, n_head=1)
    y = torch.randn(1, 5, 32)
    with torch.no_grad():
        c1, _ = tek(y)
        ham, _ = ScaledDotProductAttention()(tek.w_q(y), tek.w_k(y), tek.w_v(y))
        c2 = tek.w_o(ham)
    s.append(kontrol("n_head=1 tek kafaliya esit", torch.allclose(c1, c2, atol=1e-6),
                     f"max fark {(c1 - c2).abs().max():.2e}"))

    with torch.no_grad():
        _, a_mask = mha(x, mask=nedensel_mask(S))
    s.append(kontrol("causal mask tum kafalarda", float(a_mask.triu(diagonal=1).abs().max()) == 0.0))

    # --- Gorsel: kafalar ---
    fig, eksenler = plt.subplots(1, n_head, figsize=(13, 3.4), facecolor=YUZEY)
    for h, ax in enumerate(eksenler):
        im = ax.imshow(a_mask[0, h].numpy(), cmap=ARDISIK, vmin=0, vmax=a_mask[0].max().item(),
                       interpolation="nearest")
        eksen_duzenle(ax, f"kafa {h}", "anahtar", "sorgu" if h == 0 else "")
        ax.grid(False)
        ax.set_xticks([0, 5, 9]); ax.set_yticks([0, 5, 9])
    cb = fig.colorbar(im, ax=eksenler, fraction=0.02, pad=0.02)
    cb.ax.tick_params(colors=IKINCIL, labelsize=8); cb.outline.set_visible(False)
    fig.suptitle("Nedensel maskeli 4 kafa - egitilmemis, fark yalnizca rastgele baslatmadan",
                 color=MURK, fontsize=10, x=0.02, ha="left")
    fig.savefig(GORSELLER / "gorev6_kafalar.png", dpi=150, facecolor=YUZEY, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  -> gorseller/gorev6_kafalar.png")
    return s


# ===========================================================================
def gorev7() -> list[bool]:
    baslik("GOREV 7: Transformer Block")
    s = []
    torch.manual_seed(0)
    d_model, n_head, B, S = 128, 4, 2, 16
    blok = TransformerBlock(d_model, n_head)
    x = torch.randn(B, S, d_model)
    cikti, agirlik = blok(x, agirlik_dondur=True)

    s.append(kontrol("blok sekli korur (ust uste konabilir)", cikti.shape == x.shape))
    p_dikkat = sum(p.numel() for p in blok.dikkat.parameters())
    p_ffn = sum(p.numel() for p in blok.ffn.parameters())
    s.append(kontrol("FFN parametrelerin cogunlugu", p_ffn > p_dikkat,
                     f"FFN {p_ffn:,} vs MHA {p_dikkat:,}"))

    for n in (1, 3, 6, 12):
        y = BlokYigini(d_model, n_head, n)
        with torch.no_grad():
            c = y(x)
        s.append(kontrol(f"{n} blok ust uste calisiyor", c.shape == x.shape,
                         f"parametre {sum(p.numel() for p in y.parameters()):,}"))

    derinlikler = (2, 4, 8, 12, 16)
    var = [gradyan_olcumu(n, residual=True)[2] for n in derinlikler]
    yok = [gradyan_olcumu(n, residual=False)[2] for n in derinlikler]
    s.append(kontrol("residual VAR: oran derinlikle cokmuyor", min(var) > 0.9,
                     f"{var[0]:.3f} -> {var[-1]:.3f}"))
    s.append(kontrol("residual YOK: oran derinlikle cokuyor", yok[-1] < 0.3,
                     f"{yok[0]:.3f} -> {yok[-1]:.3f}"))

    pre = [gradyan_olcumu(n, residual=True, pre_ln=True)[2] for n in derinlikler]
    post = [gradyan_olcumu(n, residual=True, pre_ln=False)[2] for n in derinlikler]
    s.append(kontrol("Pre-LN, Post-LN'den iyi sinyal tasiyor",
                     all(a >= b for a, b in zip(pre, post)),
                     f"{pre[-1]:.2f} vs {post[-1]:.2f} @ {derinlikler[-1]} katman"))

    # --- Gorsel ---
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4), facecolor=YUZEY)
    for i, (deg, etiket) in enumerate([(var, "residual VAR"), (yok, "residual YOK")]):
        ax1.plot(derinlikler, deg, marker="o", markersize=8, linewidth=2,
                 color=SERI[i], label=etiket, zorder=3)
        ax1.annotate(etiket, xy=(derinlikler[-1], deg[-1]), xytext=(-4, 10 if i == 0 else -20),
                     textcoords="offset points", color=SERI[i], fontsize=9, ha="right", weight="bold")
    ax1.axhline(1.0, color=IKINCIL, linewidth=1, linestyle=":", zorder=2)
    eksen_duzenle(ax1, "Residual: gradyan dibe kadar iniyor mu", "blok sayisi",
                  "ilk blok / son blok gradyan normu")
    # Legend "center left"te noktali 1.0 referans cizgisinin ve verinin
    # uzerine biniyordu; ust sol bosluga alindi.
    ax1.legend(frameon=False, fontsize=8, labelcolor=IKINCIL, loc="upper left")
    ax1.set_ylim(0, max(var) * 1.2)

    for i, (deg, etiket) in enumerate([(pre, "Pre-LN"), (post, "Post-LN")]):
        ax2.plot(derinlikler, deg, marker="o", markersize=8, linewidth=2,
                 color=SERI[i], label=etiket, zorder=3)
        ax2.annotate(etiket, xy=(derinlikler[-1], deg[-1]), xytext=(-4, 12 if i == 0 else -24),
                     textcoords="offset points", color=SERI[i], fontsize=9, ha="right", weight="bold")
    ax2.axhline(1.0, color=IKINCIL, linewidth=1, linestyle=":", zorder=2)
    eksen_duzenle(ax2, "LayerNorm yeri (ikisinde de residual var)", "blok sayisi",
                  "ilk blok / son blok gradyan normu")
    ax2.legend(frameon=False, fontsize=8, labelcolor=IKINCIL, loc="upper left")
    ax2.set_ylim(0, max(pre) * 1.2)
    fig.tight_layout()
    fig.savefig(GORSELLER / "gorev7_residual.png", dpi=150, facecolor=YUZEY)
    plt.close(fig)
    print(f"\n  -> gorseller/gorev7_residual.png")
    return s


if __name__ == "__main__":
    hepsi: list[bool] = []
    for f in (gorev4, gorev5, gorev6, gorev7):
        hepsi += f()
    baslik("OZET")
    print(f"  {sum(hepsi)}/{len(hepsi)} kontrol gecti")
    if not all(hepsi):
        sys.exit(1)
