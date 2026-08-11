"""
Gorev 7: Transformer Block.

Bir blok iki alt katman: Multi-Head Attention ve Feed-Forward. Her ikisi de
LayerNorm ve residual (artik) baglanti ile sariliyor. Sarma duzeni iki turlu
olabilir ve bu secim egitimin kararliligini belirliyor:

    Post-LN (orijinal, 2017) : x = LN(x + AltKatman(x))
    Pre-LN  (modern, GPT-2+) : x = x + AltKatman(LN(x))

Fark ince ama sonucu buyuk: Pre-LN'de girdiden ciktiya KESINTISIZ bir toplama
yolu var (residual "otoyol"), Post-LN'de her katmanda LayerNorm o yolu
kesiyor. Derinlestikce Post-LN warmup'siz yuksek learning rate'te iraksiyor.

MiniGPT 3 katmani warmup'siz lr=1e-3 ile egitecegi icin varsayilan Pre-LN.
Iddia asagida olculuyor.

    ../../.venv/bin/python transformer_blok.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from cok_kafali_dikkat import MultiHeadAttention  # noqa: E402
from dikkat import nedensel_mask  # noqa: E402


class FeedForward(nn.Module):
    """
    Iki linear + aktivasyon. Ic boyut genelde 4*d_model.

    Attention token'lar ARASINDA bilgi tasir; FFN her token'i KENDI ICINDE
    isler. Blogun parametrelerinin ~2/3'u burada.
    """

    def __init__(self, d_model: int, d_ff: int | None = None,
                 dropout: float = 0.0, aktivasyon: str = "relu") -> None:
        super().__init__()
        d_ff = d_ff or 4 * d_model
        self.net = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.ReLU() if aktivasyon == "relu" else nn.GELU(),
            nn.Linear(d_ff, d_model),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class TransformerBlock(nn.Module):
    def __init__(
        self,
        d_model: int,
        n_head: int,
        d_ff: int | None = None,
        dropout: float = 0.0,
        pre_ln: bool = True,
        residual: bool = True,
    ) -> None:
        super().__init__()
        self.pre_ln = pre_ln
        self.residual = residual  # yalnizca olcum icin kapatilabiliyor

        self.ln1 = nn.LayerNorm(d_model)
        self.dikkat = MultiHeadAttention(d_model, n_head, dropout)
        self.ln2 = nn.LayerNorm(d_model)
        self.ffn = FeedForward(d_model, d_ff, dropout)
        self.dropout = nn.Dropout(dropout)

    def forward(
        self, x: torch.Tensor, mask: torch.Tensor | None = None,
        agirlik_dondur: bool = False,
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        if self.pre_ln:
            a, agirlik = self.dikkat(self.ln1(x), mask=mask)
            x = x + self.dropout(a) if self.residual else self.dropout(a)
            f = self.ffn(self.ln2(x))
            x = x + f if self.residual else f
        else:
            a, agirlik = self.dikkat(x, mask=mask)
            x = self.ln1(x + self.dropout(a)) if self.residual else self.ln1(self.dropout(a))
            f = self.ffn(x)
            x = self.ln2(x + f) if self.residual else self.ln2(f)

        return (x, agirlik) if agirlik_dondur else x


class BlokYigini(nn.Module):
    """n_layer blogu ust uste koyar. Pre-LN'de sona bir LayerNorm gerekir."""

    def __init__(self, d_model: int, n_head: int, n_layer: int,
                 pre_ln: bool = True, residual: bool = True, dropout: float = 0.0) -> None:
        super().__init__()
        self.bloklar = nn.ModuleList([
            TransformerBlock(d_model, n_head, dropout=dropout, pre_ln=pre_ln, residual=residual)
            for _ in range(n_layer)
        ])
        # Pre-LN'de son blogun ciktisi hic normalize edilmeden cikar; bu yuzden
        # yigin sonunda tek bir LayerNorm standarttir (GPT-2'deki ln_f).
        self.ln_son = nn.LayerNorm(d_model) if pre_ln else nn.Identity()

    def forward(self, x: torch.Tensor, mask: torch.Tensor | None = None) -> torch.Tensor:
        for b in self.bloklar:
            x = b(x, mask=mask)
        return self.ln_son(x)


# --------------------------------------------------------------------------
# Kanit: residual baglanti neden onemli
# --------------------------------------------------------------------------
def gradyan_olcumu(n_layer: int, residual: bool, pre_ln: bool = True,
                   d_model: int = 64, n_head: int = 4, seq_len: int = 16,
                   tohum: int = 0) -> tuple[float, float, float]:
    """
    n_layer bloklu bir yigin kurar, ileri-geri gecis yapar ve ILK ile SON
    blogun gradyan normlarini doner. Asil olculen sey ORAN (ilk/son): gradyan
    tepeden dibe inerken ne kadar soguyor?

    Olcum tasarimi iki kez duzeltildi:

    1. Ilk halinde yigin sonundaki LayerNorm ve kayip = cikti.pow(2).mean()
       kullaniliyordu. LayerNorm ciktiyi birim varyansa cektigi icin bu kayip
       parametrelerden neredeyse BAGIMSIZ - butun gradyanlar 1e-6 mertebesinde
       gurultu cikiyordu. Ustelik residual aktivasyonlari buyuttugu icin
       LayerNorm onun gradyanini daha cok kisiyor ve sonuc TERSINE donuyordu
       (residual'siz gradyan daha buyuk gorunuyordu).

    2. Mutlak norm karsilastirmak da yaniltici: residual'li ve residual'siz
       aglar farkli olcekte aktivasyon uretiyor. Derinligin etkisini yalitmak
       icin ayni agin ilk/son katman orani kullaniliyor.

    Bu yuzden burada yigin ELLE kuruluyor (son LayerNorm yok) ve gercek bir
    hedefe karsi MSE kaybi aliniyor.
    """
    torch.manual_seed(tohum)
    bloklar = nn.ModuleList([
        TransformerBlock(d_model, n_head, pre_ln=pre_ln, residual=residual)
        for _ in range(n_layer)
    ])
    kafa = nn.Linear(d_model, d_model)

    x = torch.randn(4, seq_len, d_model)
    hedef = torch.randn(4, seq_len, d_model)

    h = x
    for b in bloklar:
        h = b(h)
    kayip = nn.functional.mse_loss(kafa(h), hedef)
    kayip.backward()

    def norm(m: nn.Module) -> float:
        t = sum(p.grad.pow(2).sum() for p in m.parameters() if p.grad is not None)
        return float(torch.as_tensor(t).sqrt())

    ilk, son = norm(bloklar[0]), norm(bloklar[-1])
    return ilk, son, (ilk / son if son > 0 else float("nan"))


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 74)
    print("GOREV 7: Transformer Block")
    print("=" * 74)

    d_model, n_head, B, S = 128, 4, 2, 16
    blok = TransformerBlock(d_model, n_head)
    x = torch.randn(B, S, d_model)
    cikti, agirlik = blok(x, agirlik_dondur=True)

    print(f"\nd_model={d_model} n_head={n_head} d_ff={4*d_model} (varsayilan)")
    print(f"  girdi      : {tuple(x.shape)}")
    print(f"  cikti      : {tuple(cikti.shape)}   (blok sekli korur - ust uste konabilmesi icin)")
    print(f"  agirliklar : {tuple(agirlik.shape)}")
    assert cikti.shape == x.shape

    print("\n  Parametre dagilimi:")
    p_dikkat = sum(p.numel() for p in blok.dikkat.parameters())
    p_ffn = sum(p.numel() for p in blok.ffn.parameters())
    p_ln = sum(p.numel() for p in blok.ln1.parameters()) + sum(p.numel() for p in blok.ln2.parameters())
    top = p_dikkat + p_ffn + p_ln
    for ad, p in [("MultiHeadAttention", p_dikkat), ("FeedForward", p_ffn), ("2x LayerNorm", p_ln)]:
        print(f"    {ad:<22}{p:>9,}  (%{100*p/top:.1f})")
    print(f"    {'TOPLAM':<22}{top:>9,}")

    # --- Ust uste koyma ---
    print("\n--- Birkac blok ust uste ---")
    for n in (1, 3, 6, 12):
        y = BlokYigini(d_model, n_head, n)
        with torch.no_grad():
            c = y(x)
        print(f"  n_layer={n:<3} cikti={tuple(c.shape)}  "
              f"parametre={sum(p.numel() for p in y.parameters()):>9,}  "
              f"cikti std={c.std():.4f}")

    # --- Causal mask ile ---
    m = nedensel_mask(S)
    with torch.no_grad():
        c_mask = BlokYigini(d_model, n_head, 3)(x, mask=m)
    print(f"\n  nedensel mask ile 3 blok: {tuple(c_mask.shape)} - MiniGPT bu sekilde kullanacak")

    # --- KANIT: residual ---
    print("\n--- KANIT: Residual baglanti neden onemli ---")
    print("  Olculen: ilk blogun gradyan normu / son blogun gradyan normu.")
    print("  1'e yakin = sinyal dibe kadar sagliki iniyor. 0'a yakin = ilk")
    print("  katmanlar egitim sinyali almiyor.\n")
    print(f"  {'n_layer':>8} | {'RESIDUAL VAR':^30} | {'RESIDUAL YOK':^30}")
    print(f"  {'':>8} | {'ilk':>10}{'son':>10}{'oran':>10} | {'ilk':>10}{'son':>10}{'oran':>10}")
    print(f"  {'-'*8}-+-{'-'*30}-+-{'-'*30}")
    for n in (2, 4, 8, 12, 16):
        vi, vs, vo = gradyan_olcumu(n, residual=True)
        yi, ys, yo = gradyan_olcumu(n, residual=False)
        print(f"  {n:>8} | {vi:>10.2e}{vs:>10.2e}{vo:>10.3f} | "
              f"{yi:>10.2e}{ys:>10.2e}{yo:>10.3f}")
    print("\n  Residual yokken oran derinlikle cokuyor: gradyan her katmanda")
    print("  agirlik matrisleriyle carpilarak geriye gidiyor ve sonuyor.")
    print("  Residual varsa d(cikti)/d(girdi) = I + (...) oldugundan en az bir")
    print("  birim gecis her zaman kaliyor - oran 1 civarinda kaliyor.")

    # --- Pre-LN vs Post-LN ---
    print("\n--- Pre-LN vs Post-LN (ayni olcum, ikisinde de residual var) ---")
    print(f"  {'n_layer':>8} {'Pre-LN oran':>16} {'Post-LN oran':>16}")
    print(f"  {'-'*8} {'-'*16} {'-'*16}")
    for n in (2, 6, 12, 24):
        _, _, pre = gradyan_olcumu(n, residual=True, pre_ln=True)
        _, _, post = gradyan_olcumu(n, residual=True, pre_ln=False)
        print(f"  {n:>8} {pre:>16.3f} {post:>16.3f}")
    print("\n  Post-LN'de LayerNorm residual yolunun UZERINDE oldugu icin")
    print("  otoyol her katmanda kesiliyor; Pre-LN'de yol kesintisiz.")
