"""
Gorev 5: Scaled Dot-Product Attention.

    Attention(Q, K, V) = softmax( QK^T / sqrt(d_k) + mask ) V

Uc tasarim detayi:
  1. sqrt(d_k) bolmesi keyfi degil - olcumu asagida.
  2. Mask, softmax'tan ONCE -inf olarak eklenir (sonra sifirlamak yanlis:
     softmax normalizasyonu bozulur, satir toplami 1 olmaz).
  3. Agirliklar da dondurulur - modelin neye baktigini gormenin tek yolu.

    ../../.venv/bin/python dikkat.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class ScaledDotProductAttention(nn.Module):
    def __init__(self, dropout: float = 0.0) -> None:
        super().__init__()
        self.dropout = nn.Dropout(dropout)

    def forward(
        self,
        q: torch.Tensor,
        k: torch.Tensor,
        v: torch.Tensor,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        q, k : (..., sorgu, d_k) / (..., anahtar, d_k)
        v    : (..., anahtar, d_v)
        mask : (..., sorgu, anahtar) bool - True = GORULEBILIR, False = gizle
               (yayilabilir/broadcast edilebilir)

        Doner: (cikti, agirliklar)
          cikti      : (..., sorgu, d_v)
          agirliklar : (..., sorgu, anahtar), son eksende toplam 1
        """
        d_k = q.size(-1)

        # (..., sorgu, d_k) @ (..., d_k, anahtar) -> (..., sorgu, anahtar)
        skor = q @ k.transpose(-2, -1) / math.sqrt(d_k)

        if mask is not None:
            # -inf degil, dtype'in en kucuk degeri: -inf ile tamamen maskeli
            # bir satir olusursa softmax NaN uretir. Burada 0 satiri cikar.
            skor = skor.masked_fill(~mask, torch.finfo(skor.dtype).min)

        agirlik = torch.softmax(skor, dim=-1)
        agirlik = self.dropout(agirlik)
        return agirlik @ v, agirlik


def nedensel_mask(seq_len: int, device: torch.device | None = None) -> torch.Tensor:
    """
    Causal (nedensel) mask: pozisyon i yalnizca <= i'yi gorur.
    Alt ucgen True. MiniGPT'de gelecegi gizlemek icin bu kullaniliyor.

        [[1, 0, 0],
         [1, 1, 0],
         [1, 1, 1]]
    """
    return torch.tril(torch.ones(seq_len, seq_len, dtype=torch.bool, device=device))


def dolgu_mask(uzunluklar: list[int], seq_len: int) -> torch.Tensor:
    """Padding mask: gercek token'lar True, dolgu False. (batch, 1, seq_len)."""
    b = len(uzunluklar)
    m = torch.zeros(b, seq_len, dtype=torch.bool)
    for i, u in enumerate(uzunluklar):
        m[i, :u] = True
    return m.unsqueeze(1)


# --------------------------------------------------------------------------
# Kanit 1: sqrt(d_k) bolmesi neden gerekli
# --------------------------------------------------------------------------
def olcekleme_testi(d_k_listesi: tuple[int, ...] = (8, 32, 128, 512),
                    seq_len: int = 16, ornek: int = 256, tohum: int = 0) -> list[dict[str, float]]:
    """
    q, k ~ N(0, 1) ise nokta carpiminin varyansi d_k ile BUYUR. Olceklenmezse
    skorlar d_k arttikca yayilir, softmax doyar (tek bir token'a ~1.0 verir),
    gradyan kaybolur.

    Entropi ile olculuyor: yuksek entropi = dagilmis dikkat,
    0'a yakin entropi = doymus (tek noktaya kilitlenmis).

    ornek=256: tek bir rastgele cekimde entropi gurultulu cikiyor ve trend
    monoton gorunmuyordu (d_k=512, d_k=128'den yuksek entropi veriyordu).
    Ortalama alinca d_k'ya bagli asil egilim ortaya cikiyor.
    """
    sonuc = []
    for d_k in d_k_listesi:
        torch.manual_seed(tohum)
        q = torch.randn(ornek, seq_len, d_k)
        k = torch.randn(ornek, seq_len, d_k)

        ham = q @ k.transpose(-2, -1)
        olcekli = ham / math.sqrt(d_k)

        def entropi(s: torch.Tensor) -> tuple[float, float]:
            a = torch.softmax(s, dim=-1)
            e = -(a * torch.log(a.clamp_min(1e-12))).sum(-1).mean()
            # max agirligin ORTALAMASI; tek bir global max cekimden cekime ziplar
            return e.item(), a.max(dim=-1).values.mean().item()

        e_ham, max_ham = entropi(ham)
        e_olc, max_olc = entropi(olcekli)
        sonuc.append({
            "d_k": d_k,
            "skor_std_ham": ham.std().item(),
            "skor_std_olcekli": olcekli.std().item(),
            "entropi_ham": e_ham,
            "entropi_olcekli": e_olc,
            "max_agirlik_ham": max_ham,
            "max_agirlik_olcekli": max_olc,
        })
    return sonuc


# --------------------------------------------------------------------------
# Kanit 2: yorumlanabilir bir test case
# --------------------------------------------------------------------------
def yorumlanabilir_ornek(tohum: int = 3) -> tuple[torch.Tensor, list[str]]:
    """
    Elle kurulmus bir ornek: 5 token, her biri birbirine dik (ortonormal) bir
    vektor. Sorgu bilerek 2. token'in anahtarina esitleniyor, dolayisiyla
    dikkatin oraya kilitlenmesi GEREKIR. Beklenti kod tarafindan degil,
    kurulumdan geliyor - yani gercek bir tahmin.
    """
    torch.manual_seed(tohum)
    etiketler = ["<bas>", "kar", "yagdi", "sokaga", "<son>"]
    n, d_k = len(etiketler), 8

    # Ortonormal taban: token'lar arasi benzerlik tam olarak 0.
    # BUYUKLUK onemli: birim normlu vektorlerde eslesme logiti 1/sqrt(8)=0.35
    # kaliyor ve softmax neredeyse duz cikiyor (0.26 vs 0.18). Gercek modelde
    # Q/K projeksiyonlari daha buyuk genlikli skor uretir; burada ayni etkiyi
    # tabani ALFA ile olcekleyerek kuruyoruz. Dikkatin keskinligi Q/K
    # normuna bagli - LayerNorm'un attention oncesi durmasinin bir sebebi bu.
    alfa = 3.0
    taban = torch.linalg.qr(torch.randn(d_k, d_k))[0][:n] * alfa  # (n, d_k)
    k = v = taban.unsqueeze(0)

    # Sorgu 1: tam olarak 2. token'in anahtari
    # Sorgu 2: 1. ve 3. token'in karisimi
    q = torch.stack([taban[2], (taban[1] + taban[3]) / math.sqrt(2.0)]).unsqueeze(0)

    dikkat = ScaledDotProductAttention()
    _, agirlik = dikkat(q, k, v)
    return agirlik[0], etiketler


if __name__ == "__main__":
    torch.manual_seed(0)
    dikkat = ScaledDotProductAttention()

    print("=" * 70)
    print("GOREV 5: Scaled Dot-Product Attention")
    print("=" * 70)

    # --- Temel sekil kontrolu ---
    b, s, d_k, d_v = 2, 6, 16, 16
    q, k, v = torch.randn(b, s, d_k), torch.randn(b, s, d_k), torch.randn(b, s, d_v)
    cikti, agirlik = dikkat(q, k, v)
    print(f"\nsekiller: q={tuple(q.shape)} k={tuple(k.shape)} v={tuple(v.shape)}")
    print(f"          cikti={tuple(cikti.shape)}  agirlik={tuple(agirlik.shape)}")
    print(f"agirlik satir toplami = 1 : {torch.allclose(agirlik.sum(-1), torch.ones(b, s))}")

    # --- Causal mask ---
    print("\n--- Nedensel (causal) mask ---")
    m = nedensel_mask(s)
    _, a_mask = dikkat(q, k, v, mask=m)
    ust_ucgen = a_mask[0].triu(diagonal=1)
    print(f"  ust ucgen agirliklarin toplami : {ust_ucgen.sum():.3e}  (0 olmali)")
    print(f"  ust ucgen max                  : {ust_ucgen.max():.3e}")
    print(f"  maskeliyken satir toplami = 1  : {torch.allclose(a_mask.sum(-1), torch.ones(b, s))}")
    print(f"  ilk token yalniz kendini gorur : {a_mask[0, 0, 0]:.4f}")
    print("\n  Agirlik matrisi (1. ornek, satir=sorgu, sutun=anahtar):")
    for i in range(s):
        print("    " + " ".join(f"{a_mask[0, i, j]:5.2f}" for j in range(s)))

    # --- Padding mask ---
    print("\n--- Dolgu (padding) mask ---")
    pm = dolgu_mask([6, 3], s)
    _, a_pad = dikkat(q, k, v, mask=pm)
    print(f"  2. ornek 3 gercek token; dolgu agirliklari toplami: {a_pad[1, :, 3:].sum():.3e}")

    # --- Olcekleme kaniti ---
    print("\n--- KANIT 1: sqrt(d_k) bolmesi neden var ---")
    print(f"  {'d_k':>5} {'skor std':>18} {'entropi (nat)':>22} {'max agirlik':>20}")
    print(f"  {'':>5} {'ham':>8}{'olcekli':>10} {'ham':>10}{'olcekli':>12} {'ham':>9}{'olcekli':>11}")
    for r in olcekleme_testi():
        print(f"  {r['d_k']:>5} {r['skor_std_ham']:>8.2f}{r['skor_std_olcekli']:>10.2f}"
              f" {r['entropi_ham']:>10.3f}{r['entropi_olcekli']:>12.3f}"
              f" {r['max_agirlik_ham']:>9.3f}{r['max_agirlik_olcekli']:>11.3f}")
    print(f"\n  Duzgun dagilimin entropisi (16 token): {math.log(16):.3f} nat")
    print("  Olceklenmemis skorlarda d_k buyudukce entropi cokuyor: dikkat tek")
    print("  token'a kilitleniyor, softmax doyuyor, gradyan kayboluyor.")
    print("  Olcekli sutunda entropi d_k'dan bagimsiz kaliyor - amac buydu.")

    # --- Yorumlanabilir ornek ---
    print("\n--- KANIT 2: attention bu ornekte neye odaklandi ---")
    a, etiketler = yorumlanabilir_ornek()
    print("  Kurulum: token'lar birbirine dik; sorgular bilerek secildi.")
    print(f"\n  {'sorgu':<28}" + "".join(f"{e:>9}" for e in etiketler))
    aciklama = ["= 'yagdi' anahtari", "= ('kar'+'sokaga')/sqrt2"]
    for i in range(a.size(0)):
        print(f"  {aciklama[i]:<28}" + "".join(f"{a[i, j]:9.3f}" for j in range(len(etiketler))))
    print(f"\n  1. sorgu -> en yuksek: '{etiketler[a[0].argmax()]}' ({a[0].max():.3f})")
    ilk_iki = a[1].topk(2)
    print(f"  2. sorgu -> en yuksek iki: "
          f"'{etiketler[ilk_iki.indices[0]]}' ({ilk_iki.values[0]:.3f}), "
          f"'{etiketler[ilk_iki.indices[1]]}' ({ilk_iki.values[1]:.3f})")
