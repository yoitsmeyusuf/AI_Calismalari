"""
Gorev 6: Multi-Head Attention.

Tek bir attention, her pozisyon icin TEK bir agirlikli ortalama uretir.
Farkli iliski turlerini (sozdizimsel bagimlilik, konu benzerligi, yerel
komsuluk...) ayni anda temsil edemez - hepsi tek dagilima sikisir.

Multi-head cozumu: d_model'i n_head parcaya bolup her parcada AYRI bir
attention calistirmak. Maliyet ayni kaliyor (d_k = d_model / n_head), cunku
kafalar daha genis degil daha DAR bir uzayda calisiyor.

    ../../.venv/bin/python cok_kafali_dikkat.py
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).resolve().parent))
from dikkat import ScaledDotProductAttention, nedensel_mask  # noqa: E402


class MultiHeadAttention(nn.Module):
    def __init__(self, d_model: int, n_head: int, dropout: float = 0.0, bias: bool = True) -> None:
        super().__init__()
        if d_model % n_head != 0:
            raise ValueError(f"d_model ({d_model}) n_head'e ({n_head}) tam bolunmeli")
        self.d_model = d_model
        self.n_head = n_head
        self.d_k = d_model // n_head

        self.w_q = nn.Linear(d_model, d_model, bias=bias)
        self.w_k = nn.Linear(d_model, d_model, bias=bias)
        self.w_v = nn.Linear(d_model, d_model, bias=bias)
        self.w_o = nn.Linear(d_model, d_model, bias=bias)

        self.dikkat = ScaledDotProductAttention(dropout)
        self.cikti_dropout = nn.Dropout(dropout)

    def _bol(self, x: torch.Tensor) -> torch.Tensor:
        """(B, S, d_model) -> (B, n_head, S, d_k)."""
        b, s, _ = x.shape
        # view ile son ekseni ikiye ayir, sonra kafa eksenini one al ki
        # attention son iki eksende (S, d_k) calissin.
        return x.view(b, s, self.n_head, self.d_k).transpose(1, 2)

    def _birlestir(self, x: torch.Tensor) -> torch.Tensor:
        """(B, n_head, S, d_k) -> (B, S, d_model)."""
        b, h, s, d_k = x.shape
        # transpose sonrasi bellek bitisik degil; view'dan once contiguous.
        return x.transpose(1, 2).contiguous().view(b, s, h * d_k)

    def forward(
        self,
        sorgu: torch.Tensor,
        anahtar: torch.Tensor | None = None,
        deger: torch.Tensor | None = None,
        mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        sorgu/anahtar/deger : (B, S, d_model). anahtar/deger verilmezse
        self-attention (hepsi sorgu'dan turer).

        mask : (S, S) veya (B, 1, S, S) - True = gorulebilir.
               (S, S) verilirse kafa eksenine yayilir.

        Doner: (cikti (B, S, d_model), agirliklar (B, n_head, S, S))
        """
        anahtar = sorgu if anahtar is None else anahtar
        deger = sorgu if deger is None else deger

        q = self._bol(self.w_q(sorgu))
        k = self._bol(self.w_k(anahtar))
        v = self._bol(self.w_v(deger))

        if mask is not None and mask.dim() == 2:
            mask = mask.unsqueeze(0).unsqueeze(0)  # (1, 1, S, S) -> yayilir

        cikti, agirlik = self.dikkat(q, k, v, mask)
        cikti = self.w_o(self._birlestir(cikti))
        return self.cikti_dropout(cikti), agirlik


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 74)
    print("GOREV 6: Multi-Head Attention")
    print("=" * 74)

    # --- Odevin istedigi boyut kontrolu: n_head=4, d_model=128 ---
    B, S, d_model, n_head = 2, 10, 128, 4
    mha = MultiHeadAttention(d_model, n_head)
    x = torch.randn(B, S, d_model)

    print(f"\nd_model={d_model}  n_head={n_head}  d_k={mha.d_k}  batch={B}  seq_len={S}\n")
    print(f"  {'adim':<38}{'sekil':<26}{'aciklama'}")
    print(f"  {'-'*38}{'-'*26}{'-'*22}")
    q_duz = mha.w_q(x)
    q_bol = mha._bol(q_duz)
    cikti, agirlik = mha(x)
    ciktilar = [
        ("girdi x", tuple(x.shape), "(B, S, d_model)"),
        ("w_q(x)", tuple(q_duz.shape), "projeksiyon, boyut ayni"),
        ("view(B,S,n_head,d_k)", (B, S, n_head, mha.d_k), "son eksen bolundu"),
        ("transpose(1,2)", tuple(q_bol.shape), "(B, n_head, S, d_k)"),
        ("agirliklar", tuple(agirlik.shape), "(B, n_head, S, S)"),
        ("birlestir", (B, S, d_model), "kafalar geri toplandi"),
        ("w_o(...) = cikti", tuple(cikti.shape), "girdiyle ayni sekil"),
    ]
    for ad, sekil, aciklama in ciktilar:
        print(f"  {ad:<38}{str(sekil):<26}{aciklama}")

    assert cikti.shape == x.shape
    assert agirlik.shape == (B, n_head, S, S)
    print(f"\n  agirlik satir toplami = 1 : "
          f"{torch.allclose(agirlik.sum(-1), torch.ones(B, n_head, S), atol=1e-6)}")
    print(f"  parametre sayisi          : {sum(p.numel() for p in mha.parameters()):,}"
          f"  (4 x {d_model}x{d_model} + bias)")

    # --- Maliyet n_head'den bagimsiz mi? ---
    print("\n--- Parametre sayisi n_head'e gore ---")
    for h in (1, 2, 4, 8, 16):
        m = MultiHeadAttention(d_model, h)
        print(f"  n_head={h:<3} d_k={m.d_k:<4} parametre={sum(p.numel() for p in m.parameters()):,}")
    print("  -> Ayni. Kafalar uzayi BOLUYOR, buyutmuyor.")

    # --- n_head=1 tek kafali ile ayni mi? ---
    print("\n--- Dogrulama: n_head=1, tek kafali attention'a esit olmali ---")
    torch.manual_seed(1)
    tek = MultiHeadAttention(32, n_head=1)
    y = torch.randn(1, 5, 32)
    with torch.no_grad():
        c1, _ = tek(y)
        sdpa = ScaledDotProductAttention()
        ham, _ = sdpa(tek.w_q(y), tek.w_k(y), tek.w_v(y))
        c2 = tek.w_o(ham)
    print(f"  max fark: {(c1 - c2).abs().max():.3e}  -> {torch.allclose(c1, c2, atol=1e-6)}")

    # --- Causal mask butun kafalarda ---
    print("\n--- Nedensel mask butun kafalarda gecerli mi ---")
    m = nedensel_mask(S)
    _, a_mask = mha(x, mask=m)
    print(f"  ust ucgen toplami (tum kafalar): {a_mask.triu(diagonal=1).sum():.3e}")

    # --- Kafalar farkli seylere mi bakiyor? ---
    print("\n--- Kafalar birbirinden farkli mi ---")
    a0 = agirlik[0]  # (n_head, S, S)
    print("  Kafa ciftleri arasi ortalama mutlak fark:")
    for i in range(n_head):
        for j in range(i + 1, n_head):
            print(f"    kafa{i} vs kafa{j}: {(a0[i] - a0[j]).abs().mean():.4f}")
    ent = -(a0 * a0.clamp_min(1e-12).log()).sum(-1).mean(-1)
    print("  Kafa basina dikkat entropisi (dusuk = odakli, yuksek = dagilmis):")
    for i in range(n_head):
        print(f"    kafa{i}: {ent[i]:.3f} nat")
    print(f"  (duzgun dagilim ust siniri: {torch.tensor(float(S)).log():.3f} nat)")
    print("\n  Bunlar EGITILMEMIS kafalar - fark yalnizca rastgele baslatmadan")
    print("  geliyor. Egitimde bu fark uzmanlasmaya donusuyor; buradaki olcum")
    print("  sadece kafalarin bagimsiz parametrelendigini dogruluyor.")
