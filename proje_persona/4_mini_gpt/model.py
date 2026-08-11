"""
MiniGPT: Gorev 4-7'de yazilan modulleri birlestiren karakter duzeyinde GPT.

TransformerBlock, MultiHeadAttention ve ScaledDotProductAttention 2_transformer
klasorunden IMPORT ediliyor - kopyalanmiyor. Klasor adlari rakamla basladigi
icin paket olarak import edilemiyor, o yuzden sys.path uzerinden.

Pozisyon konusunda bir sapma var ve bilincli: Gorev 4'te sinus-kosinus PE
yazildi ve dogrulandi, ama burada OGRENILEBILIR pozisyon gommesi kullaniliyor
(nanoGPT gibi). Sebep Gorev 4'te olculdu: nn.Embedding'in N(0,1) baslatmasi +
sqrt(d_model) carpani gomme std'sini PE std'sinin 21.5 katina cikariyor, yani
pozisyon sinyali baslangicta iceriğin altinda kaliyor. Sifirdan egitilen kucuk
bir modelde bu egitimi yavaslatir. Sinus PE'yi kullanmak isteyen
`ogrenilebilir_pozisyon=False` verebilir; iki mod da destekleniyor.
"""
from __future__ import annotations

import sys
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F

KOK = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(KOK))
sys.path.insert(0, str(KOK / "2_transformer"))

from dikkat import nedensel_mask  # noqa: E402
from pozisyon_kodlama import PositionalEncoding  # noqa: E402
from transformer_blok import TransformerBlock  # noqa: E402


class MiniGPT(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        n_embd: int = 128,
        n_head: int = 4,
        n_layer: int = 3,
        block_size: int = 128,
        dropout: float = 0.0,
        ogrenilebilir_pozisyon: bool = True,
        agirlik_baglama: bool = True,
    ) -> None:
        super().__init__()
        self.block_size = block_size
        self.vocab_size = vocab_size
        self.ogrenilebilir_pozisyon = ogrenilebilir_pozisyon

        self.token_gomme = nn.Embedding(vocab_size, n_embd)
        if ogrenilebilir_pozisyon:
            self.pozisyon_gomme = nn.Embedding(block_size, n_embd)
            self.sinus_pe = None
        else:
            self.pozisyon_gomme = None
            self.sinus_pe = PositionalEncoding(n_embd, max_len=block_size)
        self.dropout = nn.Dropout(dropout)

        self.bloklar = nn.ModuleList([
            TransformerBlock(n_embd, n_head, dropout=dropout, pre_ln=True)
            for _ in range(n_layer)
        ])
        self.ln_son = nn.LayerNorm(n_embd)
        self.lm_kafa = nn.Linear(n_embd, vocab_size, bias=False)

        # Agirlik baglama: girdi gommesi ile cikti projeksiyonu ayni matris.
        # 113 x 128 = 14.5K parametre tasarrufu degil asil fayda, kucuk
        # korpusta duzenlilestirme etkisi.
        if agirlik_baglama:
            self.lm_kafa.weight = self.token_gomme.weight

        # Nedensel mask sabit; her ileri gecişte yeniden uretmemek icin buffer.
        self.register_buffer("mask", nedensel_mask(block_size), persistent=False)
        self.apply(self._baslat)

    @staticmethod
    def _baslat(m: nn.Module) -> None:
        # nanoGPT'nin baslatmasi: std=0.02. Varsayilan N(0,1) hem pozisyon
        # sinyalini bastiriyor hem baslangic kaybini gereksiz yukseltiyor.
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, mean=0.0, std=0.02)

    def forward(
        self, idx: torch.Tensor, hedef: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """idx: (B, T) tam sayi. Doner: (logits (B, T, vocab), kayip|None)."""
        B, T = idx.shape
        if T > self.block_size:
            raise ValueError(f"dizi {T} > block_size {self.block_size}")

        x = self.token_gomme(idx)  # (B, T, n_embd)
        if self.ogrenilebilir_pozisyon:
            poz = torch.arange(T, device=idx.device)
            x = x + self.pozisyon_gomme(poz)  # yayilir: (1, T, n_embd)
        else:
            x = self.sinus_pe(x)
        x = self.dropout(x)

        mask = self.mask[:T, :T]
        for blok in self.bloklar:
            x = blok(x, mask=mask)
        x = self.ln_son(x)
        logits = self.lm_kafa(x)

        kayip = None
        if hedef is not None:
            kayip = F.cross_entropy(
                logits.view(B * T, self.vocab_size), hedef.reshape(B * T)
            )
        return logits, kayip

    @torch.no_grad()
    def generate(
        self,
        idx: torch.Tensor,
        max_new_tokens: int,
        temperature: float = 1.0,
        top_k: int | None = None,
    ) -> torch.Tensor:
        """
        idx: (B, T) baslangic baglami. Her adimda bir karakter ornekliyor.

        Baglami block_size ile kirpmak zorunlu: pozisyon gommesi bundan uzunu
        tanimiyor. Model 128 karakterlik pencereden ilerisini goremez.
        """
        self.eval()
        for _ in range(max_new_tokens):
            kirpik = idx[:, -self.block_size:]
            logits, _ = self(kirpik)
            logits = logits[:, -1, :] / max(temperature, 1e-6)
            if top_k is not None:
                k = min(top_k, logits.size(-1))
                esik = torch.topk(logits, k, dim=-1).values[:, [-1]]
                logits = logits.masked_fill(logits < esik, float("-inf"))
            olasilik = F.softmax(logits, dim=-1)
            sonraki = torch.multinomial(olasilik, num_samples=1)
            idx = torch.cat((idx, sonraki), dim=1)
        return idx

    def parametre_sayisi(self, gomme_haric: bool = False) -> int:
        n = sum(p.numel() for p in self.parameters())
        if gomme_haric and self.pozisyon_gomme is not None:
            n -= self.pozisyon_gomme.weight.numel()
        return n


if __name__ == "__main__":
    torch.manual_seed(0)
    m = MiniGPT(vocab_size=113)
    idx = torch.randint(0, 113, (2, 64))
    logits, kayip = m(idx, hedef=idx)
    print(f"parametre        : {m.parametre_sayisi():,}")
    print(f"logits           : {tuple(logits.shape)}")
    print(f"baslangic kaybi  : {kayip:.4f}  (rastgele beklenti ln(113) = "
          f"{torch.tensor(113.0).log():.4f})")
    uretim = m.generate(torch.zeros((1, 1), dtype=torch.long), max_new_tokens=20)
    print(f"generate cikti   : {tuple(uretim.shape)}")
