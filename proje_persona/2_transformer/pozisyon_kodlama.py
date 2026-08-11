"""
Gorev 4: Positional Encoding + Embedding.

Transformer RNN'den farkli olarak diziyi paralel isler; self-attention
permutasyona esdeger (permutation-equivariant) bir islemdir. Yani girdiyi
karistirirsaniz cikti da ayni sekilde karisir - MODEL SIRAYI GORMEZ.
Pozisyon bilgisi bu yuzden ayrica enjekte edilir.

Bu dosya tek basina calistirilirsa iddiayi olcerek gosterir:
    ../../.venv/bin/python pozisyon_kodlama.py
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn


class PositionalEncoding(nn.Module):
    """
    Klasik sinus-kosinus pozisyon kodlamasi (Vaswani ve ark., 2017).

        PE[pos, 2i]   = sin(pos / 10000^(2i/d_model))
        PE[pos, 2i+1] = cos(pos / 10000^(2i/d_model))

    Ogrenilebilir parametre YOK; matris bir kez hesaplanip buffer'da tutulur.
    """

    def __init__(self, d_model: int, max_len: int = 5000, dropout: float = 0.0) -> None:
        super().__init__()
        if d_model % 2 != 0:
            raise ValueError(f"d_model cift olmali (sin/cos ciftleri icin), verilen: {d_model}")
        self.d_model = d_model
        self.dropout = nn.Dropout(dropout)

        pe = torch.zeros(max_len, d_model)
        pozisyon = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)  # (max_len, 1)

        # Bolen: 10000^(2i/d_model). Log uzayinda hesaplaniyor cunku dogrudan
        # ussalma buyuk d_model'de tasar (10000^1 = 1e4 sorun degil ama
        # ara adimlar float hassasiyetini zorluyor).
        bolen = torch.exp(
            torch.arange(0, d_model, 2, dtype=torch.float) * (-math.log(10000.0) / d_model)
        )  # (d_model/2,)

        pe[:, 0::2] = torch.sin(pozisyon * bolen)  # cift indisler
        pe[:, 1::2] = torch.cos(pozisyon * bolen)  # tek indisler

        # register_buffer: parametre degil ama modelin durumu. state_dict'e
        # girer, .to(device) ile tasinir, optimizer gormez.
        self.register_buffer("pe", pe.unsqueeze(0))  # (1, max_len, d_model)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: (batch, seq_len, d_model) -> ayni sekil.

        Pozisyon kodlamasi TOPLANIYOR, birlestirilmiyor (concat). Toplama
        boyutu buyutmuyor ve embedding uzayinin ayni eksenlerini paylasiyor;
        model hangi bilesenin ne kadarini kullanacagini kendisi ogreniyor.
        """
        if x.dim() != 3:
            raise ValueError(f"(batch, seq_len, d_model) bekleniyor, gelen: {tuple(x.shape)}")
        seq_len = x.size(1)
        if seq_len > self.pe.size(1):
            raise ValueError(f"dizi {seq_len} > max_len {self.pe.size(1)}")
        x = x + self.pe[:, :seq_len]
        return self.dropout(x)


class GommeVePozisyon(nn.Module):
    """
    Embedding + PositionalEncoding - odevin "embedding katmani ile birlikte
    test edin" maddesi.

    sqrt(d_model) ile olcekleme orijinal makaleden: embedding agirliklari
    ~N(0, 1) baslatilinca degerleri PE'nin [-1, 1] araligina gore kucuk
    kaliyor ve pozisyon sinyali icerigi bastiriyor.
    """

    def __init__(self, vocab_size: int, d_model: int, max_len: int = 5000,
                 dropout: float = 0.0, olcekle: bool = True) -> None:
        super().__init__()
        self.gomme = nn.Embedding(vocab_size, d_model)
        self.pozisyon = PositionalEncoding(d_model, max_len, dropout)
        self.d_model = d_model
        self.olcekle = olcekle

    def forward(self, idx: torch.Tensor) -> torch.Tensor:
        """idx: (batch, seq_len) tam sayi -> (batch, seq_len, d_model)."""
        x = self.gomme(idx)
        if self.olcekle:
            x = x * math.sqrt(self.d_model)
        return self.pozisyon(x)


# --------------------------------------------------------------------------
# Kanit: PE olmadan self-attention sirayi gormuyor
# --------------------------------------------------------------------------
def permutasyon_testi(d_model: int = 32, seq_len: int = 6, tohum: int = 0) -> dict[str, float]:
    """
    Ayni token'lari farkli sirada verip self-attention ciktisini karsilastirir.

    PE YOKKEN: cikti da ayni sekilde permute olur, yani sirayi geri alinca
    fark ~0. Model "kedi baligi yedi" ile "baligi kedi yedi" arasinda hicbir
    fark gormez.

    PE VARKEN: fark buyuk.
    """
    torch.manual_seed(tohum)
    gomme = nn.Embedding(10, d_model)
    pe = PositionalEncoding(d_model)

    idx = torch.arange(seq_len).unsqueeze(0)  # (1, seq_len)
    perm = torch.randperm(seq_len)
    ters = torch.argsort(perm)

    def dikkat(x: torch.Tensor) -> torch.Tensor:
        # Agirliksiz (Q=K=V=x) self-attention: permutasyon esdegerligini
        # gostermek icin projeksiyonlara gerek yok.
        skor = x @ x.transpose(-2, -1) / math.sqrt(d_model)
        return torch.softmax(skor, dim=-1) @ x

    with torch.no_grad():
        # --- PE yok ---
        duz = gomme(idx)
        cikti_duz = dikkat(duz)
        cikti_karisik = dikkat(gomme(idx[:, perm]))
        fark_pesiz = (cikti_duz - cikti_karisik[:, ters]).abs().max().item()

        # --- PE var ---
        cikti_pe = dikkat(pe(gomme(idx)))
        cikti_pe_karisik = dikkat(pe(gomme(idx[:, perm])))
        fark_peli = (cikti_pe - cikti_pe_karisik[:, ters]).abs().max().item()

    return {"pe_yok_fark": fark_pesiz, "pe_var_fark": fark_peli}


if __name__ == "__main__":
    torch.manual_seed(0)

    print("=" * 70)
    print("GOREV 4: Positional Encoding + Embedding")
    print("=" * 70)

    d_model, max_len = 128, 512
    pe = PositionalEncoding(d_model, max_len)
    print(f"\nPE matrisi sekli : {tuple(pe.pe.shape)}  (1, max_len, d_model)")
    print(f"Ogrenilebilir parametre : {sum(p.numel() for p in pe.parameters())}")
    print(f"Buffer (state_dict'te)  : {list(pe.state_dict().keys())}")

    print("\nIlk 4 pozisyon, ilk 8 boyut:")
    print("      " + "".join(f"{f'd{i}':>9}" for i in range(8)))
    for p in range(4):
        print(f"  pos{p} " + "".join(f"{pe.pe[0, p, i]:9.4f}" for i in range(8)))

    print("\nDogrulama:")
    print(f"  pos=0, cift indisler (sin 0)   : {pe.pe[0, 0, 0::2].abs().max():.6f}  (0 olmali)")
    print(f"  pos=0, tek indisler  (cos 0)   : {pe.pe[0, 0, 1::2].min():.6f}  (1 olmali)")
    print(f"  butun degerler [-1, 1] icinde  : {bool(pe.pe.abs().max() <= 1.0)}")
    normlar = pe.pe[0].norm(dim=-1)
    print(f"  pozisyon vektoru normu sabit   : {normlar.min():.4f} - {normlar.max():.4f}")

    # Embedding ile birlikte
    print("\n--- Embedding katmani ile birlikte ---")
    kat = GommeVePozisyon(vocab_size=50, d_model=d_model)
    idx = torch.randint(0, 50, (2, 10))
    cikti = kat(idx)
    print(f"  girdi (idx)  : {tuple(idx.shape)}  dtype={idx.dtype}")
    print(f"  cikti        : {tuple(cikti.shape)}  dtype={cikti.dtype}")
    assert cikti.shape == (2, 10, d_model)

    sadece_gomme = kat.gomme(idx) * math.sqrt(d_model)
    print(f"  gomme std    : {sadece_gomme.std():.4f}")
    print(f"  PE std       : {pe.pe[0, :10].std():.4f}")
    print(f"  toplam std   : {cikti.std():.4f}")

    # Kanit
    print("\n--- KANIT: PE olmadan sira gorunmuyor ---")
    s = permutasyon_testi()
    print(f"  PE YOK : token'lari karistirip geri al -> max fark {s['pe_yok_fark']:.2e}")
    print(f"  PE VAR : ayni islem                    -> max fark {s['pe_var_fark']:.4f}")
    print("\n  PE'siz fark sayisal gurultu seviyesinde: self-attention")
    print("  permutasyona ESDEGER, yani sirayi hic gormuyor.")
