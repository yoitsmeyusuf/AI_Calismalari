"""
10. Hafta: arama katmani (ortak modul).

Hem `ara.py` (elle sorgu) hem `4_esik_analizi.py` (30 soruluk benchmark) bunu
kullaniyor - ikisinin ayni prompt'u ve ayni benzerlik tanimini kullandigindan
emin olmak icin.

Chroma "distance" dondurur, benzerlik degil. Kosinus uzayinda:

    benzerlik = 1 - distance

Bunu karistirmak bu odevin klasik hatasi: esigi distance sanip "< 0.7" yazmak
tam ters filtre kurar.

Chroma HNSW ile *yaklasik* arama yapar. 3138 vektorde fark beklenmez ama esik
analizi buna dayandigi icin `tam_kosinus()` ile numpy uzerinden kesin sonuc da
hesaplanabiliyor; `4_esik_analizi.py` ikisini karsilastiriyor.
"""
from functools import lru_cache

import numpy as np

from ayarlar import (
    BENZERLIK_ESIGI,
    CHROMA_DIZINI,
    EMBEDDING_MODELI,
    GOMME_DTYPE,
    KOLEKSIYON_ADI,
    RET_MESAJI,
    SORGU_PROMPT,
    VARSAYILAN_K,
    VEKTOR_PARQUET,
)


@lru_cache(maxsize=1)
def _model():
    import torch
    from sentence_transformers import SentenceTransformer

    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    return SentenceTransformer(
        EMBEDDING_MODELI,
        device=cihaz,
        model_kwargs={"dtype": getattr(torch, GOMME_DTYPE)},
    )


@lru_cache(maxsize=1)
def _koleksiyon():
    import chromadb

    if not CHROMA_DIZINI.exists():
        raise SystemExit(
            f"{CHROMA_DIZINI} yok - once 2_gom_ve_indeksle.py calistirin."
        )
    istemci = chromadb.PersistentClient(path=str(CHROMA_DIZINI))
    return istemci.get_collection(KOLEKSIYON_ADI)


@lru_cache(maxsize=1)
def _parquet():
    """Tam kosinus icin tum vektorleri bellege alir (3138 x 768 -> ~9 MB)."""
    import pyarrow.parquet as pq

    tablo = pq.read_table(VEKTOR_PARQUET)
    vektorler = np.array(tablo.column("chunk_vector").to_pylist(), dtype=np.float32)
    return tablo, vektorler


def soruyu_gom(soru: str) -> np.ndarray:
    """Soruyu `query` prompt'u ile gomer.

    Dokuman tarafi `document` prompt'unu kullaniyordu; model asimetrik oldugu
    icin bu ikisi karistirilmamali.
    """
    vektor = _model().encode(
        [soru], prompt_name=SORGU_PROMPT, convert_to_numpy=True
    )[0]
    return np.asarray(vektor, dtype=np.float32)


def ara(soru: str, k: int = VARSAYILAN_K) -> list[dict]:
    """Chroma uzerinden en yakin k chunk'i benzerlik skoruyla dondurur."""
    vektor = soruyu_gom(soru)
    sonuc = _koleksiyon().query(
        query_embeddings=[vektor.tolist()],
        n_results=k,
        include=["documents", "metadatas", "distances"],
    )
    return [
        {
            "chunk_id": sonuc["ids"][0][i],
            "benzerlik": 1.0 - sonuc["distances"][0][i],  # distance -> benzerlik
            "chunk_text": sonuc["documents"][0][i],
            **sonuc["metadatas"][0][i],
        }
        for i in range(len(sonuc["ids"][0]))
    ]


def tam_kosinus(soru: str, k: int = VARSAYILAN_K) -> list[dict]:
    """HNSW'siz, numpy ile kesin kosinus. Chroma sonucunu dogrulamak icin."""
    tablo, vektorler = _parquet()
    skorlar = vektorler @ soruyu_gom(soru)  # vektorler L2-normalize -> nokta carpimi
    ust = np.argsort(-skorlar)[:k]
    idler = tablo.column("chunk_id").to_pylist()
    metinler = tablo.column("chunk_text").to_pylist()
    urller = tablo.column("url").to_pylist()
    return [
        {
            "chunk_id": idler[i],
            "benzerlik": float(skorlar[i]),
            "chunk_text": metinler[i],
            "url": urller[i],
        }
        for i in ust
    ]


def cevapla(soru: str, k: int = VARSAYILAN_K, esik: float = BENZERLIK_ESIGI) -> dict:
    """Esik kontrollu arama.

    En yuksek benzerlik esigin altindaysa hic chunk dondurmuyoruz - LLM'e
    verilecek bir sey olmadigi icin uydurma ihtimali de ortadan kalkiyor.
    """
    sonuclar = ara(soru, k=k)
    en_iyi = sonuclar[0]["benzerlik"] if sonuclar else 0.0
    if en_iyi < esik:
        return {"cevaplandi": False, "en_iyi_skor": en_iyi, "mesaj": RET_MESAJI, "kaynaklar": []}
    return {
        "cevaplandi": True,
        "en_iyi_skor": en_iyi,
        "mesaj": "",
        "kaynaklar": [s for s in sonuclar if s["benzerlik"] >= esik],
    }
