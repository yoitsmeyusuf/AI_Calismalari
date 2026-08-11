"""
10. Hafta - 2. adim: chunk'lari vektore cevir, ChromaDB'ye yaz.

Cikti:
    chroma_db/                    kalici Chroma dizini (arama icin)
    veri/chunks_vektorlu.parquet  url + chunk_text + chunk_vector (teslim icin)

Iki tuzak burada:
  1. Chroma'nin varsayilan mesafesi L2, kosinus degil -> hnsw:space acikca
     "cosine" veriliyor.
  2. Chroma benzerlik degil *distance* dondurur -> benzerlik = 1 - distance
     (ara.py'de donusturuluyor).

Calistirma:
    ../.venv/bin/python hafta10_vektor_veritabani/2_gom_ve_indeksle.py
"""
import json

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

from ayarlar import (
    CHROMA_DIZINI,
    CHROMA_MESAFE,
    CHUNK_JSONL,
    DOKUMAN_PROMPT,
    EMBEDDING_MODELI,
    GOMME_BATCH,
    GOMME_DTYPE,
    KOLEKSIYON_ADI,
    VEKTOR_BOYUTU,
    VEKTOR_PARQUET,
)
from chunklama import gomulecek_metin


def chunklari_oku() -> list[dict]:
    if not CHUNK_JSONL.exists():
        raise SystemExit(f"{CHUNK_JSONL} yok - once 1_veri_hazirla.py calistirin.")
    with open(CHUNK_JSONL, encoding="utf-8") as f:
        return [json.loads(satir) for satir in f]


def gom(chunklar: list[dict]) -> np.ndarray:
    import torch
    from sentence_transformers import SentenceTransformer

    cihaz = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Model yukleniyor: {EMBEDDING_MODELI} ({cihaz}, {GOMME_DTYPE})")
    model = SentenceTransformer(
        EMBEDDING_MODELI,
        device=cihaz,
        model_kwargs={"dtype": getattr(torch, GOMME_DTYPE)},
    )

    # Baslik enjeksiyonu: chunk_text ham kalir, vektor baslikli metinden uretilir.
    metinler = [
        gomulecek_metin(c["title"], c["bolum_basligi"], c["chunk_text"]) for c in chunklar
    ]
    print(f"{len(metinler)} chunk gomuluyor (prompt='{DOKUMAN_PROMPT}')...")
    vektorler = model.encode(
        metinler,
        prompt_name=DOKUMAN_PROMPT,
        batch_size=GOMME_BATCH,
        show_progress_bar=True,
        convert_to_numpy=True,
    )
    return np.asarray(vektorler, dtype=np.float32)


def dogrula(vektorler: np.ndarray) -> None:
    if vektorler.shape[1] != VEKTOR_BOYUTU:
        raise SystemExit(f"boyut {vektorler.shape[1]}, beklenen {VEKTOR_BOYUTU}")
    normlar = np.linalg.norm(vektorler, axis=1)
    # Model 4_Normalize katmani tasiyor; kosinus = nokta carpimi varsayimi buna
    # dayaniyor, o yuzden dogruluyoruz.
    if not np.allclose(normlar, 1.0, atol=1e-3):
        raise SystemExit(
            f"vektorler L2-normalize degil (norm araligi "
            f"{normlar.min():.4f}-{normlar.max():.4f})"
        )
    print(f"OK  {vektorler.shape[0]} x {vektorler.shape[1]}, L2-normalize")


def chromaya_yaz(chunklar: list[dict], vektorler: np.ndarray) -> None:
    import chromadb

    if CHROMA_DIZINI.exists():
        import shutil

        shutil.rmtree(CHROMA_DIZINI)  # bastan kur: yarim indeks kalmasin

    istemci = chromadb.PersistentClient(path=str(CHROMA_DIZINI))
    koleksiyon = istemci.create_collection(
        name=KOLEKSIYON_ADI,
        metadata={"hnsw:space": CHROMA_MESAFE},
    )

    for bas in range(0, len(chunklar), 1000):  # Chroma tek seferde sinirli alir
        dilim = chunklar[bas : bas + 1000]
        koleksiyon.add(
            ids=[c["chunk_id"] for c in dilim],
            embeddings=vektorler[bas : bas + 1000].tolist(),
            documents=[c["chunk_text"] for c in dilim],
            metadatas=[
                {
                    "url": c["url"],
                    "title": c["title"],
                    "bolum_basligi": c["bolum_basligi"],
                    "parent_id": c["parent_id"],
                    "__source": c["__source"],
                    "token_sayisi": c["token_sayisi"],
                }
                for c in dilim
            ],
        )
    print(f"OK  Chroma: {koleksiyon.count()} kayit -> {CHROMA_DIZINI} ({CHROMA_MESAFE})")


def parquet_yaz(chunklar: list[dict], vektorler: np.ndarray) -> None:
    tablo = pa.table(
        {
            "chunk_id": [c["chunk_id"] for c in chunklar],
            "url": [c["url"] for c in chunklar],
            "chunk_text": [c["chunk_text"] for c in chunklar],
            "chunk_vector": pa.array(vektorler.tolist(), type=pa.list_(pa.float32())),
            "title": [c["title"] for c in chunklar],
            "bolum_basligi": [c["bolum_basligi"] for c in chunklar],
            "parent_id": [c["parent_id"] for c in chunklar],
            "__source": [c["__source"] for c in chunklar],
            "token_sayisi": [c["token_sayisi"] for c in chunklar],
        }
    )
    pq.write_table(tablo, VEKTOR_PARQUET, compression="zstd")
    boyut = VEKTOR_PARQUET.stat().st_size / 1e6
    print(f"OK  parquet: {tablo.num_rows} satir, {boyut:.1f} MB -> {VEKTOR_PARQUET}")


def main() -> None:
    chunklar = chunklari_oku()
    vektorler = gom(chunklar)
    dogrula(vektorler)
    chromaya_yaz(chunklar, vektorler)
    parquet_yaz(chunklar, vektorler)


if __name__ == "__main__":
    main()
