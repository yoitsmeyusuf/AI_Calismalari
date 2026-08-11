"""
Asama 1: chunk'la -> vektorlestir -> ChromaDB'ye indeksle.

10. haftadan devralinan iki tuzak (ayarlar.py'de de yazili):
  1. Chroma'nin varsayilan mesafesi L2, kosinus degil -> hnsw:space=cosine
  2. Chroma "distance" dondurur, benzerlik degil -> benzerlik = 1 - distance

    ../../.venv/bin/python 1_chunkla_gom.py
"""
from __future__ import annotations

import json
import sys
import time
from pathlib import Path

import torch

BURASI = Path(__file__).resolve().parent
sys.path.insert(0, str(BURASI))
sys.path.insert(0, str(BURASI.parent))

import ayarlar  # noqa: E402
from chunklama import korpusu_chunkla  # noqa: E402

CHUNK_JSONL = BURASI / "chunklar.jsonl"


def main() -> None:
    import chromadb
    from sentence_transformers import SentenceTransformer

    chunklar = korpusu_chunkla()
    print(f"chunk : {len(chunklar)}  ({len({c['sarki'] for c in chunklar})} sarki)")

    aygit = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"aygit : {aygit}")
    print(f"model : {ayarlar.EMBEDDING_MODELI}")

    # float32: bf16'da L2 normlar 0.996-1.004 arasinda kayiyor ve
    # "kosinus = nokta carpimi" varsayimi tam tutmuyor (10. hafta olcumu).
    model = SentenceTransformer(
        ayarlar.EMBEDDING_MODELI, device=aygit,
        model_kwargs={"dtype": getattr(torch, ayarlar.GOMME_DTYPE)},
    )

    t0 = time.time()
    vektorler = model.encode(
        [c["gomme_metni"] for c in chunklar],
        prompt_name=ayarlar.DOKUMAN_PROMPT,   # asimetrik model: document prompt'u
        batch_size=64,
        normalize_embeddings=True,
        show_progress_bar=False,
        convert_to_numpy=True,
    )
    print(f"gomme : {vektorler.shape} - {time.time()-t0:.1f} sn")

    normlar = (vektorler ** 2).sum(axis=1) ** 0.5
    print(f"L2 norm: {normlar.min():.6f} - {normlar.max():.6f}  (1.0 olmali)")
    assert vektorler.shape[1] == ayarlar.VEKTOR_BOYUTU

    # --- Chroma ---
    istemci = chromadb.PersistentClient(path=str(ayarlar.CHROMA_DIZINI))
    try:
        istemci.delete_collection(ayarlar.KOLEKSIYON_ADI)
    except Exception:
        pass
    koleksiyon = istemci.create_collection(
        ayarlar.KOLEKSIYON_ADI,
        metadata={"hnsw:space": ayarlar.CHROMA_MESAFE},
    )
    koleksiyon.add(
        ids=[c["id"] for c in chunklar],
        embeddings=vektorler.tolist(),
        documents=[c["chunk_text"] for c in chunklar],
        metadatas=[{"sarki": c["sarki"], "bar_sayisi": c["bar_sayisi"],
                    "karakter": c["karakter"]} for c in chunklar],
    )
    print(f"chroma: {koleksiyon.count()} kayit -> {ayarlar.CHROMA_DIZINI.name}/")

    with CHUNK_JSONL.open("w", encoding="utf-8") as f:
        for c in chunklar:
            f.write(json.dumps(c, ensure_ascii=False) + "\n")
    print(f"  -> {CHUNK_JSONL.name}")

    # --- HNSW dogrulamasi: yaklasik arama kesin kosinusle ortusuyor mu ---
    import numpy as np

    sorgu = model.encode(
        ["yalnızlık ve hayal kırıklığı üzerine ne düşünüyorsun"],
        prompt_name=ayarlar.SORGU_PROMPT, normalize_embeddings=True,
        convert_to_numpy=True,
    )
    chroma_sonuc = koleksiyon.query(query_embeddings=sorgu.tolist(), n_results=5)
    chroma_benzerlik = [1 - d for d in chroma_sonuc["distances"][0]]
    kesin = (vektorler @ sorgu[0])
    kesin_ilk5 = np.sort(kesin)[::-1][:5]
    print("\nHNSW dogrulamasi (Chroma yaklasik arama vs numpy kesin kosinus):")
    for i, (c, k) in enumerate(zip(chroma_benzerlik, kesin_ilk5)):
        print(f"  {i+1}. chroma {c:.6f}   kesin {k:.6f}   fark {abs(c-k):.2e}")


if __name__ == "__main__":
    main()
