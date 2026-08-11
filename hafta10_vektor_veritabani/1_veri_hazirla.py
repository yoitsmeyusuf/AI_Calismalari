"""
10. Hafta - 1. adim: kaynak makaleleri indir, ornekle ve chunk'la.

Cikti: veri/chunks.jsonl (her satir bir chunk).


Calistirma:
    ../.venv/bin/python hafta10_vektor_veritabani/1_veri_hazirla.py
"""
import json
import random
import statistics
import sys

import pyarrow.parquet as pq
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import GatedRepoError
from transformers import AutoTokenizer

from ayarlar import (
    CHUNK_JSONL,
    EMBEDDING_MODELI,
    HASTANE_BASINA_MAKALE,
    HASTANELER,
    KAYNAK_DATASET,
    ORNEKLEM_TOHUMU,
    VERI,
)
from chunklama import MAX_TOK, MIN_TOK, chunkla


def split_indir(hastane: str) -> list[dict]:
    yol = hf_hub_download(
        KAYNAK_DATASET,
        f"data/{hastane}-00000-of-00001.parquet",
        repo_type="dataset",
    )
    return pq.read_table(yol).to_pylist()


def makaleleri_sec(satirlar: list[dict], rastgele: random.Random) -> list[dict]:
    """Bos/cok kisa ve tekrar eden makaleleri eleyip orneklem cikarir."""
    gorulen: set[str] = set()
    temiz = []
    for satir in satirlar:
        metin = (satir.get("text") or "").strip()
        if len(metin) < 400 or metin in gorulen:
            continue
        if not (satir.get("url") or "").strip():
            continue
        gorulen.add(metin)
        temiz.append(satir)

    if len(temiz) < HASTANE_BASINA_MAKALE:
        raise SystemExit(
            f"{len(temiz)} temiz makale var, {HASTANE_BASINA_MAKALE} isteniyor."
        )
    return rastgele.sample(temiz, HASTANE_BASINA_MAKALE)


def main() -> None:
    print(f"Tokenizer yukleniyor: {EMBEDDING_MODELI}")
    tokenizer = AutoTokenizer.from_pretrained(EMBEDDING_MODELI)

    def say(metin: str) -> int:
        return len(tokenizer.encode(metin, add_special_tokens=False))

    rastgele = random.Random(ORNEKLEM_TOHUMU)
    VERI.mkdir(exist_ok=True)

    tum_chunklar: list[dict] = []
    for hastane in HASTANELER:
        try:
            satirlar = split_indir(hastane)
        except GatedRepoError:
            sys.exit(
                f"HATA: {KAYNAK_DATASET} gated.\n"
                f"  https://huggingface.co/datasets/{KAYNAK_DATASET} sayfasinda\n"
                "  'Agree and access' tiklayip tekrar deneyin."
            )

        makaleler = makaleleri_sec(satirlar, rastgele)
        onceki = len(tum_chunklar)

        for makale_no, makale in enumerate(makaleler):
            makale_id = f"{hastane}-{makale_no:04d}"
            for chunk in chunkla(makale["text"], say):
                tum_chunklar.append(
                    {
                        "chunk_id": f"{makale_id}-{chunk['bolum']:02d}-{chunk['parca']}",
                        "url": makale["url"],
                        "title": (makale.get("title") or "").strip(),
                        "bolum_basligi": chunk["baslik"],
                        "chunk_text": chunk["metin"],
                        # Ayni bolumden cikan chunk'lar ayni parent'i paylasir:
                        # arama child uzerinde, cevap uretimi parent uzerinde.
                        "parent_id": f"{makale_id}-{chunk['bolum']:02d}",
                        "makale_id": makale_id,
                        "__source": hastane,
                        "token_sayisi": say(chunk["metin"]),
                    }
                )

        uretilen = tum_chunklar[onceki:]
        boylar = sorted(c["token_sayisi"] for c in uretilen)
        print(
            f"  {hastane:10} {len(makaleler):4} makale -> {len(uretilen):5} chunk "
            f"(medyan {statistics.median(boylar):.0f} tok, max {boylar[-1]})"
        )

    with open(CHUNK_JSONL, "w", encoding="utf-8") as f:
        for chunk in tum_chunklar:
            f.write(json.dumps(chunk, ensure_ascii=False) + "\n")

    boylar = sorted(c["token_sayisi"] for c in tum_chunklar)
    asan = sum(1 for b in boylar if b > MAX_TOK)
    kisa = sum(1 for b in boylar if b < MIN_TOK)
    print(
        f"\nTOPLAM {len(tum_chunklar)} chunk -> {CHUNK_JSONL}\n"
        f"  token: medyan {statistics.median(boylar):.0f} | "
        f"p90 {boylar[int(len(boylar) * 0.9)]} | "
        f"p99 {boylar[int(len(boylar) * 0.99)]} | max {boylar[-1]}\n"
        f"  tavani ({MAX_TOK}) asan: {asan} | tabanin ({MIN_TOK}) altinda: {kisa}\n"
        f"  basligi olan: {sum(1 for c in tum_chunklar if c['bolum_basligi']) / len(tum_chunklar):.0%}"
    )
    if asan:
        print("  UYARI: tavani asan chunk var, chunklama.py kontrol edilmeli.")


if __name__ == "__main__":
    main()
