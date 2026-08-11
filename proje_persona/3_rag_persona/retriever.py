"""
Ortak arama katmani: Chroma sorgusu + esik kontrolu.

Hem 2_esik_analizi.py hem persona.py bunu kullaniyor ki esik ve prompt
mantigi tek yerde dursun.
"""
from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import ayarlar  # noqa: E402


@dataclass
class Sonuc:
    id: str
    metin: str
    sarki: str
    benzerlik: float


class Retriever:
    """
    aygit: gomme modelinin kosacagi yer.

    Varsayilan cuda, AMA yerel LLM arka ucu kullanilirken CPU'ya alinmali:
    8 GB VRAM'de 7B-4bit (~5.5 GB) ile gomme modeli (200M float32, ~1 GB)
    ayni anda durunca device_map="auto" katmanlari CPU'ya tasiyor ve
    bitsandbytes bunu reddediyor. Retriever sorgu basina TEK kisa metin
    gomuyor; CPU'da farki hissedilmiyor.

    EMBED_AYGIT ortam degiskeniyle de verilebilir.
    """

    def __init__(self, aygit: str | None = None) -> None:
        import os

        istenen = aygit or os.environ.get("EMBED_AYGIT")
        if not istenen and (os.environ.get("TOOL_BACKEND", "").lower() == "yerel"):
            istenen = "cpu"
        self.aygit = istenen or ("cuda" if torch.cuda.is_available() else "cpu")
        self._model = None
        self._koleksiyon = None

    def yukle(self) -> None:
        if self._model is not None:
            return
        import chromadb
        from sentence_transformers import SentenceTransformer

        self._model = SentenceTransformer(
            ayarlar.EMBEDDING_MODELI, device=self.aygit,
            model_kwargs={"dtype": getattr(torch, ayarlar.GOMME_DTYPE)},
        )
        istemci = chromadb.PersistentClient(path=str(ayarlar.CHROMA_DIZINI))
        self._koleksiyon = istemci.get_collection(ayarlar.KOLEKSIYON_ADI)

    def goml(self, metinler: list[str]):
        self.yukle()
        # Model asimetrik: SORGU tarafi query prompt'undan gecmek zorunda.
        # Atlanirsa model yine calisir ama skor dagilimi kayar ve esik
        # analizi anlamsizlasir (10. hafta bulgusu).
        return self._model.encode(
            metinler, prompt_name=ayarlar.SORGU_PROMPT,
            normalize_embeddings=True, convert_to_numpy=True,
            show_progress_bar=False,
        )

    def ara(self, mesaj: str, k: int | None = None) -> list[Sonuc]:
        self.yukle()
        k = k or ayarlar.VARSAYILAN_K
        v = self.goml([mesaj])
        ham = self._koleksiyon.query(query_embeddings=v.tolist(), n_results=k)
        return [
            # Chroma DISTANCE dondurur, benzerlik degil. Esigi distance
            # sanip "< 0.7" yazmak tam ters filtre kurar.
            Sonuc(id=i, metin=b, sarki=m.get("sarki", ""), benzerlik=1.0 - d)
            for i, b, m, d in zip(ham["ids"][0], ham["documents"][0],
                                  ham["metadatas"][0], ham["distances"][0])
        ]

    def esikle_ara(self, mesaj: str, esik: float | None = None,
                   k: int | None = None) -> tuple[list[Sonuc], bool]:
        """Doner: (esigi gecen sonuclar, yeterli_baglam_var_mi)."""
        esik = ayarlar.BENZERLIK_ESIGI if esik is None else esik
        sonuclar = self.ara(mesaj, k)
        gecenler = [s for s in sonuclar if s.benzerlik >= esik]
        return gecenler, bool(gecenler)


if __name__ == "__main__":
    r = Retriever()
    for mesaj in sys.argv[1:] or ["bugün kendimi çok yalnız hissediyorum"]:
        sonuclar = r.ara(mesaj)
        print(f"\n> {mesaj}")
        for s in sonuclar:
            print(f"  {s.benzerlik:.4f}  [{s.sarki}]  ({len(s.metin)} karakter)")
