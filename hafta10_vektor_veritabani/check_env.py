"""
10. hafta ortam kontrolu: paketler, gated veri setine erisim, uretilen dosyalar.
Hicbir sey yazmaz, salt okunur.

Calistirma:
    ../.venv/bin/python hafta10_vektor_veritabani/check_env.py
"""
import importlib
import json

from ayarlar import (
    CHROMA_DIZINI,
    CHUNK_JSONL,
    EMBEDDING_MODELI,
    ESIK_SONUC_JSON,
    HASTANELER,
    KAYNAK_DATASET,
    KOLEKSIYON_ADI,
    SORULAR_JSON,
    VEKTOR_BOYUTU,
    VEKTOR_PARQUET,
)

PAKETLER = [
    "torch",
    "transformers",
    "sentence_transformers",
    "chromadb",
    "datasets",
    "huggingface_hub",
    "pyarrow",
    "numpy",
    "matplotlib",
]


def check_paketler() -> list[str]:
    print("--- Kutuphaneler ---")
    eksik = []
    for ad in PAKETLER:
        try:
            mod = importlib.import_module(ad)
            print(f"  OK     {ad} ({getattr(mod, '__version__', '?')})")
        except ImportError:
            print(f"  EKSIK  {ad}")
            eksik.append(ad)
    if eksik:
        print("         uv pip install --python <venv>/bin/python "
              "-r hafta10_vektor_veritabani/requirements.txt")
    return eksik


def check_gpu() -> None:
    print("--- Donanim ---")
    try:
        import torch
    except ImportError:
        print("  ATLA   torch yok")
        return
    if torch.cuda.is_available():
        ad = torch.cuda.get_device_name(0)
        vram = torch.cuda.get_device_properties(0).total_memory / 1e9
        print(f"  OK     CUDA: {ad} ({vram:.1f} GB)")
    else:
        print("  UYARI  CUDA yok - CPU'da calisir, gomme yavaslar")


def check_kaynak() -> None:
    print(f"--- Kaynak veri seti ({KAYNAK_DATASET}) ---")
    try:
        from huggingface_hub import HfApi
        from huggingface_hub.errors import GatedRepoError, HfHubHTTPError
    except ImportError:
        print("  ATLA   huggingface_hub yok")
        return

    try:
        HfApi().dataset_info(KAYNAK_DATASET, files_metadata=False)
    except GatedRepoError:
        print("  HATA   gated ve erisim yok.")
        print(f"         https://huggingface.co/datasets/{KAYNAK_DATASET}")
        print("         sayfasinda 'Agree and access' tiklayin (otomatik onay).")
        return
    except HfHubHTTPError as hata:
        print(f"  HATA   {hata}")
        return

    # dataset_info gated repoda da calisabiliyor; asil test dosya indirme.
    try:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            KAYNAK_DATASET,
            f"data/{HASTANELER[0]}-00000-of-00001.parquet",
            repo_type="dataset",
        )
        print(f"  OK     erisim var ({', '.join(HASTANELER)})")
    except GatedRepoError:
        print("  HATA   gated - 'Agree and access' tiklanmamis.")
    except Exception as hata:  # noqa: BLE001
        print(f"  HATA   {type(hata).__name__}: {hata}")


def check_model() -> None:
    print(f"--- Embedding modeli ({EMBEDDING_MODELI}) ---")
    try:
        from huggingface_hub import hf_hub_download
    except ImportError:
        print("  ATLA   huggingface_hub yok")
        return
    try:
        yol = hf_hub_download(EMBEDDING_MODELI, "config_sentence_transformers.json")
        with open(yol, encoding="utf-8") as f:
            ayar = json.load(f)
        promptlar = ayar.get("prompts", {})
        for ad in ("query", "document"):
            if ad in promptlar:
                print(f"  OK     prompt '{ad}': {promptlar[ad]!r}")
            else:
                print(f"  HATA   prompt '{ad}' yok - arama asimetrik calismaz")
    except Exception as hata:  # noqa: BLE001
        print(f"  HATA   {type(hata).__name__}: {hata}")


def check_ciktilar() -> None:
    print("--- Uretilen dosyalar ---")

    if CHUNK_JSONL.exists():
        with open(CHUNK_JSONL, encoding="utf-8") as f:
            n = sum(1 for _ in f)
        print(f"  OK     chunks.jsonl ({n} chunk)")
    else:
        print("  YOK    chunks.jsonl - 1_veri_hazirla.py")

    if VEKTOR_PARQUET.exists():
        import pyarrow.parquet as pq

        tablo = pq.read_table(VEKTOR_PARQUET)
        boyut = len(tablo.column("chunk_vector")[0])
        durum = "OK    " if boyut == VEKTOR_BOYUTU else "HATA  "
        print(f"  {durum} chunks_vektorlu.parquet ({tablo.num_rows} satir, {boyut} boyut)")
    else:
        print("  YOK    chunks_vektorlu.parquet - 2_gom_ve_indeksle.py")

    if CHROMA_DIZINI.exists():
        try:
            import chromadb

            koleksiyon = chromadb.PersistentClient(
                path=str(CHROMA_DIZINI)
            ).get_collection(KOLEKSIYON_ADI)
            mesafe = (koleksiyon.metadata or {}).get("hnsw:space", "?")
            durum = "OK    " if mesafe == "cosine" else "HATA  "
            print(f"  {durum} chroma_db ({koleksiyon.count()} kayit, mesafe={mesafe})")
        except Exception as hata:  # noqa: BLE001
            print(f"  HATA   chroma_db okunamadi: {hata}")
    else:
        print("  YOK    chroma_db - 2_gom_ve_indeksle.py")

    if SORULAR_JSON.exists():
        with open(SORULAR_JSON, encoding="utf-8") as f:
            sorular = json.load(f)
        poz, neg = len(sorular["pozitif"]), len(sorular["negatif"])
        durum = "OK    " if (poz, neg) == (20, 10) else "UYARI "
        print(f"  {durum} sorular.json ({poz} pozitif + {neg} negatif)")
    else:
        print("  YOK    sorular.json")

    if ESIK_SONUC_JSON.exists():
        with open(ESIK_SONUC_JSON, encoding="utf-8") as f:
            sonuc = json.load(f)
        en_iyi = max(t["toplam_dogru"] for t in sonuc["tablo"])
        print(f"  OK     esik_sonuclari.json (esik {sonuc['en_iyi_esik']:.2f}, {en_iyi}/30)")
    else:
        print("  YOK    esik_sonuclari.json - 4_esik_analizi.py")


def main() -> None:
    eksik = check_paketler()
    check_gpu()
    if not eksik:
        check_kaynak()
        check_model()
    check_ciktilar()


if __name__ == "__main__":
    main()
