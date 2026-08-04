"""
Ortam doğrulama: gerekli kütüphaneler, GPU ve Hugging Face girişi kontrolü.

Çalıştırma:
    .venv/bin/python check_env.py
"""
import importlib
import os

from dotenv import load_dotenv

load_dotenv()

REQUIRED_PACKAGES = [
    "torch",
    "transformers",
    "datasets",
    "tokenizers",
    "trl",
    "peft",
    "unsloth",
    "huggingface_hub",
    "dotenv",
    "playwright",
    "bs4",
]


def check_packages():
    print("--- Kütüphaneler ---")
    missing = []
    for name in REQUIRED_PACKAGES:
        try:
            mod = importlib.import_module(name)
            version = getattr(mod, "__version__", "?")
            print(f"  OK  {name} ({version})")
        except ImportError:
            print(f"  EKSIK  {name}")
            missing.append(name)
    return missing


def check_gpu():
    print("--- GPU ---")
    import torch

    if torch.cuda.is_available():
        name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  OK  {name} ({total_mem:.1f} GB VRAM)")
    else:
        print("  UYARI  CUDA GPU bulunamadı, eğitim CPU'da çok yavaş olur.")


def check_hf_auth():
    print("--- Hugging Face girişi ---")
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN") or None
    try:
        who = HfApi().whoami(token=token)
        print(f"  OK  Giriş yapılmış: {who['name']}")
    except Exception as e:
        print("  UYARI  Giriş yapılmamış. Şunlardan birini yapın:")
        print("         1) terminalde: hf auth login")
        print("         2) .env dosyasında HF_TOKEN=... doldurun")
        print(f"         (detay: {e})")


def check_hafta7():
    """7. hafta (tool calling + Spaces) bagimsiz calisir, ayri kontrol edilir."""
    print("--- 7. hafta: tool calling / Gradio ---")
    for name in ("gradio", "transformers", "requests"):
        try:
            mod = importlib.import_module(name)
            print(f"  OK  {name} ({getattr(mod, '__version__', '?')})")
        except ImportError:
            print(f"  EKSIK  {name} — kurmak için:")
            print("         uv pip install --python .venv/bin/python -r hafta7_tool_calling/requirements.txt")

    arka_uc = (os.environ.get("TOOL_BACKEND") or "yerel").strip().lower()
    print(f"  TOOL_BACKEND: {arka_uc}", end="")
    if arka_uc in ("yerel", "local", "transformers", "zerogpu"):
        try:
            import torch

            print(
                f" (model yerelde çalışacak; CUDA: "
                f"{'var' if torch.cuda.is_available() else 'YOK — küçük model ya da TOOL_BACKEND=api kullanın'})"
            )
        except ImportError:
            print(" — UYARI: torch kurulu değil, yerel arka uç çalışmaz")
    else:
        print(" (model Inference Providers üzerinden çağrılacak)")

    if not os.environ.get("HF_TOKEN"):
        print(
            "  NOT  HF_TOKEN ortamda yok; yerelde `hf auth login` cache'i kullanılır. "
            "Space'e deploy ederken token'ı elle secret olarak girmeniz gerekir."
        )


def check_hafta8():
    """8. hafta (SQLite'a yazan tool calling ajani) - veritabani kontrolu."""
    print("--- 8. hafta: veritabanı ajanı ---")
    import sqlite3
    import sys
    from pathlib import Path

    hafta8 = Path(__file__).resolve().parent / "hafta8_veritabani_ajani"
    sys.path.insert(0, str(hafta8))
    try:
        import veritabani as db

        yol = db.db_kur()
        with db.baglanti(yol) as conn:
            (kitap,) = conn.execute("SELECT COUNT(*) FROM kitaplar").fetchone()
            (siparis,) = conn.execute("SELECT COUNT(*) FROM siparisler").fetchone()
        print(f"  OK  {yol} ({kitap} kitap, {siparis} sipariş)")
    except (ImportError, sqlite3.Error) as e:
        print(f"  HATA  veritabanı kurulamadı: {type(e).__name__}: {e}")
    finally:
        sys.path.remove(str(hafta8))


def check_env_file():
    print("--- .env dosyası ---")
    required_keys = [
        "DATASET_REPO_ID",
        "TOKENIZER_REPO_ID",
        "LORA_REPO_ID",
        "IDENTITY_DATASET_REPO_ID",
        "IDENTITY_LORA_REPO_ID",
        "BASE_MODEL",
        "FELSEFE_BENCHMARK_REPO_ID",
        "SPACE_REPO_ID",
        "TOOL_BACKEND",
        "TOOL_MODEL",
        "SPACE_REPO_ID_KITAPCI",
    ]
    for key in required_keys:
        value = os.environ.get(key, "")
        placeholder = "kullanici-adi" in value or not value
        status = "TODO (doldurulmamış)" if placeholder else value
        print(f"  {key}: {status}")


def check_playwright_browser():
    print("--- Playwright / Chromium (1. hafta: scrape_reddit.py) ---")
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            browser.close()
        print("  OK  Chromium başlatılabiliyor.")
    except Exception as e:
        print("  EKSIK  Chromium kurulu değil ya da başlatılamıyor. Şunu çalıştırın:")
        print("         .venv/bin/python -m playwright install chromium")
        print(f"         (detay: {e})")


if __name__ == "__main__":
    missing = check_packages()
    check_gpu()
    check_hf_auth()
    check_playwright_browser()
    check_hafta7()
    check_hafta8()
    check_env_file()
    if missing:
        raise SystemExit(f"\nEksik kütüphaneler var: {missing}")
