"""
9. hafta ortam kontrolü: paketler, CSV şeması ve HF girişi.
Hub'a hiçbir şey yazmaz, salt okunur.

Çalıştırma:
    ../.venv/bin/python hafta9_benzerlik_veri_seti/check_env.py
"""
import csv
import importlib

from push_dataset import CSV_YOLU, PUAN_MAX, PUAN_MIN, REPO_ID

SUTUNLAR = ["sentence1", "sentence2", "score"]
PAKETLER = ["datasets", "huggingface_hub", "pandas"]


def check_paketler():
    print("--- Kütüphaneler ---")
    eksik = []
    for ad in PAKETLER:
        try:
            mod = importlib.import_module(ad)
            print(f"  OK  {ad} ({getattr(mod, '__version__', '?')})")
        except ImportError:
            print(f"  EKSIK  {ad}")
            eksik.append(ad)
    if eksik:
        print("         uv pip install --python <venv>/bin/python -r hafta9_benzerlik_veri_seti/requirements.txt")
    return eksik


def check_csv():
    print("--- veri/ciftler.csv ---")
    if not CSV_YOLU.exists():
        print(f"  EKSIK  {CSV_YOLU} yok.")
        return
    with open(CSV_YOLU, encoding="utf-8", newline="") as f:
        satirlar = list(csv.DictReader(f))
    basliklar = list(satirlar[0].keys()) if satirlar else []

    if basliklar != SUTUNLAR:
        print(f"  HATA  sütunlar {basliklar}, beklenen {SUTUNLAR}")
        return

    hatali = []
    puansiz = 0
    for i, satir in enumerate(satirlar, start=2):  # 1. satır başlık
        if not satir["sentence1"].strip() or not satir["sentence2"].strip():
            hatali.append(f"satır {i}: boş cümle")
        try:
            puan = float((satir["score"] or "").strip())
        except ValueError:
            hatali.append(f"satır {i}: score sayı değil ({satir['score']!r})")
            continue
        if puan != puan:  # nan
            puansiz += 1
        elif not PUAN_MIN <= puan <= PUAN_MAX:
            hatali.append(f"satır {i}: score {puan} aralık dışı [{PUAN_MIN}, {PUAN_MAX}]")

    print(f"  OK  şema {SUTUNLAR}, {len(satirlar)} satır")
    if puansiz:
        print(f"  NOT  {puansiz}/{len(satirlar)} satırda score = nan (puanlama yapılmadı)")
    for h in hatali:
        print(f"  HATA  {h}")


def check_hf():
    print("--- Hugging Face ---")
    from huggingface_hub import HfApi

    try:
        who = HfApi().whoami()
        rol = (who.get("auth") or {}).get("accessToken", {}).get("role", "?")
        print(f"  OK  giriş: {who['name']} (token rolü: {rol})")
        if rol == "read":
            print("  UYARI  salt-okunur token — push için 'write' izinli token gerekir.")
    except Exception as e:
        print("  UYARI  giriş yok. `hf auth login` çalıştırın.")
        print(f"         (detay: {e})")
    print(f"  hedef repo: {REPO_ID}")


if __name__ == "__main__":
    eksik = check_paketler()
    check_csv()
    check_hf()
    if eksik:
        raise SystemExit(f"\nEksik kütüphaneler var: {eksik}")
