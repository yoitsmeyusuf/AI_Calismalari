"""
6. Hafta: Felsefeye ozel 100 soruluk benchmark - 5 model karsilastirmasi.

hafta5_mmlu_benchmark/mmlu_benchmark.py'nin genel/indirilen MMLU verisi yerine, bu
script sifirdan yazilan felsefe_sorulari.py'deki 100 soruyu (13 kategori) 5
farkli modelde calistirir:
  1. taban_model      -> BASE_MODEL         (unsloth/Qwen3.5-4B, taban)
  2. finetune_lora     -> LORA_REPO_ID       (yoitsmeyusuf/felsefe-lora, 3. haftada egitilen)
  3-5. uc farkli aileden hazir instruct model (karsilastirma icin referans):
       unsloth/Qwen2.5-7B-Instruct-bnb-4bit, unsloth/gemma-2-2b-it-bnb-4bit,
       unsloth/Llama-3.2-3B-Instruct-bnb-4bit

Degerlendirme mantigi (harf esleme + belirsiz durumda sentence-transformers
ile anlamsal benzerlik) hafta5_mmlu_benchmark/mmlu_benchmark.py ile birebir aynidir.

Calistirma:
    .venv/bin/python hafta6_felsefe_benchmark/felsefe_benchmark.py
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

import pandas as pd
import torch
from dotenv import load_dotenv

from felsefe_sorulari import SORULAR, KATEGORILER

load_dotenv()

HARFLER = ["A", "B", "C", "D", "E"]
OUT_DIR = Path(__file__).resolve().parent / "sonuclar"
OUT_DIR.mkdir(exist_ok=True)

PROMPT_ONEK = (
    "Sana bir felsefe sorusu ve seçenekleri veriyorum. Sadece hangi seçeneğin "
    "doğru cevap olduğunu yaz. Örneğin 'A' veya 'B' gibi. Lütfen herhangi bir "
    "açıklama yapma!\nSoru: "
)

_anlamsal_model = None


def anlamsal_model():
    global _anlamsal_model
    if _anlamsal_model is None:
        from sentence_transformers import SentenceTransformer

        _anlamsal_model = SentenceTransformer("paraphrase-multilingual-mpnet-base-v2")
    return _anlamsal_model


def cevap_dogru_mu(dogru_cevap_index, verilen_cevap, secenekler):
    """hafta5_mmlu_benchmark/mmlu_benchmark.py ile birebir ayni mantik."""
    dogru_harf = HARFLER[dogru_cevap_index]
    verilen_cevap = verilen_cevap.upper().strip()

    if dogru_harf == verilen_cevap:
        return True
    elif len(verilen_cevap) > 1 and verilen_cevap[1] in [" ", ":", ")", "=", "-", "."]:
        return dogru_harf == verilen_cevap[0]
    else:
        model = anlamsal_model()
        encoded_cevap = model.encode([verilen_cevap])
        encoded_secenekler = model.encode(secenekler)
        benzerlik_listesi = model.similarity(encoded_cevap, encoded_secenekler).tolist()[0]
        en_yuksek = max(benzerlik_listesi)
        en_yuksek_index = benzerlik_listesi.index(en_yuksek)
        return en_yuksek_index == dogru_cevap_index


def soru_promptu(soru):
    metin = soru["soru"] + "\n"
    for j, secenek in enumerate(soru["secenekler"]):
        metin += HARFLER[j] + ": " + secenek + "\n"
    return PROMPT_ONEK + metin


def ilerleme_cubugu(guncel, toplam, uzunluk=40):
    ilerleme = guncel / toplam
    blok = int(uzunluk * ilerleme)
    return f"[{'#' * blok}{'-' * (uzunluk - blok)}] {ilerleme * 100:.2f}%"


def modeli_yukle(model_name, token):
    from unsloth import FastModel

    model, tokenizer = FastModel.from_pretrained(
        model_name=model_name,
        max_seq_length=2048,
        load_in_4bit=True,
        token=token,
    )
    FastModel.for_inference(model)
    return model, tokenizer


def modeli_test_et(etiket, model_name, multimodal, token):
    print(f"\n=== {etiket} ({model_name}) yukleniyor ===")
    model, tokenizer = modeli_yukle(model_name, token)

    toplam = len(SORULAR)
    kategori_sonuc = {k: {"dogru": 0, "toplam": 0} for k in KATEGORILER}
    dogru_sayisi = 0
    cevaplar = []
    baslama = time.time()

    for i, soru in enumerate(SORULAR):
        prompt = soru_promptu(soru)
        if multimodal:
            content = [{"type": "text", "text": prompt}]
        else:
            content = prompt
        messages = [{"role": "user", "content": content}]
        inputs = tokenizer.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_tensors="pt",
            enable_thinking=False,
        ).to(model.device)

        with torch.no_grad():
            out = model.generate(
                input_ids=inputs,
                max_new_tokens=20,
                do_sample=False,
                pad_token_id=tokenizer.eos_token_id,
            )
        cevap = tokenizer.decode(out[0][inputs.shape[1]:], skip_special_tokens=True).strip()

        kategori = soru["kategori"]
        kategori_sonuc[kategori]["toplam"] += 1

        dogru = cevap_dogru_mu(soru["cevap"], cevap, soru["secenekler"])
        if dogru:
            dogru_sayisi += 1
            kategori_sonuc[kategori]["dogru"] += 1

        cevaplar.append({
            "soru_id": soru["id"],
            "kategori": kategori,
            "cevap_verilen": cevap,
            "dogru_cevap": HARFLER[soru["cevap"]],
            "dogru_mu": dogru,
        })

        if (i + 1) % 10 == 0 or (i + 1) == toplam:
            gecen = time.time() - baslama
            print(
                f"\r{etiket}: {i + 1}/{toplam} | dogru: {dogru_sayisi} | "
                f"basari: {round(dogru_sayisi / (i + 1) * 100, 2)}% | "
                f"gecen: {round(gecen)}s | {ilerleme_cubugu(i + 1, toplam)}",
                end="",
            )

    print()
    genel_basari = round(dogru_sayisi / toplam * 100, 2)

    sonuc = {
        "etiket": etiket,
        "model_name": model_name,
        "toplam_soru": toplam,
        "dogru_sayisi": dogru_sayisi,
        "genel_basari": genel_basari,
        "sure_saniye": round(time.time() - baslama, 1),
        "kategori_basari": {
            k: round(v["dogru"] / v["toplam"] * 100, 2) for k, v in kategori_sonuc.items()
        },
    }

    with open(OUT_DIR / f"{etiket}.json", "w", encoding="utf-8") as f:
        json.dump(sonuc, f, ensure_ascii=False, indent=2)
    pd.DataFrame(cevaplar).to_csv(OUT_DIR / f"{etiket}_cevaplar.csv", index=False)

    del model
    gc.collect()
    torch.cuda.empty_cache()

    return sonuc


def main():
    token = os.environ.get("HF_TOKEN")
    base_model = os.environ.get("BASE_MODEL", "unsloth/Qwen3.5-4B")
    lora_repo = os.environ["LORA_REPO_ID"]

    modeller = [
        {"etiket": "taban_model", "model_name": base_model, "multimodal": True},
        {"etiket": "finetune_lora", "model_name": lora_repo, "multimodal": True},
        {"etiket": "qwen2_5_7b_instruct", "model_name": "unsloth/Qwen2.5-7B-Instruct-bnb-4bit", "multimodal": False},
        {"etiket": "gemma2_2b_it", "model_name": "unsloth/gemma-2-2b-it-bnb-4bit", "multimodal": False},
        {"etiket": "llama3_2_3b_instruct", "model_name": "unsloth/Llama-3.2-3B-Instruct-bnb-4bit", "multimodal": False},
    ]

    print(f"Felsefe benchmark: {len(SORULAR)} soru, {len(KATEGORILER)} kategori, {len(modeller)} model.")

    sonuclar = []
    for m in modeller:
        sonuclar.append(modeli_test_et(m["etiket"], m["model_name"], m["multimodal"], token))

    with open(OUT_DIR / "karsilastirma.json", "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, ensure_ascii=False, indent=2)

    print("\n=== SONUC ===")
    for s in sorted(sonuclar, key=lambda x: -x["genel_basari"]):
        print(f"{s['etiket']} ({s['model_name']}): %{s['genel_basari']} ({s['dogru_sayisi']}/{s['toplam_soru']}) - {s['sure_saniye']}s")

    print("\nTUM SONUCLAR TAMAMLANDI.")


if __name__ == "__main__":
    main()
