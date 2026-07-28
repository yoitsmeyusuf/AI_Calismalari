"""
5. Hafta: MMLU benchmark karsilastirmasi (taban model vs LoRA fine-tune).

Turkce MMLU veri seti (6200 soru / 62 bolum) uzerinde calisir. Dogruluk
kontrolu icin harf eslesmesi + belirsiz durumda sentence-transformers ile
anlamsal benzerlik kullanilir. Model, Ollama gibi bir ara katmana ihtiyac
olmadan dogrudan unsloth.FastModel ile yuklenip calistirilir; boylece hem
taban model hem de HF'deki LoRA adaptoru (GGUF'a cevirmeden) test edilebilir.

Calistirma:
    .venv/bin/python hafta5_mmlu_benchmark/mmlu_benchmark.py
"""
import gc
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import pandas as pd
import torch
from dotenv import load_dotenv

load_dotenv()

HARFLER = ["A", "B", "C", "D", "E"]
OUT_DIR = Path(__file__).resolve().parent / "sonuclar"
OUT_DIR.mkdir(exist_ok=True)

PROMPT_ONEK = (
    "Sana soru ve seçenekleri veriyorum. sadece hangi seçeneğin sorunun doğru "
    "cevabı olduğunu yaz. Örneğin 'A' veya 'B' gibi. Lütfen herhangi bir "
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
    """Harf eslesmesi, sonra gerekirse anlamsal benzerlik ile dogruluk kontrolu."""
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


def soru_promptu(row):
    soru = row["soru"] + "\n"
    for j, secenek in enumerate(row["secenekler"]):
        soru += HARFLER[j] + ": " + secenek + "\n"
    return PROMPT_ONEK + soru


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


def modeli_test_et(etiket, model_name, mmlu_veri, token):
    print(f"\n=== {etiket} ({model_name}) yukleniyor ===")
    model, tokenizer = modeli_yukle(model_name, token)

    veri = mmlu_veri.reset_index(drop=True)
    toplam = len(veri)

    bolum_sonuc = {}
    dogru_sayisi = 0
    cevaplar = []
    baslama = time.time()

    for i in range(toplam):
        row = veri.iloc[i]
        prompt = soru_promptu(row)
        messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
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

        bolum = row["bolum"]
        bolum_sonuc.setdefault(bolum, {"dogru": 0, "toplam": 0})
        bolum_sonuc[bolum]["toplam"] += 1

        dogru = cevap_dogru_mu(row["cevap"], cevap, row["secenekler"])
        if dogru:
            dogru_sayisi += 1
            bolum_sonuc[bolum]["dogru"] += 1

        cevaplar.append({"soru_index": i, "bolum": bolum, "cevap_verilen": cevap, "dogru_mu": dogru})

        if (i + 1) % 25 == 0 or (i + 1) == toplam:
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
        "bolum_basari": {
            b: round(v["dogru"] / v["toplam"] * 100, 2) for b, v in bolum_sonuc.items()
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

    # Her bolumden dengeli (stratified) ornek aliniyor; tum 6200 soruyu
    # calistirmak yerine temsili bir alt kume kullanmak icin.
    sample_per_bolum = os.environ.get("SAMPLE_PER_BOLUM")
    sample_per_bolum = int(sample_per_bolum) if sample_per_bolum else 15

    print("Turkce MMLU verisi indiriliyor...")
    mmlu_veri_tam = pd.read_parquet(
        "hf://datasets/alibayram/yapay_zeka_turkce_mmlu_model_cevaplari/data/train-00000-of-00001.parquet"
    )
    print(f"{len(mmlu_veri_tam)} soru, {mmlu_veri_tam['bolum'].nunique()} bolum yuklendi.")

    if sample_per_bolum:
        parcalar = [
            grup.sample(n=min(sample_per_bolum, len(grup)), random_state=42)
            for _, grup in mmlu_veri_tam.groupby("bolum")
        ]
        mmlu_veri = pd.concat(parcalar, ignore_index=True)
        print(
            f"Stratified ornek: bolum basina {sample_per_bolum} soru -> "
            f"toplam {len(mmlu_veri)} soru ({mmlu_veri['bolum'].nunique()} bolum)."
        )
    else:
        mmlu_veri = mmlu_veri_tam

    sonuclar = []
    sonuclar.append(modeli_test_et("taban_model", base_model, mmlu_veri, token))
    sonuclar.append(modeli_test_et("finetune_lora", lora_repo, mmlu_veri, token))

    with open(OUT_DIR / "karsilastirma.json", "w", encoding="utf-8") as f:
        json.dump(sonuclar, f, ensure_ascii=False, indent=2)

    print("\n=== SONUC ===")
    for s in sonuclar:
        print(f"{s['etiket']} ({s['model_name']}): %{s['genel_basari']} ({s['dogru_sayisi']}/{s['toplam_soru']}) - {s['sure_saniye']}s")

    print("\nTUM SONUCLAR TAMAMLANDI.")


if __name__ == "__main__":
    main()
