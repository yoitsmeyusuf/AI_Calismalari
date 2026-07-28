"""
6. Hafta: 100 soruluk felsefe benchmark verisini + 5 model karsilastirma
sonuclarini Hugging Face Hub'a bir Dataset olarak pusklar.

Pushlanan icerik:
- Sorular: felsefe_sorulari.py -> HF Dataset (id, kategori, soru, secenekler, cevap)
- README.md: sonuclar/karsilastirma.json okunarak otomatik uretilen, 5 modelin
  genel ve kategori bazli basari tablosunu iceren dataset karti.
- Kod ve ham sonuclar: felsefe_benchmark.py, felsefe_sorulari.py, sonuclar/*
  (tekrarlanabilirlik icin).

Calistirma (once felsefe_benchmark.py calistirilip sonuclar/ doldurulmus olmali):
    .venv/bin/python hafta6_felsefe_benchmark/push_dataset.py
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

from felsefe_sorulari import SORULAR, KATEGORILER

load_dotenv()

HERE = Path(__file__).resolve().parent
SONUCLAR_DIR = HERE / "sonuclar"


def sonuclari_yukle():
    path = SONUCLAR_DIR / "karsilastirma.json"
    if not path.exists():
        return None
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def sonuc_tablosu_markdown(sonuclar):
    if not sonuclar:
        return "_Henüz benchmark çalıştırılmadı — `felsefe_benchmark.py`'yi çalıştırıp bu dosyayı yeniden pushlayın._"

    siralanmis = sorted(sonuclar, key=lambda s: -s["genel_basari"])
    satirlar = [
        "| Model | Etiket | Doğru | Toplam | Başarı | Süre |",
        "|---|---|---|---|---|---|",
    ]
    for s in siralanmis:
        satirlar.append(
            f"| `{s['model_name']}` | {s['etiket']} | {s['dogru_sayisi']} | "
            f"{s['toplam_soru']} | **%{s['genel_basari']}** | {s['sure_saniye']}s |"
        )
    genel = "\n".join(satirlar)

    kategori_satirlar = ["| Kategori | " + " | ".join(s["etiket"] for s in siralanmis) + " |",
                          "|---" * (len(siralanmis) + 1) + "|"]
    for k in KATEGORILER:
        row = [k] + [f"%{s['kategori_basari'].get(k, 0)}" for s in siralanmis]
        kategori_satirlar.append("| " + " | ".join(row) + " |")
    kategori = "\n".join(kategori_satirlar)

    return f"### Genel sonuç\n\n{genel}\n\n### Kategori bazlı sonuç\n\n{kategori}"


def dataset_karti(sonuclar) -> str:
    kategori_liste = "\n".join(
        f"- {k}: {sum(1 for s in SORULAR if s['kategori'] == k)} soru" for k in KATEGORILER
    )
    return f"""---
license: apache-2.0
language:
- tr
task_categories:
- question-answering
- multiple-choice
pretty_name: Felsefe Benchmark (100 Soru, TR)
tags:
- felsefe
- philosophy
- benchmark
- turkish
- mcq
---

# Felsefe Benchmark — 100 Soru (Türkçe)

Yalnızca felsefe konularına odaklanan, 13 alt kategoriye yayılmış 100
soruluk çoktan seçmeli (A-E) bir değerlendirme seti. `hafta5_mmlu_benchmark`'taki
genel Türkçe MMLU testinden farklı olarak, bu set
[yoitsmeyusuf/felsefe-lora](https://huggingface.co/yoitsmeyusuf/felsefe-lora)
adaptörünü taban modeliyle ve farklı model aileleriyle karşılaştırmak için
hazırlandı.

## Kategoriler

{kategori_liste}

## Alanlar

- `id`: 1-100 arası soru kimliği
- `kategori`: yukarıdaki 13 kategoriden biri
- `soru`: soru metni (Türkçe)
- `secenekler`: 5 elemanlı seçenek listesi (A-E sırasıyla)
- `cevap`: doğru seçeneğin 0-indeksli konumu (0=A, 1=B, ..., 4=E)

## Değerlendirme yöntemi

`felsefe_benchmark.py` script'i (bu repoda), [hafta5_mmlu_benchmark/mmlu_benchmark.py](https://huggingface.co/yoitsmeyusuf/felsefe-lora/blob/main/mmlu_benchmark.py)
ile birebir aynı doğruluk kontrolünü kullanır: model cevabı doğru harfle
doğrudan eşleşmiyorsa, `sentence-transformers/paraphrase-multilingual-mpnet-base-v2`
ile cevabın seçeneklere anlamsal benzerliğine bakılır. Greedy decoding
(`do_sample=False`), `enable_thinking=False`, `max_new_tokens=20`.

## Test edilen 5 model

1. **Taban model** — `unsloth/Qwen3.5-4B`
2. **Fine-tune (LoRA)** — [`yoitsmeyusuf/felsefe-lora`](https://huggingface.co/yoitsmeyusuf/felsefe-lora) (3. haftada felsefe verisiyle eğitildi)
3. `unsloth/Qwen2.5-7B-Instruct-bnb-4bit`
4. `unsloth/gemma-2-2b-it-bnb-4bit`
5. `unsloth/Llama-3.2-3B-Instruct-bnb-4bit`

## Sonuçlar

{sonuc_tablosu_markdown(sonuclar)}

Ham cevaplar ve tam JSON sonuçları: [`sonuclar/`](https://huggingface.co/datasets/{os.environ.get("FELSEFE_BENCHMARK_REPO_ID", "kullanici-adi/felsefe-benchmark-100")}/tree/main/sonuclar).
Benchmark kodu: [`felsefe_benchmark.py`](https://huggingface.co/datasets/{os.environ.get("FELSEFE_BENCHMARK_REPO_ID", "kullanici-adi/felsefe-benchmark-100")}/blob/main/felsefe_benchmark.py).

## Kullanım

```python
from datasets import load_dataset

ds = load_dataset("{os.environ.get("FELSEFE_BENCHMARK_REPO_ID", "kullanici-adi/felsefe-benchmark-100")}", split="train")
print(ds[0])
```
"""


def main():
    from datasets import Dataset
    from huggingface_hub import HfApi

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ["FELSEFE_BENCHMARK_REPO_ID"]

    ds = Dataset.from_list(SORULAR)
    print(ds)
    ds.push_to_hub(repo_id, token=token)

    sonuclar = sonuclari_yukle()
    if sonuclar is None:
        print(
            "UYARI: sonuclar/karsilastirma.json bulunamadı — dataset kartı sonuç "
            "tablosu olmadan pushlanıyor. felsefe_benchmark.py'yi çalıştırıp bu "
            "script'i tekrar çalıştırınca kart otomatik güncellenir."
        )

    api = HfApi(token=token)
    api.upload_file(
        path_or_fileobj=dataset_karti(sonuclar).encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(HERE / "felsefe_benchmark.py"),
        path_in_repo="felsefe_benchmark.py",
        repo_id=repo_id,
        repo_type="dataset",
    )
    api.upload_file(
        path_or_fileobj=str(HERE / "felsefe_sorulari.py"),
        path_in_repo="felsefe_sorulari.py",
        repo_id=repo_id,
        repo_type="dataset",
    )
    if SONUCLAR_DIR.exists() and any(SONUCLAR_DIR.iterdir()):
        api.upload_folder(
            folder_path=str(SONUCLAR_DIR),
            path_in_repo="sonuclar",
            repo_id=repo_id,
            repo_type="dataset",
        )

    print(f"Yüklendi: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
