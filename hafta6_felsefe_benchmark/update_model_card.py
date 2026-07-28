"""
6. Hafta: yoitsmeyusuf/felsefe-lora model kartina, 5. haftadaki genel MMLU
bolumunun yanina, bu klasordeki felsefeye ozel 100 soruluk 5-model
karsilastirmasini ekleyen bir "Felsefe Benchmark (Ozel, 100 Soru)" bolumu
ekler ve HF Hub'a pushlar.

hafta5_mmlu_benchmark'in "MMLU Benchmark" bolumunu DEGISTIRMEZ, sadece onun
altina/yanina yeni bir bolum ekler. Dosya adi carpismasini onlemek icin
(5. haftanin sonuclar/ klasoruyle ayni isimler) bu calismanin ciktilari repoda
ayri bir "felsefe_benchmark_hafta6/" alt klasorune pushlanir.

Calistirma (once felsefe_benchmark.py calistirilip sonuclar/ doldurulmus olmali):
    .venv/bin/python hafta6_felsefe_benchmark/update_model_card.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

from push_dataset import sonuc_tablosu_markdown, sonuclari_yukle

load_dotenv()

HERE = Path(__file__).resolve().parent
SONUCLAR_DIR = HERE / "sonuclar"

BOLUM_BASLIGI = "## Felsefe Benchmark (Özel, 100 Soru)"


def yeni_bolum(sonuclar, repo_id) -> str:
    tablo = sonuc_tablosu_markdown(sonuclar)
    return f"""{BOLUM_BASLIGI}

5. haftadaki genel Türkçe MMLU testine ek olarak, model sıfırdan hazırlanmış,
yalnızca felsefe konularına odaklanan 100 soruluk (13 kategori) özel bir
benchmark'ta da test edildi ve taban modelin yanı sıra 3 farklı model
ailesinden referans modellerle karşılaştırıldı. Soru seti ve değerlendirme
kodu: [`felsefe_benchmark.py`](https://huggingface.co/{repo_id}/blob/main/felsefe_benchmark_hafta6/felsefe_benchmark.py),
[`felsefe_sorulari.py`](https://huggingface.co/{repo_id}/blob/main/felsefe_benchmark_hafta6/felsefe_sorulari.py)
(bu repoda `felsefe_benchmark_hafta6/` altında; asıl geliştirme deposu: [hafta6_felsefe_benchmark](https://huggingface.co/datasets/{os.environ.get("FELSEFE_BENCHMARK_REPO_ID", "")})).

Yöntem `mmlu_benchmark.py` ile birebir aynı: harf eşleşmesi + belirsiz
durumda `sentence-transformers/paraphrase-multilingual-mpnet-base-v2` ile
anlamsal benzerlik, greedy decoding (`do_sample=False`), `enable_thinking=False`.

{tablo}

Ham cevaplar ve tam sonuç JSON'ları: [`felsefe_benchmark_hafta6/sonuclar/`](https://huggingface.co/{repo_id}/tree/main/felsefe_benchmark_hafta6/sonuclar).
"""


def main():
    from huggingface_hub import HfApi, hf_hub_download

    token = os.environ.get("HF_TOKEN")
    repo_id = os.environ["LORA_REPO_ID"]
    api = HfApi(token=token)

    sonuclar = sonuclari_yukle()
    if sonuclar is None:
        raise SystemExit(
            "sonuclar/karsilastirma.json bulunamadı. Önce felsefe_benchmark.py "
            "çalıştırılmalı."
        )

    readme_path = hf_hub_download(repo_id, "README.md", token=token)
    readme = Path(readme_path).read_text(encoding="utf-8")

    bolum = yeni_bolum(sonuclar, repo_id)

    if BOLUM_BASLIGI in readme:
        # Onceki calistirmadan kalan bolumu guncelle (bastan sona degistir).
        before, _, rest = readme.partition(BOLUM_BASLIGI)
        _, _, after = rest.partition("\n## Kullanım")
        yeni_readme = before + bolum + "\n## Kullanım" + after
    elif "\n## Kullanım" in readme:
        yeni_readme = readme.replace("\n## Kullanım", "\n" + bolum + "\n## Kullanım", 1)
    else:
        yeni_readme = readme.rstrip() + "\n\n" + bolum

    api.upload_file(
        path_or_fileobj=yeni_readme.encode("utf-8"),
        path_in_repo="README.md",
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj=str(HERE / "felsefe_benchmark.py"),
        path_in_repo="felsefe_benchmark_hafta6/felsefe_benchmark.py",
        repo_id=repo_id,
    )
    api.upload_file(
        path_or_fileobj=str(HERE / "felsefe_sorulari.py"),
        path_in_repo="felsefe_benchmark_hafta6/felsefe_sorulari.py",
        repo_id=repo_id,
    )
    if SONUCLAR_DIR.exists() and any(SONUCLAR_DIR.iterdir()):
        api.upload_folder(
            folder_path=str(SONUCLAR_DIR),
            path_in_repo="felsefe_benchmark_hafta6/sonuclar",
            repo_id=repo_id,
        )

    print(f"Model kartı güncellendi: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
