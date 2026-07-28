"""
1. Hafta: Veri Seti (Dataset) Hazırlama

Akış:
1) scrape_reddit.py (r/felsefe, Playwright ile, hesapsız) ve/veya
   scrape_eksisozluk.py (hesapsız) çalıştırılır; ikisi de aynı dosyaya
   (data/raw/scraped_turkish_qa.jsonl) satır ekler — veri elle yazılmak
   yerine tamamen scraping ile toplanıyor.
2) (opsiyonel) seed_examples.py'ye ekleyeceğiniz elle yazılmış örnekler de
   dahil edilir (kalite artırmak / scraping'in kaçırdığı konuları eklemek için).
3) (opsiyonel) augment_with_model() ile küçük bir alt küme çoğaltılabilir —
   scraping zaten organik veri sağladığından genelde gerekmez.
4) Ortak mesaj şemasında (bkz. common/hf_dataset_schema.py) tek dilli
   ("turkish") bir Dataset oluşturulur.
5) Hugging Face Hub'a push edilir (DATASET_REPO_ID).

Çalıştırma:
    .venv/bin/python hafta1_veri_seti/scrape_reddit.py       # r/felsefe'den topla
    .venv/bin/python hafta1_veri_seti/scrape_eksisozluk.py   # (opsiyonel) ek kaynak
    .venv/bin/python hafta1_veri_seti/build_dataset.py        # sonra push et
"""
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

from common.hf_dataset_schema import build_dataset_dict, make_conversation, make_message
from seed_examples import TURKISH_SEEDS

load_dotenv()

SCRAPED_PATH = Path(__file__).resolve().parent / "data" / "raw" / "scraped_turkish_qa.jsonl"

# Sentetik çoğaltma açık/kapalı. Scraping ile toplanan veri zaten organik
# olduğundan varsayılan olarak kapalı; küçük kalan bir konuyu genişletmek
# isterseniz True yapabilirsiniz.
AUGMENT_WITH_MODEL = False
VARIATIONS_PER_SEED = 5


def load_scraped_pairs(path: Path) -> list[dict]:
    if not path.exists():
        return []
    pairs = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            row = json.loads(line)
            pairs.append({"soru": row["soru"], "cevap": row["cevap"]})
    return pairs


def augment_with_model(seeds: list[dict], n_variations: int, model_name: str | None = None) -> list[dict]:
    """Elle yazılan/seçilen örnekleri yerel bir model ile parafraz ederek genişletir.

    NOT: BASE_MODEL (.env) varsayılanı Qwen3.5-4B, image-text-to-text olarak
    paketli olduğu için standart `transformers.pipeline("text-generation", ...)`
    ile doğrudan çalışmayabilir. Augmentation için ayrı, sade bir metin modeli
    (ör. "unsloth/Qwen2.5-3B-Instruct") vermeniz önerilir.
    """
    from transformers import pipeline

    model_name = model_name or "unsloth/Qwen2.5-3B-Instruct"
    generator = pipeline("text-generation", model=model_name, device_map="auto")

    augmented = list(seeds)
    for seed in seeds:
        prompt = (
            f"Aşağıdaki soru-cevap örneğini aynı konuda ama farklı kelimelerle "
            f"{n_variations} farklı şekilde yeniden yaz. Sadece yeni soru-cevapları listele.\n\n"
            f"Soru: {seed['soru']}\nCevap: {seed['cevap']}"
        )
        output = generator(prompt, max_new_tokens=512, do_sample=True, temperature=0.8)
        # TODO: output[0]["generated_text"] içeriğini parse edip
        # {"soru": ..., "cevap": ...} sözlüklerine dönüştürün ve augmented'e ekleyin.

    return augmented


def seeds_to_conversations(seeds: list[dict]) -> list:
    conversations = []
    for item in seeds:
        conversations.append(
            make_conversation(
                make_message("user", item["soru"]),
                make_message("assistant", item["cevap"], thinking=item.get("dusunce", "")),
            )
        )
    return conversations


def main():
    pairs = load_scraped_pairs(SCRAPED_PATH) + TURKISH_SEEDS

    if not pairs:
        raise SystemExit(
            f"Veri bulunamadı. Önce 'scrape_reddit.py' ve/veya 'scrape_eksisozluk.py' "
            f"çalıştırıp {SCRAPED_PATH}'ı doldurun, veya seed_examples.py'ye elle örnek ekleyin."
        )

    if AUGMENT_WITH_MODEL:
        pairs = augment_with_model(pairs, VARIATIONS_PER_SEED)

    dataset_dict = build_dataset_dict({"turkish": seeds_to_conversations(pairs)})
    print(dataset_dict)
    print(f"Toplam örnek: {len(pairs)} (scraped: {len(pairs) - len(TURKISH_SEEDS)}, elle: {len(TURKISH_SEEDS)})")

    repo_id = "yoitsmeyusuf/felsefe_finetune"
    token = os.environ.get("HF_TOKEN")
    dataset_dict.push_to_hub(repo_id, token=token)
    print(f"Yüklendi: https://huggingface.co/datasets/{repo_id}")


if __name__ == "__main__":
    main()
