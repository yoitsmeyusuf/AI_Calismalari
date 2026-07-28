"""
2. Hafta: BPE Tokenizer Oluşturma

Kendi korpusunuz (ör. 1. haftada topladığınız domain metinleri) üzerinde
sıfırdan bir BPE tokenizer eğitir ve Hugging Face Hub'a yükler.

Çalıştırma:
    .venv/bin/python hafta2_tokenizer/train_bpe_tokenizer.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
from transformers import PreTrainedTokenizerFast

load_dotenv()

VOCAB_SIZE = 8000  # korpus küçük (~380K karakter, ~527 satır) — 32000 için yeterli sinyal yok, ezberlemeye kayar
SPECIAL_TOKENS = ["<unk>", "<pad>", "<|endoftext|>", "<|user|>", "<|assistant|>", "<|system|>"]

# 1. haftanın temizlenmiş veri setinden (hafta1_veri_seti/data/raw/scraped_turkish_qa.jsonl)
# üretildi: her satırın soru+cevap metni ayrı satırlar halinde.
CORPUS_PATHS: list[str] = [
    str(Path(__file__).resolve().parent / "data" / "corpus.txt"),
]


def get_training_corpus(paths: list[str]):
    for path in paths:
        with open(path, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    yield line


def main():
    if not CORPUS_PATHS:
        raise SystemExit(
            "CORPUS_PATHS boş. Tokenizer'ı eğitmek için en az bir metin dosyası yolu ekleyin."
        )

    tokenizer = Tokenizer(models.BPE(unk_token="<unk>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()

    trainer = trainers.BpeTrainer(
        vocab_size=VOCAB_SIZE,
        special_tokens=SPECIAL_TOKENS,
        min_frequency=2,
    )
    tokenizer.train_from_iterator(get_training_corpus(CORPUS_PATHS), trainer=trainer)

    fast_tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="<unk>",
        pad_token="<pad>",
        bos_token="<|endoftext|>",
        eos_token="<|endoftext|>",
    )

    repo_id = os.environ["TOKENIZER_REPO_ID"]
    token = os.environ.get("HF_TOKEN")
    fast_tokenizer.push_to_hub(repo_id, token=token)
    print(f"Yüklendi: https://huggingface.co/{repo_id}")


if __name__ == "__main__":
    main()
