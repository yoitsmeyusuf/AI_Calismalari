"""
3. Hafta: LoRA adaptörünü test etme.

Base model + LORA_REPO_ID adaptörünü yükleyip birkaç felsefe sorusu sorar.

Çalıştırma:
    .venv/bin/python hafta3_finetune/test_inference.py
"""
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from dotenv import load_dotenv

load_dotenv()

QUESTIONS = [
    "Nietzsche'nin üst-insan (übermensch) kavramı nedir?",
    "Varoluşçuluk nedir, temsilcileri kimlerdir?",
    "Descartes'ın 'düşünüyorum öyleyse varım' önermesi ne anlama gelir?",
]


def main():
    from unsloth import FastModel
    from transformers import TextStreamer

    token = os.environ.get("HF_TOKEN")
    lora_repo = os.environ["LORA_REPO_ID"]

    model, tokenizer = FastModel.from_pretrained(
        model_name=lora_repo,
        max_seq_length=2048,
        load_in_4bit=True,
        token=token,
    )
    FastModel.for_inference(model)

    for q in QUESTIONS:
        print("\n" + "=" * 60)
        print("SORU:", q)
        print("=" * 60)
        messages = [{"role": "user", "content": [{"type": "text", "text": q}]}]
        inputs = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True, tokenize=True, return_tensors="pt"
        ).to(model.device)
        streamer = TextStreamer(tokenizer, skip_prompt=True)
        model.generate(
            input_ids=inputs,
            streamer=streamer,
            max_new_tokens=256,
            temperature=0.7,
            top_p=0.9,
        )


if __name__ == "__main__":
    main()
