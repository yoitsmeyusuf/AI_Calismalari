"""
unsloth ile LoRA fine-tuning için ortak eğitim fonksiyonu.
3. Hafta (domain modeli) ve 4. Hafta (identity modeli) bu fonksiyonu,
sadece farklı dataset_repo/output_repo değerleriyle çağırır.
"""
import os

MAX_SEQ_LENGTH = 2048


def _conversation_to_text(example: dict, tokenizer) -> dict:
    messages = [{"role": m["role"], "content": m["content"]} for m in example["train"]]
    return {"text": tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)}


def train_lora(
    dataset_repo: str,
    output_repo: str,
    base_model: str,
    split: str | None = None,
    num_train_epochs: float = 1.0,
    lora_r: int = 16,
) -> None:
    # unsloth diğer script'lerde (ör. tokenizer eğitimi) gereksiz olduğundan
    # import'u fonksiyon içinde tutuyoruz.
    # FastModel (FastLanguageModel değil) kullanıyoruz: Qwen3.5 image-text-to-text
    # olarak paketlendiği için genel yükleyici gerekiyor, biz sadece metinle eğitiyoruz.
    from unsloth import FastModel
    from datasets import load_dataset, concatenate_datasets
    from trl import SFTTrainer, SFTConfig

    token = os.environ.get("HF_TOKEN")

    model, tokenizer = FastModel.from_pretrained(
        model_name=base_model,
        max_seq_length=MAX_SEQ_LENGTH,
        load_in_4bit=True,
        token=token,
    )

    model = FastModel.get_peft_model(
        model,
        r=lora_r,
        lora_alpha=lora_r,
        lora_dropout=0,
        bias="none",
        target_modules=["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
        use_gradient_checkpointing="unsloth",
        random_state=3407,
    )

    raw = load_dataset(dataset_repo, token=token)
    if split is not None:
        train_split = raw[split]
    elif len(raw) == 1:
        train_split = list(raw.values())[0]
    else:
        # turkish + english splitlerinin ikisiyle de eğit
        train_split = concatenate_datasets(list(raw.values()))

    dataset = train_split.map(lambda ex: _conversation_to_text(ex, tokenizer))

    trainer = SFTTrainer(
        model=model,
        tokenizer=tokenizer,
        train_dataset=dataset,
        dataset_text_field="text",
        max_seq_length=MAX_SEQ_LENGTH,
        args=SFTConfig(
            per_device_train_batch_size=2,
            gradient_accumulation_steps=4,
            warmup_steps=10,
            num_train_epochs=num_train_epochs,
            learning_rate=2e-4,
            logging_steps=1,
            optim="adamw_8bit",
            output_dir="outputs",
            report_to="none",
            seed=3407,
        ),
    )

    trainer.train()

    model.push_to_hub(output_repo, token=token)
    tokenizer.push_to_hub(output_repo, token=token)
    print(f"Yüklendi: https://huggingface.co/{output_repo}")
