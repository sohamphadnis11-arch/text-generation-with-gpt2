# -*- coding: utf-8 -*-
"""
Refactored GPT-2 Text Generation Script
Prodigy Infotech Internship Task
"""

# ==========================================================
#  🔧 STEP 1 — Install Dependencies
# ==========================================================
!pip install -U pip setuptools wheel
!pip install tokenizers==0.13.3 --only-binary=:all:
!pip install transformers==4.31.0 datasets torch --quiet

# ==========================================================
#  📦 STEP 2 — Import Required Libraries
# ==========================================================
import os
os.environ["WANDB_DISABLED"] = "true"

from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    Trainer,
    TrainingArguments,
    DataCollatorForLanguageModeling,
)
from datasets import Dataset
import torch

# ==========================================================
#  🧠 STEP 3 — Model & Tokenizer Loader
# ==========================================================
def load_model_and_tokenizer(model_name="gpt2"):
    tokenizer = GPT2Tokenizer.from_pretrained(model_name)
    model = GPT2LMHeadModel.from_pretrained(model_name)

    tokenizer.pad_token = tokenizer.eos_token
    model.config.pad_token_id = model.config.eos_token_id

    return model, tokenizer


# ==========================================================
#  📚 STEP 4 — Dataset Creation & Tokenization
# ==========================================================
def create_dataset():
    text_samples = [
        "Artificial intelligence is changing the world of technology.",
        "Machine learning helps computers learn from experience.",
        "Natural language processing enables communication with computers.",
        "Data science combines math and coding to solve real problems.",
        "AI will make automation smarter and more efficient in the future.",
    ]
    return Dataset.from_dict({"text": text_samples})


def tokenize_dataset(dataset, tokenizer):
    def encode(batch):
        return tokenizer(
            batch["text"],
            truncation=True,
            padding="max_length",
            max_length=64
        )
    return dataset.map(encode, batched=True)


# ==========================================================
#  ⚙️ STEP 5 — Training Setup
# ==========================================================
def configure_training():
    return TrainingArguments(
        output_dir="./results",
        overwrite_output_dir=True,
        num_train_epochs=2,
        per_device_train_batch_size=2,
        save_steps=500,
        save_total_limit=2,
        logging_steps=5,
    )


# ==========================================================
#  🏋️ STEP 6 — Train the Model
# ==========================================================
def train_model(model, tokenizer, train_dataset, training_args):
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

    trainer = Trainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        data_collator=data_collator,
    )

    print("🚀 Training initiated...")
    trainer.train()

    return trainer


# ==========================================================
#  ✨ STEP 7 — Text Generation
# ==========================================================
def generate_text(model, tokenizer, prompt="Artificial intelligence"):
    encoded = tokenizer(prompt, return_tensors="pt")

    output_tokens = model.generate(
        **encoded,
        max_length=80,
        temperature=0.7,
        top_p=0.9,
        do_sample=True,
    )

    generated = tokenizer.decode(output_tokens[0], skip_special_tokens=True)
    print("\n🧠 Generated Text:\n")
    print(generated)
    return generated


# ==========================================================
#  💾 STEP 8 — Save Model
# ==========================================================
def save_model(model, tokenizer, path="./fine_tuned_model"):
    model.save_pretrained(path)
    tokenizer.save_pretrained(path)
    print(f"\n✅ Model saved to: {path}")


# ==========================================================
#  ▶️ MAIN EXECUTION PIPELINE
# ==========================================================
if __name__ == "__main__":

    # Load GPT-2 + tokenizer
    model, tokenizer = load_model_and_tokenizer()

    # Create and tokenize dataset
    dataset = create_dataset()
    tokenized_data = tokenize_dataset(dataset, tokenizer)

    # Training configuration
    training_args = configure_training()

    # Train model
    trainer = train_model(model, tokenizer, tokenized_data, training_args)

    # Generate output text
    generate_text(model, tokenizer)

    # Save model
    save_model(model, tokenizer)

    print("\n🎉 Training complete! You can now generate custom text!")

