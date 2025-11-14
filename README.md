# GPT-2 Text Generation — Fine-Tuning Project
Prodigy Infotech Internship Task
Author: Pranay Phepade

--------------------------------------------

## Project Overview
This project fine-tunes a GPT-2 language model using custom text data. It demonstrates how to:

- Load and configure GPT-2
- Create and tokenize a dataset
- Train the model using HuggingFace Trainer
- Generate text using the fine-tuned model
- Save the trained model for future use

--------------------------------------------

## Installation
Install all required dependencies:

pip install -U pip setuptools wheel
pip install tokenizers==0.13.3 --only-binary=:all:
pip install transformers==4.31.0 datasets torch

--------------------------------------------

## Model and Tokenizer Setup
Load GPT-2 and its tokenizer:

model, tokenizer = load_model_and_tokenizer("gpt2")

Padding tokens are configured automatically to avoid training errors.

--------------------------------------------

## Dataset
The dataset contains sample AI-related sentences:

text_samples = [
    "Artificial intelligence is changing the world of technology.",
    "Machine learning helps computers learn from experience.",
    "Natural language processing enables communication with computers.",
    "Data science combines math and coding to solve real problems.",
    "AI will make automation smarter and more efficient in the future."
]

The dataset is tokenized using:

tokenize_dataset(dataset, tokenizer)

--------------------------------------------

## Training Configuration
Training settings include:

TrainingArguments(
    output_dir="./results",
    overwrite_output_dir=True,
    num_train_epochs=2,
    per_device_train_batch_size=2,
    save_steps=500,
    save_total_limit=2,
    logging_steps=5
)

--------------------------------------------

## Training the Model
Training is done using HuggingFace Trainer:

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_data,
    data_collator=data_collator
)
trainer.train()

--------------------------------------------

## Text Generation
Generate text after training:

output = model.generate(
    **encoded,
    max_length=80,
    temperature=0.7,
    top_p=0.9,
    do_sample=True
)

--------------------------------------------

## Saving the Fine-Tuned Model
Save both the model and tokenizer:

model.save_pretrained("./fine_tuned_model")
tokenizer.save_pretrained("./fine_tuned_model")

--------------------------------------------

## Running the Complete Pipeline
The script automatically:
1. Loads GPT-2
2. Creates dataset
3. Tokenizes dataset
4. Configures training
5. Trains the model
6. Generates sample text
7. Saves the fine-tuned model

Run using:

python your_script_name.py

--------------------------------------------

## Project Structure

script.py
results/
fine_tuned_model/
README.md

--------------------------------------------

## Conclusion
This project provides a complete workflow for fine-tuning GPT-2 using the HuggingFace Transformers library. You can expand the dataset and tune hyperparameters to enhance performance.

