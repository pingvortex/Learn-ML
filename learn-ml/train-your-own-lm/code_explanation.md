# Explanation of the Code

This section provides a detailed, line-by-line breakdown of the code for fine-tuning a language model, connecting to concepts from [Large Language Models (LLMs)](./large_language_models.md) and tools from [Libraries Used in the Code](libraries.md). It’s designed to help you understand the code and prepare for [Training Your Own Model](training_your_model.md).

## Overview

The code fine-tunes a pre-trained language model (`PingVortex/VLM-1`) on a text dataset (`stas/openwebtext-10k`) to improve its performance on a specific task, such as text generation. It includes anti-overfitting measures and visualizes training progress.

## Code Breakdown

### 1. Install Required Packages

```python
!pip install transformers datasets matplotlib torch
```

- Installs the libraries discussed in [Libraries Used in the Code](libraries.md).
- The `!` prefix runs the command in the shell (common in Jupyter notebooks).

### 2. Import Libraries

```python
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling,
    EarlyStoppingCallback
)
from datasets import load_dataset
import matplotlib.pyplot as plt
import os
```

- Imports specific classes from `transformers` for tokenization, modeling, and training.
- Imports `datasets` for data handling, `matplotlib` for plotting, and `os` for file operations.

### 3. Configuration

```python
MODEL_NAME = "PingVortex/VLM-1"
DATASET_NAME = "stas/openwebtext-10k"
OUTPUT_DIR = "./vlm_finetuned"
FINAL_MODEL_DIR = "./final_model"
NUM_EPOCHS = 3
BATCH_SIZE = 8
LEARNING_RATE = 2e-5
EARLY_STOPPING_PATIENCE = 2
BLOCK_SIZE = 512
```

- Defines key parameters:
  - **Model and Dataset**: Specify the pre-trained model and training data.
  - **Directories**: Where to save checkpoints and the final model.
  - **Hyperparameters**: Control training duration, batch size, learning rate, and context length.
  - **Early Stopping**: Stops training if validation loss doesn’t improve for 2 epochs.

### 4. Load Dataset

```python
dataset = load_dataset(DATASET_NAME)
```

- Uses `datasets` to load `stas/openwebtext-10k`, a text dataset for training.

### 5. Load Tokenizer and Model

```python
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token
```

- Loads the tokenizer and model from Hugging Face’s model hub.
- Sets the padding token to the end-of-sequence token if missing, ensuring proper batch training.

### 6. Tokenization

```python
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length",
        return_attention_mask=True
    )

tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)
```

- Defines a function to tokenize text, truncating or padding to `BLOCK_SIZE` (512 tokens).
- Applies tokenization to the dataset in batches, removing original text columns to save memory.

### 7. Data Splitting

```python
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]
```

- Splits the tokenized dataset into 90% training and 10% validation sets to monitor performance and prevent overfitting.

### 8. Data Collator

```python
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False
)
```

- Sets up a data collator for causal language modeling (predicting the next token).
- `mlm=False` indicates it’s not masked language modeling (like BERT).

### 9. Training Arguments

```python
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,
    eval_strategy="epoch",
    save_strategy="epoch",
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    logging_steps=50,
    report_to="none",
    fp16=True,
    gradient_accumulation_steps=2,
)
```

- Configures the training process with:
  - **Anti-overfitting**: Weight decay (L2 regularization), early stopping, and validation checks.
  - **Efficiency**: Mixed precision (`fp16`) and gradient accumulation (effective batch size = `BATCH_SIZE * 2`).
  - **Monitoring**: Logs metrics every 50 steps and evaluates/saves each epoch.
  - **Best Model**: Loads the model with the lowest validation loss.

### 10. Trainer Initialization

```python
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
)
```

- Initializes the `Trainer` with the model, arguments, datasets, and early stopping callback.

### 11. Training

```python
try:
    print("Starting training...")
    train_result = trainer.train()
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current progress...")
```

- Starts training, with a try-catch block to handle interruptions (e.g., Ctrl+C) gracefully.

### 12. Save Model

```python
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"Model and tokenizer saved to {FINAL_MODEL_DIR}")
```

- Saves the final model and tokenizer to `FINAL_MODEL_DIR` for later use.

### 13. Plot Training Metrics

```python
history = trainer.state.log_history

train_loss = [entry["loss"] for entry in history if "loss" in entry]
eval_loss = [entry["eval_loss"] for entry in history if "eval_loss" in entry]

plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Training Loss", marker="o")
plt.plot(eval_loss, label="Validation Loss", marker="o")
plt.title("Training Progress")
plt.xlabel("Epochs")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.savefig(os.path.join(FINAL_MODEL_DIR, "training_plot.png"))
print("Training plot saved to:", os.path.join(FINAL_MODEL_DIR, "training_plot.png"))
```

- Extracts training and validation loss from the training history.
- Uses `matplotlib` to create a plot, saving it to `FINAL_MODEL_DIR`.

### 14. Print Final Metrics

```python
print("\nTraining completed with metrics:")
print(f"Final Training Loss: {train_loss[-1]:.4f}")
print(f"Final Validation Loss: {eval_loss[-1]:.4f}")
```

- Displays the final loss values for reference.

## Key Features

- **Fine-tuning**: Adapts a pre-trained model to a specific dataset, as discussed in [Large Language Models (LLMs)](./large_language_models.md).
- **Anti-overfitting**: Uses validation, early stopping, and weight decay.
- **Efficiency**: Employs mixed precision and gradient accumulation to manage GPU memory.
- **Monitoring**: Visualizes loss curves to assess training progress.

## Why This Code?

It provides a robust, beginner-friendly framework for fine-tuning, balancing performance and resource constraints. It’s a practical application of concepts from [How Neural Networks Work](./neural_networks.md) and [Libraries Used in the Code](libraries.md).

## Next Steps

Try running the code yourself with [Training Your Own Model](training_your_model.md), or revisit [Large Language Models (LLMs)](./large_language_models.md) for more context.

[Back to README](./README.md)
