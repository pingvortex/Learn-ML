# Install required packages
# REMOVE if you are not using Google Colab and paste it in the terminal (without the exclamation mark)
!pip install transformers datasets matplotlib torch 'accelerate>=0.26.0'
# Import required libraries
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

# ====================== IMPORTANT CONFIGURATION ======================
MODEL_NAME = "distilgpt2"          # Model to fine-tune
DATASET_NAME = "stas/openwebtext-10k"    # Dataset to use
OUTPUT_DIR = "./model_finetuned"           # Where to save checkpoints
FINAL_MODEL_DIR = "./final_model"        # Final model save path
NUM_EPOCHS = 3                           # Max training epochs
BATCH_SIZE = 8                           # Adjust based on GPU memory
LEARNING_RATE = 2e-5                     # Learning rate
EARLY_STOPPING_PATIENCE = 2              # Stop after 2 eval loss increases
BLOCK_SIZE = 512                         # Context length
# =====================================================================

# Load dataset
dataset = load_dataset(DATASET_NAME)

# Load tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

# Add padding token if missing (required for batch training)
if tokenizer.pad_token is None:
    tokenizer.pad_token = tokenizer.eos_token

# Tokenization function
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=BLOCK_SIZE,
        padding="max_length",
        return_attention_mask=True
    )

# Process dataset
tokenized_dataset = dataset.map(
    tokenize_function,
    batched=True,
    remove_columns=dataset["train"].column_names
)

# Split into train/validation (10% validation)
split_dataset = tokenized_dataset["train"].train_test_split(test_size=0.1)
train_dataset = split_dataset["train"]
eval_dataset = split_dataset["test"]

# Data collator for language modeling
data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False  # Causal language modeling
)

# Training arguments with anti-overfitting measures
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    overwrite_output_dir=True,
    num_train_epochs=NUM_EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    learning_rate=LEARNING_RATE,
    weight_decay=0.01,                   # L2 regularization
    eval_strategy="epoch",               # Match evaluation with save strategy
    save_strategy="epoch",               # Save checkpoints each epoch
    load_best_model_at_end=True,         # For early stopping
    metric_for_best_model="eval_loss",
    greater_is_better=False,             # Lower eval_loss is better
    logging_steps=50,                    # Log metrics every 50 steps
    report_to="none",                    # Disable external logging
    fp16=True,                           # Use mixed precision if available
    gradient_accumulation_steps=2,       # Effective batch size = BATCH_SIZE * 2
)

# Initialize Trainer with early stopping
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
    eval_dataset=eval_dataset,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=EARLY_STOPPING_PATIENCE)]
)

# Start training with graceful interruption handling
try:
    print("Starting training...")
    train_result = trainer.train()
except KeyboardInterrupt:
    print("\nTraining interrupted. Saving current progress...")

# Ensure model and tokenizer are saved in both cases
trainer.save_model(FINAL_MODEL_DIR)
tokenizer.save_pretrained(FINAL_MODEL_DIR)
print(f"Model and tokenizer saved to {FINAL_MODEL_DIR}")

# Plot training metrics
history = trainer.state.log_history

# Extract training and validation loss
train_loss = [entry["loss"] for entry in history if "loss" in entry]
eval_loss = [entry["eval_loss"] for entry in history if "eval_loss" in entry]

# Create plot
plt.figure(figsize=(10, 6))
plt.plot(train_loss, label="Training Loss", marker="o")
plt.plot(eval_loss, label="Validation Loss", marker="o")
plt.title("Training Progress")
plt.xlabel("Epochs" if training_args.eval_strategy == "epoch" else "Steps")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

# Save and show plot
plt.savefig(os.path.join(FINAL_MODEL_DIR, "training_plot.png"))
print("Training plot saved to:", os.path.join(FINAL_MODEL_DIR, "training_plot.png"))
plt.show()

# Print final metrics
print("\nTraining completed with metrics:")
print(f"Final Training Loss: {train_loss[-1]:.4f}")
print(f"Final Validation Loss: {eval_loss[-1]:.4f}")
