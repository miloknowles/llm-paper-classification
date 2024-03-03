import sys; sys.path.append("..")
import argparse

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from classifier.paths import data_folder, models_folder

import numpy as np
import evaluate


metrics = dict(
  accuracy=evaluate.load("accuracy"),
  f1=evaluate.load("f1"),
  precision=evaluate.load("precision"),
  recall=evaluate.load("recall"),
)


def compute_metrics(pred_eval: tuple[torch.Tensor, torch.Tensor]) -> dict[str, float]:
  """Takes a tuple of logits and labels and returns a dictionary of metrics."""
  logits, labels = pred_eval
  predictions = np.argmax(logits, axis=-1)
  return {name: metric.compute(predictions=predictions, references=labels) for name, metric in metrics.items()}


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_name", type=str, default="finetune_debug", help="The name of the run.")
  parser.add_argument("--model_name_or_path", type=str, default="distilbert-base-uncased", help="The name of the model to use (if downloading from Hugging Face). Otherwise, a path to a locally trained model.")
  parser.add_argument("--tokenizer_name", type=str, default="distilbert-base-uncased-arxiv-32k", help="The name of the tokenizer to use. This should exist in the `models/tokenizers` folder.")
  parser.add_argument("--epochs", type=int, default=50, help="The number of epochs to train for.")
  parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use.")
  parser.add_argument("--context_length", type=int, default=512, help="The maximum length of the context.")
  parser.add_argument("--lr", type=float, default=2e-5, help="The learning rate to use.")
  parser.add_argument("--fast", action="store_true", help="Run the script in fast mode.")
  args = parser.parse_args()

  num_labels = 2

  id2label = {0: "False", 1: "True"}
  label2id = {"False": 0, "True": 1}

  model = AutoModelForSequenceClassification.from_pretrained(
    args.model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
  )
  # https://stackoverflow.com/questions/69842980/asking-to-truncate-to-max-length-but-no-maximum-length-is-provided-and-the-model
  tokenizer = AutoTokenizer.from_pretrained(args.model_name, model_max_length=args.context_length)
  print("Context window size:", tokenizer.model_max_length)

  # This dataset has columns: `text` and `label`.
  dataset = load_dataset("json", data_files={
    "train": str(data_folder / "finetuning" / "augmented_train.jsonl"),
    "val": str(data_folder / "finetuning" / "val.jsonl"),
  }).select_columns(["text", "label"])

  def convert_labels(examples: dict[str, list[int | str]]):
    """Convert the `label` field to a numeric value (it's "True" or "False" in the raw data)."""
    return {"label": [{"True": 1, "False": 0}[label] for label in examples["label"]]}

  def tokenize(examples: dict[str, list[int | str]]):
    """Tokenize the `text` field of all examples."""
    return tokenizer(examples["text"], truncation=True, padding="max_length")

  dataset = dataset.map(convert_labels, batched=True)
  dataset = dataset.map(tokenize, batched=True).shuffle(seed=42)
  print("Dataset:")
  print(dataset)

  training_args = TrainingArguments(
    output_dir=models_folder / args.run_name,
    evaluation_strategy="epoch",
    num_train_epochs=args.epochs,
    per_device_train_batch_size=args.batch_size,
    per_device_eval_batch_size=args.batch_size,
    save_strategy="epoch",
    learning_rate=args.lr,
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics,
    tokenizer=tokenizer,
  )

  trainer.train()


if __name__ == "__main__":
  main()