import sys; sys.path.append("..")
import logging

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from classifier.paths import data_folder, models_folder
from classifier.train_utils import get_best_system_device

import numpy as np
import evaluate


logger = logging.getLogger(__name__)

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
  num_labels = 2
  num_train_epochs = 50
  batch_size = 16

  model_name = "distilbert/distilbert-base-uncased"
  # model_name = "bert-base-uncased"
  run_name = "debugging"
  device = get_best_system_device()

  id2label = {0: "False", 1: "True"}
  label2id = {"False": 0, "True": 1}

  model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
  )
  # https://stackoverflow.com/questions/69842980/asking-to-truncate-to-max-length-but-no-maximum-length-is-provided-and-the-model
  tokenizer = AutoTokenizer.from_pretrained(model_name, model_max_length=512)

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

  training_args = TrainingArguments(
    output_dir=models_folder / run_name,
    evaluation_strategy="epoch",
    num_train_epochs=num_train_epochs,
    per_device_train_batch_size=batch_size,
    per_device_eval_batch_size=batch_size,
    save_strategy="epoch",
    # learning_rate=1e-4,
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