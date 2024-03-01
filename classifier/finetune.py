import sys; sys.path.append("..")
import logging

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments
from transformers import TrainingArguments, Trainer

from datasets import load_dataset, DatasetDict
from classifier.arxiv_dataset import create_dataset_dict
from classifier.paths import data_folder, models_folder

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
  print(logits.shape, labels.shape)
  print(logits)
  print(labels)
  predictions = np.argmax(logits, axis=-1)
  return {name: metric.compute(predictions=predictions, references=labels) for name, metric in metrics.items()}


def get_max_steps(train_path: str) -> int:
  """Get the maximum number of training steps.
  
  This is required for the `TrainingArguments` object, since we're using an
  iterable dataset that is backed by a generator.
  """
  with open(train_path, 'r') as f:
    for count, _ in enumerate(f):
      pass
  return count


def main():
  num_labels = 2
  model_name = 'albert-base-v2'
  run_name = "debugging"

  model = AutoModelForSequenceClassification.from_pretrained(
    model_name, num_labels=num_labels
  )
  tokenizer = AutoTokenizer.from_pretrained(model_name)

  dataset = create_dataset_dict(
    data_folder / "finetuning" / "train.jsonl",
    data_folder / "finetuning" / "val.jsonl",
    data_folder / "finetuning" / "test.jsonl",
  )

  def convert_labels(examples: dict[str, list[int | str]]):
    """Convert the `label` field to a numeric value (it's "True" or "False" in the raw data)."""
    return {"label": [{"True": 1, "False": 0}[label] for label in examples["label"]]}

  def tokenize(examples: dict[str, list[int | str]]):
    """Tokenize the `text` field of all examples."""
    return tokenizer(examples["text"], padding="max_length", truncation=True)

  dataset = dataset.map(convert_labels, batched=True)
  dataset = dataset.map(tokenize, batched=True)

  training_args = TrainingArguments(
    output_dir=models_folder / run_name,
    evaluation_strategy="epoch",
    num_train_epochs=10,
    save_strategy="epoch",
    max_steps=get_max_steps(data_folder / "finetuning" / "train.jsonl"),
  )

  print(training_args)

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics,
  )

  trainer.train()


if __name__ == "__main__":
  main()