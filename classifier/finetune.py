import sys; sys.path.append("..")
import logging

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer, DataCollatorWithPadding
from transformers import TrainingArguments, Trainer

from datasets import load_dataset, DatasetDict
from classifier.arxiv_dataset import load_dataset_splits
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
  predictions = np.argmax(logits, axis=-1)
  return {name: metric.compute(predictions=predictions, references=labels) for name, metric in metrics.items()}


def get_max_steps(train_path: str, num_train_epochs: int, batch_size: int) -> int:
  """Get the maximum number of training steps.
  
  This is required for the `TrainingArguments` object, since we're using an
  iterable dataset that is backed by a generator with unknown length.
  """
  with open(train_path, 'r') as f:
    for n_examples, _ in enumerate(f):
      pass
  return (n_examples + 1) * num_train_epochs // batch_size


def main():
  num_labels = 2
  num_train_epochs = 10
  batch_size = 16

  # model_name = "distilbert/distilbert-base-uncased"
  model_name = "google-bert/bert-base-uncased"

  run_name = "debugging"

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

  dataset = load_dataset_splits(
    data_folder / "finetuning" / "train.jsonl",
    data_folder / "finetuning" / "val.jsonl",
    data_folder / "finetuning" / "test.jsonl",
  )

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
    max_steps=get_max_steps(data_folder / "finetuning" / "train.jsonl", num_train_epochs, batch_size),
  )

  trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=dataset["train"],
    eval_dataset=dataset["val"],
    compute_metrics=compute_metrics,
    # data_collator=data_collator,
    tokenizer=tokenizer,
  )

  trainer.train()


if __name__ == "__main__":
  main()