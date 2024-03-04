import sys; sys.path.append("..")
import argparse
import json

import torch

from transformers import AutoModelForSequenceClassification, AutoTokenizer
from transformers import TrainingArguments, Trainer
from datasets import load_dataset

from classifier.paths import data_folder, models_folder, output_folder

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
  predictions = logits.argmax(axis=-1)
  return {name: metric.compute(predictions=predictions, references=labels) for name, metric in metrics.items()}


def save_predictions(dataset, trainer, id2label):
  """Predict labels for the test set and save them to disk."""
  logits = trainer.predict(dataset.get("test")).predictions
  predictions = logits.argmax(axis=-1)

  # Count the number of true and false predictions as a sanity check.
  predicted_true = predictions.sum()
  predicted_false = len(predictions) - predicted_true
  print(f"Predicted True: {predicted_true} ({100 * predicted_true / len(predictions):.2f}%), Predicted False: {predicted_false} ({100 * predicted_false / len(predictions):.2f}%)")

  # Associate the predictions with the original examples.
  testset = dataset.map(lambda _, idx: {"label": id2label[predictions[idx]]}, with_indices=True)
  testset = testset.select_columns(["text", "meta", "label"])

  # Write out the test set with labels included.
  with open(output_folder / "test_predictions.jsonl", "w") as f:
    for example in testset["test"]:
      f.write(json.dumps(example) + "\n")


def main():
  parser = argparse.ArgumentParser()
  parser.add_argument("--run_name", type=str, default="finetune_debug", help="The name of the run.")
  parser.add_argument("--model", type=str, default="distilbert-base-uncased",
                      help="The name of the model to use (if downloading from Hugging Face). Otherwise, a path to a locally trained model.")
  parser.add_argument("--tokenizer", type=str, default="distilbert-base-uncased-arxiv-32k",
                      help="The name of a tokenizer to use (if downloading from Hugging Face). Otherwise, a path to a locally trained tokenizer.")
  parser.add_argument("--epochs", type=int, default=50, help="The number of epochs to train for.")
  parser.add_argument("--batch_size", type=int, default=16, help="The batch size to use.")
  parser.add_argument("--context_length", type=int, default=512, help="The maximum length of the context.")
  parser.add_argument("--lr", type=float, default=2e-5, help="The learning rate to use.")
  parser.add_argument("--fast", action="store_true", help="Run the script in fast mode.")
  parser.add_argument("--validate", action="store_true", help="Run the script in validation mode.")
  parser.add_argument("--test", action="store_true", help="Run the script in test mode.")
  args = parser.parse_args()

  num_labels = 2

  id2label = {0: "False", 1: "True"}
  label2id = {"False": 0, "True": 1}

  model = AutoModelForSequenceClassification.from_pretrained(
    args.model,
    num_labels=num_labels,
    id2label=id2label,
    label2id=label2id,
  )

  # https://stackoverflow.com/questions/69842980/asking-to-truncate-to-max-length-but-no-maximum-length-is-provided-and-the-model
  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer, model_max_length=args.context_length)
  print("Context window size:", tokenizer.model_max_length)

  def convert_labels(examples: dict[str, list[int | str]]):
    """Convert the `label` field to a numeric value (it's "True" or "False" in the raw data)."""
    return {"label": [{"True": 1, "False": 0}[label] for label in examples["label"]]}

  def tokenize(examples: dict[str, list[int | str]]):
    """Tokenize the `text` field of all examples."""
    return tokenizer(examples["text"], truncation=True, padding="max_length")
  
  # This dataset has columns: `text` and `label`.
  data_files = {}
  if args.validate:
    data_files["val"] = str(data_folder / "finetuning" / "val.jsonl")
  elif args.test:
    data_files["test"] = str(data_folder / "finetuning" / "test.jsonl")
  else:
    data_files = {
      "train": str(data_folder / "finetuning" / "augmented_train.jsonl"),
      "val": str(data_folder / "finetuning" / "val.jsonl"),
    }

  dataset = load_dataset("json", data_files=data_files) # .select_columns(["text", "label"] if not args.test else ["text"])

  # At test time, there are no labels to convert, so this would throw an error.
  if not args.test:
    dataset = dataset.map(convert_labels, batched=True)

  dataset = dataset.map(tokenize, batched=True)

  # Keep test predictions in the same order as the inputs.
  if not args.test:
    dataset = dataset.shuffle(seed=42)

  print("Loaded dataset:")
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
    train_dataset=dataset.get("train"),
    eval_dataset=dataset.get("val"),
    compute_metrics=compute_metrics if not args.test else None,
    tokenizer=tokenizer,
  )

  if args.validate:
    metrics = trainer.evaluate()
    print(json.dumps(metrics, indent=2))

  elif args.test:
    save_predictions(dataset, trainer, id2label)

  else:
    trainer.train()


if __name__ == "__main__":
  main()