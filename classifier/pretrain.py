import sys; sys.path.append("..")
import argparse

import torch

from transformers import (
  AutoConfig,
  AutoModelForMaskedLM,
  AutoTokenizer,
  DataCollatorForLanguageModeling,
  set_seed,
)
from transformers import TrainingArguments, Trainer
from datasets import load_from_disk
import evaluate

from classifier.paths import data_folder, models_folder


metrics = dict(
  accuracy=evaluate.load("accuracy"),
  # ppl=evaluate.load("perplexity"),
)


def compute_metrics(eval_preds: tuple[torch.Tensor, torch.Tensor]):
  """Takes a tuple of logits and labels and returns a dictionary of metrics.

  Note that this function is called *after* the logits have been preprocessed (argmax).
  """
  preds, labels = eval_preds
  labels = labels.reshape(-1)
  preds = preds.reshape(-1)
  mask = labels != -100
  labels = labels[mask]
  preds = preds[mask]
  return {name: metric.compute(predictions=preds, references=labels) for name, metric in metrics.items()}


def preprocess_logits_for_metrics(logits, labels):
  if isinstance(logits, tuple):
    logits = logits[0]
  return logits.argmax(dim=-1)



def main():
  """Pretrain a masked language model on the arXiv dataset."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--data_folder", type=str, help="The folder where the pretraining data is stored.")
  parser.add_argument("--run_name", type=str, default="pretrain_debug", help="The name of the run.")
  parser.add_argument("--model_name", type=str, default="distilbert/distilbert-base-uncased", help="The name of the model to use. This should be a Hugging Face model name.")
  parser.add_argument("--tokenizer_path", type=str, default="distilbert-base-uncased-arxiv", help="The name of the tokenizer to use (if remote), otherwise a path to a local saved tokenizer.")
  parser.add_argument("--epochs", type=int, default=50, help="The number of epochs to train for.")
  parser.add_argument("--batch_size", type=int, default=32, help="The batch size to use.")
  parser.add_argument("--mlm_probability", type=float, default=0.15, help="The probability of masking tokens.")
  parser.add_argument("--context_length", type=int, default=512, help="The maximum length of the context.")
  parser.add_argument("--lr", type=float, default=2e-5, help="The learning rate to use.")
  parser.add_argument("--fast", action="store_true", help="Run the script in fast mode.")
  args = parser.parse_args()

  set_seed(42)

  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, model_max_length=args.context_length)
  dataset = load_from_disk(args.data_folder)

  if args.fast:
    dataset["train"] = dataset["train"].select(range(10))
    dataset["val"] = dataset["val"].select(range(10))

  print("Loaded dataset:")
  print(dataset)

  # https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertConfig
  config = AutoConfig.from_pretrained(
    args.model_name,
    vocab_size=len(tokenizer),
  )
  model = AutoModelForMaskedLM.from_config(config)

  model_size = sum(t.numel() for t in model.parameters())
  print(f"Model size: {model_size/1000**2:.1f}M parameters")

  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=args.mlm_probability,
  )

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
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
  )

  trainer.train()


if __name__ == "__main__":
  main()