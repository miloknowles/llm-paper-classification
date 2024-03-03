import sys; sys.path.append("..")

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

from classifier.paths import data_folder, models_folder

import evaluate


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


def main(fast_mode: bool = False):
  """Pretrain a masked language model on the arXiv dataset."""
  set_seed(42)

  run_name = "pretrain_debug"
  model_name = "distilbert-base-uncased"
  mlm_probability = 0.15 # same as the default used by BERT
  context_length = 512
  num_train_epochs = 50

  # NOTE(milo): I'm able to use a batch size of 32 on an L4 GPU, but only 16 locally.
  batch_size = 32
  tokenizer_path = models_folder / "tokenizers" / "distilbert-base-uncased-arxiv"

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=context_length)

  dataset = load_from_disk(str(data_folder / "pretraining" / "tokenized"))

  if fast_mode:
    dataset["train"] = dataset["train"].select(range(10))
    dataset["val"] = dataset["val"].select(range(10))

  print("Loaded dataset:")
  print(dataset)

  # https://huggingface.co/docs/transformers/model_doc/distilbert#transformers.DistilBertConfig
  config = AutoConfig.from_pretrained(
    model_name,
    vocab_size=len(tokenizer),
  )
  model = AutoModelForMaskedLM.from_config(config)

  model_size = sum(t.numel() for t in model.parameters())
  print(f"Model size: {model_size/1000**2:.1f}M parameters")

  data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=True,
    mlm_probability=mlm_probability,
  )

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
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
    preprocess_logits_for_metrics=preprocess_logits_for_metrics,
  )

  trainer.train()


if __name__ == "__main__":
  main()