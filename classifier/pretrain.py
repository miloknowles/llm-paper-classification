import sys; sys.path.append("..")

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


def main():
  """Pretrain a masked language model on the arXiv dataset."""
  set_seed(42)

  run_name = "pretrain_debug"
  model_name = "distilbert-base-uncased"
  mlm_probability = 0.15 # same as the default used by BERT
  context_length = 512
  num_train_epochs = 1
  batch_size = 16
  tokenizer_path = models_folder / "tokenizers" / "distilbert-base-uncased-arxiv"

  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, model_max_length=context_length)

  dataset = load_from_disk(str(data_folder / "pretraining" / "tokenized"))
  print("Loaded dataset:")
  print(dataset)

  config = AutoConfig.from_pretrained(
    model_name,
    vocab_size=len(tokenizer),
  )
  model = AutoModelForMaskedLM.from_config(config)

  model_size = sum(t.numel() for t in model.parameters())
  print(f"Model size: {model_size/1000**2:.1f}M parameters")

  def preprocess_logits_for_metrics(logits, labels):
    if isinstance(logits, tuple):
      logits = logits[0]
    return logits.argmax(dim=-1)

  metric = evaluate.load("accuracy")

  def compute_metrics(eval_preds):
    """Takes a tuple of logits and labels and returns a dictionary of metrics.

    Note that this function is called *after* the logits have been preprocessed (argmax).
    """
    preds, labels = eval_preds
    labels = labels.reshape(-1)
    preds = preds.reshape(-1)
    mask = labels != -100
    labels = labels[mask]
    preds = preds[mask]
    return metric.compute(predictions=preds, references=labels)

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
    preprocess_logits_for_metrics=preprocess_logits_for_metrics
  )

  # trainer.train()


if __name__ == "__main__":
  main()