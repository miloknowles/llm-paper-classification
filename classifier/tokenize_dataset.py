import sys; sys.path.append("..")

from transformers import AutoTokenizer
from datasets import load_dataset, DatasetDict

from classifier.paths import data_folder, models_folder


def load_pretraining_text_dataset(num_proc: int = 4) -> DatasetDict:
  """Loads the raw arXiv data and maps it to a dataset with a single `text` column.
  
  The title and abstract are joined with a "." as in the finetuning step.
  """
  raw = load_dataset("json", data_files={
    "train": str(data_folder / "pretraining" / "train.jsonl"),
    "val": str(data_folder / "pretraining" / "val.jsonl")
  }).select_columns(["title", "abstract"])

  raw = raw.map(
    lambda example: {"text": ". ".join([example["title"], example["abstract"]])},
    remove_columns=["title", "abstract"],
    num_proc=num_proc
  )

  return raw


def save_pretraining_tokenized_dataset(
  context_length: int = 512,
  tokenizer_path: str = models_folder / "tokenizers" / "distilbert-base-uncased-arxiv",
  num_proc: int = 4
):
  """Tokenizes the pretraining dataset and saves it to disk.
  
  This allows us to skip this time-consuming step at training time.

  Notes
  -----
  Tokenized text is chunked into batches of `context_length` tokens. This means
  that some examples will be split into multiple batches.

  Output
  ------
  The tokenized dataset is saved to `data_folder/pretraining/tokenized`, in
  Arrow format. You can read the data using `datasets.load_from_disk`.
  """
  tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, max_length=context_length)
  tokenizer.model_max_length = context_length

  raw_datasets = load_pretraining_text_dataset(num_proc=num_proc)
  print("LOADED")

  def tokenize(element):
    """Tokenize the `text` field of this example into chunks of `context_length` tokens."""
    outputs = tokenizer(
      element["text"],
      truncation=True,
      max_length=context_length,
      return_overflowing_tokens=True,
      return_length=True,
    )
    # Chunks that don't reach the `context_length` are discarded.
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
      if length == context_length:
        input_batch.append(input_ids)
    return {"input_ids": input_batch}

  tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,
    remove_columns=raw_datasets["train"].column_names,
    num_proc=num_proc
  )

  print(tokenized_datasets)
  tokenized_datasets.save_to_disk(data_folder / "pretraining" / "tokenized")
  print("DONE")


if __name__ == "__main__":
  save_pretraining_tokenized_dataset(
    context_length=512,
    tokenizer_path=models_folder / "tokenizers" / "distilbert-base-uncased-arxiv",
    num_proc=4
  )