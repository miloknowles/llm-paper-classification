import sys; sys.path.append("..")
import argparse

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


def save_pretraining_tokenized_dataset():
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
  parser = argparse.ArgumentParser()
  parser.add_argument("--dataset_name", type=str, default="arxiv", help="The name of the dataset to use.")
  parser.add_argument("--context_length", type=int, default=512, help="The maximum length of the context window. The dataset is chunked into this size.")
  parser.add_argument("--tokenizer_path", type=str, default="../models/tokenizers/distilbert-base-uncased-arxiv-32k", help="The path to the tokenizer to use.")
  parser.add_argument("--num_proc", type=int, default=4, help="The number of processes to use.")
  args = parser.parse_args()

  if args.dataset_name is None:
    raise ValueError("Must include a `dataset_name`!")

  tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path, model_max_length=args.context_length)
  print("Context window size:", tokenizer.model_max_length)

  raw_datasets = load_pretraining_text_dataset(num_proc=args.num_proc)
  print("LOADED")

  def tokenize(element):
    """Tokenize the `text` field of this example into chunks of `context_length` tokens.

    
    Notes
    -----
    It's very important that we both truncate and pad the input below. There are
    many arXiv papers that are shorter than the `context_length`, and we want to
    include them. There are also some papers that are longer, and we want to
    truncate those and include their pieces as well.

    The `element["text"]` field is a `list[str]` with a batch size of `1000` by
    default.
    """
    outputs = tokenizer(
      element["text"],
      truncation=True,
      padding="max_length",
      max_length=args.context_length,
      return_overflowing_tokens=True,
      return_length=True,
    )

    # Chunks that don't reach the `context_length` are discarded.
    input_batch = []
    for length, input_ids in zip(outputs["length"], outputs["input_ids"]):
      if length == args.context_length:
        input_batch.append(input_ids)

    return {"input_ids": input_batch}

  tokenized_datasets = raw_datasets.map(
    tokenize,
    batched=True,
    batch_size=8, # 
    remove_columns=raw_datasets["train"].column_names,
    num_proc=args.num_proc
  )
  print("TOKENIZED")

  print(tokenized_datasets)
  # tokenized_datasets.save_to_disk(data_folder / "pretraining" / "tokenized" / args.dataset_name)
  print("DONE")


if __name__ == "__main__":
  save_pretraining_tokenized_dataset()