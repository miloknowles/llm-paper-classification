import sys; sys.path.append('..')
import argparse

from datasets import load_dataset
from classifier.paths import data_folder, models_folder

from tqdm import tqdm
from transformers import AutoTokenizer


def main():
  """Train a tokenizer on the pretraining dataset."""
  parser = argparse.ArgumentParser()
  parser.add_argument("--model_id", type=str, default="distilbert/distilbert-base-uncased",
                      help="The name of the model to use. This should be a Hugging Face model name.")
  parser.add_argument("--tokenizer_id", type=str, default="distilbert-base-uncased-arxiv-32k",
                      help="What to name the tokenizer. This will be saved to `models/tokenizers`.")
  parser.add_argument("--vocab_size", type=int, default=32_000,
                      help="The size of the vocabulary to build.")
  args = parser.parse_args()

  # Note that this will still be a `dict` with a "train" key.
  corpus = load_dataset(
    "json",
    data_files=str(data_folder / "pretraining" / "train.jsonl")
  ).select_columns(["title", "abstract"])

  # Similar to the finetuning dataset, join the title and abstract together with a "." in between.
  # You can create a new field and remove old ones in one step:
  # https://huggingface.co/docs/datasets/en/process#map
  corpus = corpus.map(
    lambda example: {"text": ". ".join([example["title"], example["abstract"]])},
    remove_columns=["title", "abstract"]
  )

  def batch_iterator(batch_size: int = 10000):
    """Iterate through the dataset in batches."""
    for i in tqdm(range(0, len(corpus["train"]), batch_size)):
      yield corpus["train"][i : i + batch_size]["text"]

  # Create a tokenizer from existing one to re-use special tokens.
  tokenizer = AutoTokenizer.from_pretrained(args.model_id, use_fast=True)

  # Train the tokenizer and save it.
  bert_tokenizer = tokenizer.train_new_from_iterator(text_iterator=batch_iterator(), vocab_size=32_000)
  bert_tokenizer.save_pretrained(models_folder / "tokenizers" / args.tokenizer_id)


if __name__ == "__main__":
  main()