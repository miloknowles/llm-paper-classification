import json

from datasets import IterableDataset, IterableDatasetDict


def create_iterable_dataset(jsonl_path: str) -> IterableDataset:
  """Creates an iterable dataset from a path to a JSONL file.
  
  Notes
  -----
  This avoids reading the entire file into memory at once.
  
  The limitation of iterable datasets is that their length isn't known ahead of
  time. If you don't specify a `max-steps` argument in the `TrainingArguments`,
  you'll get an error about the learning rate scheduler.

  In the future, we may want to see if __len__ can be overridden to get the
  best of both worlds.
  ```
  """
  def _generator():
    with open(jsonl_path, 'r') as f:
      for line in f:
        yield json.loads(line)

  return IterableDataset.from_generator(_generator)


def load_dataset_splits(
  train_path: str, val_path: str, test_path: str
) -> IterableDatasetDict:
  """Creates a `DatasetDict` from paths to JSONL split files."""
  return IterableDatasetDict({
    "train": create_iterable_dataset(train_path),
    "val": create_iterable_dataset(val_path),
    "test": create_iterable_dataset(test_path),
  })

