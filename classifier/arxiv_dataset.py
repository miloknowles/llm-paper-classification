import json

from torch.utils.data import IterableDataset as _BaseIterableDataset
from datasets import IterableDataset, IterableDatasetDict, Dataset



def create_dataset(jsonl_path: str) -> IterableDataset:
  """Creates an iterable dataset from a path to a JSONL file.
  
  This avoids reading the entire file into memory at once.
  """
  def _generator():
    with open(jsonl_path, 'r') as f:
      for line in f:
        yield json.loads(line)
  
  # Get the features from the first example.
  # TODO(milo): Should we infer the features here?
  # features = list(next(_generator()).keys())

  return IterableDataset.from_generator(_generator)


# class ArxivDataset(_BaseIterableDataset):
#   def __init__(
#     self,
#     jsonl_path: str
#   ):
#     super(_BaseIterableDataset, self).__init__()
#     self._path = jsonl_path

#     with open(self._path, 'r') as f:
#       for count, line in enumerate(f):
#         pass

#     self._length = count + 1

#   def __iter__(self):
#     with open(self._path, 'r') as f:
#       for line in f:
#         yield json.loads(line)

#   def __len__(self) -> int:
#     return self._length
  

def create_dataset_dict(
  train_path: str, val_path: str, test_path: str
) -> IterableDatasetDict:
  """Creates a `DatasetDict` from paths to JSONL split files."""
  return IterableDatasetDict({
    "train": create_dataset(train_path),
    "val": create_dataset(val_path),
    "test": create_dataset(test_path),
    # "train": ArxivDataset(train_path),
    # "val": ArxivDataset(val_path),
    # "test": ArxivDataset(test_path),
  })

