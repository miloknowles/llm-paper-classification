import pathlib


def top_folder():
  current_path = pathlib.Path(__file__).resolve()
  repository_path = current_path.parents[1]
  return repository_path


data_folder = top_folder() / "data"
models_folder = top_folder() / "models"
output_folder = top_folder() / "output"