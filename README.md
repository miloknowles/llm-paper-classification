# Training an arXiv Paper Classifier using BERT-Tiny

Pre-training and finetuning language models to classify their relevance to a topic, built using the [HuggingFace library](https://huggingface.co/docs/transformers/en/index).

## Getting Started

To install dependencies, run:
```
pipenv install --dev
```

## Dataset

Download the arXiv dataset from [here](https://www.kaggle.com/datasets/Cornell-University/arxiv), the `mv` it to the `data` folder and `unzip archive.zip`.

I renamed the file format to `.jsonl` to be more precise.

The classification dataset can be found [here](https://drive.google.com/drive/u/1/folders/1re_PhEZzIxe8rAnO1cRcwVSAzNZv8teP).

## Pre-Training Pipeline

To replicate the results in this project, go through the following steps.

### Dataset Splits

Use the notebook `notebooks/pretrain_dataset.ipynb` to split the arXiv pretraining data into training, validation, and testing splits.

### Train the Tokenizer

Instead of using the original model's tokenizer out of the box, we train a new one. The motivation for this is to better capture the types of tokens in arXiv abstracts, which might be different from `wikipedia` and `bookcorpus`. For example, there could be special AI-related jargon in the abstracts that we want to be able to represent with the model.

Run `python train_tokenizer.py` to do this step.

### Tokenize the Dataset

To save time during training, we pre-process the training and validation splits into a tokenized form, and save them to disk.

Run `python tokenize_dataset.py` to do this step.

### Pre-Train a BERT Model

Run `python pretrain.py` to train the model from scratch. Weights will be saved at each epoch, and evalution metrics will be printed.

## Finetuning Pipeline

### Finetune from a Pre-Trained Model

Finetune a model by specifying either a name (from HuggingFace) or a path to a local pre-trained model. See the script for all arguments.
```python
# This uses the pre-trained model found on HuggingFace. You can replace the --model and --tokenizer with a path to the local model.
python finetune.py --run_name finetune_debug --model prajjwal1/bert-tiny --tokenizer prajjwal1/bert-tiny --epochs 10
```

### Evaluate

Run the `finetune.py` script in `--validate` mode:
```python
python finetune.py --validate --model ../models/bert-tiny-test/checkpoint-944/ --tokenizer ../models/bert-tiny-test/checkpoint-944
```

This will quickly run the model on the validation set and print out evaluation metrics.

### Test Set Inference

Finally, generate predictions on the held out test set.

Run the `finetune.py` script in `--test` mode:
```python
python finetune.py --test --model ../models/bert-tiny-test/checkpoint-944/ --tokenizer ../models/bert-tiny-test/checkpoint-944
```

This will save to `output/test_predictions.jsonl`.