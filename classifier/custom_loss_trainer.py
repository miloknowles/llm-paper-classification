from transformers import Trainer


class CustomLossTrainer(Trainer):
  """Subclasses `Trainer` to use a custom loss function.
  
  This allows us to use a weighted cross entropy loss to deal with class imbalance.
  """
  def __init__(
    self,
    compute_loss_fn: callable,
    *args,
    **kwargs
  ):
    super().__init__(*args, **kwargs)
    self.compute_loss_fn = compute_loss_fn

  def compute_loss(self, model, inputs, return_outputs=False):
    """Compute using the custom loss function."""
    labels = inputs.pop("labels")
    outputs = model(**inputs)
    logits = outputs.logits
    loss = self.compute_loss_fn(logits, labels)
    return (loss, outputs) if return_outputs else loss
