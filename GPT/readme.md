### GPT-2 Implementation

This is an implementation of GPT-2. The implementation is heavily inspired by the GPT-2 implementation by Andrej Karpathy and Chapter 5 of the book *"Build a Large Language Model"* by Sebastian Raschka.

In this implementation, I did not apply the original weight initialization from the GPT-2 paper, as it led to worse results during training. I also chose not to apply weight tying, which resulted in a model with approximately 164M parameters instead of the original 124M.

#### To-do:
- Add a validation set.
- Train the model on a larger dataset.
- Include more information on how the model processes the data (Mechanistic Interpretability).
