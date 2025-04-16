# Simple PyTorch Transformer Chatbot

This repository contains a simple implementation of a transformer-based chatbot using PyTorch.

## Requirements

- Python 3.6+
- PyTorch 1.9.0+
- NumPy 1.19.5+

You can install the required packages using:

```bash
pip install -r requirements.txt
```

## Usage

To train and test the chatbot, simply run:

```bash
python train_transformer_chatbot.py
```

The script will:
1. Create a vocabulary from the sample conversations
2. Train a transformer model on these conversations
3. Save the trained model to `transformer_chatbot.pth`
4. Test the model with a few example inputs

## Model Architecture

The chatbot uses a standard transformer architecture with:
- Word embeddings
- Positional encoding
- Multi-head self-attention
- Feed-forward networks
- Layer normalization

## Customization

To customize the chatbot:
- Modify the `conversations` list to include your own training data
- Adjust hyperparameters in the `main()` function
- Expand the vocabulary and training data for better performance

## Advanced Usage

To use a trained model for inference:

```python
import torch
from train_transformer_chatbot import TransformerChatbot, generate_response

# Load the saved model
checkpoint = torch.load('transformer_chatbot.pth')
vocab = checkpoint['vocab']
idx2word = checkpoint['idx2word']
vocab_size = len(vocab)

# Initialize the model
model = TransformerChatbot(vocab_size)
model.load_state_dict(checkpoint['model_state_dict'])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)

# Generate responses
while True:
    user_input = input("You: ")
    if user_input.lower() == "quit":
        break
    response = generate_response(model, user_input, vocab, idx2word, device)
    print(f"Bot: {response}")
``` 