import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from transformers import AutoTokenizer
import glob
import os

# Tokenizer and constants
TOKENIZER_NAME = "microsoft/codebert-base"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
MAX_LEN = 20

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, max_seq_length=100):
        super(PositionalEncoding, self).__init__()
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0))
    def forward(self, x):
        return x + self.pe[:, :x.size(1)]

class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=8, num_decoder_layers=8, dim_feedforward=8192, dropout=0.1):
        super(TransformerChatbot, self).__init__()
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout
        )
        self.output_layer = nn.Linear(d_model, vocab_size)
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        src_embedded = self.dropout(self.positional_encoding(self.embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.embedding(tgt)))
        src_key_padding_mask = (src == tokenizer.pad_token_id)
        tgt_key_padding_mask = (tgt == tokenizer.pad_token_id)
        tgt_len = tgt.size(1)
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        output = self.transformer(
            src_embedded, 
            tgt_embedded, 
            src_mask=src_mask, 
            tgt_mask=subsequent_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        return self.output_layer(output)
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

def generate_response(model, input_text, tokenizer, device, max_length=20, temperature=1.0, top_k=50):
    model.eval()
    input_enc = tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    input_tensor = input_enc["input_ids"].to(device)
    decoder_input = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).to(device)
    output_ids = []
    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_tensor, decoder_input)
        next_token_logits = output[:, -1, :]
        logits = next_token_logits / temperature
        topk_vals, topk_indices = torch.topk(logits, top_k)
        filter_mask = logits < topk_vals.min()
        logits[filter_mask] = -float('Inf')
        probs = F.softmax(logits, dim=-1)
        next_token_id = torch.multinomial(probs, 1).item()
        if next_token_id == tokenizer.sep_token_id or next_token_id == tokenizer.pad_token_id:
            break
        output_ids.append(next_token_id)
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)
    return tokenizer.decode(output_ids, skip_special_tokens=True)

def find_latest_checkpoint():
    """
    Finds the latest model checkpoint in the models directory.
    Returns the path to the most recently modified .pth file, or None if none found.
    """
    checkpoint_files = glob.glob('models/*.pth')
    if checkpoint_files:
        return max(checkpoint_files, key=os.path.getctime)
    return None

def main():
    device = get_device()
    ckpt_path = find_latest_checkpoint()
    if not ckpt_path:
        print("No checkpoint found. Please train the model first.")
        return
    print(f"Loading checkpoint: {ckpt_path}")
    checkpoint = torch.load(ckpt_path, map_location=device)
    model = TransformerChatbot(tokenizer.vocab_size)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    print("\nWelcome to the Transformer Chatbot! Type 'quit' or 'exit' to stop.\n")
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in {"quit", "exit"}:
            print("Goodbye!")
            break
        response = generate_response(model, user_input, tokenizer, device, temperature=1.0, top_k=50)
        print(f"Bot: {response}")

if __name__ == "__main__":
    main() 