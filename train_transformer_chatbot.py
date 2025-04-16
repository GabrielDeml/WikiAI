import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import numpy as np
import random
import math
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Device selection: MPS (Apple Silicon GPU) > CUDA > CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Tokenizer
TOKENIZER_NAME = "bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
MAX_LEN = 20

# Wikipedia streaming and tokenization
def pairwise_sentences(article):
    import re
    sentence_splitter = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_splitter.split(article["text"].replace("\n", " "))
    for i in range(len(sentences) - 1):
        yield sentences[i], sentences[i+1]

def gen_tokenized_pairs(max_pairs=5000):
    wiki = load_dataset("wikipedia", "20220301.en", split="train", streaming=True, trust_remote_code=True)
    count = 0
    for article in tqdm(wiki, total=max_pairs//2, desc="Streaming Wikipedia"):
        for inp, resp in pairwise_sentences(article):
            if len(inp.split()) > 2 and len(resp.split()) > 2:
                input_enc = tokenizer(inp, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
                resp_enc = tokenizer(resp, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
                yield {
                    "input": input_enc["input_ids"].squeeze(0),
                    "response_input": torch.cat([
                        torch.tensor([tokenizer.cls_token_id]),
                        resp_enc["input_ids"].squeeze(0)[:-1]
                    ]),
                    "response_target": resp_enc["input_ids"].squeeze(0)
                }
                count += 1
                if count >= max_pairs:
                    return

# Custom Dataset
class WikiChatDataset(Dataset):
    def __init__(self, max_pairs=5000):
        self.data = list(gen_tokenized_pairs(max_pairs))
    def __len__(self):
        return len(self.data)
    def __getitem__(self, idx):
        return self.data[idx]

# Positional Encoding
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

# Transformer Chatbot Model
class TransformerChatbot(nn.Module):
    def __init__(self, vocab_size, d_model=64, nhead=4, num_encoder_layers=2, num_decoder_layers=2, dim_feedforward=128):
        super(TransformerChatbot, self).__init__()
        
        # Embeddings
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model)
        
        # Transformer
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True
        )
        
        # Output layer
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        # Embedding and positional encoding
        src_embedded = self.positional_encoding(self.embedding(src))
        tgt_embedded = self.positional_encoding(self.embedding(tgt))
        
        # Create masks for padding tokens
        src_key_padding_mask = (src == tokenizer.pad_token_id)
        tgt_key_padding_mask = (tgt == tokenizer.pad_token_id)
        
        # Generate subsequent mask for target
        tgt_len = tgt.size(1)
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Transformer forward
        output = self.transformer(
            src_embedded, 
            tgt_embedded, 
            src_mask=src_mask, 
            tgt_mask=subsequent_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Pass through output layer
        return self.output_layer(output)
    
    def generate_square_subsequent_mask(self, sz):
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# Training function with mixed precision for MPS
def train(model, dataloader, optimizer, criterion, device, epochs=10):
    model.train()
    for epoch in range(epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            src = batch["input"].to(device)
            tgt_input = batch["response_input"].to(device)
            tgt_output = batch["response_target"].to(device)
            optimizer.zero_grad()
            # Only use autocast for CUDA devices, not for MPS
            if device.type == "cuda":
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(src, tgt_input)
                    output_flat = output.contiguous().view(-1, output.size(-1))
                    tgt_output_flat = tgt_output.contiguous().view(-1)
                    loss = criterion(output_flat, tgt_output_flat)
            else:
                output = model(src, tgt_input)
                output_flat = output.contiguous().view(-1, output.size(-1))
                tgt_output_flat = tgt_output.contiguous().view(-1)
                loss = criterion(output_flat, tgt_output_flat)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
    return model

# Generate response function (tokenizer-based)
def generate_response(model, input_text, tokenizer, device, max_length=20):
    model.eval()
    input_enc = tokenizer(input_text, truncation=True, padding="max_length", max_length=max_length, return_tensors="pt")
    input_tensor = input_enc["input_ids"].to(device)
    decoder_input = torch.tensor([[tokenizer.cls_token_id]], dtype=torch.long).to(device)
    output_ids = []
    for _ in range(max_length):
        with torch.no_grad():
            output = model(input_tensor, decoder_input)
        next_token_logits = output[:, -1, :]
        next_token_id = next_token_logits.argmax(-1).item()
        if next_token_id == tokenizer.sep_token_id or next_token_id == tokenizer.pad_token_id:
            break
        output_ids.append(next_token_id)
        decoder_input = torch.cat([decoder_input, torch.tensor([[next_token_id]], dtype=torch.long).to(device)], dim=1)
    return tokenizer.decode(output_ids, skip_special_tokens=True)

# Main training procedure
def main():
    batch_size = 4
    lr = 0.001
    max_pairs = int(os.environ.get("WIKI_PAIRS", 5000))
    dataset = WikiChatDataset(max_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    model = TransformerChatbot(tokenizer.vocab_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id)
    print("Training model...")
    model = train(model, dataloader, optimizer, criterion, device)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
        'tokenizer': tokenizer.name_or_path
    }, 'transformer_chatbot.pth')
    print("Model saved to transformer_chatbot.pth")
    print("\nTesting the model:")
    test_inputs = ["What is artificial intelligence?", "Tell me about the moon.", "Who was Albert Einstein?", "Explain quantum mechanics."]
    for test_input in test_inputs:
        response = generate_response(model, test_input, tokenizer, device)
        print(f"Input: {test_input}")
        print(f"Response: {response}")
        print()

if __name__ == "__main__":
    main() 