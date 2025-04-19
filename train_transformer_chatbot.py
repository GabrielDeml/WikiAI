import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import math
import os
from tqdm import tqdm
from datasets import load_dataset
from transformers import AutoTokenizer
from transformers import AdamW, get_linear_schedule_with_warmup
import glob

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Device selection: Prefer MPS (Apple Silicon GPU), then CUDA, then CPU
if torch.backends.mps.is_available():
    device = torch.device("mps")
elif torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")
print(f"Using device: {device}")

# Tokenizer setup
TOKENIZER_NAME = "bert-base-uncased"  # Pretrained BERT tokenizer
# Load the tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
MAX_LEN = 128  # Maximum sequence length for input/output

# --- Wikipedia streaming and tokenization utilities ---

def pairwise_sentences(article):
    """
    Splits the article text into sentences and yields consecutive sentence pairs.
    Used to create (input, response) pairs for chatbot training.
    """
    import re
    sentence_splitter = re.compile(r'(?<=[.!?]) +')
    sentences = sentence_splitter.split(article["text"].replace("\n", " "))
    for i in range(len(sentences) - 1):
        yield sentences[i], sentences[i+1]

def gen_tokenized_pairs(max_pairs=100000):
    """
    Streams Wikipedia articles and yields tokenized (input, response) pairs up to max_pairs.
    Each pair is tokenized and padded to MAX_LEN.
    """
    wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, trust_remote_code=True)
    count = 0
    for article in tqdm(wiki, total=max_pairs//2, desc="Streaming Wikipedia"):
        for inp, resp in pairwise_sentences(article):
            if len(inp.split()) > 2 and len(resp.split()) > 2:
                input_enc = tokenizer(inp, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
                resp_enc = tokenizer(resp, truncation=True, padding="max_length", max_length=MAX_LEN, return_tensors="pt")
                yield {
                    "input": input_enc["input_ids"].squeeze(0),
                    # Decoder input starts with [CLS] and is shifted right
                    "response_input": torch.cat([
                        torch.tensor([tokenizer.cls_token_id]),
                        resp_enc["input_ids"].squeeze(0)[:-1]
                    ]),
                    # Decoder target is the actual response
                    "response_target": resp_enc["input_ids"].squeeze(0)
                }
                count += 1
                if count >= max_pairs:
                    return

class WikiChatIterableDataset(IterableDataset):
    """
    Iterable PyTorch dataset that streams Wikipedia and yields tokenized (input, response) pairs.
    """
    def __init__(self, max_pairs=100000):
        super().__init__()
        self.max_pairs = max_pairs

    def __iter__(self):
        """
        Iterates through the dataset, yielding tokenized input-response pairs.
        """
        count = 0
        wiki = load_dataset("wikimedia/wikipedia", "20231101.en", split="train", streaming=True, trust_remote_code=True)
        for article in wiki:
            for inp, resp in pairwise_sentences(article):
                if len(inp.split()) > 2 and len(resp.split()) > 2:
                    input_enc = tokenizer(inp, truncation=True, padding=False, max_length=MAX_LEN, return_tensors="pt")
                    resp_enc = tokenizer(resp, truncation=True, padding=False, max_length=MAX_LEN, return_tensors="pt")
                    yield {
                        "input": input_enc["input_ids"].squeeze(0),
                        # Decoder input starts with [CLS] and is shifted right
                        "response_input": torch.cat([
                            torch.tensor([tokenizer.cls_token_id]),
                            resp_enc["input_ids"].squeeze(0)[:-1]
                        ]),
                        # Decoder target is the actual response
                        "response_target": resp_enc["input_ids"].squeeze(0)
                    }
                    count += 1
                    if count >= self.max_pairs:
                        return

    def __len__(self):
        """Returns the number of pairs in the dataset."""
        return self.max_pairs

def collate_fn(batch):
    """
    Pads a batch of samples to the same length for input, response_input, and response_target.
    Returns a dictionary of padded tensors.
    """
    inputs = pad_sequence([item["input"] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    resp_inputs = pad_sequence([item["response_input"] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    resp_targets = pad_sequence([item["response_target"] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    return {"input": inputs, "response_input": resp_inputs, "response_target": resp_targets}

# --- Positional Encoding for Transformer ---
class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for transformer models.
    """
    def __init__(self, d_model, max_seq_length=100):
        super(PositionalEncoding, self).__init__()
        
        pe = torch.zeros(max_seq_length, d_model)
        position = torch.arange(0, max_seq_length, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        self.register_buffer('pe', pe.unsqueeze(0))  # Register as buffer so it's saved with the model
        
    def forward(self, x):
        # Add positional encoding to input tensor x
        return x + self.pe[:, :x.size(1)]

# --- Transformer Chatbot Model ---
class TransformerChatbot(nn.Module):
    """
    Sequence-to-sequence transformer model for chatbot response generation.
    """
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1):
        super(TransformerChatbot, self).__init__()
        
        # Embedding layer for tokens
        self.embedding = nn.Embedding(vocab_size, d_model)
        # Positional encoding
        self.positional_encoding = PositionalEncoding(d_model)
        self.dropout = nn.Dropout(dropout)
        
        # Transformer encoder-decoder
        self.transformer = nn.Transformer(
            d_model=d_model,
            nhead=nhead,
            num_encoder_layers=num_encoder_layers,
            num_decoder_layers=num_decoder_layers,
            dim_feedforward=dim_feedforward,
            batch_first=True,
            dropout=dropout
        )
        
        # Output projection to vocabulary size
        self.output_layer = nn.Linear(d_model, vocab_size)
        
    def forward(self, src, tgt, src_mask=None, tgt_mask=None):
        """
        Forward pass for the transformer chatbot.
        src: input tensor (batch, seq_len)
        tgt: decoder input tensor (batch, seq_len)
        Returns logits for each token in the output sequence.
        """
        # Embedding and positional encoding
        src_embedded = self.dropout(self.positional_encoding(self.embedding(src)))
        tgt_embedded = self.dropout(self.positional_encoding(self.embedding(tgt)))
        
        # Create masks for padding tokens
        src_key_padding_mask = (src == tokenizer.pad_token_id)
        tgt_key_padding_mask = (tgt == tokenizer.pad_token_id)
        
        # Generate subsequent mask for target (prevents attending to future tokens)
        tgt_len = tgt.size(1)
        subsequent_mask = self.generate_square_subsequent_mask(tgt_len).to(tgt.device)
        
        # Transformer forward pass
        output = self.transformer(
            src_embedded, 
            tgt_embedded, 
            src_mask=src_mask, 
            tgt_mask=subsequent_mask,
            src_key_padding_mask=src_key_padding_mask,
            tgt_key_padding_mask=tgt_key_padding_mask
        )
        
        # Project to vocabulary logits
        return self.output_layer(output)
    
    def generate_square_subsequent_mask(self, sz):
        """
        Generates a square mask for the sequence. Used in the decoder to mask future positions.
        """
        mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
        mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
        return mask

# --- Training function with mixed precision for MPS ---
def train(model, dataloader, optimizer, criterion, device, scheduler, start_epoch=0, epochs=10):
    """
    Trains the transformer chatbot model.
    Supports mixed precision on CUDA, and saves checkpoints after each epoch.
    """
    model.train()
    for epoch in range(start_epoch, epochs):
        total_loss = 0
        for batch in tqdm(dataloader, desc=f"Epoch {epoch+1}"):
            src = batch["input"].to(device)
            tgt_input = batch["response_input"].to(device)
            tgt_output = batch["response_target"].to(device)
            optimizer.zero_grad()
            
            # Only use autocast for CUDA devices (mixed precision)
            if device.type == "cuda":
                with torch.autocast(device_type='cuda', dtype=torch.float16):
                    output = model(src, tgt_input)
                    output_flat = output.contiguous().view(-1, output.size(-1))
                    tgt_output_flat = tgt_output.contiguous().view(-1)
                    loss = criterion(output_flat, tgt_output_flat)
            else:
                # Normal forward pass for CPU and MPS
                output = model(src, tgt_input)
                output_flat = output.contiguous().view(-1, output.size(-1))
                tgt_output_flat = tgt_output.contiguous().view(-1)
                loss = criterion(output_flat, tgt_output_flat)
                
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Gradient clipping
            optimizer.step()
            total_loss += loss.item()
        print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader):.4f}")
        # Save checkpoint after each epoch
        os.makedirs("models", exist_ok=True)  # Create models directory if it doesn't exist
        checkpoint_path = f"models/chatbot_epoch_{epoch+1}.pth"
        torch.save({
            'epoch': epoch + 1,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tokenizer': tokenizer.name_or_path
        }, checkpoint_path)
        # Also save latest checkpoint for test script compatibility
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'tokenizer': tokenizer.name_or_path
        }, 'transformer_chatbot.pth')
        print(f"Checkpoint saved to {checkpoint_path}")
        scheduler.step()
    return model

# --- Generate response function (tokenizer-based) ---
def generate_response(model, input_text, tokenizer, device, max_length=20, temperature=1.0, top_k=50):
    """
    Generates a response from the model given an input text string.
    Uses top-k sampling and temperature for diversity.
    """
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
        # Top-k filtering
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

# --- Main training procedure ---
def main():
    """
    Main function to set up data, model, optimizer, and start training.
    Handles checkpoint loading and saving.
    """
    batch_size = 16
    lr = 5e-5
    max_pairs = int(os.environ.get("WIKI_PAIRS", 100000))
    dataset = WikiChatIterableDataset(max_pairs)
    dataloader = DataLoader(dataset, batch_size=batch_size, collate_fn=collate_fn, num_workers=2, shuffle=False)
    model = TransformerChatbot(tokenizer.vocab_size).to(device)
    optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)

    # Linear warmup schedule: 10% warmup, then linear decay
    total_steps = len(dataloader) * 50
    scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(0.1 * total_steps), num_training_steps=total_steps)

    # Create models directory if it doesn't exist
    os.makedirs("models", exist_ok=True)
    
    # Find latest checkpoint
    checkpoint_files = glob.glob('models/chatbot_epoch_*.pth')
    if checkpoint_files:
        latest_ckpt = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        start_epoch = checkpoint.get('epoch', 0)
    else:
        print("No checkpoint found. Starting from scratch.")
        start_epoch = 0

    epochs = 50  # You can change this or make it configurable
    print("Training model...")
    model = train(model, dataloader, optimizer, criterion, device, scheduler, start_epoch=start_epoch, epochs=start_epoch+epochs)
    print("Training complete.")

if __name__ == "__main__":
    main()
