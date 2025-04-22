import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup
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
from torch.amp import autocast, GradScaler
import subprocess
import time
try:
    import bitsandbytes as bnb
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

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
if device.type == "cuda":
    torch.backends.cudnn.benchmark = True

# Tokenizer setup
TOKENIZER_NAME = "microsoft/codebert-base"  # Code-aware tokenizer
# Load the tokenizer from HuggingFace
tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
MAX_LEN = 128  # Maximum sequence length for input/output

# --- Wikipedia streaming and tokenization utilities ---

def pairwise_sentences(article):
    """
    Splits code content into consecutive line pairs.
    Used to create (input, response) pairs for chatbot training.
    """
    raw = article.get("content", "")
    if not raw:
        return
    # Discard very large files to avoid long streams
    if len(raw) > 200_000:  # skip files over ~200k chars
        return
    # Split on non-empty lines
    lines = [line for line in raw.split("\n") if line.strip()]
    # Yield consecutive line pairs
    for i in range(len(lines) - 1):
        yield lines[i], lines[i + 1]

def gen_tokenized_pairs(max_pairs=100000):
    """
    Streams Wikipedia articles and yields tokenized (input, response) pairs up to max_pairs.
    Each pair is tokenized and padded to MAX_LEN.
    """
    wiki = load_dataset("bigcode/the-stack", split="train", streaming=True, trust_remote_code=True)
    count = 0
    for article in tqdm(wiki, total=max_pairs//2, desc="Streaming Wikipedia"):
        raw = article.get("content", "")
        if not raw or len(raw) > 200_000:
            continue
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
        wiki = load_dataset("bigcode/the-stack", split="train", streaming=True, trust_remote_code=True)
        for article in wiki:
            raw = article.get("content", "")
            if not raw or len(raw) > 200_000:
                continue
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
    def __init__(self, vocab_size, d_model=512, nhead=8, num_encoder_layers=8, num_decoder_layers=8, dim_feedforward=8192, dropout=0.1):
        
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
        
        # Ensure mask type matches key_padding_mask type (bool)
        if subsequent_mask.dtype != torch.bool:
            subsequent_mask = subsequent_mask == 0
        
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
def train(model, train_loader, val_loader, optimizer, criterion, device, scheduler, total_steps):
    """
    Trains the transformer chatbot model for a fixed number of steps.
    Runs validation and checkpointing every CHECKPOINT_STEPS iterations.
    """
    best_val_loss = float('inf')
    checkpoint_interval = int(os.environ.get("CHECKPOINT_STEPS", 1000))
    iteration = 0
    scaler = GradScaler(device if device.type == "cuda" else None)
    from itertools import islice, cycle
    from math import ceil

    pbar = tqdm(total=total_steps, desc="Training")
    model.train()
    cumulative_loss = 0.0
    for batch in islice(cycle(train_loader), total_steps):
        src = batch["input"].to(device)
        tgt_input = batch["response_input"].to(device)
        tgt_output = batch["response_target"].to(device)

        optimizer.zero_grad()
        with autocast(device_type=device.type if device.type in ("cuda", "cpu", "mps") else "cpu"):
            output = model(src, tgt_input)
            output_flat = output.view(-1, output.size(-1))
            tgt_output_flat = tgt_output.view(-1)
            loss = criterion(output_flat, tgt_output_flat)

        scaler.scale(loss).backward()
        scaler.unscale_(optimizer)
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        scaler.step(optimizer)
        scaler.update()
        scheduler.step()

        iteration += 1
        cumulative_loss += loss.item()
        avg_loss = cumulative_loss / iteration
        pbar.update(1)
        pbar.set_postfix(loss=f"{avg_loss:.4f}")

        if iteration % checkpoint_interval == 0:
            # Validation
            val_loss = 0.0
            val_steps = ceil(val_loader.dataset.max_pairs / val_loader.batch_size)
            val_pbar = tqdm(total=val_steps, desc="Validation")
            model.eval()
            val_batches = 0
            with torch.no_grad():
                for vbatch in islice(cycle(val_loader), val_steps):
                    vsrc = vbatch["input"].to(device)
                    vtgt_in = vbatch["response_input"].to(device)
                    vtgt_out = vbatch["response_target"].to(device)
                    # Skip batch if all targets are padding
                    if (vtgt_out != tokenizer.pad_token_id).sum() == 0:
                        continue
                    vout = model(vsrc, vtgt_in)
                    vloss = criterion(vout.view(-1, vout.size(-1)), vtgt_out.view(-1))
                    # Skip batch if loss is nan or inf
                    if torch.isnan(vloss) or torch.isinf(vloss):
                        print("[Validation] NaN or Inf loss detected, skipping batch.")
                        continue
                    val_loss += vloss.item()
                    val_batches += 1
                    val_pbar.update(1)
            val_pbar.close()
            avg_val = val_loss / max(val_batches, 1)
            print(f"\nIter {iteration}: Validation Loss: {avg_val:.4f}")
            ckpt_path = f"models/iter_{iteration}.pth"
            os.makedirs("models", exist_ok=True)
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer': tokenizer.name_or_path
            }, ckpt_path)
            print(f"Checkpoint saved to {ckpt_path}")
            model.train()
    pbar.close()
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
    # Allow batch size override via env, default to 512 for speed
    train_batch_size = int(os.environ.get("BATCH_SIZE", 512))
    val_batch_size = int(os.environ.get("VAL_BATCH_SIZE", 32))
    lr = 0.001
    max_pairs = int(os.environ.get("WIKI_PAIRS", 5000))
    train_dataset = WikiChatIterableDataset(max_pairs)
    train_loader = DataLoader(
        train_dataset,
        batch_size=train_batch_size,
        collate_fn=collate_fn,
        num_workers=8,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False
    )

    # Validation DataLoader with larger dataset and smaller batch size
    val_max_pairs = int(os.environ.get("VAL_WIKI_PAIRS", 10000))
    val_dataset = WikiChatIterableDataset(val_max_pairs)
    val_loader = DataLoader(
        val_dataset,
        batch_size=val_batch_size,
        collate_fn=collate_fn,
        num_workers=4,
        pin_memory=True,
        persistent_workers=True,
        shuffle=False
    )

    # Find latest checkpoint
    checkpoint_files = glob.glob('models/chatbot_epoch_*.pth')
    if checkpoint_files:
        latest_ckpt = max(checkpoint_files, key=os.path.getctime)
        print(f"Loading checkpoint: {latest_ckpt}")
        checkpoint = torch.load(latest_ckpt, map_location=device)
        model = TransformerChatbot(tokenizer.vocab_size).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        print("No checkpoint found. Starting from scratch.")
        model = TransformerChatbot(tokenizer.vocab_size).to(device)

    # Total training steps
    total_steps = int(os.environ.get("TRAIN_STEPS", 100000))
    # Use bitsandbytes AdamW8bit if available and on CUDA
    if BNB_AVAILABLE and device.type == "cuda":
        print("Using bitsandbytes AdamW8bit optimizer for speed and memory efficiency.")
        optimizer = bnb.optim.AdamW8bit(model.parameters(), lr=lr, weight_decay=0.01)
    else:
        optimizer = AdamW(model.parameters(), lr=lr, weight_decay=0.01)
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    print(f"Training for {total_steps} steps with checkpoints every {os.environ.get('CHECKPOINT_STEPS', 1000)} iterations...")
    criterion = nn.CrossEntropyLoss(ignore_index=tokenizer.pad_token_id, label_smoothing=0.1)

    # Log GPU utilization every 100 steps
    def log_gpu_util():
        if device.type == "cuda":
            try:
                util = subprocess.check_output(["nvidia-smi", "--query-gpu=utilization.gpu,memory.used,memory.total", "--format=csv,noheader,nounits"])
                print("[GPU]", util.decode().strip())
            except Exception as e:
                print("[GPU] nvidia-smi not available:", e)

    # Try to auto-tune batch size if OOM
    try:
        model = train(model, train_loader, val_loader, optimizer, criterion, device, scheduler, total_steps)
    except RuntimeError as e:
        if "out of memory" in str(e):
            print("CUDA OOM: Try reducing BATCH_SIZE env variable.")
        raise

    print("Training complete.")
    # Suggest using local dataset for speed if streaming is slow
    print("Tip: For even faster training, consider downloading a local subset of the dataset and training from disk instead of streaming.")

if __name__ == "__main__":
    main()
