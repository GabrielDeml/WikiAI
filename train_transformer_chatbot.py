"""
Code Completion Model Training Script with Transformer Architecture

This script trains a transformer-based model for code completion using datasets from Hugging Face.

To run this script with Hugging Face authentication (required for accessing gated datasets):
    python train_transformer_chatbot.py --hf_token YOUR_HF_TOKEN

Or set the token as an environment variable:
    export HF_TOKEN=your_token
    python train_transformer_chatbot.py

You can get a token from: https://huggingface.co/settings/tokens
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader, IterableDataset
from torch.optim import AdamW
from transformers import get_linear_schedule_with_warmup, GPT2Tokenizer, RobertaTokenizer
from torch.nn.utils.rnn import pad_sequence
import numpy as np
import random
import math
import os
import time
import logging
from tqdm import tqdm
from datasets import load_dataset, config
from transformers import AutoTokenizer
import glob
import requests
from requests.exceptions import ReadTimeout, ConnectionError
from torch.cuda.amp import autocast, GradScaler
from huggingface_hub import login
import argparse

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Set random seed for reproducibility
seed = 42
torch.manual_seed(seed)
random.seed(seed)
np.random.seed(seed)

# Configure Hugging Face datasets
# Increase timeouts and retries for more reliable downloads
config.HF_DATASETS_OFFLINE = int(os.environ.get("HF_DATASETS_OFFLINE", 0))
config.HF_DATASETS_CACHE = os.environ.get("HF_DATASETS_CACHE", os.path.expanduser("~/.cache/huggingface/datasets"))
config.DOWNLOADED_DATASETS_PATH = os.environ.get("DOWNLOADED_DATASETS_PATH", os.path.expanduser("~/.cache/huggingface/downloads"))

# Hugging Face Login function
def login_to_huggingface(token=None):
    """
    Login to HuggingFace Hub.
    If token is not provided, will try to get it from HF_TOKEN environment variable.
    If still not available, guide the user to get a token.
    """
    if token is None:
        token = os.environ.get("HF_TOKEN")
    
    if token:
        try:
            # Login to Hugging Face
            login(token=token)
            logger.info("Successfully logged in to Hugging Face Hub.")
            return True
        except Exception as e:
            logger.error(f"Failed to login to Hugging Face Hub: {e}")
            return False
    else:
        logger.warning(
            "No Hugging Face token provided. To access gated datasets, please:\n"
            "1. Create an account at https://huggingface.co/\n"
            "2. Generate a token at https://huggingface.co/settings/tokens\n"
            "3. Provide it using --hf_token argument or set HF_TOKEN environment variable"
        )
        return False

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

# Set code-specific parameters
MAX_LEN = 512  # Maximum sequence length for code snippets - increased for code completion
TARGET_LANGUAGES = ["python", "javascript", "java", "go", "ruby", "rust", "cpp"]  # Languages to include

# Tokenizer setup for code
TOKENIZER_NAME = "microsoft/codebert-base"  # Better for code tokenization
max_retries = 5
for attempt in range(max_retries):
    try:
        tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
        # Add special tokens for code
        tokenizer.add_special_tokens({
            'additional_special_tokens': ['<filename>', '<code>', '</code>', '<comment>', '</comment>']
        })
        break
    except (ConnectionError, ReadTimeout) as e:
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            logger.warning(f"Error loading tokenizer (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
            time.sleep(wait_time)
        else:
            logger.error(f"Failed to load tokenizer after {max_retries} attempts. Using fallback method.")
            tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
            break

# --- Code snippet processing functions ---

def process_code_snippet(snippet):
    """
    Processes a code snippet for the model.
    For code completion, we want to generate predictions for the next tokens.
    """
    # Clean up code - remove excessive newlines, normalize spacing
    snippet = snippet.replace('\r\n', '\n')
    snippet = '\n'.join([line for line in snippet.split('\n') if line.strip()])
    
    # If snippet is too large, take a random window
    if len(snippet) > MAX_LEN * 4:  # Rough character estimation
        start_idx = random.randint(0, len(snippet) - MAX_LEN * 2)
        snippet = snippet[start_idx:start_idx + MAX_LEN * 2]
    
    return snippet

def safe_load_dataset(dataset_name, dataset_config=None, split="train", streaming=True, max_retries=5, timeout=30, data_dir=None):
    """
    Safely load a dataset with retries and proper error handling.
    """
    for attempt in range(max_retries):
        try:
            # Try to load the dataset with provided parameters
            if data_dir:
                dataset = load_dataset(
                    dataset_name,
                    data_dir=data_dir,
                    split=split,
                    streaming=streaming,
                    trust_remote_code=True,
                    download_timeout=timeout
                )
            elif dataset_config:
                dataset = load_dataset(
                    dataset_name,
                    dataset_config,
                    split=split,
                    streaming=streaming,
                    trust_remote_code=True,
                    download_timeout=timeout
                )
            else:
                dataset = load_dataset(
                    dataset_name,
                    split=split,
                    streaming=streaming,
                    trust_remote_code=True,
                    download_timeout=timeout
                )
            return dataset
        except (ConnectionError, ReadTimeout, requests.exceptions.RequestException) as e:
            wait_time = min(2 ** attempt, 60)  # Exponential backoff with max 60s
            if attempt < max_retries - 1:
                logger.warning(f"Error loading dataset (attempt {attempt+1}/{max_retries}): {e}. Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                logger.error(f"Failed to load dataset after {max_retries} attempts: {e}")
                if "gated dataset" in str(e).lower():
                    logger.error("This is a gated dataset. Please provide a Hugging Face token with --hf_token or set the HF_TOKEN environment variable.")
                raise

class CodeCompletionDataset(IterableDataset):
    """
    Dataset for code completion using The Stack dataset.
    """
    def __init__(self, max_samples=10000, languages=None, context_length=384, completion_length=128):
        super().__init__()
        self.max_samples = max_samples
        self.languages = languages or TARGET_LANGUAGES
        self.context_length = context_length  
        self.completion_length = completion_length
        self.total_length = context_length + completion_length
        self.auth_failure_count = 0  # Track authentication failures
        
    def __iter__(self):
        """
        Iterates through the dataset, yielding tokenized code samples for completion.
        """
        count = 0
        datasets_loaded = 0  # Track how many datasets were successfully loaded
        
        # Process each selected language
        for lang in self.languages:
            logger.info(f"Loading The Stack dataset for language: {lang}")
            
            try:
                # Try multiple approaches to load the dataset
                try:
                    # First approach: with data_dir parameter
                    dataset = safe_load_dataset(
                        "bigcode/the-stack", 
                        split="train", 
                        timeout=60, 
                        data_dir=f"data/{lang}"
                    )
                except Exception as e1:
                    logger.warning(f"First dataset loading approach failed: {e1}")
                    
                    # Check if this is an authentication issue
                    if "gated dataset" in str(e1).lower():
                        self.auth_failure_count += 1
                        if self.auth_failure_count >= 3:  # After 3 failures, give a clearer message
                            logger.error("\n" + "="*80)
                            logger.error("AUTHENTICATION ERROR: Cannot access gated datasets")
                            logger.error("To fix this, run the script with a Hugging Face token:")
                            logger.error("    python train_transformer_chatbot.py --hf_token YOUR_TOKEN")
                            logger.error("Or set the token as an environment variable:")
                            logger.error("    export HF_TOKEN=your_token")
                            logger.error("="*80 + "\n")
                    
                    # Second approach: with language as the config
                    try:
                        dataset = safe_load_dataset(
                            "bigcode/the-stack", 
                            dataset_config=lang,
                            split="train", 
                            timeout=60
                        )
                    except Exception as e2:
                        logger.warning(f"Second dataset loading approach failed: {e2}")
                        
                        # Third approach: load the dataset directly
                        try:
                            dataset = load_dataset(
                                "bigcode/the-stack", 
                                streaming=True,
                                split="train"
                            ).filter(lambda x: x.get("lang") == lang)
                        except Exception as e3:
                            logger.warning(f"Third dataset loading approach failed: {e3}")
                            
                            # If all approaches for the-stack fail, try starcoderdata as a final fallback
                            try:
                                logger.info(f"Trying starcoderdata for {lang}")
                                dataset = load_dataset(
                                    "bigcode/starcoderdata", 
                                    data_dir=lang,
                                    split="train",
                                    streaming=True
                                )
                            except Exception as e4:
                                logger.error(f"All dataset loading approaches failed for {lang}")
                                raise e4
                
                # If we reach here, we successfully loaded a dataset
                datasets_loaded += 1
                
                for sample in dataset:
                    try:
                        # Handle different dataset structures
                        code_content = None
                        
                        # Check what fields are available in the sample
                        if "content" in sample:
                            code_content = sample["content"]
                        elif "code" in sample:
                            code_content = sample["code"]
                        elif "text" in sample:
                            code_content = sample["text"]
                        else:
                            # Try to find any field that might contain a string
                            for key, value in sample.items():
                                if isinstance(value, str) and len(value) > 10:
                                    code_content = value
                                    break
                        
                        if not code_content or not isinstance(code_content, str) or len(code_content) < 10:
                            continue
                            
                        # Process code snippet
                        code = process_code_snippet(code_content)
                        
                        # Tokenize the code
                        tokens = tokenizer.encode(code, truncation=True, max_length=self.total_length)
                        
                        if len(tokens) < 32:  # Skip very short snippets
                            continue
                            
                        # For a causal language model, input and target are the same sequence
                        # but target is shifted by one position
                        input_ids = tokens[:-1]
                        target_ids = tokens[1:]
                        
                        if len(input_ids) < 10:  # Skip samples with too few tokens
                            continue
                        
                        # Create sample with both the input and target
                        yield {
                            "input_ids": torch.tensor(input_ids),
                            "target_ids": torch.tensor(target_ids),
                            "language": lang
                        }
                        
                        count += 1
                        if count >= self.max_samples:
                            logger.info(f"Reached max_samples limit ({self.max_samples})")
                            return
                            
                    except Exception as e:
                        logger.warning(f"Error processing code sample: {e}. Skipping...")
                        continue
                        
            except Exception as e:
                logger.error(f"Error loading dataset for {lang}: {e}")
                continue
                
        # If we couldn't get enough samples, log a warning
        if count < self.max_samples:
            logger.warning(f"Could only retrieve {count} samples, less than requested {self.max_samples}")
            
        # If no datasets were loaded successfully, we need to inform the user
        if datasets_loaded == 0:
            logger.error("\n" + "="*80)
            logger.error("FAILED TO LOAD ANY DATASETS")
            logger.error("This is likely an authentication issue with Hugging Face.")
            logger.error("Please ensure you have provided a valid token using --hf_token.")
            logger.error("You can create a token at: https://huggingface.co/settings/tokens")
            logger.error("="*80 + "\n")

    def __len__(self):
        """Returns the maximum number of samples"""
        return self.max_samples

def collate_fn(batch):
    """
    Collate function for DataLoader that handles variable-length sequences.
    """
    # Pad input and target sequences to the same length within batch
    input_ids = pad_sequence([item["input_ids"] for item in batch], batch_first=True, padding_value=tokenizer.pad_token_id)
    target_ids = pad_sequence([item["target_ids"] for item in batch], batch_first=True, padding_value=-100)  # -100 is ignored in loss
    
    # Create attention masks
    attention_mask = (input_ids != tokenizer.pad_token_id).float()
    
    return {
        "input_ids": input_ids,
        "target_ids": target_ids,
        "attention_mask": attention_mask,
        "languages": [item["language"] for item in batch]
    }

# --- Positional Encoding for Transformer ---
class PositionalEncoding(nn.Module):
    """
    Implements sinusoidal positional encoding for transformer models.
    """
    def __init__(self, d_model, max_seq_length=1024):
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

# --- Code Completion Transformer Model ---
class CodeCompletionModel(nn.Module):
    """
    Transformer-based model for code completion. Uses a decoder-only architecture
    similar to GPT models for causal language modeling of code.
    """
    def __init__(self, vocab_size, d_model=768, nhead=12, num_layers=12, 
                 dim_feedforward=3072, dropout=0.1, max_seq_length=1024):
        super(CodeCompletionModel, self).__init__()
        
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.positional_encoding = PositionalEncoding(d_model, max_seq_length)
        self.dropout = nn.Dropout(dropout)
        
        # Create decoder layers manually for more control
        self.layers = nn.ModuleList([
            nn.TransformerDecoderLayer(
                d_model=d_model,
                nhead=nhead,
                dim_feedforward=dim_feedforward,
                dropout=dropout,
                batch_first=True
            ) for _ in range(num_layers)
        ])
        
        # Mask to prevent looking at future tokens
        self.register_buffer("causal_mask", torch.triu(torch.ones(max_seq_length, max_seq_length) * float('-inf'), diagonal=1))
        
        # Output projection to vocabulary
        self.output_layer = nn.Linear(d_model, vocab_size)
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize the weights for better training stability"""
        if isinstance(module, (nn.Linear, nn.Embedding)):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if isinstance(module, nn.Linear) and module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.LayerNorm):
            module.bias.data.zero_()
            module.weight.data.fill_(1.0)
        
    def forward(self, input_ids, attention_mask=None):
        """
        Forward pass for causal language modeling (code completion)
        
        input_ids: Input token ids [batch_size, seq_len]
        attention_mask: Mask for padding [batch_size, seq_len]
        """
        seq_len = input_ids.size(1)
        
        # Get embeddings
        x = self.embedding(input_ids)
        x = self.positional_encoding(x)
        x = self.dropout(x)
        
        # Create a causal mask for the sequence length we're using
        mask = self.causal_mask[:seq_len, :seq_len]
        
        # Apply each transformer decoder layer
        for layer in self.layers:
            # For a decoder-only model, we use the same inputs for both arguments
            # but rely on the causal mask to prevent information leakage
            x = layer(x, x, tgt_mask=mask, tgt_key_padding_mask=~attention_mask.bool() if attention_mask is not None else None)
        
        # Project to vocabulary
        logits = self.output_layer(x)
        
        return logits

# --- Training function with mixed precision ---
def train(model, train_loader, val_loader, optimizer, criterion, device, scheduler, total_steps):
    """
    Trains the code completion model for a fixed number of steps.
    Runs validation and checkpointing at regular intervals.
    """
    best_val_loss = float('inf')
    checkpoint_interval = int(os.environ.get("CHECKPOINT_STEPS", 1000))
    iteration = 0
    
    # Setup gradient scaler for mixed precision training
    if device.type == "cuda":
        scaler = torch.amp.GradScaler('cuda')
    else:
        scaler = GradScaler()
        
    from itertools import islice
    from math import ceil

    pbar = tqdm(total=total_steps, desc="Training")
    model.train()
    cumulative_loss = 0.0
    
    # Main training loop with robust error handling
    while iteration < total_steps:
        try:
            # Get a batch from the dataloader with timeout handling
            batch = next(islice(train_loader, 1))
            
            input_ids = batch["input_ids"].to(device)
            target_ids = batch["target_ids"].to(device)
            attention_mask = batch.get("attention_mask", None)
            if attention_mask is not None:
                attention_mask = attention_mask.to(device)

            optimizer.zero_grad()
            
            # Proper amp context manager based on device type
            if device.type == "cuda":
                with torch.amp.autocast('cuda'):
                    logits = model(input_ids, attention_mask)
                    # Reshape for loss calculation
                    shift_logits = logits.view(-1, logits.size(-1))
                    shift_targets = target_ids.view(-1)
                    loss = criterion(shift_logits, shift_targets)
            else:
                with autocast(enabled=(device.type=="cuda")):
                    logits = model(input_ids, attention_mask)
                    # Reshape for loss calculation
                    shift_logits = logits.view(-1, logits.size(-1))
                    shift_targets = target_ids.view(-1)
                    loss = criterion(shift_logits, shift_targets)

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
                run_validation(model, val_loader, criterion, device, iteration, best_val_loss)
                model.train()
                
        except (StopIteration, IndexError):
            logger.warning("Dataloader exhausted or error encountered. Recreating dataloader...")
            # Recreate dataloader if it gets exhausted
            train_loader = recreate_dataloader(train_loader.dataset.max_samples, batch_size=train_loader.batch_size)
            continue
            
        except Exception as e:
            logger.error(f"Error during training iteration {iteration}: {e}")
            # Save emergency checkpoint
            emergency_path = f"models/emergency_iter_{iteration}.pth"
            os.makedirs("models", exist_ok=True)
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer': tokenizer.name_or_path
            }, emergency_path)
            logger.info(f"Emergency checkpoint saved to {emergency_path}")
            
            # Allow training to continue despite errors
            time.sleep(1)  # Brief pause before continuing
            continue
            
    pbar.close()
    return model

def run_validation(model, val_loader, criterion, device, iteration, best_val_loss):
    """Separate validation function with error handling"""
    try:
        val_loss = 0.0
        val_steps = 0
        val_pbar = tqdm(total=ceil(val_loader.dataset.max_samples / val_loader.batch_size), desc="Validation")
        model.eval()
        
        # Keep track of processed validation batches
        with torch.no_grad():
            for _ in range(ceil(val_loader.dataset.max_samples / val_loader.batch_size)):
                try:
                    batch = next(iter(val_loader))
                    input_ids = batch["input_ids"].to(device)
                    target_ids = batch["target_ids"].to(device)
                    attention_mask = batch.get("attention_mask", None)
                    if attention_mask is not None:
                        attention_mask = attention_mask.to(device)
                    
                    logits = model(input_ids, attention_mask)
                    shift_logits = logits.view(-1, logits.size(-1))
                    shift_targets = target_ids.view(-1)
                    vloss = criterion(shift_logits, shift_targets)
                    val_loss += vloss.item()
                    val_steps += 1
                    val_pbar.update(1)
                except (StopIteration, IndexError):
                    logger.warning("Validation dataloader exhausted early.")
                    break
                except Exception as e:
                    logger.error(f"Error during validation: {e}")
                    continue
                    
        val_pbar.close()
        
        if val_steps > 0:
            avg_val = val_loss / val_steps
            logger.info(f"\nIter {iteration}: Validation Loss: {avg_val:.4f}")
            
            # Save checkpoint
            ckpt_path = f"models/code_iter_{iteration}.pth"
            os.makedirs("models", exist_ok=True)
            torch.save({
                'iteration': iteration,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'tokenizer': tokenizer.name_or_path,
                'val_loss': avg_val
            }, ckpt_path)
            logger.info(f"Checkpoint saved to {ckpt_path}")
            
            # Update best validation loss
            if avg_val < best_val_loss:
                best_val_loss = avg_val
                best_path = f"models/best_code_model.pth"
                torch.save({
                    'iteration': iteration,
                    'model_state_dict': model.state_dict(),
                    'val_loss': avg_val
                }, best_path)
                logger.info(f"New best model saved to {best_path}")
                
    except Exception as e:
        logger.error(f"Validation failed: {e}")

def recreate_dataloader(max_samples, batch_size=4, languages=None):
    """Recreate a dataloader with the same parameters"""
    dataset = CodeCompletionDataset(max_samples=max_samples, languages=languages or TARGET_LANGUAGES)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True
    )

# --- Generate code completion function ---
def generate_code_completion(model, prompt, tokenizer, device, max_length=100, temperature=0.8, top_p=0.95, top_k=50):
    """
    Generates code completion for a given prompt using the trained model.
    Uses nucleus (top-p) sampling combined with top-k sampling for better quality.
    """
    model.eval()
    
    # Tokenize the prompt
    input_ids = tokenizer.encode(prompt, return_tensors="pt").to(device)
    attention_mask = torch.ones_like(input_ids)
    
    # Track generated tokens
    all_tokens = input_ids.clone()
    
    # Generate tokens one by one
    for _ in range(max_length):
        with torch.no_grad():
            # Get predictions
            outputs = model(all_tokens, attention_mask)
            next_token_logits = outputs[:, -1, :] / temperature
            
            # Filter out special tokens (except EOS)
            for special_id in tokenizer.all_special_ids:
                if special_id != tokenizer.eos_token_id:
                    next_token_logits[:, special_id] = -float('inf')
            
            # Apply top-k filtering
            if top_k > 0:
                indices_to_remove = next_token_logits < torch.topk(next_token_logits, top_k)[0][..., -1, None]
                next_token_logits[indices_to_remove] = -float('inf')
            
            # Apply nucleus (top-p) filtering
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                
                # Remove tokens with cumulative probability above the threshold
                sorted_indices_to_remove = cumulative_probs > top_p
                # Shift the indices to the right to keep also the first token above the threshold
                sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                sorted_indices_to_remove[..., 0] = 0
                
                indices_to_remove = sorted_indices[sorted_indices_to_remove]
                next_token_logits[0, indices_to_remove] = -float('inf')
            
            # Sample from the filtered distribution
            probs = F.softmax(next_token_logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            # Stop if we predict the end of sequence token
            if next_token.item() == tokenizer.eos_token_id:
                break
                
            # Add to generated tokens
            all_tokens = torch.cat([all_tokens, next_token], dim=1)
            attention_mask = torch.cat([attention_mask, torch.ones_like(next_token)], dim=1)
    
    # Decode the generated tokens
    completed_code = tokenizer.decode(all_tokens[0], skip_special_tokens=True)
    # Return just the completion (not including the original prompt)
    completion = completed_code[len(prompt):] if len(completed_code) > len(prompt) else ""
    
    return completion

# --- Main training procedure ---
def main():
    """
    Main function to set up data, model, optimizer, and start training.
    Handles checkpoint loading and saving.
    """
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Train a code completion transformer model")
    parser.add_argument("--batch_size", type=int, default=int(os.environ.get("BATCH_SIZE", 8)),
                        help="Batch size for training")
    parser.add_argument("--learning_rate", type=float, default=float(os.environ.get("LEARNING_RATE", 5e-5)),
                        help="Learning rate")
    parser.add_argument("--max_samples", type=int, default=int(os.environ.get("CODE_SAMPLES", 50000)),
                        help="Maximum number of code samples to use for training")
    parser.add_argument("--languages", type=str, default=os.environ.get("CODE_LANGUAGES", "python,javascript,java"),
                        help="Comma-separated list of programming languages")
    parser.add_argument("--total_steps", type=int, default=int(os.environ.get("TRAIN_STEPS", 50000)),
                        help="Total number of training steps")
    parser.add_argument("--hf_token", type=str, default=None,
                        help="Hugging Face authentication token for accessing gated datasets")
    
    args = parser.parse_args()
    
    # Login to Hugging Face if token is provided
    login_to_huggingface(args.hf_token)
    
    # Extract parameters from args
    batch_size = args.batch_size
    lr = args.learning_rate
    max_samples = args.max_samples
    
    # Configure code languages to use
    languages = args.languages.split(",")
    if not languages or languages[0] == "":
        languages = TARGET_LANGUAGES
    
    logger.info(f"Creating dataset with {max_samples} samples from languages: {languages}")
    dataset = CodeCompletionDataset(max_samples=max_samples, languages=languages)
    
    # Configure DataLoader with proper error handling
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=min(8, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True
    )

    # Validation DataLoader with smaller size
    val_max_samples = min(int(os.environ.get("VAL_SAMPLES", 5000)), max_samples // 5)
    val_dataset = CodeCompletionDataset(max_samples=val_max_samples, languages=languages)
    val_dataloader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        collate_fn=collate_fn,
        num_workers=min(4, os.cpu_count() or 1),
        pin_memory=True,
        persistent_workers=True
    )

    # Find latest checkpoint
    checkpoint_files = glob.glob('models/code_iter_*.pth') + glob.glob('models/best_code_model.pth')
    model = None
    
    if checkpoint_files:
        try:
            latest_ckpt = max(checkpoint_files, key=os.path.getctime)
            logger.info(f"Loading checkpoint: {latest_ckpt}")
            checkpoint = torch.load(latest_ckpt, map_location=device)
            model = CodeCompletionModel(tokenizer.vocab_size).to(device)
            model.load_state_dict(checkpoint['model_state_dict'])
            logger.info(f"Successfully loaded checkpoint from {latest_ckpt}")
        except Exception as e:
            logger.error(f"Error loading checkpoint: {e}. Starting from scratch.")
            model = CodeCompletionModel(tokenizer.vocab_size).to(device)
    
    if model is None:
        logger.info("No checkpoint found. Starting from scratch.")
        model = CodeCompletionModel(tokenizer.vocab_size).to(device)

    # Total training steps
    total_steps = args.total_steps
    
    # Configure optimizer with weight decay
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {'params': [p for n, p in model.named_parameters() if not any(nd in n for nd in no_decay)],
         'weight_decay': 0.01},
        {'params': [p for n, p in model.named_parameters() if any(nd in n for nd in no_decay)],
         'weight_decay': 0.0}
    ]
    optimizer = AdamW(optimizer_grouped_parameters, lr=lr)
    
    # Learning rate scheduler
    scheduler = get_linear_schedule_with_warmup(
        optimizer,
        num_warmup_steps=total_steps // 10,
        num_training_steps=total_steps
    )
    
    logger.info(f"Training for {total_steps} steps with checkpoints every {os.environ.get('CHECKPOINT_STEPS', 1000)} iterations...")
    
    # Set criterion with label smoothing for better generalization
    criterion = nn.CrossEntropyLoss(ignore_index=-100, label_smoothing=0.1)
    
    try:
        model = train(model, dataloader, val_dataloader, optimizer, criterion, device, scheduler, total_steps)
        logger.info("Training complete.")
        
        # Test the model with a code completion example
        test_prompt = "def fibonacci(n):\n    # Returns the nth Fibonacci number\n    "
        completion = generate_code_completion(model, test_prompt, tokenizer, device)
        logger.info(f"Test completion example:\nPrompt: {test_prompt}\nCompletion: {completion}")
        
    except KeyboardInterrupt:
        logger.info("Training interrupted by user. Saving final checkpoint...")
        # Save final checkpoint on keyboard interrupt
        final_path = f"models/interrupted_final.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, final_path)
        logger.info(f"Interrupted checkpoint saved to {final_path}")
    except Exception as e:
        logger.error(f"Training failed with error: {e}")
        # Save emergency checkpoint
        emergency_path = f"models/emergency_final.pth"
        torch.save({
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
        }, emergency_path)
        logger.info(f"Emergency checkpoint saved to {emergency_path}")
        raise

if __name__ == "__main__":
    main()
