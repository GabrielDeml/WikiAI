"""
AI Code Completion App

Launches a Gradio interface for code completion using a pretrained Transformer model.
"""
import os
import glob
import torch
import gradio as gr
from transformers import AutoTokenizer
from train_transformer_chatbot import TransformerChatbot, generate_response

# Configuration
TOKENIZER_NAME = os.environ.get("TOKENIZER_NAME", "microsoft/codebert-base")
DEFAULT_MAX_LENGTH = int(os.environ.get("MAX_COMPLETION_LENGTH", 128))

def get_device():
    if torch.backends.mps.is_available():
        return torch.device("mps")
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def find_latest_checkpoint():
    """
    Finds the latest model checkpoint (.pth) in the models directory.
    """
    checkpoint_files = glob.glob('models/*.pth')
    if not checkpoint_files:
        return None
    return max(checkpoint_files, key=os.path.getctime)

# Load model and tokenizer
device = get_device()
checkpoint_path = find_latest_checkpoint()
if checkpoint_path is None:
    raise FileNotFoundError("No model checkpoint found in 'models' directory. Please train the model first.")

tokenizer = AutoTokenizer.from_pretrained(TOKENIZER_NAME)
model = TransformerChatbot(tokenizer.vocab_size)
state = torch.load(checkpoint_path, map_location=device)
# state may contain a dict with 'model_state_dict' or be the state dict itself
model_state = state.get('model_state_dict', state)
model.load_state_dict(model_state)
model.to(device)
model.eval()

def complete(code_prompt, max_length=DEFAULT_MAX_LENGTH, temperature=1.0, top_k=50):
    """
    Generates a code completion for the given prompt.
    """
    prompt = code_prompt or ""
    completion = generate_response(
        model,
        prompt,
        tokenizer,
        device,
        max_length=max_length,
        temperature=temperature,
        top_k=top_k
    )
    return completion

def main():
    title = "AI Code Completion"
    description = (
        "Generate code completions using a pretrained Transformer model. "
        "Adjust parameters for max length, temperature, and top-k sampling."
    )
    inputs = [
        gr.Textbox(lines=10, placeholder="Enter your code prompt here...", label="Code Prompt"),
        gr.Slider(minimum=1, maximum=512, step=1, value=DEFAULT_MAX_LENGTH, label="Max Completion Length"),
        gr.Slider(minimum=0.1, maximum=2.0, step=0.1, value=1.0, label="Temperature"),
        gr.Slider(minimum=1, maximum=100, step=1, value=50, label="Top-K Sampling")
    ]
    output = gr.Textbox(lines=10, label="Completion")

    iface = gr.Interface(
        fn=complete,
        inputs=inputs,
        outputs=output,
        title=title,
        description=description,
        allow_flagging="never"
    )
    iface.launch()

if __name__ == "__main__":
    main()