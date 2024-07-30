import os
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
import torch

os.environ['PYTORCH_ENABLE_MPS_FALLBACK'] = '1'

# Detect the available device
device = torch.device("cuda" if torch.cuda.is_available() else ("mps" if torch.backends.mps.is_available() else "cpu"))
print(f"Using device: {device}")

# Load the base model and tokenizer
base_model_name = "NousResearch/Llama-2-7b-chat-hf"
tokenizer = AutoTokenizer.from_pretrained(base_model_name)
base_model = AutoModelForCausalLM.from_pretrained(base_model_name, device_map='auto', torch_dtype=torch.float16)

# Load the fine-tuned adapter weights
parent = os.path.dirname(os.path.abspath(__file__))
adapter_path = f'{parent}/checkpoint-1070'
print(f"adapter_path: {adapter_path}")
print(f"Parent directory: {parent}")
try:
    base_model = PeftModel.from_pretrained(base_model, adapter_path)  # No .to(device)
except KeyError as e:
    print(f"KeyError encountered: {e}")

base_model.to(device)

# def get_model_and_tokenizer():
#     return base_model, tokenizer, device