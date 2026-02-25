import os, torch
from transformers import AutoConfig, AutoModel, AutoTokenizer

from generate import generate

token = os.getenv("HF_TOKEN")

# model_name = "GSAI-ML/LLaDA-8B-Instruct"
model_name = "FunAGI/LLaDA-8B-Base-gptqmodel-4bit"
device = "cuda" if torch.cuda.is_available() else "cpu"

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True, token=token)
model = AutoModel.from_pretrained(
    model_name, trust_remote_code=True, torch_dtype=torch.bfloat16, token=token
).to(device)
model.eval()

if tokenizer.padding_side != "left":
    tokenizer.padding_side = "left"
if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

assert tokenizer.pad_token_id != 126336

# Prepare model input
prompt = "Give me a short introduction to diffusion language models."
messages = [{"role": "user", "content": prompt}]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
encoded = tokenizer(
    [text],
    add_special_tokens=False,
    padding=True,
    return_tensors="pt",
)
input_ids = encoded["input_ids"].to(device)
attention_mask = encoded["attention_mask"].to(device)

# Conduct text generation
generated_ids = generate(
    model,
    input_ids,
    attention_mask,
    steps=64,
    gen_length=64,
    block_length=64,
    temperature=0.0,
    cfg_scale=0.0,
    remasking="low_confidence",
)
output_ids = generated_ids[:, input_ids.shape[1]:]

content = tokenizer.decode(output_ids[0], skip_special_tokens=True)

print("content:", content)
