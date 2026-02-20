import os

from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

token = os.environ['HF_TOKEN']

# model_name = "Qwen/Qwen3-30B-A3B-Instruct-2507"
model_name = "Qwen/Qwen3-8B-FP8"
offload_folder = os.path.join(os.path.dirname(__file__), model_name, ".offload_qwen3")
os.makedirs(offload_folder, exist_ok=True)
config = AutoConfig.from_pretrained(model_name, token=token)
config.tie_word_embeddings = False

# load the tokenizer and the model
try:
    tokenizer = AutoTokenizer.from_pretrained(model_name, token=token)
except Exception as exc:
    # Older tokenizers/transformers builds may fail to parse newer Qwen tokenizer.json.
    if "ModelWrapper" in str(exc):
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False, token=token)
    else:
        raise
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    config=config,
    torch_dtype="auto",
    device_map="auto",
    offload_folder=offload_folder,
    token=token
)

# prepare the model input
prompt = "Give me a short introduction to large language model."
messages = [
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

# conduct text completion
generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512
)
output_ids = generated_ids[0][len(model_inputs.input_ids[0]):].tolist() 

content = tokenizer.decode(output_ids, skip_special_tokens=True)

print("content:", content)
