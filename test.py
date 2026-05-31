# import torch
# from transformers import AutoModelForCausalLM, AutoTokenizer

# device = torch.device("mps")
# model = AutoModelForCausalLM.from_pretrained(
#     "google/gemma-2-2b-it",
#     torch_dtype=torch.float32,
#     low_cpu_mem_usage=True
# ).to(device)

# tokenizer = AutoTokenizer.from_pretrained("google/gemma-2-2b-it")
# inputs = tokenizer("Hello, how are you?", return_tensors="pt").to(device)

# with torch.no_grad():
#     out = model.generate(**inputs, max_new_tokens=20)

# print(tokenizer.decode(out[0], skip_special_tokens=True))

import anthropic

from config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

if not ANTHROPIC_API_KEY:
    raise RuntimeError("ANTHROPIC_API_KEY not found in root .env")

client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)
r = client.messages.create(
    model=ANTHROPIC_MODEL,
    max_tokens=10,
    messages=[{"role": "user", "content": "Say hi"}]
)
print(r.content[0].text)