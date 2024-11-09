from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


model = AutoModelForCausalLM.from_pretrained("mistralai/Mistral-7B-v0.1" , device_map = 'auto')
tokenizer = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1" , padding_side ="left" )
tokenizer.pad_token = tokenizer.eos_token

sentences = [
    "A list of colors: red, blue",
    "Portugal is a country in Europe.",
    "The sun rises in the east.",
    "What is the capital of France?"
]


model_inputs = tokenizer(sentences, return_tensors="pt").to("cpu")
generated_ids = model.generate(**model_inputs)
tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

tokenizer.pad_token = tokenizer.eos_token  # Most LLMs don't have a pad token by default
model_inputs = tokenizer(
    ["A list of colors: red, blue", "Portugal is"], return_tensors="pt", padding=True
).to("cuda")
generated_ids = model.generate(**model_inputs)
generated_text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)

for idx, sentence in enumerate(sentences):
    print(f"Input: {sentence}")
    print(f"Generated: {generated_text[idx]}")
    print("-" * 50)