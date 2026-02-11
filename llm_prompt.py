from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import transformers

# ---- CLI PARSER ----
parser = argparse.ArgumentParser()

parser.add_argument(
    "-m",
    "--model",
    default="Qwen/Qwen2.5-1.5B-Instruct",
    type=str,
    help="Hugging Face model name or path",
)
parser.add_argument(
    "-p",
    "--prompt",
    type=str,
    required=True,
    help="Prompt to send to the model",
)
parser.add_argument(
    "--max_new_tokens",
    type=int,
    default=128,
    help="Maximum number of tokens to generate",
)
parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    default=0.7,
    help="Sampling temperature (0.0 = deterministic)",
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default=0,
    help="Random seed",
)

args = parser.parse_args()

# ---- INIT MODEL / DISPLAY RESPONSE ----
transformers.set_seed(args.seed)

model = AutoModelForCausalLM.from_pretrained(
    args.model,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(args.model)

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": args.prompt},
]

text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

do_sample = args.temperature > 0.0

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=args.max_new_tokens,
    temperature=args.temperature,
    do_sample=do_sample,
)

generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)
