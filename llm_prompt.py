from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import transformers

# INIT MODEL / DISPLAY RESPONSE --

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
seed = 0

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

transformers.set_seed(seed)

prompt = "Introduce yourself!"
messages = [
    {"role": "system", "content": "You are an AI assistant made to answer questions. Depending on the prompt, you will either answer with 'Yes' or 'No', or 'True' or 'False'."},
    {"role": "user", "content": prompt}
]
text = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True
)
model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

generated_ids = model.generate(
    **model_inputs,
    max_new_tokens=512,
    temperature=0.7,
    do_sample=True
)
generated_ids = [
    output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
]

response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
print(response)


#  CLI PARSER ---

parser = argparse.ArgumentParser()
parser.add_argument(
    "-m",
    "--model",
    default = "Qwen/Qwen2.5-1.5B-Instruct",
    type=str,
    help ="Set model",
)
parser.add_argument(
    "-p",
    "--prompt",
    type = str,
)
parser.add_argument(
    "--max_new_tokens",
    type = int,
    default = 128
)
parser.add_argument(
    "-t",
    "--temperature",
    type=float,
    default = 0.7
)
parser.add_argument(
    "-s",
    "--seed",
    type=int,
    default = 0
)
args = parser.parse_args()

if args.model:
    model_name = args.model
if args.prompt:
    prompt = args.prompt
if args.max_new_tokens:
    max_new_tokens = args.max_new_tokens
if args.temperature:
    temperature = args.temperature
if args.seed:
    seed = args.seed
