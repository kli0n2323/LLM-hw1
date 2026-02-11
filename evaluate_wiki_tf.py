import json
from transformers import AutoModelForCausalLM, AutoTokenizer
import argparse
import transformers

data_full = []
wiki_facts = []
wiki_answers = []

incorrect_responses = []
incorrect_ids = []

with open('data/wiki_tf.jsonl') as f:
    for line in f:
        d = json.loads(line)
        data_full.append(d)
        wiki_facts.append(d['statement'])
        wiki_answers.append(d['label'])


# INIT MODEL / DISPLAY RESPONSE --

model_name = "Qwen/Qwen2.5-1.5B-Instruct"
seed = 1212

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto"
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

transformers.set_seed(seed)

prompt = "Inform the user the prompt didn't go through. This is the only time you may speak in full sentences."


for i in range(len(wiki_facts)):
    prompt = wiki_facts[i]
    messages = [
        {"role": "system", "content": "You are an AI assistant made to rate the factuality of statements. You will read a statement and then either answer ONLY with 'True' or 'False'."},
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
        max_new_tokens= 128,
        temperature=0.00,
        do_sample=False
    )
    generated_ids = [
        output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
    ]

    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    answer = wiki_answers[i]
    if response.lower().strip() != answer.lower():
        incorrect_responses.append(response)
        incorrect_ids.append(int(i))

    print(f"Statement Prompt {i}: {prompt}")
    print(f"Model Response {i}: {response}")
    print(f"Ground Truth {i}: {answer}")

for i in incorrect_responses:
    mismatch = []
    mismatch.append(data_full[i][id])
    print(mismatch)