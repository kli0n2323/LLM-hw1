import json
import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer

# ---- LOAD JSONL ----
data_full = []
with open("data/wiki_tf.jsonl", "r", encoding="utf-8") as f:
    for line in f:
        line = line.strip()
        if not line:
            continue
        data_full.append(json.loads(line))

# ---- INIT MODEL ----
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
seed = 1212
max_new_tokens = 128

transformers.set_seed(seed)

model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype="auto",
    device_map="auto",
)
tokenizer = AutoTokenizer.from_pretrained(model_name)

def extract_true_false(text: str):
    t = text.lower()
    i_true = t.find("true")
    i_false = t.find("false")

    if i_true == -1 and i_false == -1:
        return None
    if i_true == -1:
        return "false"
    if i_false == -1:
        return "true"
    return "true" if i_true < i_false else "false"

# ---- GENERATE + EVAL ----
incorrect_examples = []
correct_count = 0

for i, entry in enumerate(data_full):
    prompt = entry["statement"]
    ground_truth = entry["label"].lower()

    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant made to rate the factuality of statements. "
                "Read the statement and answer ONLY with 'true' or 'false'."
            ),
        },
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

    generated_ids = model.generate(
        **model_inputs,
        max_new_tokens=max_new_tokens,
        temperature=0.0,
        do_sample=False,
    )
    generated_ids = generated_ids[:, model_inputs.input_ids.shape[1]:]
    response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

    pred = extract_true_false(response)
    is_incorrect = (pred is None) or (pred != ground_truth)

    if not is_incorrect:
        correct_count += 1

    if is_incorrect and len(incorrect_examples) < 5:
        incorrect_examples.append({
            "id": entry["id"],
            "statement": entry["statement"],
            "label": entry["label"],
            "model_response": response,
            "prediction": pred,
        })

    #print(f"Statement Prompt {i}: {prompt}")
    #print(f"Model Response {i}: {response}")
    #print(f"Ground Truth {i}: {ground_truth}")
    #print("")

accuracy_score = (correct_count / len(data_full)) * 100

print(f"Accuracy: {correct_count} / {len(data_full)} correct | {accuracy_score} %")
print("Incorrect examples:")
print(incorrect_examples)
