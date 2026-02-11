import transformers
from transformers import AutoModelForCausalLM, AutoTokenizer
import datasets

# ---- SETTINGS ----
model_name = "Qwen/Qwen2.5-1.5B-Instruct"
seed = 1212
n = 75  # subset size
max_new_tokens = 128

transformers.set_seed(seed)

# ---- LOAD BOOLQ SUBSET ----
boolq = datasets.load_dataset("boolq")
boolq_subset = boolq["train"].shuffle(seed=seed).select(range(n))

# ---- INIT MODEL ----
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

# =========================
# EXPERIMENT A: WITH PASSAGE
# =========================
incorrect_responses = []
incorrect_examples = []
correct_count = 0

for i, entry in enumerate(boolq_subset):
    prompt = f"Question: {entry['question']}? | Passage: {entry['passage']}"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant made to answer questions about a literary passage. "
                "You will read a passage and then either answer ONLY with 'true' or 'false'."
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
    ground_truth = "true" if entry["answer"] else "false"

    is_incorrect = (pred is None) or (pred != ground_truth)

    if not is_incorrect:
        correct_count += 1

    if is_incorrect and len(incorrect_examples) < 5:
        incorrect_examples.append({
            "id": i,
            "question": entry["question"],
            "ground_truth": ground_truth,
            "model_output": response,
            "parsed_pred": pred,
        })

    #print(f"Question Prompt {i}: {prompt}")
    #print(f"Model Response {i}: {response}")
    #print(f"Ground Truth {i}: {ground_truth}")
    #print("")
accuracy_score = (correct_count / n) * 100

print("INCORRECT RESPONSES WITH PASSAGES:")
print(incorrect_examples)
print(f"Accuracy: {correct_count} / {n} correct | ({accuracy_score} %)")
print("")

# ============================
# EXPERIMENT B: WITHOUT PASSAGE
# ============================
incorrect_examples = []
correct_count = 0

for i, entry in enumerate(boolq_subset):
    prompt = f"Question: {entry['question']}?"
    messages = [
        {
            "role": "system",
            "content": (
                "You are an AI assistant. Answer the question and either answer ONLY with 'true' or 'false'."
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
    ground_truth = "true" if entry["answer"] else "false"

    is_incorrect = (pred is None) or (pred != ground_truth)

    if not is_incorrect:
        correct_count += 1

    if is_incorrect and len(incorrect_examples) < 5:
        incorrect_examples.append({
            "id": i,
            "question": entry["question"],
            "ground_truth": ground_truth,
            "model_output": response,
            "prediction": pred,
        })

    #print(f"Passage Prompt {i}: {prompt}")
    #print(f"Model Response {i}: {response}")
    #print(f"Ground Truth {i}: {ground_truth}")
    #print("")

accuracy_score = (correct_count / n) * 100

print("INCORRECT RESPONSES WITHOUT PASSAGES:")
print(incorrect_examples)
print(f"Accuracy: {correct_count} / {n} correct | ({accuracy_score} %)")
