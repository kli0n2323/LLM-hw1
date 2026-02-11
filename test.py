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

print(data_full)