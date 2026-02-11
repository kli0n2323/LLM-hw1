HOW TO RUN EACH SCRIPT:
`python -m script_name.py`

CHOSEN MODEL:
`"Qwen/Qwen2.5-1.5B-Instruct"`
-- Chosen for small parameter size (1.5b) and instruction tuning.

WIKI FACTS ACCURACY:
-- `80%` (8 / 10 correct across 10 runs)
BOOLQ ACCURACY (with passages):
-- 
BOOLQ ACCURACY (without passages):
--

REPORT:
These experiments evaluated a small LLM (Qwen 2.5 Instruct) in its ability to reason about the factuality of given statements/questions. By running multiple experiments where the LLM is given varying levels of context (none in pt2 and pt3.2 vs. given passage in pt3.1), we examine how in-prompt context affects the LLM's ability to accurately label a statement as true or false. 