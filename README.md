HOW TO RUN EACH SCRIPT:
`python -m script_name.py`

CHOSEN MODEL:
`"Qwen/Qwen2.5-1.5B-Instruct"`
-- Chosen for small parameter size (1.5b) and instruction tuning.

WIKI FACTS ACCURACY:
-- `80%` (8 / 10 correct across 10 runs)
BOOLQ ACCURACY (with passages):
-- `54.6%` (avg. 42 / 75 correct across 3 runs, n = 75)
BOOLQ ACCURACY (without passages):
-- `56%` (avg. 43 / 75 correct across 3 runs, n = 75)

REPORT:
These experiments evaluated a small LLM (Qwen 2.5 Instruct) in its ability to reason about the factuality of given statements/questions. By running multiple experiments where the LLM is given varying levels of context (none in pt2 and pt3.2 vs. given passage in pt3.1), we examine how in-prompt context affects the LLM's ability to accurately label a statement as true or false. The LLM could predict the factuality of the Wikipedia statements 8 out of 10 times across 10 runs with no in-prompt context. For the BOOLQ tests, it answered the questions without in-prompt context correcly ~54.6% of the time, and with the in-prompt context passage, ~56% of the time on average. This was between three runs each of 75 total prompts. I was expecting the score boost from the in-prompt context to be much higher, but it seems like the model struggles to pull from this context when it isn't extremely explicit. I believe that the main limitation is the size of the model, and a larger model would see a much larger boost in score.