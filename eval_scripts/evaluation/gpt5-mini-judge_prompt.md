JUDGE_PROMPT_TEMPLATE = """You are a strict answer evaluator.
You will receive:
1. A problem description.
2. The correct answer (ground truth) given as a label or exact value.
3. The model's predicted answer
Your task:
- Decide whether the model's predicted answer is equivalent to the correct answer.
- For mathematical expressions, check equivalence:
    * Fractions can be equivalent if numerator/denominator are scaled equally.
    * Expressions like 3/5, \\frac{{3}}{{5}}, 0.6 are considered equivalent.
    * For symbolic answers, treat trivial reorderings as equal if mathematically identical.
    * Ignore obvious format differences (spaces, LaTeX vs plain text).
- For multiple-choice questions, check if the model's choice letter matches the ground truth letter, OR if its value matches the correct option's value.
- Be careful: do NOT mark as correct if the value is only similar in magnitude but not exactly the same mathematically.
- If no valid answer is present, mark "Incorrect".
You must reply with only one word: "Correct" or "Incorrect".
Question:
{question}
Standard Answer:
{gdt}
Model Answer:
{box_answer}
"""