from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage, SystemMessage
import json

model = ChatOpenAI(
    model="gpt-4o",
    temperature=0
)

evaluation_prompt_without_gt = """You are tasked with evaluating a new generated answer based on a given question, old answer and feedback of this old answer. Your task is described below.
- Check whether the new generated answer solves the problem of the old answer.
- Ignore differences in punctuation and phrasing between the old answer and new generated answer.

Your response must strictly follow the format of a JSON object:
{{
  reason: <string>, # a short explanation of the correctness of the generated answer
  correctness: the evaluation with binary scores (0 for Wrong, 1 for Correct)
  feedback: <string> # a short new feedback for the new generated answer
}}

Question:
```
{question}
```

Old Answer:
```
{old_answer}
```

Feedback:
```
{feedback}
```
Thought Process:
```
{think}
```
New Generated Answer: 
```
{new_response}
```

# JSON Output:"""

evaluation_prompt = """You are tasked with evaluating a generated answer based on a given question and an ground truth answer. Your task is described below.
Correctness:
    - Check whether the facts in generated answer **contradicts** any facts in ground truth answer. You should also lightly penalize omission of detail, and focus on the main idea. Lack of detail is OK, but **contradicting facts** are not.
    - Check whether the generated answer is strictly following the old feedback. All parts of the feedback should be followed in the generated answer.
    - Ignore differences in punctuation and phrasing between the ground truth answer and generated answer.
    - **It is OK if the generated answer contains MORE information than the ground truth answer, as long as it does not contain any conflicting statements.**
    - Assess if there are any discrepancies in values, or information between the generated and ground truth answers.
    - The generated answer do not need to cover all the details in the ground truth answer, but it should not contradict any of them.
    - Assign a corrective score as a float from 1.0 (completely incorrect) to 10.0 (perfectly aligned), indicating how much change is required for the generated answer to meet the ground truth and feedback.
    - More penalty for any conflicting statements and not following any part of the feedback.
Reason: explain why each point is incorrect or contradictory.
Feedback: describe how the answer should be revised to align with the ground truth. If applicable, suggest an improved version of the answer.

Your response must strictly follow the format of a JSON object:
{{
  reason: <string>, # an explanation of the correctness of the generated answer in Vietnamese
  correctness: the evaluation with scores (0 to 10)
  feedback: <string> # a detail feedback for improving the generated answer in Vietnamese
}}

Question:
```
{question}
```

Thought Process:
```
{think}
```

Old Feedback:
```
{feedback}
```

Ground Truth Answer: 
```
{ground_truth_answer}
```

Generated Answer: 
```
{generated_answer}
```

# JSON Output:"""


async def grade_bot_response(question, bot_response, feedback, think, ground_truth):

    prompt = evaluation_prompt.format(
        question=question,
        ground_truth_answer=ground_truth, 
        feedback=feedback,
        think=think,
        generated_answer=bot_response
    )

    messages = [
        SystemMessage(content=prompt)
    ]

    response = model.invoke(messages)
    try:
        response_json = json.loads(response.content)
        correctness = response_json["correctness"]
        reason = response_json["reason"]
        feedback = response_json["feedback"]
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        print(f"Response content: {response.content}")
        correctness = 0
        reason = "Failed to decode JSON response"
        feedback = "Please check the response format"

    return correctness, reason, feedback