from langmem import Prompt
from loguru import logger
from .core import get_bot_response
from .evaluate import grade_bot_response


async def optimize_prompt(
    optimizer,
    prompt,
    question,
    bot_response,
    feedback,
    config=None
):
    max_reflection_steps = config.get("max_reflection_steps", 3)

    better_prompt = prompt
    correctness = 0
    for i in range(max_reflection_steps):
        if correctness > 7:
            logger.debug("Better prompt found, stopping optimization.")
            better_prompt = improved_prompt
            break
        logger.debug(
            f"Reflection step {i + 1}/{max_reflection_steps}\n"
            f"Question: {question}\n"
            f"Current response: {bot_response}\n"
            f"Feedback: {feedback}\n"
        )
        logger.debug("Starting optimization process...")
        # Complex conversation that needs better structure
        conversation = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": bot_response},
        ]
        trajectories = [(conversation, feedback)]
        optimizer_prompt = Prompt(
            name="chatbot_optimization",
            prompt=prompt,
            # update_instructions="ONLY make a minimal changes, only address where"
            # " errors have occurred after reasoning over why they occur. Try not to use examples.",
            update_instructions="Make a HIGH impact change to the prompt, do whatever you think is necessary to improve the performance of the chatbot."
            " Make the response strictly follow the feedback and the groundtruth provided."
            " No need to keep the original instructions or the structure of the prompt."
            " Address the errors that have occurred after reasoning over why they occur."
            " Try not to use examples.",
            # when_to_update="If there seem to be errors in recall of named entities.",
        )

        improved_prompt = await optimizer(
            trajectories=trajectories,
            prompt=optimizer_prompt
        )

        logger.debug("End of optimization process. Starting evaluation...")
        think, new_response = await get_bot_response(
            prompt=improved_prompt,
            question=question
        )

        logger.debug(
            f"Thought process: {think}\n"
            f"New response from bot: {new_response}"
        )

        correctness, reason, feedback = await grade_bot_response(
            question,
            old_response=bot_response,
            feedback=feedback,
            think=think,
            new_response=new_response
        )
        logger.debug(
            f"Evaluation result: {correctness}\n"
            f"Reason: {reason}\n"
            f"Feedback: {feedback}"
        )
    if correctness <=7:
        logger.debug("No better prompt found, using original.")
        better_prompt = prompt
    
    logger.debug("Writing better prompt to file: improved_prompt.txt")
    with open("improved_prompt.txt", "w", encoding="utf-8") as f:
        f.write(better_prompt)
    return better_prompt
