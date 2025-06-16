from .evaluate import grade_bot_response
import re
from datetime import datetime
import json
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    MessagesPlaceholder,
    HumanMessagePromptTemplate
)
from langchain_core.prompts.prompt import PromptTemplate
from langfuse.callback import CallbackHandler as LangfuseCallbackHandler
from namigen_chat_gpt.workflow.nami_bot import NamiBot

call_backs = [LangfuseCallbackHandler()]
namibot = NamiBot()
async def run_rag_chain(question):
    rag_chain = namibot.get_rag_chain()
    user_input = {
        "question": question,
        "history": ""
    }

    outputs = await rag_chain.ainvoke(
        user_input,
        {"callbacks": call_backs}
    )

    tool_outputs = ""
    for output in outputs:
        if isinstance(output, list):
            if isinstance(output[0], str):
                tool_outputs += "\n".join(output)
            elif isinstance(output[0], dict):
                tool_outputs += "\n".join(
                    [
                        json.dumps(
                            out,
                            indent=4,
                            ensure_ascii=False
                        )
                        for out in output
                    ]
                )
            else:
                raise ValueError("Invalid output type")
        elif isinstance(output, str):
            tool_outputs += output
        elif isinstance(output, dict):
            tool_outputs += json.dumps(
                output,
                indent=4,
                ensure_ascii=False
            )
    return tool_outputs
ending_prompt = """
<Documents>
{context}
</Documents>

<UserInfo>
{user_info}
</UserInfo>

<Start conversation>
{chat_history}
</End conversation>

Current time (yyyy-mm-dd): {now}
New Customer's inquiry: {question}

BIDV: """

async def get_bot_response(
    prompt,
    question
):
    namibot.faq_prompt = ChatPromptTemplate.from_messages(
        [
        SystemMessagePromptTemplate(
            prompt=PromptTemplate(
                template=prompt + ending_prompt,
            )
        ),
    ])

    user_input = {
        "now": datetime.now().strftime("%Y-%m-%d"),
        "question": question,
        "intent": question,
        "chat_history": "",
        # "example": example,
        "context": await run_rag_chain(question),
        "user_info": ""
    }

    qa_chain = namibot.get_qa_chain(is_response=True, bot_type="normal")

    qa_response = await qa_chain.ainvoke(
        user_input,
        config={"callbacks": call_backs},
        include_run_info=True
    )
    # if qa_response.content:
    bot_response = qa_response.content
    
    matches = re.findall(r"<think>(.*?)</think>", bot_response, flags=re.DOTALL)
    think = ""
    for match in matches:
        think += match.strip()
    final_response = re.sub(r"<think>.*?</think>\s*", "", bot_response, flags=re.DOTALL)

    return think, final_response