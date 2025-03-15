import asyncio
import os
import pandas as pd
from langchain_experimental.tools.python.tool import PythonAstREPLTool
from autogen_ext.tools.langchain import LangChainToolAdapter
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_agentchat.messages import TextMessage
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.ui import Console
from autogen_core import CancellationToken
from dotenv import load_dotenv
from autogen_core.models import ModelFamily

load_dotenv()

async def main() -> None:
    df = pd.read_csv("titanic2.csv")  # type: ignore
    tool = LangChainToolAdapter(PythonAstREPLTool(locals={"df": df}))
    model_client = OpenAIChatCompletionClient(
        #base_url='http://127.0.0.1:1234/v1', # omit for OpenAI calls
        model='gpt-4o-mini',
        api_key=os.getenv("OPEN_AI_API_KEY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.GPT_4O,
        }
    )
    agent = AssistantAgent(
        "assistant",
        tools=[tool],
        model_client=model_client,
        system_message="Use the `df` variable to access the dataset.",
    )
    await Console(
        agent.on_messages_stream(
            [TextMessage(content="Which passengers have a Description that includes boxing?", source="user")], CancellationToken()
        )
    )


asyncio.run(main())
