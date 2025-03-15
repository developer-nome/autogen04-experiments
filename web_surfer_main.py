import asyncio
import os
from autogen_agentchat.agents import AssistantAgent, UserProxyAgent
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_agentchat.teams import RoundRobinGroupChat
from autogen_agentchat.ui import Console
from autogen_core.models import ModelFamily, ModelInfo
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.agents.web_surfer import MultimodalWebSurfer
from datetime import datetime
from dotenv import load_dotenv

load_dotenv()

async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        base_url='http://127.0.0.1:1234/v1', # omit for OpenAI calls
        model='gemma-3-4b-it',
        api_key=os.getenv("OPEN_AI_API_KEY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.GPT_4O,
        }
    )
    # Find information about the 2025 St. Patricks Day Parade in St. Louis, MO, and write a short summary
    assistant = AssistantAgent("assistant", model_client, system_message="")
    web_surfer = MultimodalWebSurfer("web_surfer", model_client)
    user_proxy = UserProxyAgent("user_proxy")
    #termination = TextMentionTermination("exit") # Type 'exit' to end the conversation.
    termination =  MaxMessageTermination(5) | TextMentionTermination("TERMINATE")
    team = RoundRobinGroupChat([web_surfer, assistant], termination_condition=termination)
    # await Console(team.run_stream(task="Find information about current weather in St. Charles, MO, and write a short summary."))
    stream = team.run_stream(task="Find information about hiking trails near St. Charles, MO, and write a short summary.")
    async for message in stream:
        if str(type(message)) == "<class 'autogen_agentchat.messages.TextMessage'>":
            print(type(message))
            print("--Begin message--")
            print(message.content)
            print("--End message--")

asyncio.run(main())

