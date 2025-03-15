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

def calculator(a: float, b: float, operator: str) -> str:
    print('Calc invoked...')
    try:
        if operator == '+':
            return str(a + b)
        elif operator == '-':
            return str(a - b)
        elif operator == '*':
            return str(a * b)
        elif operator == '/':
            if b == 0:
                return 'Error: Division by zero'
            return str(a / b)
        else:
            return 'Error: Invalid operator. Please use +, -, *, or /'
    except Exception as e:
        return f'Error: {str(e)}'
    
def get_current_time():
    """Returns the current time in 12-hour format with AM/PM indicator.
    Returns:
        str: The current time formatted as HH:MM AM/PM
    """
    return datetime.now().strftime("%I:%M %p")

async def main() -> None:
    model_client = OpenAIChatCompletionClient(
        base_url='http://127.0.0.1:1234/v1',
            model='gemma-3-4b-it',
            api_key=os.getenv("OPEN_AI_API_KEY"),
            model_info={
                "vision": False,
                "function_calling": True,
                "json_output": True,
                "family": ModelFamily.GPT_4O,
            }
    )
    agent1 = AssistantAgent("assistant", 
                            model_client=model_client, 
                            tools=[calculator, get_current_time],
                            system_message="""You are a helpful assistant that has the ability to get the current time and perform calulations as function calls"""
                            )
    #agent2 = AssistantAgent("Assistant2", model_client=model_client)
    termination = MaxMessageTermination(5)
    team = RoundRobinGroupChat([agent1], termination_condition=termination)

    stream = team.run_stream(task="If I start with 3 apple and I get 3 new apples, how many apples would I now have?")
    async for message in stream:
        if str(type(message)) == "<class 'autogen_agentchat.messages.ToolCallSummaryMessage'>":
            print(type(message))
            print("--Begin message--")
            print(message.content)
            print("--End message--")

    # # Run the team again without a task to continue the previous task.
    # stream = team.run_stream()
    # async for message in stream:
    #     print(message)

asyncio.run(main())