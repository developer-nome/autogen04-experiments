import asyncio
import os
from autogen_ext.models.openai import OpenAIChatCompletionClient
from autogen_ext.tools.mcp import StdioServerParams, mcp_server_tools
from autogen_agentchat.agents import AssistantAgent
from autogen_agentchat.teams import RoundRobinGroupChat, SelectorGroupChat, Swarm
from autogen_agentchat.conditions import MaxMessageTermination, TextMentionTermination
from autogen_core.models import ModelFamily, ModelInfo
from autogen_core import CancellationToken
from autogen_agentchat.ui import Console
from dotenv import load_dotenv

load_dotenv()

async def main() -> None:
    samples_dir = "/Users/billhorn/code/python/autogen004/sample_files"

    # fetch_mcp_server = StdioServerParams(command="npx", args=["-y", "@modelcontextprotocol/server-filesystem@2025.3.28", "/Users/billhorn/code/python/autogen004/sample_files"])
    fetch_mcp_server = StdioServerParams(
        command="docker", args=[
            "run",
            "-i",
            "--rm",
            "--mount", f"type=bind,source={samples_dir},target=/samples_files",
            "mcp/filesystem",
            "/samples_files"
        ]
    )
    filesystem_tool = await mcp_server_tools(fetch_mcp_server)

    model_client = OpenAIChatCompletionClient(
        # base_url='http://127.0.0.1:11434/v1', # omit for OpenAI calls
        # NOTE 4.1 nano did not work with the MCP server-filesystem tool
        # gpt-4.1-2025-04-14     = $2.00 / $ 8.00
        # gpt-5-2025-08-07       = $1.25 / $10.00
        # gpt-5-mini-2025-08-07  = $0.25 / $ 2.00
        model='gpt-4.1-2025-04-14',
        api_key=os.getenv("OPEN_AI_API_KEY"),
        model_info={
            "vision": False,
            "function_calling": True,
            "json_output": True,
            "family": ModelFamily.GPT_4O,
        }
    )

    assistant_agent = AssistantAgent(
        name="assistant_agent",
        model_client=model_client,
        handoffs=["filesystem_tool_expert"],
        system_message="""You are an organization assistant agent.
        The filesystem_tool_expert is responsible for managing file system operations.
        If you need to get information about the file system, you can ask the filesystem_tool_expert,
        your last response should have only the relevant answer.
        Use TERMINATE when the task is complete.
        """
    )

    filesystem_tool_expert = AssistantAgent(
        name="filesystem_tool_expert",
        model_client=model_client,
        tools=filesystem_tool,
        system_message="""You are an agent specialized in file system operations.
        You use the filesystem tool to retrieve and. manage files.
        When the transaction is complete, handoff to the assistant_agent to finalize the response.
        """
    )

    termination = MaxMessageTermination(
        max_messages=7) | TextMentionTermination("TERMINATE")

    team = Swarm([assistant_agent, filesystem_tool_expert], termination_condition=termination)

    last_content = "Incomplete or error response"
    try:
        result = await team.run(task="List all files in the sample_files directory", cancellation_token=CancellationToken())

        for msg in reversed(result.messages):
            if hasattr(msg, "content"):
                last_content = msg.content
                break
        print(last_content)
    except AssertionError as e:
        print(f"AssertionError: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")

if __name__ == "__main__":
    asyncio.run(main())
