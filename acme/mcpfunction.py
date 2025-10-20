# mcpfunction.py

import asyncio
import os
import shutil

from agents import Agent, Runner, gen_trace_id, trace
from agents.mcp import MCPServer, MCPServerStdio
from dotenv import load_dotenv

load_dotenv()

async def run(mcp_server: MCPServer, message: str):
    agent = Agent(
        name="Assistant",
        instructions="Use the tools to return flight info.",
        mcp_servers=[mcp_server],
    )

    #message = "Find the answer to: " + message
    print(f"\n\nRunning: {message}")
    result = await Runner.run(starting_agent=agent, input=message)
    print(result.final_output)
    return result.final_output

async def run_mcp(message: str):
    # current_dir = os.path.dirname(os.path.abspath(__file__))
    # samples_dir = os.path.join(current_dir, "data")

    if not shutil.which("npx"):
        raise RuntimeError("npx is not installed. Please install it with `npm install -g npx`.")

    async with MCPServerStdio(
        name="Flight Info Bot",
        params={
            "command": "npx",
            "args": ["/Users/billhorn/code/javascript/acme-air-demo", message],
            "cache_tools_list": "True"
        },
    ) as server:
        trace_id = gen_trace_id()
        with trace(workflow_name="MCP Filesystem Example", trace_id=trace_id):
            print(f"View trace: https://platform.openai.com/traces/{trace_id}\n")
            return await run(server, message)
