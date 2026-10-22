"""
MCP Client + OpenAI Agent：连接 task2_server.py，通过 Agent 调用 MCP 工具。

运行方式：
    python task2_client.py          # 默认 stdio：客户端自动以子进程方式启动 task2_server.py
    python task2_server.py --sse    # 另开一个终端先启动 SSE 服务
    python task2_client.py --sse    # 通过 http://127.0.0.1:8000/sse 连接
"""
import asyncio
import logging
import os
import sys

# 从环境变量读取 Key，不要把密钥写死在代码里
API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("请先设置环境变量 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = BASE_URL

from agents import Agent, Runner, AsyncOpenAI, OpenAIChatCompletionsModel
from agents import set_default_openai_api, set_tracing_disabled
from agents.mcp import MCPServerSse, MCPServerStdio

SERVER_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "task2_server.py")

MESSAGES = [
    # 一次调用触发 3 个新增工具
    "请帮我做三件事：1）现在北京时间几点？2）计算 (23 + 17) * 3 / 4；"
    "3）统计这段文本的字符数和中英文词数：Hello world，你好世界！",
    # 复用原有工具
    "摇 3 次骰子，并告诉我最新新闻。",
]


async def run(mcp_server) -> None:
    external_client = AsyncOpenAI(
        api_key=os.environ["OPENAI_API_KEY"],
        base_url=os.environ["OPENAI_BASE_URL"],
    )

    agent = Agent(
        name="Assistant",
        instructions=(
            "你是一个可以调用 MCP 工具的助手。请根据用户的问题选择合适的工具，"
            "必要时可以连续调用多个工具，最后用中文简洁地总结结果。"
        ),
        mcp_servers=[mcp_server],
        model=OpenAIChatCompletionsModel(
            model="qwen-max",
            openai_client=external_client,
        ),
    )

    for message in MESSAGES:
        print(f"\nRunning: {message}")
        result = await Runner.run(starting_agent=agent, input=message)
        print(result.final_output)


async def main() -> None:
    set_default_openai_api("chat_completions")
    set_tracing_disabled(True)
    logging.getLogger("httpx").setLevel(logging.WARNING)

    if "--sse" in sys.argv:
        # SSE：需要先在另一个终端运行 python task2_server.py --sse
        server = MCPServerSse(
            name="Homework MCP Server (SSE)",
            params={"url": "http://127.0.0.1:8000/sse"},
        )
    else:
        # stdio：客户端把 server 脚本作为子进程启动，标准输入/输出即通信通道
        server = MCPServerStdio(
            name="Homework MCP Server (stdio)",
            params={
                "command": sys.executable,
                "args": [SERVER_PATH],
                "cwd": os.path.dirname(SERVER_PATH),
            },
        )

    async with server:
        print("已连接 MCP Server，可用工具：")
        for tool in await server.list_tools():
            print(f"  - {tool.name}: {tool.description.splitlines()[0]}")
        await run(server)


if __name__ == "__main__":
    asyncio.run(main())
