"""
按照下openai agents框架，实现一个应用需求：用户输入文本，判断下文本的语种（是什么语言的）
然后如果是非中文的，则翻译为中文；如果是中文，则翻译为英文。最终输出格式需要为如下json：
{
    "query_input": "",
    "language": "",
    "result": ""
}
"""
import asyncio
import json
import os
import sys

from openai import AsyncOpenAI
from pydantic import BaseModel

# 从环境变量读取 Key，不要把密钥写死在代码里
API_KEY = os.getenv("DASHSCOPE_API_KEY") or os.getenv("OPENAI_API_KEY")
if not API_KEY:
    raise SystemExit("请先设置环境变量 DASHSCOPE_API_KEY（或 OPENAI_API_KEY）")
BASE_URL = os.getenv("OPENAI_BASE_URL", "https://dashscope.aliyuncs.com/compatible-mode/v1")

os.environ["OPENAI_API_KEY"] = API_KEY
os.environ["OPENAI_BASE_URL"] = BASE_URL

from agents import Agent, Runner, SQLiteSession, OpenAIChatCompletionsModel
from agents import set_default_openai_api, set_tracing_disabled

set_default_openai_api("chat_completions")
set_tracing_disabled(True)


class TranslationResult(BaseModel):
    """智能体最终返回的结构，保证输出格式固定为题目要求的 JSON。"""

    query_input: str
    language: str
    result: str


# 使用兼容 OpenAI 协议的 DashScope 客户端
client = AsyncOpenAI(
    api_key=os.environ["OPENAI_API_KEY"],
    base_url=os.environ["OPENAI_BASE_URL"],
)

model = OpenAIChatCompletionsModel(model="qwen-plus", openai_client=client)

translator_agent = Agent(
    name="语种识别与翻译助手",
    instructions=(
        "你是一个语种识别与翻译助手。用户会输入一段文本，请你完成：\n"
        "1. 识别这段文本的语种，language 字段填写该语言的中文名称（如“英语”“日语”“中文”）；\n"
        "2. 如果文本不是中文，将其翻译成中文；如果文本是中文，将其翻译成英文；\n"
        "3. query_input 字段原样保留用户输入的文本，result 字段填写翻译结果。\n"
        "翻译要准确、通顺，不要添加任何解释性内容。"
    ),
    model=model,
    output_type=TranslationResult,
)

session = SQLiteSession("conversation_1")


async def main() -> None:
    # 支持命令行参数（方便测试），否则交互式输入
    query_input = " ".join(sys.argv[1:]).strip()
    if not query_input:
        query_input = input("请输入文本：").strip()
    if not query_input:
        print("输入不能为空")
        return

    result = await Runner.run(translator_agent, query_input, session=session)
    output: TranslationResult = result.final_output

    print(json.dumps(output.model_dump(), ensure_ascii=False, indent=4))


if __name__ == "__main__":
    asyncio.run(main())
