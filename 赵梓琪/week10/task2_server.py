"""
MCP Server：在原示例 roll_dice、get_news 的基础上，增加 3 个工具：

1. get_current_time —— 获取指定时区的当前时间
2. calculate        —— 安全地计算数学表达式
3. text_statistics  —— 统计文本的字符数、中文字数和英文单词数

运行方式：
    python task2_server.py          # stdio 模式（由 task2_client.py 自动拉起）
    python task2_server.py --sse    # SSE 模式，监听 http://127.0.0.1:8000/sse
"""
import ast
import operator
import random
import re
import sys
from datetime import datetime
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

# 用 mcp 官方 SDK 内置的 FastMCP（与当前环境 mcp 1.30.0 匹配；
# 环境中 fastmcp 4.0.3 与 mcp 1.30.0 不兼容，原 server.py 也会因此导入失败）
from mcp.server.fastmcp import FastMCP

mcp = FastMCP(name="Homework Tools", host="127.0.0.1", port=8000, log_level="WARNING")


@mcp.tool()
def roll_dice(n_dice: int) -> list[int]:
    """Roll `n_dice` 6-sided dice and return the results."""
    return [random.randint(1, 6) for _ in range(n_dice)]


@mcp.tool()
def get_news() -> str:
    """Get latest news."""
    return "最新消息 Qwen4 is coming!"


# ----------------------------- 新增工具 1 -----------------------------
@mcp.tool()
def get_current_time(timezone: str = "Asia/Shanghai") -> str:
    """获取指定时区的当前时间。

    Args:
        timezone: IANA 时区名，例如 "Asia/Shanghai"、"Asia/Tokyo"、"UTC"。
    """
    try:
        tz = ZoneInfo(timezone)
    except (ZoneInfoNotFoundError, ValueError):
        return f"未知时区：{timezone}，请使用 IANA 时区名，例如 Asia/Shanghai"
    return datetime.now(tz).strftime("%Y-%m-%d %H:%M:%S %Z")


# ----------------------------- 新增工具 2 -----------------------------
_BIN_OPS = {
    ast.Add: operator.add,
    ast.Sub: operator.sub,
    ast.Mult: operator.mul,
    ast.Div: operator.truediv,
    ast.FloorDiv: operator.floordiv,
    ast.Mod: operator.mod,
    ast.Pow: operator.pow,
}
_UNARY_OPS = {
    ast.UAdd: operator.pos,
    ast.USub: operator.neg,
}


def _safe_eval(node: ast.AST) -> float:
    """只允许数字、四则运算和括号，避免 eval 执行任意代码。"""
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _BIN_OPS:
        return _BIN_OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _UNARY_OPS:
        return _UNARY_OPS[type(node.op)](_safe_eval(node.operand))
    raise ValueError("表达式只支持数字、+ - * / // % ** 和括号")


@mcp.tool()
def calculate(expression: str) -> str:
    """计算数学表达式并返回结果。

    Args:
        expression: 例如 "(23 + 17) * 3 / 4"。
    """
    try:
        value = _safe_eval(ast.parse(expression, mode="eval").body)
    except (SyntaxError, ValueError, ZeroDivisionError, TypeError) as e:
        return f"计算失败：{e}"
    return f"{expression.strip()} = {value}"


# ----------------------------- 新增工具 3 -----------------------------
@mcp.tool()
def text_statistics(text: str) -> dict[str, int]:
    """统计文本的基本信息。

    Args:
        text: 需要统计的文本。

    Returns:
        总字符数、中文字符数、英文单词数。
    """
    chinese_chars = sum(1 for ch in text if "一" <= ch <= "鿿")
    english_words = len(re.findall(r"[A-Za-z]+(?:['-][A-Za-z]+)*", text))
    return {
        "total_chars": len(text),
        "chinese_chars": chinese_chars,
        "english_words": english_words,
    }


if __name__ == "__main__":
    if "--sse" in sys.argv:
        print("MCP Server (SSE) 启动中：http://127.0.0.1:8000/sse")
        mcp.run(transport="sse")
    else:
        # stdio 模式：标准输入/输出就是与客户端通信的通道
        mcp.run(transport="stdio")
