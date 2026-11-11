import sys; sys.stdout.reconfigure(encoding="utf-8")

from langchain_core.tools import tool
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from typing import List

# 도구 정의
@tool
def multiply(a: int, b: int) -> int:
    """두 정수를 곱합니다."""
    return a * b

@tool
def add(a: int, b: int) -> int:
    """두 정수를 더합니다."""
    return a + b

def run_tool_loop(
    tools: list,
    ai_message: AIMessage,
    max_iterations: int = 5
) -> List[ToolMessage]:
    """도구 실행 루프: tool_calls를 실행하고 ToolMessage 리스트를 반환합니다.

    Args:
        tools: 사용 가능한 도구 리스트
        ai_message: LLM이 반환한 AIMessage (tool_calls 포함)
        max_iterations: 최대 반복 횟수 (무한 루프 방지)

    Returns:
        실행 결과 ToolMessage 리스트
    """
    tool_map = {t.name: t for t in tools}
    tool_messages = []

    if not ai_message.tool_calls:
        print("도구 호출 없음 - 일반 텍스트 응답")
        return tool_messages

    for i, tc in enumerate(ai_message.tool_calls):
        if i >= max_iterations:
            print(f"최대 반복 횟수({max_iterations}) 도달, 중단")
            break

        tool_name = tc["name"]
        tool_args = tc["args"]
        tool_id = tc["id"]

        if tool_name not in tool_map:
            content = f"Error: '{tool_name}' 도구를 찾을 수 없습니다."
        else:
            try:
                result = tool_map[tool_name].invoke(tool_args)
                content = str(result)
            except Exception as e:
                content = f"Error: {type(e).__name__}: {str(e)}"

        tool_msg = ToolMessage(
            content=content,
            tool_call_id=tool_id,
            name=tool_name
        )
        tool_messages.append(tool_msg)
        print(f"  실행 [{i+1}]: {tool_name}({tool_args}) -> {content}")

    return tool_messages

# === 테스트 ===
tools = [multiply, add]

# 테스트 1: 단일 도구 호출
print("=== 테스트 1: 단일 도구 호출 ===")
ai_msg1 = AIMessage(
    content="",
    tool_calls=[{"name": "multiply", "args": {"a": 7, "b": 8}, "id": "c1", "type": "tool_call"}]
)
results1 = run_tool_loop(tools, ai_msg1)
print(f"  결과 ToolMessage 수: {len(results1)}")

# 테스트 2: 다중 도구 호출
print("\n=== 테스트 2: 다중 도구 호출 ===")
ai_msg2 = AIMessage(
    content="",
    tool_calls=[
        {"name": "multiply", "args": {"a": 3, "b": 4}, "id": "c2", "type": "tool_call"},
        {"name": "add", "args": {"a": 10, "b": 20}, "id": "c3", "type": "tool_call"}
    ]
)
results2 = run_tool_loop(tools, ai_msg2)
print(f"  결과 ToolMessage 수: {len(results2)}")

# 테스트 3: 존재하지 않는 도구
print("\n=== 테스트 3: 존재하지 않는 도구 ===")
ai_msg3 = AIMessage(
    content="",
    tool_calls=[{"name": "unknown_tool", "args": {}, "id": "c4", "type": "tool_call"}]
)
results3 = run_tool_loop(tools, ai_msg3)
print(f"  결과 ToolMessage 수: {len(results3)}")

# 테스트 4: 도구 호출 없는 일반 응답
print("\n=== 테스트 4: 도구 호출 없음 ===")
ai_msg4 = AIMessage(content="안녕하세요!")
results4 = run_tool_loop(tools, ai_msg4)
print(f"  결과 ToolMessage 수: {len(results4)}")
