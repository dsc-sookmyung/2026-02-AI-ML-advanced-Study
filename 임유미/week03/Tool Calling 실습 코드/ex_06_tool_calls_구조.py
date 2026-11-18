from langchain_core.messages import AIMessage, ToolCall

# tool_calls 구조를 직접 생성하여 이해
# (실제로는 model_with_tools.invoke()의 AIMessage에서 가져옵니다)

# 단일 도구 호출 예시
single_call = AIMessage(
    content="",  # 도구 호출 시 content는 빈 문자열
    tool_calls=[
        {
            "name": "multiply",
            "args": {"a": 3, "b": 12},
            "id": "call_abc123",
            "type": "tool_call"
        }
    ]
)

print("=== 단일 도구 호출 ===")
print(f"content: '{single_call.content}' (빈 문자열)")
print(f"tool_calls 수: {len(single_call.tool_calls)}")
for tc in single_call.tool_calls:
    print(f"  도구 이름: {tc['name']}")
    print(f"  인자: {tc['args']}")
    print(f"  호출 ID: {tc['id']}")

# 다중 도구 호출 예시
multi_call = AIMessage(
    content="",
    tool_calls=[
        {
            "name": "get_weather",
            "args": {"city": "서울"},
            "id": "call_def456",
            "type": "tool_call"
        },
        {
            "name": "get_weather",
            "args": {"city": "부산"},
            "id": "call_ghi789",
            "type": "tool_call"
        }
    ]
)

print("\n=== 다중 도구 호출 ===")
print(f"tool_calls 수: {len(multi_call.tool_calls)}")
for i, tc in enumerate(multi_call.tool_calls, 1):
    print(f"  [{i}] {tc['name']}({tc['args']}) - ID: {tc['id']}")

# tool_calls가 없는 경우 (일반 텍스트 응답)
normal_response = AIMessage(content="안녕하세요! 무엇을 도와드릴까요?")
print(f"\n=== 일반 응답 ===")
print(f"content: '{normal_response.content}'")
print(f"tool_calls: {normal_response.tool_calls} (빈 리스트)")
